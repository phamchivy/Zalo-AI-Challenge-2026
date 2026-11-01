"""Kernel Python giữ trạng thái cho agent nhiều bước.

Mỗi shop có vài tiến trình con (worker) đã nạp sẵn dữ liệu (pack pickle) MỘT lần. Mỗi lời giải của agent là một
"session" gắn với một worker: biến tạo ở bước trước còn dùng được ở bước sau (như notebook), không phải nạp lại dữ liệu.

An toàn:
  - mỗi lệnh có timeout (signal.alarm trong worker); worker treo / chết -> giết, khởi động lại, session báo lỗi
  - giới hạn RAM mỗi worker (RLIMIT_AS)
  - DataFrame dùng chung giữa các session nhưng pandas >= 3 luôn copy-on-write và mỗi lần truy cập `canon[...]`,
    `tables[...]` trả bản sao nông -> code của session này sửa bảng không ảnh hưởng session khác
Code được cấp: pd, np, re, math, json, date, datetime, timedelta, tables, canon, meta, fold, to_num, parse_date,
month_end, round_money, sql(query) (DuckDB trên các bảng canon / tables), answer(type, value, reason).
"""
from __future__ import annotations

import asyncio
import io
import multiprocessing as mp
import os
import time
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from datamate.core.log import get_logger

log = get_logger('kernel')
ANSWER_TYPES = {'number', 'percent', 'string', 'date', 'boolean', 'list', 'table', 'unanswerable', 'abstain'}


@dataclass
class CellResult:
    status: str                      # ok | error | timeout | crash
    stdout: str = ''
    error: Optional[str] = None
    answer: Optional[dict] = None    # {'type', 'value', 'reason'} nếu code gọi answer()
    elapsed: float = 0.0


# ----------------------------------------------------------------------------- worker (tiến trình con)
class _Frames(dict):
    """dict DataFrame: mỗi lần lấy trả bản sao nông (copy-on-write) để session không sửa dữ liệu dùng chung."""

    def __getitem__(self, k):
        v = super().__getitem__(k)
        try:
            return v.copy(deep=False)
        except AttributeError:
            return v

    def get(self, k, default=None):
        return self[k] if k in self else default

    def items(self):
        return [(k, self[k]) for k in self.keys()]

    def values(self):
        return [self[k] for k in self.keys()]


def _worker_main(conn, pack_path: str, mem_gb: float, out_limit: int):   # pragma: no cover - chạy trong tiến trình con
    import json
    import math
    import pickle
    import re
    import signal
    import sys
    from contextlib import redirect_stdout
    from datetime import date, datetime, timedelta
    # Máy nhiều nhân (96): numpy/OpenBLAS/DuckDB tự mở 1 luồng mỗi nhân, mỗi luồng giữ vài chục MB bộ nhớ ảo
    # -> vượt RLIMIT_AS ngay khi import ('cannot allocate memory for thread-local data'). Worker chỉ cần 1–2 luồng.
    for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[_k] = '1'
    os.environ['MALLOC_ARENA_MAX'] = '2'
    try:
        import resource
        lim = int(mem_gb * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
    except Exception:  # noqa: BLE001
        pass
    import numpy as np
    import pandas as pd
    pd.set_option('display.width', 200)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.max_colwidth', 40)
    # pandas 3 (Copy-on-Write): `df[c].fillna(..., inplace=True)` không sửa df, chỉ in cảnh báo ra stderr (agent không
    # thấy) -> kết quả sai âm thầm. Biến thành lỗi để agent đọc được trong kết quả chạy và viết lại.
    import warnings
    _cae = getattr(pd.errors, 'ChainedAssignmentError', None)
    if _cae is not None:
        warnings.simplefilter('error', _cae)
    from datamate.core.textutil import fold, month_end, parse_date, to_num
    from datamate.reasoning.executor import round_money
    with open(pack_path, 'rb') as f:
        pack = pickle.load(f)
    tables = _Frames({k: v for k, v in (pack.get('tables') or {}).items() if v is not None})
    canon = _Frames({k: v for k, v in (pack.get('canon') or {}).items() if v is not None})
    meta = dict(pack.get('meta') or {})

    def _sql_factory():
        try:
            import duckdb
        except Exception:  # noqa: BLE001
            def nosql(q):
                raise RuntimeError('duckdb chưa cài: dùng pandas')
            return nosql
        con = duckdb.connect(config={'threads': 2})
        registered = set()

        names = {}
        for name in list(canon.keys()) + list(tables.keys()):
            names.setdefault(re.sub(r'\W+', '_', name).strip('_').lower(), name)

        def sql(query: str):
            """Chạy SQL (DuckDB) trên các bảng canon (lines, orders, products, ...) và bảng gốc (tên trong tables,
            ký tự đặc biệt đổi thành '_')."""
            for tok in set(re.findall(r'[A-Za-z_][\w]*', query)):
                key = tok.lower()
                if key in names and key not in registered:
                    src = names[key]
                    df = canon[src] if src in canon else tables[src]
                    if hasattr(df, 'columns') and len(df.columns):
                        con.register(key, df)
                        registered.add(key)
            return con.execute(query).df()
        return sql

    sql = _sql_factory()

    class _Timeout(Exception):
        pass

    def _alarm(signum, frame):
        raise _Timeout()

    signal.signal(signal.SIGALRM, _alarm)
    sessions: dict[str, dict] = {}

    def new_ns():
        ns = {'pd': pd, 'np': np, 're': re, 'math': math, 'json': json, 'date': date, 'datetime': datetime,
              'timedelta': timedelta, 'tables': tables, 'canon': canon, 'meta': dict(meta), 'fold': fold, 'to_num': to_num,
              'parse_date': parse_date, 'month_end': month_end, 'round_money': round_money, 'sql': sql,
              '__name__': '__agent__'}

        def answer(type, value=None, reason=None):  # noqa: A002 - tên tham số theo prompt
            ns['__answer__'] = {'type': str(type).strip().lower(), 'value': value, 'reason': reason}
            return 'OK: đã ghi đáp án'
        ns['answer'] = answer
        return ns

    conn.send({'ready': True})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            break
        op = msg.get('op')
        if op == 'close':
            sessions.pop(msg.get('sid'), None)
            continue
        if op == 'stop':
            break
        if op != 'exec':
            continue
        sid, code, timeout = msg['sid'], msg['code'], float(msg.get('timeout', 30))
        ns = sessions.get(sid)
        if ns is None:
            ns = sessions[sid] = new_ns()
            if msg.get('meta'):
                ns['meta'].update(msg['meta'])
        buf = io.StringIO()
        t0 = time.time()
        res = {'status': 'ok', 'error': None}
        ns.pop('__answer__', None)
        signal.alarm(max(1, int(math.ceil(timeout))))
        try:
            with redirect_stdout(buf), warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter('always')
                if _cae is not None:
                    warnings.simplefilter('error', _cae)
                exec(compile(code, '<cell>', 'exec'), ns)
            # cảnh báo (chia cho 0 ra NaN, ngày không rõ định dạng ...) đưa vào kết quả cho agent thấy, không in ra console
            seen = []
            for w in wlist:
                ln = f'{w.category.__name__}: {str(w.message).split(chr(10))[0][:160]}'
                if ln not in seen:
                    seen.append(ln)
            if seen:
                buf.write('\nCẢNH BÁO:\n' + '\n'.join(seen[:4]) + '\n')
        except _Timeout:
            res = {'status': 'timeout', 'error': f'quá {timeout:.0f}s: tính toán quá nặng, hãy lọc dữ liệu trước'}
        except MemoryError:
            res = {'status': 'error', 'error': 'MemoryError: dữ liệu trung gian quá lớn'}
        except Exception:  # noqa: BLE001
            tb = traceback.format_exc()
            # chỉ giữ phần liên quan tới code của agent
            keep = [ln for ln in tb.splitlines() if ('File "<cell>"' in ln or not ln.startswith('  File'))
                    and 'exec(compile(' not in ln and not set(ln.strip()) <= set('~^')]
            err = '\n'.join(keep)
            if 'ChainedAssignmentError' in err:
                err = err.split('\n')[0] + ('\nChainedAssignmentError: pandas 3 không sửa được bảng qua df[col].method(..., '
                                            'inplace=True). Viết lại: df[col] = df[col].method(...)')
            res = {'status': 'error', 'error': err[-1800:]}
        finally:
            signal.alarm(0)
        out = buf.getvalue()
        if len(out) > out_limit:
            out = out[:out_limit // 2] + f'\n... (cắt {len(out) - out_limit} ký tự) ...\n' + out[-out_limit // 2:]
        ans = ns.pop('__answer__', None)
        try:
            json.dumps(ans, default=str)
        except Exception:  # noqa: BLE001
            ans = {'type': 'error', 'value': None, 'reason': 'answer không tuần tự hoá được'}
        res.update(stdout=out, answer=_jsonable(ans), elapsed=time.time() - t0)
        try:
            conn.send(res)
        except Exception:  # noqa: BLE001 - kết quả không gửi được (vd quá lớn) -> báo lỗi
            conn.send({'status': 'error', 'error': 'kết quả không gửi được', 'stdout': out[:2000], 'answer': None})
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0)


def _jsonable(x: Any) -> Any:
    """Đưa giá trị đáp án (numpy, pandas, date...) về kiểu JSON thuần."""
    import math
    from datetime import date, datetime
    try:
        import numpy as np
        import pandas as pd
    except Exception:  # noqa: BLE001
        np = pd = None
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return None if math.isnan(x) or math.isinf(x) else x
    if np is not None:
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return _jsonable(float(x))
        if isinstance(x, np.ndarray):
            return [_jsonable(v) for v in x.tolist()]
    if pd is not None:
        if isinstance(x, pd.Timestamp):
            return x.date().isoformat()
        if isinstance(x, pd.Series):
            return [_jsonable(v) for v in x.tolist()]
        if isinstance(x, pd.DataFrame):
            return {'columns': [str(c) for c in x.columns], 'rows': [[_jsonable(v) for v in r] for r in x.values.tolist()]}
    if isinstance(x, (datetime, date)):
        return x.isoformat()[:10]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    return str(x)


# ----------------------------------------------------------------------------- phía cha
@dataclass
class _Worker:
    pack: str
    proc: Any
    conn: Any
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    sessions: set = field(default_factory=set)
    last_used: float = field(default_factory=time.time)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.is_alive()


class KernelPool:
    """Quản lý worker theo shop. per_shop: số worker tối đa mỗi shop; max_workers: tổng (đuổi shop ít dùng nhất)."""

    def __init__(self, timeout_sec: float = 30.0, per_shop: int = 4, max_workers: int = 24, mem_gb: float = 12.0,
                 out_limit: int = 3000, start_timeout: float = 300.0):
        self.timeout = timeout_sec
        self.per_shop = per_shop
        self.max_workers = max_workers
        self.mem_gb = mem_gb
        self.out_limit = out_limit
        self.start_timeout = start_timeout
        self._ctx = mp.get_context('spawn')
        self._shops: 'OrderedDict[str, list[_Worker]]' = OrderedDict()
        self._sid_worker: dict[str, _Worker] = {}
        self._pool = ThreadPoolExecutor(max_workers=max(4, max_workers + 2), thread_name_prefix='kernel')
        self._glock = asyncio.Lock()

    # -- vòng đời worker
    def _spawn(self, pack: str) -> _Worker:
        parent, child = self._ctx.Pipe()
        env_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.environ.setdefault('PYTHONPATH', env_root)
        # glibc đọc MALLOC_ARENA_MAX lúc tiến trình khởi động -> phải đặt trước khi spawn worker
        for _k, _v in (('MALLOC_ARENA_MAX', '2'), ('OMP_NUM_THREADS', '1'), ('OPENBLAS_NUM_THREADS', '1'), ('MKL_NUM_THREADS', '1')):
            os.environ.setdefault(_k, _v)
        p = self._ctx.Process(target=_worker_main, args=(child, pack, self.mem_gb, self.out_limit), daemon=True)
        p.start()
        child.close()
        return _Worker(pack=pack, proc=p, conn=parent)

    def _kill(self, w: _Worker):
        try:
            w.conn.send({'op': 'stop'})
        except Exception:  # noqa: BLE001
            pass
        try:
            if w.proc.is_alive():
                w.proc.kill()
            w.proc.join(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        for sid in list(w.sessions):
            self._sid_worker.pop(sid, None)

    async def _ready(self, w: _Worker) -> bool:
        loop = asyncio.get_running_loop()
        try:
            ok = await loop.run_in_executor(self._pool, w.conn.poll, self.start_timeout)
            if ok:
                msg = await loop.run_in_executor(self._pool, w.conn.recv)
                return bool(msg.get('ready'))
        except Exception:  # noqa: BLE001
            pass
        return False

    async def _worker_for(self, pack: str, sid: str) -> _Worker:
        w = self._sid_worker.get(sid)
        if w is not None and w.alive():
            return w
        async with self._glock:
            ws = self._shops.setdefault(pack, [])
            self._shops.move_to_end(pack)
            ws[:] = [x for x in ws if x.alive()]
            if len(ws) < self.per_shop:
                total = sum(len(v) for v in self._shops.values())
                while total >= self.max_workers:
                    victim = next((k for k in self._shops if k != pack and self._shops[k]), None)
                    if victim is None:
                        break
                    for x in self._shops.pop(victim):
                        self._kill(x)
                    total = sum(len(v) for v in self._shops.values())
                if total < self.max_workers or not ws:
                    w = self._spawn(pack)
                    if not await self._ready(w):
                        self._kill(w)
                        raise RuntimeError('worker không khởi động được (nạp dữ liệu lỗi hoặc thiếu RAM)')
                    ws.append(w)
            w = min(ws, key=lambda x: len(x.sessions))
            w.sessions.add(sid)
            self._sid_worker[sid] = w
            return w

    # -- API
    async def execute(self, pack: str, sid: str, code: str, meta: Optional[dict] = None,
                      timeout: Optional[float] = None) -> CellResult:
        t0 = time.time()
        timeout = timeout or self.timeout
        try:
            w = await self._worker_for(pack, sid)
        except Exception as e:  # noqa: BLE001
            return CellResult(status='crash', error=str(e), elapsed=time.time() - t0)
        loop = asyncio.get_running_loop()
        async with w.lock:
            w.last_used = time.time()
            try:
                w.conn.send({'op': 'exec', 'sid': sid, 'code': code, 'timeout': timeout, 'meta': meta})
                ok = await loop.run_in_executor(self._pool, w.conn.poll, timeout + 15)
                if not ok:
                    raise TimeoutError('worker không trả lời')
                d = await loop.run_in_executor(self._pool, w.conn.recv)
            except Exception as e:  # noqa: BLE001 - worker treo / chết: thay worker mới, session mất trạng thái
                log.warning('kernel worker lỗi (%s) -> khởi động lại', e)
                self._kill(w)
                async with self._glock:
                    lst = self._shops.get(pack, [])
                    if w in lst:
                        lst.remove(w)
                return CellResult(status='crash', error='kernel bị khởi động lại (code quá nặng / hết RAM); '
                                                         'các biến trước đó đã mất, hãy tính lại từ đầu gọn hơn',
                                  elapsed=time.time() - t0)
        return CellResult(status=d.get('status', 'error'), stdout=d.get('stdout') or '', error=d.get('error'),
                          answer=d.get('answer'), elapsed=time.time() - t0)

    async def close_session(self, sid: str):
        w = self._sid_worker.pop(sid, None)
        if w is not None:
            w.sessions.discard(sid)
            try:
                w.conn.send({'op': 'close', 'sid': sid})
            except Exception:  # noqa: BLE001
                pass

    async def close(self):
        for ws in self._shops.values():
            for w in ws:
                self._kill(w)
        self._shops.clear()
        self._pool.shutdown(wait=False, cancel_futures=True)
