"""Chạy code LLM trong tiến trình con có timeout + giới hạn RAM; dữ liệu nạp từ 'pack' pickle của doanh nghiệp."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from datamate.core.log import get_logger

log = get_logger('sandbox')
RUNNER = str(Path(__file__).with_name('runner.py'))


@dataclass
class ExecResult:
    status: str                    # ok | error | timeout | empty
    result: Any = None
    error: Optional[str] = None
    stdout: str = ''
    elapsed: float = 0.0


def write_pack(cat, cache_dir: str) -> str:
    """Xuất dữ liệu doanh nghiệp cho sandbox (1 lần / doanh nghiệp)."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    from datamate.data.catalog import _code_version
    path = Path(cache_dir) / f'pack-{cat.business_id}-{hashlib.md5(cat.root.encode()).hexdigest()[:8]}-{_code_version()}.pkl'
    if path.exists():
        return str(path)
    c = cat.canon
    pack = {
        'tables': {n: t.df for n, t in cat.raw.items() if t.kind != 'error'},
        'canon': {'lines': c.lines, 'orders': c.orders, 'products': c.products, 'units': c.units,
                  'customers': c.customers, 'payments': c.payments, 'purchases': c.purchases,
                  'stock': getattr(c, 'stock', None), 'prices': getattr(c, 'prices', None), 'lots': getattr(c, 'lots', None),
                  'lot_sales': getattr(c, 'lot_sales', None), 'deliveries': getattr(c, 'deliveries', None)},
        'meta': {'business_id': cat.business_id, 'data_min': str(c.date_min), 'data_max': str(c.date_max),
                 'caps': sorted(c.caps)},
    }
    tmp = str(path) + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(pack, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return str(path)


class Sandbox:
    def __init__(self, timeout_sec: float = 30.0, max_parallel: int = 8, work_dir: Optional[str] = None):
        self.timeout = timeout_sec
        self._sem = asyncio.Semaphore(max_parallel)
        self.work_dir = work_dir or tempfile.mkdtemp(prefix='dm_sbx_')

    async def run(self, pack_path: str, code: str, meta_overrides: Optional[dict] = None) -> ExecResult:
        t0 = time.time()
        prelude = ''
        if meta_overrides:
            prelude = 'meta.update(' + json.dumps(meta_overrides, ensure_ascii=False, default=str) + ')\n'
        fd, code_path = tempfile.mkstemp(suffix='.py', dir=self.work_dir)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(prelude + code)
        async with self._sem:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-I', RUNNER, pack_path, code_path, str(self.timeout),
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=self.work_dir)
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout + 15)
            except asyncio.TimeoutError:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                return ExecResult(status='timeout', error=f'>{self.timeout}s', elapsed=time.time() - t0)
            finally:
                try:
                    os.remove(code_path)
                except OSError:
                    pass
        text = out.decode('utf-8', errors='replace')
        if '__RESULT__' not in text:
            return ExecResult(status='error', error=(err.decode('utf-8', errors='replace')[-1500:] or 'no result'),
                              stdout=text[-2000:], elapsed=time.time() - t0)
        head, _, payload = text.rpartition('__RESULT__')
        try:
            d = json.loads(payload.strip())
        except json.JSONDecodeError as e:
            return ExecResult(status='error', error=f'bad result json: {e}', stdout=head[-2000:], elapsed=time.time() - t0)
        return ExecResult(status=d.get('status', 'error'), result=d.get('result'), error=d.get('error'),
                          stdout=head[-2000:], elapsed=time.time() - t0)
