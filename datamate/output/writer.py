"""Ghi submission.jsonl tăng dần (flush từng dòng, chạy lại thì resume) + log mỗi câu ở logs/<qid>.json."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Optional

from datamate.core.types import AnswerType, Decision, Trace


def submission_row(d: Decision) -> dict:
    row = {'id': d.qid, 'type': d.type.value}
    if d.type == AnswerType.UNANSWERABLE:
        row['reason'] = (d.reason or 'Dữ liệu không đủ để trả lời')[:300]
    elif d.type != AnswerType.ABSTAIN:
        row['answer'] = d.answer
    return row


EXECUTOR_REF = 'src/datamate/reasoning/executor.py::Executor.run'


def _supporting(d: Decision, trace: Trace):
    """Các lời giải ra đúng đáp án đã chốt (đã chạy thật)."""
    from datamate.output.formatter import answer_key, normalize_answer
    if d.type in (AnswerType.ABSTAIN,):
        return []
    try:
        key = answer_key(*normalize_answer(d.type, d.answer)) if d.type != AnswerType.UNANSWERABLE else 'unanswerable'
    except Exception:  # noqa: BLE001
        return []
    out = []
    for c in trace.candidates:
        if c.status != 'ok' or c.answer_type is None:
            continue
        try:
            k = answer_key(*normalize_answer(c.answer_type, c.answer)) if c.answer_type != AnswerType.UNANSWERABLE \
                else 'unanswerable'
        except Exception:  # noqa: BLE001
            continue
        if k == key:
            out.append(c)
    return out


def _code_of(c) -> dict:
    if c.strategy.startswith('rule_plan') or c.strategy == 'guard':
        return {'strategy': c.strategy, 'code': EXECUTOR_REF if c.strategy.startswith('rule_plan')
                else 'src/datamate/reasoning/strategies/guard.py', 'inputs': c.program, 'status': c.status,
                'output': c.answer if c.status == 'ok' else c.error}
    return {'strategy': c.strategy, 'code': c.program, 'status': c.status,
            'output': (c.meta or {}).get('outputs') if c.status == 'ok' or (c.meta or {}).get('outputs') else c.error,
            'answer': c.answer}


def btc_log_record(d: Decision, trace: Trace, row: dict) -> dict:
    """Bản ghi log theo guideline BTC (mục 3.3). thinking = null: Qwen chạy với enable_thinking=False (không sinh suy luận)."""
    sup = _supporting(d, trace)
    main = [_code_of(c) for c in sup]
    if len(main) == 1:
        code, output = main[0]['code'], main[0]['output']
        inputs = main[0].get('inputs')
    elif main:
        code, output, inputs = [m['code'] for m in main], [m['output'] for m in main], [m.get('inputs') for m in main]
    else:
        code = output = inputs = None
    ans = {k: v for k, v in row.items() if k != 'id'}
    return {'id': d.qid, 'answer': ans, 'elapsed_ms': max(0, int(round(trace.elapsed_sec * 1000))), 'thinking': None,
            'code': code, 'inputs': inputs, 'output': output,
            'all_attempts': [_code_of(c) for c in trace.candidates],
            **json.loads(trace.model_dump_json())}          # trace đầy đủ ở cùng cấp (công cụ phân tích cũ đọc như trước)


class SubmissionWriter:
    def __init__(self, out_dir: str, resume: bool = True, filename: str = 'submission.jsonl'):
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'logs').mkdir(exist_ok=True)
        self.path = self.out / filename
        self.done: dict[str, dict] = {}
        if not resume:                     # --fresh: xoá kết quả + log cũ, tránh trộn lần chạy trước vào eval / sweep
            if self.path.exists():
                self.path.unlink()
            for f in (self.out / 'logs').glob('*.json'):
                f.unlink()
        if resume and self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                try:
                    r = json.loads(line)
                    self.done[r['id']] = r
                except (json.JSONDecodeError, KeyError):
                    continue
        self._lock = threading.Lock()
        self._fh = open(self.path, 'a', encoding='utf-8')

    def write(self, d: Decision, trace: Optional[Trace] = None):
        row = submission_row(d)
        with self._lock:
            self._fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self.done[d.qid] = row
        if trace is not None:
            # v3.4.0: 1 bản ghi / câu, MỘT dòng JSON, đủ trường BTC yêu cầu (id, answer, elapsed_ms, thinking, code, output)
            # + toàn bộ trace để giải trình khi BTC chạy lại.
            rec = btc_log_record(d, trace, row)
            (self.out / 'logs' / f'{d.qid}.json').write_text(json.dumps(rec, ensure_ascii=False, default=str) + '\n',
                                                             encoding='utf-8')

    def close(self):
        self._fh.close()

    def finalize(self, all_ids: list[str]):
        """Viết lại file sạch: đúng 1 dòng / id theo thứ tự đề, id thiếu = abstain."""
        self.close()
        rows = [self.done.get(i, {'id': i, 'type': 'abstain'}) for i in all_ids]
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        os.replace(tmp, self.path)
