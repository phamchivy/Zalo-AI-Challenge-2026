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
            (self.out / 'logs' / f'{d.qid}.json').write_text(trace.model_dump_json(indent=1), encoding='utf-8')

    def close(self):
        self._fh.close()

    def finalize(self, all_ids: list[str]):
        """Viết lại file sạch: đúng 1 dòng / id theo thứ tự đề, id thiếu = abstain."""
        self.close()
        rows = [self.done.get(i, {'id': i, 'type': 'abstain'}) for i in all_ids]
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
        os.replace(tmp, self.path)
