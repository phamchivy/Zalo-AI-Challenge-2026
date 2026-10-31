"""Chọn tập câu DEV nhỏ để so mô hình / cấu hình LLM nhanh (không dùng holdout, không dùng public).

Từ một lần chạy tầng A trên dev: ưu tiên câu tầng A bỏ trống (đất của LLM), rồi câu tầng A sai, thêm một phần câu tầng A
đúng (để thấy LLM có làm hỏng câu dễ không). Thứ tự cố định theo md5(id) -> chạy lại ra cùng tập.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from datamate.offline.scorer import score_one


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def make_subset(split: str, run: str, n: int = 200, shops: Optional[set] = None,
                mix: tuple[float, float, float] = (0.55, 0.25, 0.20)) -> tuple[list[str], dict]:
    qs = {q['id']: q for q in _jsonl(Path(split) / 'questions.jsonl')}
    gold = {a['id']: a for a in _jsonl(Path(split) / 'answers.jsonl')}
    pred = {p['id']: p for p in _jsonl(Path(run) / 'submission.jsonl')}
    groups = {'abstain': [], 'wrong': [], 'right': []}
    for qid, q in qs.items():
        if shops is not None and q['business_id'] not in shops:
            continue
        if qid not in gold or qid not in pred:
            continue
        s = score_one(pred[qid], gold[qid])
        groups['abstain' if s == 0 else ('wrong' if s < 0 else 'right')].append(qid)
    key = lambda x: hashlib.md5(x.encode()).hexdigest()  # noqa: E731
    out, info = [], {}
    quota = [int(round(n * m)) for m in mix]
    for (g, ids), k in zip(groups.items(), quota):
        pick = sorted(ids, key=key)[:k]
        out += pick
        info[g] = f'{len(pick)}/{len(ids)}'
    if len(out) < n:                                   # nhóm nào thiếu thì bù từ câu còn lại
        rest = sorted((x for g in groups.values() for x in g if x not in set(out)), key=key)
        out += rest[:n - len(out)]
    return sorted(out, key=key), info
