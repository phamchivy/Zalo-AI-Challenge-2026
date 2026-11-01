"""v2.3 — chọn ngưỡng bỏ phiếu trên DEV: chạy lại bộ bỏ phiếu trên các candidate đã ghi trong logs với nhiều cặp
(min_score, min_share), tính PA theo đáp án dev. Không chạy lại LLM. Không dùng holdout / public.

Cũng in bảng "số lời giải agent đồng ý -> tỉ lệ đúng" để biết nên tin agent tới mức nào.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datamate.core.config import DecisionConfig
from datamate.core.types import Candidate
from datamate.decision.voter import decide
from datamate.offline.scorer import score_one
from datamate.output.formatter import answer_key, normalize_answer

MIN_SCORE = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]
MIN_SHARE = [0.5, 0.6, 0.7, 0.8]


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def load_rows(split: str, run: str, shops: Optional[set] = None) -> list[tuple[str, list[Candidate], dict]]:
    qs = {q['id']: q for q in _jsonl(Path(split) / 'questions.jsonl')}
    gold = {a['id']: a for a in _jsonl(Path(split) / 'answers.jsonl')}
    rows = []
    sub = Path(run) / 'submission.jsonl'
    in_run = {r['id'] for r in _jsonl(sub)} if sub.exists() else None     # chỉ câu của lần chạy này
    for f in sorted((Path(run) / 'logs').glob('*.json')):
        tr = json.loads(f.read_text(encoding='utf-8'))
        qid = tr.get('qid')
        if qid not in gold or (shops is not None and qs.get(qid, {}).get('business_id') not in shops):
            continue
        if in_run is not None and qid not in in_run:
            continue
        cands = []
        for c in tr.get('candidates', []):
            try:
                cands.append(Candidate(**{k: v for k, v in c.items() if k in Candidate.model_fields}))
            except Exception:  # noqa: BLE001
                continue
        rows.append((qid, cands, gold[qid]))
    return rows


def agreement_table(rows) -> dict:
    """k lời giải agent cùng đáp án (nhóm lớn nhất) -> số câu, tỉ lệ đúng của đáp án đó."""
    agg = defaultdict(lambda: [0, 0])
    for qid, cands, g in rows:
        ag = [c for c in cands if c.strategy == 'agent' and c.status == 'ok' and c.answer_type is not None]
        if not ag:
            continue
        groups = defaultdict(list)
        for c in ag:
            try:
                t, a = normalize_answer(c.answer_type, c.answer)
                if t:
                    groups[answer_key(t, a)].append((t, a))
            except Exception:  # noqa: BLE001 - đáp án dạng lạ: bỏ qua
                continue
        if not groups:
            continue
        k, members = max(groups.items(), key=lambda kv: len(kv[1]))
        t, a = members[0]
        n_total = sum(1 for c in cands if c.strategy == 'agent')
        key = f'{len(members)}/{n_total}'
        agg[key][0] += 1
        agg[key][1] += score_one({'type': t, 'answer': a}, g) == 1
    return {k: {'n': n, 'precision': round(ok / n, 3)} for k, (n, ok) in sorted(agg.items())}


def sweep(rows, base: DecisionConfig) -> list[dict]:
    out = []
    for ms in MIN_SCORE:
        for sh in MIN_SHARE:
            cfg = base.model_copy(update={'min_score': ms, 'min_share': sh})
            ok = wrong = 0
            for qid, cands, g in rows:
                d = decide(qid, cands, cfg)
                s = score_one({'type': d.type.value, 'answer': d.answer}, g)
                ok += s == 1
                wrong += s == -1
            n = max(len(rows), 1)
            out.append({'min_score': ms, 'min_share': sh, 'PA': round((ok - wrong) / n, 4), 'correct': ok, 'wrong': wrong})
    return sorted(out, key=lambda r: -r['PA'])
