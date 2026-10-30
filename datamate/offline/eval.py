"""Đánh giá một submission trên split có đáp án (train): PA tổng + theo doanh nghiệp / type / intent / strategy,
xuất danh sách câu sai kèm plan/code để sửa (errors.jsonl)."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

from datamate.offline.scorer import score_one


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def evaluate(split_dir: str, pred_path: str, logs_dir: Optional[str] = None, out_dir: Optional[str] = None,
             only_answered_ids: bool = False) -> dict:
    split = Path(split_dir)
    qs = {q['id']: q for q in _read_jsonl(split / 'questions.jsonl')}
    gold = {a['id']: a for a in _read_jsonl(split / 'answers.jsonl')}
    preds = {p['id']: p for p in _read_jsonl(Path(pred_path))}
    ids = [i for i in qs if i in preds] if only_answered_ids else list(qs)
    logs = Path(logs_dir) if logs_dir else Path(pred_path).parent / 'logs'
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))   # key -> val -> [n, +1, 0, -1]
    errors = []
    total = [0, 0, 0, 0]
    for i in ids:
        g, p = gold.get(i), preds.get(i)
        if g is None:
            continue
        s = score_one(p, g)
        intent, policy = '', (p or {}).get('type', 'missing')
        lf = logs / f'{i}.json'
        trace = None
        if lf.exists():
            trace = json.loads(lf.read_text(encoding='utf-8'))
            intent = trace.get('intent', '')
            policy = (trace.get('decision') or {}).get('policy', policy)
        keys = {'business': qs[i]['business_id'], 'gold_type': g['type'], 'intent': intent or '?', 'policy': policy.split('(')[0]}
        for k, v in keys.items():
            row = agg[k][v]
            row[0] += 1
            row[{1: 1, 0: 2, -1: 3}[s]] += 1
        total[0] += 1
        total[{1: 1, 0: 2, -1: 3}[s]] += 1
        if s == -1:
            errors.append({'id': i, 'business_id': qs[i]['business_id'], 'question': qs[i]['question'], 'as_of': qs[i]['as_of_date'],
                           'gold': g, 'pred': p, 'intent': intent,
                           'candidates': [{k: c.get(k) for k in ('strategy', 'status', 'answer_type', 'answer', 'weight', 'error', 'program')}
                                          for c in (trace or {}).get('candidates', [])]})
    n = max(total[0], 1)
    rep = {
        'n': total[0], 'correct': total[1], 'abstain': total[2], 'wrong': total[3],
        'PA': round((total[1] - total[3]) / n, 4), 'accuracy_answered': round(total[1] / max(total[1] + total[3], 1), 4),
        'coverage': round((total[1] + total[3]) / n, 4),
        'breakdown': {k: {v: {'n': r[0], 'ok': r[1], 'abstain': r[2], 'wrong': r[3], 'PA': round((r[1] - r[3]) / max(r[0], 1), 3)}
                          for v, r in sorted(d.items(), key=lambda x: -x[1][0])} for k, d in agg.items()},
    }
    if out_dir:
        o = Path(out_dir)
        o.mkdir(parents=True, exist_ok=True)
        (o / 'eval_report.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
        with open(o / 'errors.jsonl', 'w', encoding='utf-8') as f:
            for e in errors:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + '\n')
    return rep


def format_report(rep: dict, top: int = 15) -> str:
    lines = [f"N={rep['n']}  PA={rep['PA']:.4f}  đúng={rep['correct']}  sai={rep['wrong']}  bỏ={rep['abstain']}  "
             f"acc(trả lời)={rep['accuracy_answered']:.3f}  coverage={rep['coverage']:.3f}"]
    for k in ('gold_type', 'intent', 'policy', 'business'):
        lines.append(f'\n[{k}]')
        for v, r in list(rep['breakdown'].get(k, {}).items())[:top]:
            lines.append(f"  {v:28s} n={r['n']:4d} ok={r['ok']:4d} bỏ={r['abstain']:4d} sai={r['wrong']:4d} PA={r['PA']:+.3f}")
    return '\n'.join(lines)
