"""Hiệu chỉnh độ tin cậy tầng A từ một lần chạy train "thô" (configs/no_llm_raw.yaml: trả lời mọi câu có plan).

  1. Đọc logs/<qid>.json: đáp án của rule_plan + đặc trưng (intent, hệ số từ lạ, sức khoẻ dữ liệu).
  2. Chia shop thành k fold (băm business_id, cố định). Với mỗi fold: độ chính xác theo intent tính trên CÁC FOLD KHÁC
     (cận dưới Wilson) -> mô phỏng quyết định trên fold này => PA out-of-fold = dự báo cho shop chưa thấy (public/private).
  3. Ghi calibration.yaml (bảng intent tính trên toàn bộ train) + báo cáo PA out-of-fold theo ngưỡng min_score.
Không dùng câu hỏi/đáp án public/private; không có id shop nào trong bảng kết quả.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import yaml

from datamate.decision.calibration import wilson_lower
from datamate.offline.scorer import score_one
from datamate.output.formatter import normalize_answer

THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def _fold(biz: str, k: int) -> int:
    return int(hashlib.md5(biz.encode()).hexdigest(), 16) % k


def collect(split: str, run: str, businesses: Optional[set] = None) -> list[dict]:
    split, run = Path(split), Path(run)
    qs = {q['id']: q for q in _jsonl(split / 'questions.jsonl')}
    if businesses is not None:
        qs = {k: q for k, q in qs.items() if q['business_id'] in businesses}
    gold = {a['id']: a for a in _jsonl(split / 'answers.jsonl')}
    sub = {r['id']: r for r in _jsonl(run / 'submission.jsonl')} if (run / 'submission.jsonl').exists() else {}
    rows = []
    for qid, q in qs.items():
        g = gold.get(qid)
        lf = run / 'logs' / f'{qid}.json'
        if g is None or not lf.exists():
            continue
        tr = json.loads(lf.read_text(encoding='utf-8'))
        rule = next((c for c in tr.get('candidates', []) if c['strategy'] == 'rule_plan' and c['status'] == 'ok'
                     and c.get('answer_type') not in (None, 'unanswerable')), None)
        guard = next((c for c in tr.get('candidates', []) if c['strategy'] == 'guard' and c['status'] == 'ok'), None)
        r = {'id': qid, 'biz': q['business_id'], 'intent': tr.get('intent') or '?', 'rule': None, 'guard': None}
        if rule:
            t, a = normalize_answer(rule['answer_type'], rule['answer'])
            meta = rule.get('meta') or {}
            r['rule'] = {'s': score_one({'type': t, 'answer': a} if t else None, g),
                         'prior': meta.get('base', rule.get('weight', 0.9)) if meta.get('base_src', 'prior') == 'prior' else 0.9,
                         'f': meta.get('f_unexplained', 1.0) * meta.get('f_health', 1.0)}
        if guard:
            r['guard'] = score_one({'type': 'unanswerable'}, g)
        rows.append(r)
    return rows


def table(rows: list[dict]) -> dict:
    agg = defaultdict(lambda: [0, 0])
    for r in rows:
        if r['rule'] and r['rule']['f'] > 0:
            agg[r['intent']][0] += 1
            agg[r['intent']][1] += r['rule']['s'] == 1
    return {k: {'n': n, 'ok': ok, 'wrong': n - ok, 'precision': round(ok / n, 4), 'lower': round(wilson_lower(ok, n), 4)}
            for k, (n, ok) in sorted(agg.items())}


def simulate(rows: list[dict], k: int, min_n: int) -> dict:
    """PA out-of-fold cho từng ngưỡng min_score."""
    folds = defaultdict(list)
    for r in rows:
        folds[_fold(r['biz'], k)].append(r)
    res = {t: [0, 0, 0] for t in THRESHOLDS}            # ok, wrong, n
    for f, fr in folds.items():
        other = [r for g, rs in folds.items() if g != f for r in rs]
        tab = table(other)
        for r in fr:
            for t in THRESHOLDS:
                res[t][2] += 1
                s = 0
                if r['guard'] is not None:
                    s = r['guard']                      # guard (unanswerable) chạy độc lập
                elif r['rule']:
                    row = tab.get(r['intent'])
                    base = row['lower'] if row and row['n'] >= min_n else r['rule']['prior']
                    if base * r['rule']['f'] >= t:
                        s = r['rule']['s']
                res[t][0] += s == 1
                res[t][1] += s == -1
    return {str(t): {'PA': round((ok - w) / max(n, 1), 4), 'correct': ok, 'wrong': w,
                     'precision': round(ok / max(ok + w, 1), 4)} for t, (ok, w, n) in res.items()}


def calibrate(split: str, run: str, out: str, folds: int = 5, min_n: int = 8, businesses: Optional[set] = None) -> dict:
    """businesses: chỉ dùng shop dev (không bao giờ hiệu chỉnh trên holdout)."""
    rows = collect(split, run, businesses)
    tab = table(rows)
    oof = simulate(rows, folds, min_n)
    best = max(oof.items(), key=lambda kv: kv[1]['PA'])
    doc = {'note': 'sinh bởi datamate calibrate; intent -> độ chính xác của tầng A đo trên train (lower = cận dưới Wilson z=1)',
           'shops': 'dev' if businesses is not None else 'all', 'min_n': min_n, 'folds': folds, 'questions': len(rows), 'recommended_min_score': float(best[0]),
           'oof_by_threshold': oof, 'intents': tab}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return doc
