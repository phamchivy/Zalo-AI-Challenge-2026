"""Bộ chấm offline mô phỏng đúng luật chấm của BTC (Evaluation): +1 đúng, -1 sai, 0 bỏ."""
from __future__ import annotations

import math
from typing import Any

from datamate.core.textutil import norm_answer_str


def _num_eq(a: Any, b: Any, pct: bool = False) -> bool:
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return False
    if pct:
        return round(x, 2) == round(y, 2)
    return math.isclose(x, y, rel_tol=0, abs_tol=1e-6)


def _str_eq(a: Any, ref: Any, aliases: list | None) -> bool:
    if not isinstance(a, str):
        return False
    na = norm_answer_str(a)
    forms = [ref] + list(aliases or [])
    return any(na == norm_answer_str(f) for f in forms if isinstance(f, str))


def _cell_eq(a: Any, ref: Any, aliases: list | None = None) -> bool:
    if isinstance(ref, (int, float)) and not isinstance(ref, bool):
        return _num_eq(a, ref)
    if isinstance(ref, str):
        return _str_eq(a, ref, aliases)
    return a == ref


def score_one(pred: dict | None, gold: dict) -> int:
    """pred: dòng submission (có thể None). gold: dòng answers.jsonl (type, answer, alias)."""
    if pred is None or pred.get('type') in (None, 'abstain'):
        return 0
    gt, pt = gold.get('type'), pred.get('type')
    if pt != gt:
        return -1
    if gt == 'unanswerable':
        return 1
    ga, pa, al = gold.get('answer'), pred.get('answer'), gold.get('alias') or []
    ok = False
    if gt == 'number':
        ok = _num_eq(pa, ga)
    elif gt == 'percent':
        ok = _num_eq(pa, ga, pct=True)
    elif gt in ('date', 'boolean'):
        ok = pa == ga
    elif gt == 'string':
        ok = _str_eq(pa, ga, al)
    elif gt == 'list':
        if isinstance(pa, list) and isinstance(ga, list) and len(pa) == len(ga):
            ok = all(_cell_eq(x, y, al[i] if i < len(al) and isinstance(al[i], list) else None) for i, (x, y) in enumerate(zip(pa, ga)))
    elif gt == 'table':
        ok = _table_eq(pa, ga, al)
    return 1 if ok else -1


def _table_eq(pa: Any, ga: dict, al: list) -> bool:
    if not isinstance(pa, dict) or not isinstance(pa.get('rows'), list):
        return False
    gcols = [norm_answer_str(c) for c in ga.get('columns', [])]
    pcols = [norm_answer_str(c) for c in pa.get('columns', [])]
    if len(pcols) != len(gcols):
        return False
    order = [pcols.index(c) for c in gcols] if sorted(pcols) == sorted(gcols) else list(range(len(gcols)))
    grows = ga['rows']
    prows = [[r[i] if i < len(r) else None for i in order] for r in pa['rows']]
    if len(prows) != len(grows):
        return False
    used = [False] * len(prows)
    for gi, g in enumerate(grows):
        ra = al[gi] if gi < len(al) and isinstance(al[gi], list) else None
        hit = False
        for pi, p in enumerate(prows):
            if not used[pi] and all(_cell_eq(x, y, ra if j == 0 else None) for j, (x, y) in enumerate(zip(p, g))):
                used[pi] = hit = True
                break
        if not hit:
            return False
    return True
