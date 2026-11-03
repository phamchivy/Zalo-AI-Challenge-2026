#!/usr/bin/env python
"""Chạy phần đọc câu (thời gian + intent, không cần dữ liệu shop) trên toàn bộ câu public trong log -> JSON.
python scripts/parse_harness.py <run_dir>/logs <run_dir>/rules.json <run_dir>/run_summary.json <out.json> [calendar.csv]
So hai file out (trước / sau khi sửa code) để biết câu nào đổi cách đọc."""
import json, glob, sys
from datetime import date
from pathlib import Path
from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time
from datamate.understand.intent import parse_question
from datamate.understand.entity_link import EntityIndex
from datamate.knowledge.rules import RuleSet
from datamate.knowledge.calendar import LunarCalendar
from datamate.core.textutil import fold
import re
logs, rules_p, summ_p, out = sys.argv[1:5]
rules = RuleSet.from_json(Path(rules_p).read_text(encoding='utf-8'))
cats = json.load(open(summ_p, encoding='utf-8'))['catalogs']
cal = None
for p in sys.argv[5:6]:
    try: cal = LunarCalendar.load(p)
    except Exception as e: print('cal', e)
res = {}
for f in sorted(glob.glob(logs + '/*.json')):
    t = json.load(open(f, encoding='utf-8'))
    q, asof = t['question'], date.fromisoformat(t['as_of_date'])
    qn = qnorm(q)
    qf = ' '.join(re.sub(r'(?<!\d),|,(?!\d)|[?!;"“”]', ' ', fold(q)).split())
    tp = resolve_time(qn, asof, cal)
    c = cats.get(t['business_id'], {})
    caps = set(c.get('caps', []))
    dmin = date.fromisoformat(c['date_min']) if c.get('date_min') not in (None, 'None') else None
    dmax = date.fromisoformat(c['date_max']) if c.get('date_max') not in (None, 'None') else None
    try:
        pr = parse_question(qn, qf, asof, tp, EntityIndex(), caps, dmin, dmax, cal, rules=rules, raw=q)
        plan = pr.plan
        res[t['qid']] = {'intent': pr.intent, 'metric': getattr(plan, 'metric', None), 'op': getattr(plan, 'op', None),
                         'periods': [f'{p.start}..{p.end}' for p in (plan.periods if plan else tp.ranges)], 'unexplained': (pr.features or {}).get('unexplained'), 'unused': getattr(tp, 'unused', []),
                         'reason': pr.reason[:80], 'q': q, 'biz': t['business_id']}
    except Exception as e:
        res[t['qid']] = {'intent': 'ERR', 'err': repr(e)[:120], 'q': q, 'biz': t['business_id']}
json.dump(res, open(out, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
print(len(res), 'câu')
