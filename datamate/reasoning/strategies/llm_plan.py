"""LLM điền Plan JSON (guided decoding) -> Executor deterministic. Dùng khi rule parser không chắc."""
from __future__ import annotations

import json
import time
from datetime import date

from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType, Plan, TimeRange
from datamate.reasoning.executor import Executor, PlanError, Unanswerable
from datamate.reasoning.prompts import PLAN_SCHEMA, describe_plan, plan_messages
from datamate.reasoning.strategies.base import Strategy
from datamate.reasoning.strategies.llm_code import question_block
from datamate.runtime.llm_client import extract_json

TIER_B = {'debt', 'inventory', 'expiry', 'price_effect', 'purchase', 'dead_stock', 'delivery', 'net_profit', 'special', 'lots'}


def entity_block(cat, ix, max_items: int = 150) -> str:
    lines = ['# THỰC THỂ CÓ TRONG DỮ LIỆU (key: tên)']
    seen = set()
    prods = []
    for k, d, _, _ in ix.products:
        if k not in seen:
            seen.add(k)
            prods.append(f'{k}: {d}')
    lines.append('products: ' + '; '.join(prods[:max_items]))
    for name, vals in (('category', list(ix.categories.values())), ('customer_group', list(ix.cgroups.values())),
                       ('channel', ix.channels), ('payment', ix.payments), ('staff', sorted(set(ix.staff.values())))):
        if vals:
            lines.append(f'{name}: ' + '; '.join(map(str, vals[:60])))
    c = cat.canon
    lines.append(f'Khoảng dữ liệu: {c.date_min}..{c.date_max}. caps: {", ".join(sorted(c.caps))}')
    return '\n'.join(lines)


@register_strategy('llm_plan')
class LLMPlan(Strategy):
    uses_llm = True

    async def run(self, u, cat, ix):
        env = self.env
        if not env.llm_allowed():
            return [self.cand(status='skipped', error='llm disabled or out of time')]
        if u.parse.plan is not None and u.parse.confidence >= float(self.scfg.options.get('skip_if_rule_conf', 0.8)):
            return [self.cand(status='skipped', error='rule plan confident')]
        if u.parse.tier_b_hint in TIER_B:
            return [self.cand(status='skipped', error='tier-B intent')]
        t0 = time.time()
        rb = env.kb.rulebook.section('don hang', 'thoi gian', 'xep hang', 'theo nganh', max_chars=9000)
        msgs = plan_messages(rb, entity_block(cat, ix), question_block(u))
        res = await env.llm.chat(msgs, json_schema=PLAN_SCHEMA, max_tokens=600)
        d = extract_json(res.text) if res.text else None
        if not isinstance(d, dict) or not d.get('supported'):
            return [self.cand(status='empty', error=res.error or 'unsupported', elapsed=time.time() - t0, tokens=res.tokens)]
        try:
            periods = [TimeRange(start=date.fromisoformat(p['start'][:10]), end=date.fromisoformat(p['end'][:10]))
                       for p in d.get('periods') or []]
            at = d.get('answer_type')
            plan = Plan(metric=d['metric'], op=d['op'], periods=periods, filters=d.get('filters') or {},
                        group_by=d.get('group_by'), split_by=d.get('split_by'), split_values=d.get('split_values') or [],
                        k=d.get('k'), threshold=d.get('threshold'), threshold_metric=d.get('threshold_metric'),
                        answer_type=AnswerType(at) if at in [x.value for x in AnswerType] else None, source='llm')
            t, a = Executor(cat.canon, env.kb.rulebook.params, env.kb.rules).run(plan)
            return [self.cand(program=describe_plan(plan), answer_type=t, answer=a, elapsed=time.time() - t0, tokens=res.tokens)]
        except Unanswerable as e:
            return [self.cand(answer_type=AnswerType.UNANSWERABLE, reason=str(e), weight=self.scfg.weight * 0.5,
                              elapsed=time.time() - t0, tokens=res.tokens)]
        except (PlanError, KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            return [self.cand(program=json.dumps(d, ensure_ascii=False), status='error', error=repr(e)[:300],
                              elapsed=time.time() - t0, tokens=res.tokens)]
