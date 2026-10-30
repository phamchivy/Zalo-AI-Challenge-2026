"""Tầng A: rule parser -> Plan -> Executor (deterministic). Không gọi LLM."""
from __future__ import annotations

import time

from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType
from datamate.reasoning.executor import Executor, PlanError, Unanswerable
from datamate.reasoning.prompts import describe_plan
from datamate.reasoning.strategies.base import Strategy


@register_strategy('rule_plan')
class RulePlan(Strategy):
    async def run(self, u, cat, ix):
        p = u.parse.plan
        if p is None:
            return [self.cand(status='skipped', error=u.parse.reason or 'no rule')]
        t0 = time.time()
        try:
            t, a = Executor(cat.canon, self.env.kb.rulebook.params).run(p)
            return [self.cand(program=describe_plan(p), answer_type=t, answer=a, weight=self.scfg.weight * u.parse.confidence,
                              elapsed=time.time() - t0)]
        except Unanswerable as e:
            # thiếu hẳn loại thông tin: đề xuất unanswerable với trọng số vừa phải (cần strategy khác đồng ý)
            return [self.cand(program=describe_plan(p), answer_type=AnswerType.UNANSWERABLE, reason=str(e),
                              weight=self.scfg.options.get('unanswerable_weight', 0.5), elapsed=time.time() - t0)]
        except (PlanError, KeyError, ValueError, TypeError, IndexError, ZeroDivisionError) as e:
            return [self.cand(program=describe_plan(p), status='error', error=repr(e)[:300], elapsed=time.time() - t0)]
