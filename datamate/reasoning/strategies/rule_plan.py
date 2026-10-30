"""Tầng A: rule parser -> Plan -> Executor (deterministic). Không gọi LLM.

Trọng số candidate = độ tin cậy theo đặc trưng (decision/calibration.py), không phải hằng số.
Plan cần luật mà rulebook không xác nhận / dữ liệu thiếu năng lực -> RuleMissing -> skipped (bỏ trống).
"""
from __future__ import annotations

import time

from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType
from datamate.decision.calibration import Calibration, rule_confidence
from datamate.reasoning.executor import Executor, PlanError, RuleMissing, Unanswerable
from datamate.reasoning.prompts import describe_plan
from datamate.reasoning.strategies.base import Strategy


@register_strategy('rule_plan')
class RulePlan(Strategy):
    _cal = None

    @property
    def cal(self) -> Calibration:
        if self._cal is None:
            self._cal = Calibration.load(self.env.cfg.decision.calibration_path)
        return self._cal

    async def run(self, u, cat, ix):
        p = u.parse.plan
        if p is None:
            return [self.cand(status='skipped', error=u.parse.reason or 'no rule')]
        t0 = time.time()
        ex = Executor(cat.canon, self.env.kb.rulebook.params, self.env.kb.rules)
        conf, meta = rule_confidence(u.parse, cat.canon, self.env.cfg.decision, self.cal)
        meta['intent'] = u.parse.intent
        try:
            t, a = ex.run(p)
            return [self.cand(program=describe_plan(p), answer_type=t, answer=a, weight=self.scfg.weight * conf,
                              elapsed=time.time() - t0, meta=meta)]
        except Unanswerable as e:
            # thiếu hẳn loại thông tin: đề xuất unanswerable với trọng số vừa phải (cần strategy khác đồng ý)
            return [self.cand(program=describe_plan(p), answer_type=AnswerType.UNANSWERABLE, reason=str(e),
                              weight=self.scfg.options.get('unanswerable_weight', 0.5), elapsed=time.time() - t0, meta=meta)]
        except RuleMissing as e:
            meta['rule_missing'] = str(e)[:300]
            return [self.cand(program=describe_plan(p), status='skipped', error=f'rule/caps: {e}'[:300],
                              elapsed=time.time() - t0, meta=meta)]
        except (PlanError, KeyError, ValueError, TypeError, IndexError, ZeroDivisionError, AttributeError) as e:
            return [self.cand(program=describe_plan(p), status='error', error=repr(e)[:300], elapsed=time.time() - t0, meta=meta)]
