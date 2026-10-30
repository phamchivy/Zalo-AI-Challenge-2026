"""Phát hiện câu không trả lời được (ngoài khoảng dữ liệu / ngoài phạm vi dữ liệu)."""
from __future__ import annotations

from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType
from datamate.reasoning.strategies.base import Strategy


@register_strategy('guard')
class Guard(Strategy):
    async def run(self, u, cat, ix):
        r = u.ctx.unanswerable_reason
        if not r:
            return [self.cand(status='skipped')]
        return [self.cand(answer_type=AnswerType.UNANSWERABLE, reason=r)]
