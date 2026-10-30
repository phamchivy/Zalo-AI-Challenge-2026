"""Giao diện strategy + môi trường dùng chung (config, tri thức, LLM, sandbox, cache per-business)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from datamate.core.config import Config, StrategyConfig
from datamate.core.types import Candidate


@dataclass
class Env:
    cfg: Config
    kb: Any                                  # KnowledgeBase
    llm: Any = None                          # LLMClient | MockLLM | None
    sandbox: Any = None                      # Sandbox
    packs: dict = field(default_factory=dict)        # business_id -> pack path
    shop_blocks: dict = field(default_factory=dict)  # business_id -> prompt block (cache)
    deadline: float = float('inf')           # time.time() quá mốc -> chỉ chạy strategy không dùng LLM

    def llm_allowed(self) -> bool:
        return self.llm is not None and self.cfg.llm.enabled and time.time() < self.deadline


class Strategy:
    name = 'base'
    uses_llm = False

    def __init__(self, scfg: StrategyConfig, env: Env):
        self.scfg = scfg
        self.env = env

    async def run(self, u, cat, ix) -> list[Candidate]:  # pragma: no cover - interface
        raise NotImplementedError

    def cand(self, idx: int = 0, **kw) -> Candidate:
        return Candidate(cand_id=f'{self.name}:{idx}', strategy=self.name, weight=kw.pop('weight', self.scfg.weight), **kw)
