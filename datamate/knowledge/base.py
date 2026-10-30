from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from datamate.knowledge.calendar import LunarCalendar
from datamate.knowledge.rulebook import Rulebook, load_rulebook
from datamate.knowledge.rules import RuleSet


@dataclass
class KnowledgeBase:
    """Tri thức của split đang chạy: rulebook (nguyên văn + tham số + luật dạng dữ liệu) và lịch âm."""
    rulebook: Rulebook
    calendar: LunarCalendar

    @property
    def rules(self) -> RuleSet:
        return self.rulebook.rules

    @classmethod
    def load(cls, split_dir: str | Path, rules_override: str | Path | None = None) -> 'KnowledgeBase':
        """rules_override: file rules.json (LLM/người điền cùng schema) ghi đè bộ đọc regex.
        Mặc định tìm <split>/rules.json rồi biến môi trường DATAMATE_RULES."""
        d = Path(split_dir)
        rb = load_rulebook(d / 'rulebook.md')
        ov = rules_override or os.environ.get('DATAMATE_RULES') or (d / 'rules.json')
        if ov and Path(ov).exists():
            rb.rules = rb.rules.merged(RuleSet.from_json(Path(ov).read_text(encoding='utf-8')))
        return cls(rulebook=rb, calendar=LunarCalendar.load(d / 'calendar.csv'))
