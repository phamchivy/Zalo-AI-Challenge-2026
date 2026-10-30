from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from datamate.knowledge.calendar import LunarCalendar
from datamate.knowledge.rulebook import Rulebook, load_rulebook


@dataclass
class KnowledgeBase:
    """Tri thức của split đang chạy: rulebook (nguyên văn + tham số) và lịch âm."""
    rulebook: Rulebook
    calendar: LunarCalendar

    @classmethod
    def load(cls, split_dir: str | Path) -> 'KnowledgeBase':
        d = Path(split_dir)
        return cls(rulebook=load_rulebook(d / 'rulebook.md'), calendar=LunarCalendar.load(d / 'calendar.csv'))
