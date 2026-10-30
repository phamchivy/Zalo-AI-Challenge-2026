"""Rulebook của split: giữ nguyên văn (đưa vào prompt) + rút một số tham số định lượng cho tầng A.

Mỗi split có rulebook riêng (public khác train) -> KHÔNG hardcode rulebook train; tham số dưới đây chỉ là mặc định,
được ghi đè khi đọc được từ file rulebook của split đang chạy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from datamate.core.textutil import fold
from datamate.knowledge.rules import RuleSet, compile_rules


@dataclass
class RuleParams:
    morning: tuple[str, str] = ('06:00', '13:59')
    afternoon: tuple[str, str] = ('14:00', '21:59')
    tet_start_lunar_day: int = 23        # 23 tháng Chạp
    tet_end_lunar_day: int = 10          # mùng 10 tháng Giêng
    week_start: int = 1                  # thứ Hai
    aov_round_half_up: bool = True
    shipping_counts_as_revenue: bool = True
    cancel_rate_denominator_all: bool = True
    size_suffix: bool = True             # cà phê: (M)/(L) cuối tên


@dataclass
class Rulebook:
    text: str = ''
    sections: dict[str, str] = field(default_factory=dict)
    params: RuleParams = field(default_factory=RuleParams)
    rules: RuleSet = field(default_factory=RuleSet)          # luật nghiệp vụ dạng dữ liệu (knowledge/rules.py)

    def section(self, *keywords: str, max_chars: int = 4000) -> str:
        out = []
        for h, body in self.sections.items():
            if any(k in fold(h) for k in keywords):
                out.append(f'{h}\n{body}')
        return '\n\n'.join(out)[:max_chars]


def load_rulebook(path: str | Path) -> Rulebook:
    p = Path(path)
    text = p.read_text(encoding='utf-8') if p.exists() else ''
    rb = Rulebook(text=text, rules=compile_rules(text))
    cur, buf = None, []
    for line in text.splitlines():
        if line.startswith('## ') or line.startswith('### '):
            if cur:
                rb.sections[cur] = '\n'.join(buf).strip()
            cur, buf = line.strip('# ').strip(), []
        else:
            buf.append(line)
    if cur:
        rb.sections[cur] = '\n'.join(buf).strip()
    t = fold(text)
    if m := re.search(r'ca sang (\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})', t):
        rb.params.morning = (f'{int(m[1]):02d}:{m[2]}', f'{int(m[3]):02d}:{m[4]}')
    if m := re.search(r'ca chieu (\d{1,2}):(\d{2})\s*[-–]\s*(\d{1,2}):(\d{2})', t):
        rb.params.afternoon = (f'{int(m[1]):02d}:{m[2]}', f'{int(m[3]):02d}:{m[4]}')
    if m := re.search(r'(\d{1,2}) thang chap', t):
        rb.params.tet_start_lunar_day = int(m[1])
    if m := re.search(r'mung (\d{1,2}) thang gieng', t):
        rb.params.tet_end_lunar_day = int(m[1])
    return rb
