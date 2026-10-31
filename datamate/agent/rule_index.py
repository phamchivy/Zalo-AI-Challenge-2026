"""Truy hồi các dòng luật trong rulebook liên quan tới câu hỏi (không dùng LLM).

Rulebook đã nằm nguyên văn trong system prompt (được cache), nhưng mô hình 14B dễ bỏ sót luật nằm giữa văn bản dài.
Đưa thêm 3–6 dòng luật liên quan nhất vào phần câu hỏi giúp mô hình chú ý đúng chỗ.
Điểm = tổng IDF của các từ (và cặp từ) chung giữa câu hỏi và dòng luật; từ đệm / thời gian không tính.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from datamate.core.textutil import fold
from datamate.knowledge.rules import parse_statements

_STOP = {'la', 'cua', 'va', 'cho', 'thi', 'co', 'khong', 'trong', 'theo', 'mot', 'cac', 'nhung', 'duoc', 'bao', 'nhieu',
         'nao', 'gi', 'tu', 'den', 'het', 'ngay', 'thang', 'nam', 'quy', 'tuan', 'hom', 'nay', 'qua', 'truoc', 'sau', 'khi',
         'neu', 'voi', 'tai', 'o', 'ra', 'vao', 'de', 'ma', 'so', 'tinh', 'lan', 'ca', 'cung', 'do', 'nay', 'nhu', 'hay',
         'hoac', 'khac', 'van', 'chi', 'moi', 'tat', 'hon', 'bang', 'tren', 'duoi', 'vi', 'can', 'phai', 'deu'}


@dataclass
class RuleRow:
    section: str
    name: str
    text: str
    raw: str


def _toks(s: str) -> list[str]:
    return [t for t in re.findall(r'[a-z0-9]+', fold(s)) if t not in _STOP and len(t) >= 2]


class RuleIndex:
    def __init__(self, rulebook_text: str):
        self.rows = [RuleRow(st.section, st.name, st.text, st.raw) for st in parse_statements(rulebook_text) if st.text]
        df: Counter = Counter()
        self._feats = []
        for r in self.rows:
            t = _toks(r.name + ' ' + r.text)
            f = set(t) | {f'{a}_{b}' for a, b in zip(t, t[1:])}
            name_f = set(_toks(r.name))
            self._feats.append((f, name_f))
            df.update(f)
        n = max(len(self.rows), 1)
        self.idf = {k: math.log(1 + n / v) for k, v in df.items()}

    def search(self, question: str, k: int = 5, min_score: float = 2.5) -> list[RuleRow]:
        t = _toks(question)
        q = set(t) | {f'{a}_{b}' for a, b in zip(t, t[1:])}
        scored = []
        for i, (f, name_f) in enumerate(self._feats):
            common = q & f
            if not common:
                continue
            s = sum(self.idf.get(x, 0) for x in common) + 0.5 * sum(self.idf.get(x, 0) for x in q & name_f)
            scored.append((s, i))
        scored.sort(reverse=True)
        return [self.rows[i] for s, i in scored[:k] if s >= min_score]

    def block(self, question: str, k: int = 5) -> str:
        rows = self.search(question, k)
        if not rows:
            return ''
        lines = ['Luật liên quan nhất trong RULEBOOK (đọc kỹ trước khi tính):']
        for r in rows:
            raw = re.sub(r'\s+', ' ', r.raw.strip().strip('|').replace('|', '—'))
            lines.append(f'- [{r.section}] {raw[:400]}')
        return '\n'.join(lines) + '\n'
