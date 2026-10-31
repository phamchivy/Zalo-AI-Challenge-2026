"""Profile từng bảng gốc: kiểu giá trị suy ra (num/date/id/text), tỉ lệ rỗng, mẫu, top value.
Dùng cho (1) schema mapper kiểm chứng vai trò cột bằng giá trị, (2) thẻ bảng đưa vào prompt LLM (tầng B)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from datamate.core.textutil import parse_date, to_num
from datamate.data.raw_loader import RawTable

SAMPLE = 3000


@dataclass
class ColumnProfile:
    name: str
    null_rate: float
    n_distinct: int
    num_rate: float          # tỉ lệ ô đọc được là số
    date_rate: float         # tỉ lệ ô đọc được là ngày
    serial_rate: float       # tỉ lệ ô là số serial ngày Excel (5 chữ số 36526..54789)
    dm_rate: float           # tỉ lệ ô dạng 'dd/mm' không năm (năm lấy theo sheet)
    int_like: bool
    neg_rate: float
    mean_len: float
    samples: list[str] = field(default_factory=list)
    top: dict[str, int] = field(default_factory=dict)

    @property
    def kind(self) -> str:
        if self.date_rate >= 0.8:
            return 'date'
        if self.num_rate >= 0.9:
            return 'num'
        return 'text'


@dataclass
class TableProfile:
    name: str
    n_rows: int
    columns: dict[str, ColumnProfile]
    files: list[str]
    sheets: list[str]


_DATEISH = re.compile(r'(\d{1,2}[/.\-]\d{1,2}[/.\-]\d{2,4})|(\d{4}-\d{1,2}-\d{1,2})|(thg)')
_NUMISH = re.compile(r'^\(?[-+]?\d[\d.,]*\)?\s*[^\d\s]{0,8}$')
# tiền viết tắt: '1tr440', '1tr5', '3tr450k', '1ty2'
_MONEY_SHORT = re.compile(r'^[-+]?\d{1,3}\s*(tr|triệu|trieu|tỷ|ty|tỉ|k)\s*\d{1,3}\s*(k|đ|d)?$', re.I)


def is_numberish(x) -> bool:
    """Ô là số thật sự ('38,000', '1.200.000đ', '2 hộp'), không phải tên có chứa số ('Panadol 500mg')."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return x == x
    t = str(x).strip()
    return bool(_NUMISH.match(t) or _MONEY_SHORT.match(t))


def profile_column(name: str, s: pd.Series) -> ColumnProfile:
    n = len(s)
    nn = s.dropna()
    nn = nn[nn.astype(str).str.strip() != '']
    sample = nn.head(SAMPLE)
    strs = sample.astype(str)
    nums = [to_num(x) if is_numberish(x) else None for x in sample.tolist()] if len(sample) else []
    num_ok = [x for x in nums if x is not None]
    # ngày: chỉ chuỗi có dáng ngày hoặc datetime (tránh nhầm số tiền thành serial Excel)
    date_ok = 0
    for x in sample.tolist()[:600]:
        if hasattr(x, 'year') or (isinstance(x, str) and _DATEISH.search(x) and parse_date(x)):
            date_ok += 1
    serial = sum(1 for x in strs.tolist()[:600] if re.fullmatch(r'\d{5}(\.0+)?', x.strip()) and 36526 <= float(x) <= 54789)
    dm = sum(1 for x in strs.tolist()[:600] if re.fullmatch(r'(0?[1-9]|[12]\d|3[01])[/.\-](0?[1-9]|1[0-2])', x.strip()))
    vc = strs.value_counts()
    return ColumnProfile(
        name=name,
        null_rate=round(1 - len(nn) / max(n, 1), 4),
        n_distinct=int(vc.size),
        num_rate=round(len(num_ok) / max(len(sample), 1), 3),
        date_rate=round(date_ok / max(min(len(sample), 600), 1), 3),
        serial_rate=round(serial / max(min(len(sample), 600), 1), 3),
        dm_rate=round(dm / max(min(len(sample), 600), 1), 3),
        int_like=bool(num_ok) and all(float(x).is_integer() for x in num_ok[:500]),
        neg_rate=round(sum(1 for x in num_ok if x < 0) / max(len(num_ok), 1), 4),
        mean_len=round(float(strs.str.len().mean()) if len(strs) else 0.0, 1),
        samples=[str(x)[:50] for x in vc.index[:6].tolist()] if vc.size <= 6 else [str(x)[:50] for x in sample.sample(min(6, len(sample)), random_state=0).tolist()],
        top={str(k)[:50]: int(v) for k, v in vc.head(12).items()},
    )


def profile_table(t: RawTable) -> TableProfile:
    cols = {c: profile_column(c, t.df[c]) for c in t.columns}
    return TableProfile(name=t.name, n_rows=len(t.df), columns=cols, files=t.files, sheets=t.sheets)


def table_card(tp: TableProfile, role_of: Optional[dict[str, str]] = None, max_cols: int = 40) -> str:
    """Thẻ bảng dạng text gọn cho prompt."""
    lines = [f'TABLE `{tp.name}` rows={tp.n_rows} files={tp.files[:3]}' + (f' sheets={tp.sheets[0]}..{tp.sheets[-1]}' if tp.sheets else '')]
    for c, p in list(tp.columns.items())[:max_cols]:
        role = f' role={role_of[c]}' if role_of and role_of.get(c) else ''
        vals = ', '.join(repr(v) for v in (list(p.top)[:5] if p.n_distinct <= 12 else p.samples[:4]))
        lines.append(f'  - `{c}` [{p.kind}{role}] null={p.null_rate:.0%} distinct={p.n_distinct} e.g. {vals}')
    return '\n'.join(lines)
