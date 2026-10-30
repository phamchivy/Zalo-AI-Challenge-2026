"""Đọc README.md của doanh nghiệp: từ điển cột, quy ước, ngày cuối dữ liệu, bảng quy đổi đơn vị ghi trong README.

README của bộ đề dùng một bộ từ vựng mô tả cột thống nhất ("mã đơn", "thành tiền sau giảm",
"số lượng đã đổi ra đơn vị cơ bản"...) -> đây là nguồn chính để ánh xạ cột, ổn định hơn tên cột.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from datamate.core.textutil import fold, parse_date


@dataclass
class ReadmeInfo:
    text: str = ''
    columns: dict[str, dict[str, str]] = field(default_factory=dict)   # file_key -> {col_name: desc}
    conventions: dict[str, str] = field(default_factory=dict)          # label -> text
    data_end: Optional[date] = None
    unit_conversions: list[tuple[str, str, float, str]] = field(default_factory=list)  # (product, big_unit, factor, base_unit)

    def desc_for(self, file_name: str, sheet: Optional[str], col: str) -> Optional[str]:
        """Tìm mô tả cột: ưu tiên đúng file+sheet, rồi đúng file, rồi bất kỳ file nào có cột trùng tên."""
        fcol = fold(col)
        keys = []
        fstem = fold(file_name)
        for k in self.columns:
            fk = fold(k)
            score = 0
            if fstem and fstem.split('.')[0] in fk:
                score += 2
            if sheet and fold(sheet) in fk:
                score += 1
            keys.append((score, k))
        for _, k in sorted(keys, key=lambda x: -x[0]):
            for c, d in self.columns[k].items():
                if fold(c) == fcol:
                    return d
        return None


_SECTION = re.compile(r'^##\s+Cột trong\s+(.+?)\s*$', re.M)
_ITEM = re.compile(r'^\s*-\s+\*\*(.+?)\*\*\s*:\s*(.+?)\s*$', re.M)
_END = re.compile(r'có đến hết\s+(\d{1,2}/\d{1,2}/\d{4})', re.I)
_CONV = re.compile(r'^\s*-\s+(.+?):\s*1\s+(.+?)\s*=\s*([\d.,]+)\s+(.+?)\s*$', re.M)


def parse_readme(text: str) -> ReadmeInfo:
    info = ReadmeInfo(text=text or '')
    if not text:
        return info
    # các section "## Cột trong `file`[, sheet `x`]"
    heads = list(_SECTION.finditer(text))
    for i, h in enumerate(heads):
        body = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        body = body.split('\n## ')[0]
        key = h.group(1).replace('`', '').strip()
        info.columns[key] = {m.group(1).strip(): m.group(2).strip() for m in _ITEM.finditer(body)}
    # quy ước
    q = re.search(r'^##\s+Quy ước\s*$(.+?)(?=^##\s|\Z)', text, re.M | re.S)
    if q:
        for m in _ITEM.finditer(q.group(1)):
            info.conventions[m.group(1).strip()] = m.group(2).strip()
    if m := _END.search(text):
        info.data_end = parse_date(m.group(1))
    for m in _CONV.finditer(text):
        prod, big, fac, base = m.group(1).strip(), m.group(2).strip(), m.group(3), m.group(4).strip()
        try:
            f = float(fac.replace(',', '.'))
        except ValueError:
            continue
        if len(prod) < 80 and len(base) < 20:
            info.unit_conversions.append((prod, big, f, base))
    return info


def convention_flags(info: ReadmeInfo) -> dict[str, bool]:
    """Các quy ước đặc biệt ảnh hưởng tầng A (phần còn lại để LLM đọc README)."""
    t = fold(info.text)
    return {
        'negative_qty_return': 'so luong am' in t,
        'status_in_note': bool(re.search(r'(huy|tra lai|dang giao).{0,60}(ghi o|ghi chu)', t)) or 'khong co cot trang thai' in t,
        'topping_joined': 'topping' in t and '" + "' in info.text,
        'service_fee_line': 'phi phuc vu' in t,
        'non_product_rows': 'voucher' in t and 'dung' in t and 'diem' in t,
        'abbreviations': 'ghi tat' in t,
    }
