"""Đọc biểu thức thời gian (trên câu đã qnorm) -> danh sách TimeRange theo thứ tự xuất hiện + cờ ngữ cảnh.

Hỗ trợ: hôm nay/qua/kia, ngày D, từ D1 đến D2, đến hết ngày D (snapshot), N ngày gần nhất, tuần này/trước,
tháng M/Y (cả dạng Th6/2025, T1/26, 6-2025, 8/26 khi có chữ 'tháng'), tháng M không năm, tháng này/trước,
quý Q/Y, quý này/trước, năm Y, năm nay/trước, từ đầu năm, Tết năm Y, N ngày trước Tết, chuỗi tháng
(“3 tháng liền nhau kết thúc ở tháng 6/2025”, “từ tháng 4/2025 đến tháng 6/2025”, “các tháng 11/2025, 12/2025 và 1/2026”).
Luật diễn giải lấy theo rulebook §2 (tháng không ghi năm: năm nay nếu không sau hôm nay, ngược lại năm trước...).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from datamate.core.textutil import add_months, month_end
from datamate.core.types import TimeRange
from datamate.knowledge.calendar import LunarCalendar


@dataclass
class TimeParse:
    ranges: list[TimeRange] = field(default_factory=list)
    flags: set[str] = field(default_factory=set)       # prev_period | yoy | mtd_compare | before_tet | series | snapshot
    n_before: Optional[int] = None                      # "N ngày trước Tết"


def _y(s: str) -> int:
    y = int(s)
    return 2000 + y if y < 100 else y


def _month(y: int, m: int, expr: str, label: str = 'main') -> TimeRange:
    return TimeRange(start=date(y, m, 1), end=month_end(y, m), label=label, expr=expr, granularity='month')


def _quarter(y: int, q: int, expr: str) -> TimeRange:
    return TimeRange(start=date(y, 3 * q - 2, 1), end=month_end(y, 3 * q), expr=expr, granularity='quarter')


def _day(d: date, expr: str) -> TimeRange:
    return TimeRange(start=d, end=d, expr=expr, granularity='day')


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def resolve_time(t: str, as_of: date, cal: Optional[LunarCalendar] = None) -> TimeParse:
    """t: câu đã qnorm. Trả về các khoảng theo thứ tự xuất hiện trong câu."""
    found: list[tuple[int, TimeRange]] = []
    flags: set[str] = set()
    used = [False] * (len(t) + 1)

    def take(m: re.Match) -> bool:
        a, b = m.span()
        if any(used[a:b]):
            return False
        for i in range(a, b):
            used[i] = True
        return True

    def add(pos: int, tr: TimeRange):
        found.append((pos, tr))

    D = r'(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})'
    # ---- khoảng ngày tường minh: từ D1 đến (hết) D2
    for m in re.finditer(rf'(?:tu |giai doan |trong khoang )*(?:ngay )?{D}\s*(?:den|toi|-|cho den|đen)\s*(?:het )?(?:ngay )?{D}', t):
        if take(m):
            a = _safe_date(int(m[3]), int(m[2]), int(m[1]))
            b = _safe_date(int(m[6]), int(m[5]), int(m[4]))
            if a and b:
                add(m.start(), TimeRange(start=a, end=b, expr=m[0], granularity='custom'))
    for m in re.finditer(r'tu (?:ngay )?(\d{1,2})[/.\-](\d{1,2})(?:[/.\-](\d{4}))?\s*(?:den|toi) (?:het )?(?:ngay )?(?:hom nay|nay|bay gio)', t):
        if take(m):
            y = int(m[3]) if m[3] else as_of.year
            a = _safe_date(y, int(m[2]), int(m[1]))
            if a:
                add(m.start(), TimeRange(start=a, end=as_of, expr=m[0], granularity='custom'))
    # ---- đến hết ngày D / tính đến ngày D / cuối ngày D (tình trạng)
    for m in re.finditer(rf'(?:tinh |tinh )?(?:den|toi|cho den) (?:het |cuoi )?(?:ngay )?{D}|cuoi ngay {D}|theo ton kho ngay {D}', t):
        if take(m):
            g = [x for x in m.groups() if x is not None]
            d = _safe_date(int(g[2]), int(g[1]), int(g[0]))
            if d:
                add(m.start(), TimeRange(start=date(2000, 1, 1), end=d, label='snapshot', expr=m[0], granularity='snapshot'))
                flags.add('snapshot')
    # ---- ngày D
    for m in re.finditer(rf'{D}', t):
        if take(m):
            d = _safe_date(int(m[3]), int(m[2]), int(m[1]))
            if d:
                add(m.start(), _day(d, m[0]))
    for m in re.finditer(r'\bngay (\d{1,2})[/.\-](\d{1,2})\b(?![/.\-]\d)', t):
        if take(m):
            dd, mm = int(m[1]), int(m[2])
            d = _safe_date(as_of.year, mm, dd)
            if d and d > as_of:
                d = _safe_date(as_of.year - 1, mm, dd)
            if d:
                add(m.start(), _day(d, m[0]))
    # ---- chuỗi tháng: "N thang lien (nhau|tiep) ket thuc o thang M/Y"
    for m in re.finditer(r'(\d+) thang (?:lien (?:nhau|tiep)|gan nhat|lien ke)?\s*(?:ket thuc|den het|cuoi cung la)?\s*(?:o|vao|tai)?\s*thang (\d{1,2})\s*(?:/|-|nam )\s*(\d{4}|\d{2})\b', t):
        if take(m):
            n, mm, yy = int(m[1]), int(m[2]), _y(m[3])
            for k in range(n - 1, -1, -1):
                d = add_months(date(yy, mm, 1), -k)
                add(m.start() + (n - k), _month(d.year, d.month, m[0], label=f'series_{n - 1 - k}'))
            flags.add('series')
    # ---- từ tháng A/Y (đến|tới|qua) tháng B/Y  -> dải tháng
    for m in re.finditer(r'tu thang (\d{1,2})\s*(?:/|-|nam )\s*(\d{4}|\d{2})\s*(?:den|toi|qua|sang)\s*(?:het )?thang (\d{1,2})\s*(?:/|-|nam )\s*(\d{4}|\d{2})\b', t):
        tail = t[m.end():m.end() + 40]
        if re.match(r'\s*(roi|,)\s*(?:qua |sang |den )?thang \d', tail):
            continue   # "từ tháng 4 qua tháng 5 rồi tháng 6": để bộ đọc tháng lẻ xử lý
        if take(m):
            a = date(_y(m[2]), int(m[1]), 1)
            b = date(_y(m[4]), int(m[3]), 1)
            k = 0
            while a <= b and k < 36:
                add(m.start() + k, _month(a.year, a.month, m[0], label=f'series_{k}'))
                a = add_months(a, 1)
                k += 1
            flags.add('series')
    # ---- quý
    for m in re.finditer(r'\bquy (\d)\s*(?:/|-|nam |\.)\s*(\d{4}|\d{2})\b', t):
        if take(m):
            q, yy = int(m[1]), _y(m[2])
            if 1 <= q <= 4:
                add(m.start(), _quarter(yy, q, m[0]))
    for m in re.finditer(r'\bquy (\d)\b(?!\s*(?:/|-|nam \d))', t):
        if take(m):
            q = int(m[1])
            if 1 <= q <= 4:
                y = as_of.year if date(as_of.year, 3 * q - 2, 1) <= as_of else as_of.year - 1
                add(m.start(), _quarter(y, q, m[0]))
    for m in re.finditer(r'\bquy (nay|truoc|vua roi|roi)\b', t):
        if take(m):
            q = (as_of.month - 1) // 3 + 1
            if m[1] == 'nay':
                add(m.start(), TimeRange(start=date(as_of.year, 3 * q - 2, 1), end=as_of, expr=m[0], granularity='quarter'))
            else:
                y, q = (as_of.year, q - 1) if q > 1 else (as_of.year - 1, 4)
                add(m.start(), _quarter(y, q, m[0]))
    # ---- tháng M/Y (nhiều dạng); dạng liệt kê sau chữ 'thang'
    for m in re.finditer(r'\b(?:thang|t)\s?(\d{1,2})\s*(?:/|-|nam |\.)\s*(\d{4}|\d{2})\b', t):
        if take(m):
            mm, yy = int(m[1]), _y(m[2])
            if 1 <= mm <= 12:
                add(m.start(), _month(yy, mm, m[0]))
            # liệt kê tiếp: ", 12/2025 va 1/2026"
            pos = m.end()
            while True:
                m2 = re.match(r'\s*(?:,|va|và|;|roi|qua|sang|-)\s*(?:thang |t)?(\d{1,2})\s*(?:/|-)\s*(\d{4}|\d{2})\b', t[pos:])
                if not m2:
                    break
                a, b = pos + m2.start(), pos + m2.end()
                if any(used[a:b]):
                    break
                for i in range(a, b):
                    used[i] = True
                mm2, yy2 = int(m2[1]), _y(m2[2])
                if 1 <= mm2 <= 12:
                    add(a, _month(yy2, mm2, m2[0]))
                pos = b
    for m in re.finditer(r'(?<![\d/])(\d{1,2})[/\-](\d{4})\b', t):
        if take(m):
            mm, yy = int(m[1]), int(m[2])
            if 1 <= mm <= 12:
                add(m.start(), _month(yy, mm, m[0]))
    # ---- tháng M không năm / tháng này / tháng trước
    for m in re.finditer(r'\bthang (\d{1,2})\b(?!\s*(?:/|-|nam \d|\.\d))', t):
        if take(m):
            mm = int(m[1])
            if 1 <= mm <= 12:
                y = as_of.year if date(as_of.year, mm, 1) <= as_of else as_of.year - 1
                add(m.start(), _month(y, mm, m[0]))
    for m in re.finditer(r'\b(?:tu )?(?:dau )?thang nay\b(?:\s*(?:den|toi) (?:het )?(?:hom nay|nay))?', t):
        if take(m):
            add(m.start(), TimeRange(start=as_of.replace(day=1), end=as_of, expr=m[0], granularity='month'))
    for m in re.finditer(r'\bcung ky thang truoc\b|\bcung (?:so ngay|ky) (?:cua )?thang truoc\b', t):
        if take(m):
            flags.add('mtd_compare')
    for m in re.finditer(r'\bthang (?:lien truoc|truoc nua)\b', t):
        if take(m):
            flags.add('prev_period')
    for m in re.finditer(r'\bthang truoc\b', t):
        if take(m):
            p = add_months(as_of.replace(day=1), -1)
            add(m.start(), _month(p.year, p.month, m[0]))
    # ---- tuần
    for m in re.finditer(r'\btuan nay\b', t):
        if take(m):
            add(m.start(), TimeRange(start=as_of - timedelta(days=as_of.weekday()), end=as_of, expr=m[0], granularity='week'))
    for m in re.finditer(r'\btuan truoc\b', t):
        if take(m):
            s = as_of - timedelta(days=as_of.weekday() + 7)
            add(m.start(), TimeRange(start=s, end=s + timedelta(days=6), expr=m[0], granularity='week'))
    # ---- N ngày gần nhất
    for m in re.finditer(r'\b(\d+) ngay (?:gan nhat|gan day|qua|vua qua|lien gan nhat|cuoi|gan nhat tinh ca hom nay|tro lai day)\b', t):
        if take(m):
            n = int(m[1])
            add(m.start(), TimeRange(start=as_of - timedelta(days=n - 1), end=as_of, expr=m[0], granularity='custom'))
    # ---- Tết
    for m in re.finditer(r'\b(?:dip |mua )?tet(?: am lich| nguyen dan| co truyen)?(?: nam)? (\d{4})\b', t):
        if take(m) and cal:
            w = cal.tet_window(int(m[1]))
            if w:
                add(m.start(), TimeRange(start=w[0], end=w[1], expr=m[0], granularity='tet'))
    for m in re.finditer(r'\b(\d+) ngay (?:ngay )?truoc tet\b', t):
        if take(m):
            flags.add('before_tet')
            found.append((m.start(), TimeRange(start=as_of, end=as_of, label='before_tet', expr=m[0], granularity='custom')))
    for m in re.finditer(r'\b(?:dip |mua )?tet (?:nay|vua roi|vua qua|nam nay)\b|\b(?:dip |mua )?tet nam truoc\b|\b(?:dip|mua) tet\b(?! nam)', t):
        if take(m) and cal:
            y = as_of.year + (1 if 'nam truoc' not in m[0] else 0)
            w = None
            for yy in (y, y - 1, y - 2):
                w = cal.tet_window(yy)
                if w and w[0] <= as_of:
                    if 'nam truoc' in m[0]:
                        w = cal.tet_window(yy - 1)
                    break
            if w:
                add(m.start(), TimeRange(start=w[0], end=w[1], expr=m[0], granularity='tet'))
    # ---- năm
    for m in re.finditer(r'\btu dau nam(?: \d{4})?(?: (?:toi|den) (?:nay|hom nay|het hom nay))?\b', t):
        if take(m):
            add(m.start(), TimeRange(start=date(as_of.year, 1, 1), end=as_of, expr=m[0], granularity='year'))
    for m in re.finditer(r'\bnam (\d{4})\b', t):
        if take(m):
            y = int(m[1])
            add(m.start(), TimeRange(start=date(y, 1, 1), end=date(y, 12, 31), expr=m[0], granularity='year'))
    for m in re.finditer(r'\bnam nay\b', t):
        if take(m):
            add(m.start(), TimeRange(start=date(as_of.year, 1, 1), end=as_of, expr=m[0], granularity='year'))
    for m in re.finditer(r'\bcung (?:thang|ky) nam truoc\b|\bcung ky nam (?:truoc|ngoai)\b|\bso voi nam truoc\b', t):
        if take(m):
            flags.add('yoy')
    for m in re.finditer(r'\bnam truoc\b', t):
        if take(m):
            y = as_of.year - 1
            add(m.start(), TimeRange(start=date(y, 1, 1), end=date(y, 12, 31), expr=m[0], granularity='year'))
    # ---- ngày tương đối
    for m in re.finditer(r'\bhom nay\b|\bngay hom nay\b|\bhien tai\b|\bhien gio\b', t):
        if take(m):
            add(m.start(), _day(as_of, m[0]))
    for m in re.finditer(r'\bhom qua\b', t):
        if take(m):
            add(m.start(), _day(as_of - timedelta(days=1), m[0]))
    for m in re.finditer(r'\bhom kia\b', t):
        if take(m):
            add(m.start(), _day(as_of - timedelta(days=2), m[0]))

    found.sort(key=lambda x: x[0])
    tp = TimeParse(ranges=[tr for _, tr in found], flags=flags)
    if 'before_tet' in flags:
        m = re.search(r'\b(\d+) ngay (?:ngay )?truoc tet\b', t)
        tp.n_before = int(m[1]) if m else None
        tp.ranges = [r for r in tp.ranges if r.label != 'before_tet']
    return tp
