"""Tiện ích văn bản / số / ngày dùng chung cho mọi module (deterministic, không phụ thuộc LLM)."""
from __future__ import annotations

import math
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Optional

# ----------------------------------------------------------------------------- text
_CAMEL = re.compile(r'(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')


def nfc(s: Any) -> str:
    return unicodedata.normalize('NFC', str(s))


def fold(s: Any) -> str:
    """Chữ thường, bỏ dấu, đ->d, gộp khoảng trắng. Dùng để so khớp mờ (KHÔNG dùng để xuất đáp án)."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ''
    t = unicodedata.normalize('NFD', str(s).lower())
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn').replace('đ', 'd')
    return ' '.join(t.split())


def fold_name(s: Any) -> str:
    """fold + tách camelCase + bỏ ký tự đặc biệt: 'MaHoaDon'->'ma hoa don', 'T.Tiền'->'t tien'."""
    s = _CAMEL.sub(' ', str(s)).replace('_', ' ')
    return ' '.join(re.sub(r'[^a-z0-9%]+', ' ', fold(s)).split())


def norm_answer_str(s: Any) -> str:
    """Chuẩn hoá chuỗi đúng như bộ chấm: NFC, lower, ký tự không phải chữ/số/space -> space, gộp space."""
    t = unicodedata.normalize('NFC', str(s)).lower()
    t = ''.join(ch if (ch.isalnum() or ch.isspace()) else ' ' for ch in t)
    return ' '.join(t.split())


_PHONE = re.compile(r'[\(\-–:,]?\s*\(?\+?\d[\d .]{7,}\d\)?\s*$')


def strip_phone(name: Any) -> str:
    """'Cao Long Minh (0321806537)' -> 'Cao Long Minh'; 'Hoàng Trung Hương - 0971 618 698' -> 'Hoàng Trung Hương'."""
    s = str(name).strip()
    s = _PHONE.sub('', s).strip(' -–:,(')
    return s


def customer_key(name: Any) -> str:
    return fold(strip_phone(name))


WALKIN = {'', 'khach le', 'le', 'kl', 'k l', 'khach vang lai', 'vang lai', 'khach', 'nan', 'none', '-', '0', 'khong',
          'khach le (khong luu)', 'khach khong luu'}


def is_walkin(name: Any) -> bool:
    return fold(name) in WALKIN


# ----------------------------------------------------------------------------- numbers
_NUM_GROUPED = re.compile(r'^-?\d{1,3}([.,]\d{3})+$')


def to_num(v: Any) -> Optional[float]:
    """'560,000' / '38000' / '1.200.000đ' / '1 gói' / '18.6' / '-12,5' / '407k' / '2,176tr' / '(5.000)' -> số.
    None nếu không đọc được."""
    if v is None:
        return None
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)
    s = str(v).strip().lower()
    if s in ('', 'nan', 'none', '-', 'null', 'nat'):
        return None
    neg = s.startswith('-') or s.startswith('−') or (s.startswith('(') and s.endswith(')'))
    m = re.search(r'\d[\d.,]*', s)
    if not m:
        return None
    t = m.group(0).rstrip('.,')
    tail = s[m.end():].strip()
    mult = 1.0
    if re.match(r'^(k|nghin|nghìn|ngan|ngàn)\b', tail):
        mult = 1e3
    elif re.match(r'^(tr|triệu|trieu)\b', tail):
        mult = 1e6
    elif re.match(r'^(tỷ|ty|tỉ|ti)\b', tail):
        mult = 1e9
    if mult > 1 and (t.count(',') + t.count('.')) == 1 and re.search(r'[.,]\d{1,3}$', t):
        t = t.replace(',', '.')                       # '2,176tr' = 2.176 triệu
    elif _NUM_GROUPED.match(t):
        t = re.sub(r'[.,]', '', t)
    elif t.count(',') == 1 and t.count('.') == 0:
        t = t.replace(',', '.')
    elif t.count('.') > 1 or (t.count(',') >= 1 and t.count('.') >= 1):
        last = max(t.rfind('.'), t.rfind(','))
        t = re.sub(r'[.,]', '', t[:last]) + '.' + t[last + 1:]
    try:
        x = float(t) * mult
    except ValueError:
        return None
    return -x if neg else x


# ----------------------------------------------------------------------------- dates
_VN_MONTH = re.compile(r'^(\d{1,2})\s*thg\s*(\d{1,2})\s*,?\s*(\d{4})')
_DMY = re.compile(r'^(\d{1,2})[/.\-](\d{1,2})[/.\-](\d{4})')
_YMD = re.compile(r'^(\d{4})[/.\-](\d{1,2})[/.\-](\d{1,2})')
_DM = re.compile(r'^(\d{1,2})[/.\-](\d{1,2})$')
_TIME = re.compile(r'(\d{1,2}):(\d{2})(?::(\d{2}))?')


def parse_date(v: Any, default_year: Optional[int] = None) -> Optional[date]:
    """Nhận mọi định dạng gặp trong dữ liệu: dd/mm/yyyy, yyyy-mm-dd[ hh:mm:ss], '1 thg 6 2025', serial Excel,
    'dd/mm' (lấy năm của sheet). Trả None nếu không đọc được."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        if isinstance(v, float) and math.isnan(v):
            return None
        if 20000 < v < 80000:
            return date(1899, 12, 30) + timedelta(days=int(v))
        return None
    s = str(v).strip().lower()
    if not s or s in ('nan', 'none', 'nat'):
        return None
    try:
        if m := _YMD.match(s):
            return date(int(m[1]), int(m[2]), int(m[3]))
        if m := _DMY.match(s):
            return date(int(m[3]), int(m[2]), int(m[1]))
        if m := _VN_MONTH.match(s):
            return date(int(m[3]), int(m[2]), int(m[1]))
        if re.fullmatch(r'\d{5}(\.0+)?', s):
            return date(1899, 12, 30) + timedelta(days=int(float(s)))
        if (m := _DM.match(s)) and default_year:
            return date(default_year, int(m[2]), int(m[1]))
    except ValueError:
        return None
    return None


def parse_time(v: Any) -> Optional[str]:
    """Trả 'HH:MM:SS' nếu chuỗi có giờ."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.strftime('%H:%M:%S')
    s = str(v)
    if re.fullmatch(r'0?\.\d+', s.strip()):           # Excel time fraction
        sec = round(float(s) * 86400)
        return f'{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}'
    m = _TIME.search(s)
    if not m:
        return None
    return f'{int(m[1]):02d}:{int(m[2]):02d}:{int(m[3] or 0):02d}'


def year_from_text(s: Any) -> Optional[int]:
    m = re.search(r'(20\d{2})', str(s))
    return int(m[1]) if m else None


def month_end(y: int, m: int) -> date:
    return (date(y + m // 12, m % 12 + 1, 1) - timedelta(days=1))


def add_months(d: date, k: int) -> date:
    y, m = divmod(d.month - 1 + k, 12)
    y += d.year
    m += 1
    return date(y, m, min(d.day, month_end(y, m).day))
