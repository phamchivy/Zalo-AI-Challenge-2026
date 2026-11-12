"""v3.1.0 — Hàm hỗ trợ TỔNG QUÁT cho code của LLM (chạy trong kernel).

Nguyên tắc: hàm chỉ xử lý NGHĨA CỦA DỮ LIỆU VÀ NGÔN NGỮ (thời gian, so khớp tên, xem cấu trúc bảng, lịch âm).
Hàm KHÔNG chứa luật nghiệp vụ (đơn nào tính doanh thu, nợ tính thế nào, Tết là khoảng nào...): LLM đọc RULEBOOK và
tự áp. LLM không bắt buộc dùng các hàm này; phần còn lại viết bằng pandas tự do.

  period(text, as_of)        cụm thời gian tiếng Việt -> các khoảng ngày (dùng bộ đọc thời gian đã kiểm thử của tầng A)
  find(values, text, k)      giá trị trong dữ liệu khớp nhất với cụm chữ trong câu (bỏ dấu, viết tắt, sai chính tả nhẹ)
  describe(df, name)         cột, kiểu, % trống, số giá trị khác nhau, giá trị mẫu
  lunar(d) / solar(...) / holidays(kw)   tra lịch âm-dương và ngày lễ từ calendar.csv của split
"""
from __future__ import annotations

import re
from datetime import date, datetime
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Iterable, Optional

from datamate.core.textutil import fold

MAX_VALUES = 20000


# ----------------------------------------------------------------------------- thời gian
def _as_date(x: Any) -> date:
    if x is None:
        raise ValueError('thiếu as_of')
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    if hasattr(x, 'date') and callable(x.date):          # pd.Timestamp
        return x.date()
    return date.fromisoformat(str(x)[:10])


@lru_cache(maxsize=8)
def _calendar(path: str):
    from datamate.knowledge.calendar import LunarCalendar
    return LunarCalendar.load(path) if path else LunarCalendar([])


def period(text: str, as_of: Any, calendar_path: str = '') -> list[dict]:
    """'tháng trước', 'T4/26', 'quý 2 năm nay', 'từ 3/5 đến 10/5', '7 ngày gần nhất'... -> danh sách
    {'start', 'end' (pd.Timestamp, gồm cả hai đầu), 'expr', 'label', 'granularity'} theo thứ tự xuất hiện.
    Không đọc được -> []."""
    import pandas as pd

    from datamate.understand.normalize import qnorm
    from datamate.understand.time_resolver import resolve_time
    tp = resolve_time(qnorm(str(text)), _as_date(as_of), _calendar(calendar_path))
    out = []
    for r in tp.ranges:
        out.append({'start': pd.Timestamp(r.start), 'end': pd.Timestamp(r.end), 'expr': r.expr, 'label': r.label,
                    'granularity': getattr(r, 'granularity', None)})
    return out


def lunar(d: Any, calendar_path: str = '') -> Optional[dict]:
    """Ngày dương -> {'ngay_am', 'thang_am', 'nam_am', 'nhuan', 'thu' (1=Thứ Hai), 'le'} hoặc None nếu ngoài lịch."""
    rec = _calendar(calendar_path).by_date.get(_as_date(d))
    if rec is None:
        return None
    return {'ngay_am': rec['ld'], 'thang_am': rec['lm'], 'nam_am': rec['ly'], 'nhuan': rec['leap'], 'thu': rec['weekday'],
            'le': rec['holiday']}


def solar(nam_am: int, thang_am: int, ngay_am: int, nhuan: bool = False, calendar_path: str = ''):
    """Ngày âm -> pd.Timestamp ngày dương (None nếu ngoài lịch)."""
    import pandas as pd
    d = _calendar(calendar_path).solar(int(nam_am), int(thang_am), int(ngay_am), bool(nhuan))
    return pd.Timestamp(d) if d else None


def holidays(keyword: str, calendar_path: str = '') -> list:
    """Các ngày có tên lễ chứa keyword (vd 'Tết', 'Quốc khánh') theo calendar.csv."""
    import pandas as pd
    k = fold(keyword)
    cal = _calendar(calendar_path)
    return [pd.Timestamp(d) for d, r in sorted(cal.by_date.items()) if r['holiday'] and k in fold(r['holiday'])]


# ----------------------------------------------------------------------------- so khớp tên
def _toks(s: str) -> list[str]:
    return re.findall(r'[a-z0-9]+', s)


def _score(q: str, qt: set[str], v: str) -> float:
    if not v:
        return 0.0
    if q == v:
        return 1.0
    qc, vc = q.replace(' ', ''), v.replace(' ', '')
    if qc == vc:
        return 0.98
    vt = set(_toks(v))
    if qc and (qc in vc or vc in qc):
        short, long_ = sorted((len(qc), len(vc)))
        sub = 0.75 + 0.2 * short / long_
    else:
        sub = 0.0
    cover = len(qt & vt) / len(qt) if qt else 0.0             # phần chữ của câu có trong giá trị
    prec = len(qt & vt) / len(vt) if vt else 0.0              # phần chữ của giá trị có trong câu
    ratio = SequenceMatcher(None, qc, vc).ratio()
    return max(sub, 0.5 * cover + 0.2 * prec + 0.3 * ratio)


def find(values: Iterable, text: str, k: int = 5, min_score: float = 0.5, scores: bool = False) -> list:
    """Các giá trị (khác nhau) trong `values` khớp nhất với cụm chữ `text`, tốt nhất trước.
    values: Series / list / Index (vd P['product_name'], L['unit'], O['channel']). So khớp không dấu, không phân biệt
    hoa thường, chịu viết liền / viết tắt / sai nhẹ. scores=True -> [(giá trị, điểm 0..1)]. Không có gì đủ giống -> []."""
    try:
        import pandas as pd
        ser = pd.Series(list(values) if not hasattr(values, 'dropna') else values)
        uniq = [x for x in ser.dropna().unique().tolist()][:MAX_VALUES]
    except Exception:  # noqa: BLE001
        uniq = list(dict.fromkeys(x for x in values if x is not None))[:MAX_VALUES]
    q = ' '.join(_toks(fold(text)))
    qt = set(_toks(q))
    scored = []
    for x in uniq:
        s = _score(q, qt, ' '.join(_toks(fold(x))))
        if s >= min_score:
            scored.append((round(s, 3), len(str(x)), str(x), x))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    out = [(x, s) for s, _, _, x in scored[:max(1, int(k))]]
    return out if scores else [x for x, _ in out]


# ----------------------------------------------------------------------------- xem cấu trúc bảng
def describe(df, name: str = '', k: int = 6, max_cols: int = 60) -> str:
    """Mô tả ngắn một bảng: số dòng; mỗi cột: kiểu, % trống, số giá trị khác nhau, vài giá trị mẫu (phổ biến nhất)."""
    if df is None or not hasattr(df, 'columns'):
        return f'{name}: (không có)'
    lines = [f'{name}: {len(df)} dòng, {len(df.columns)} cột']
    for c in list(df.columns)[:max_cols]:
        s = df[c]
        nn = s.notna().mean() if len(s) else 0.0
        try:
            nu = int(s.nunique(dropna=True))
            top = [str(v)[:30] for v in s.dropna().astype(str).value_counts().index[:k]]
        except Exception:  # noqa: BLE001 - cột chứa list/dict
            nu, top = -1, []
        lines.append(f'  - {c} [{s.dtype}] trống {100 * (1 - nn):.0f}%, {nu} giá trị khác nhau; vd {top}')
    if len(df.columns) > max_cols:
        lines.append(f'  ... (+{len(df.columns) - max_cols} cột)')
    return '\n'.join(lines)
