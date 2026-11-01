"""Kiểm định đáp án của agent trước khi đưa vào bỏ phiếu: đúng kiểu, giá trị hợp lý. Trả về danh sách cảnh báo;
agent được yêu cầu kiểm tra lại một lần, cảnh báo nặng (sai kiểu) thì loại candidate."""
from __future__ import annotations

import math
import re
from typing import Any, Optional

_DATE = re.compile(r'^\d{4}-\d{2}(-\d{2})?$')


def coerce(atype: str, value: Any) -> tuple[Optional[str], Any, list[str]]:
    """Chuẩn hoá giá trị theo kiểu; trả (type, value, lỗi nặng)."""
    errs: list[str] = []
    t = (atype or '').strip().lower()
    if t in ('int', 'float', 'money', 'count'):
        t = 'number'
    if t in ('pct', '%'):
        t = 'percent'
    if t in ('bool',):
        t = 'boolean'
    if t in ('str', 'text', 'name'):
        t = 'string'
    if t not in ('number', 'percent', 'string', 'date', 'boolean', 'list', 'table', 'unanswerable', 'abstain'):
        return None, value, [f'kiểu đáp án không hợp lệ: {atype!r}']
    if t in ('number', 'percent'):
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
        if isinstance(value, str):
            v2 = value.replace('%', '').replace(',', '.').strip()
            try:
                value = float(v2)
            except ValueError:
                errs.append(f'{t} nhưng giá trị là chuỗi {value!r}'[:200])
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            if not errs:
                errs.append(f'{t} nhưng giá trị không phải số: {value!r}'[:200])
        else:
            try:
                bad = math.isnan(value) or math.isinf(value) or abs(value) > 1e15
            except OverflowError:
                bad = True
            if bad:
                errs.append('giá trị là NaN / vô cực / lớn bất thường')
    elif t == 'date':
        if hasattr(value, 'isoformat') and not isinstance(value, str):      # Timestamp / date / datetime
            value = value.isoformat()[:10]
        if not isinstance(value, str) or not _DATE.match(value.strip()):
            errs.append(f"date phải dạng 'YYYY-MM-DD' hoặc 'YYYY-MM', nhận {value!r}"[:200])
    elif t == 'boolean':
        if isinstance(value, str) and value.strip().lower() in ('true', 'false', 'co', 'có', 'khong', 'không'):
            value = value.strip().lower() in ('true', 'co', 'có')
        if not isinstance(value, bool):
            errs.append(f'boolean phải là True/False, nhận {value!r}'[:200])
    elif t == 'list':
        if isinstance(value, tuple):
            value = list(value)
        if not isinstance(value, list) or not value:
            errs.append('list rỗng hoặc không phải danh sách')
    elif t == 'table':
        if not (isinstance(value, dict) and 'columns' in value and 'rows' in value):
            errs.append("table phải là {'columns': [...], 'rows': [[...]]}")
        elif not value['rows']:
            errs.append('table rỗng (không có dòng nào): bộ lọc đã loại hết dữ liệu')
    elif t == 'string':
        if value is None or (isinstance(value, str) and not value.strip()):
            errs.append('string rỗng')
        elif not isinstance(value, str):
            value = str(value)
    return t, value, errs


def sanity_warnings(t: str, value: Any, question_folded: str) -> list[str]:
    """Cảnh báo mềm: giá trị đúng kiểu nhưng đáng ngờ -> yêu cầu agent kiểm tra lại một lần."""
    w: list[str] = []
    q = question_folded
    if t == 'number' and isinstance(value, (int, float)):
        if value < 0 and re.search(r'doanh thu|ban duoc|tien hang|so don|so luong|bao nhieu don|may don', q) \
                and not re.search(r'chenh lech|tang|giam|lech|hon kem|loi nhuan|lai', q):
            w.append('kết quả âm trong khi câu hỏi về doanh thu / số lượng')
        if value == 0:
            w.append('kết quả bằng 0: kiểm tra bộ lọc (tên hàng, ngày, trạng thái) có lọc mất hết dòng không')
        if abs(value) > 1e13:
            w.append('kết quả lớn bất thường: kiểm tra đơn vị / cột tiền')
        if isinstance(value, float) and re.search(r'so don|may don|bao nhieu don|so khach|may khach|bao nhieu khach', q) \
                and abs(value - round(value)) > 1e-9:
            w.append('số đếm (đơn / khách) phải là số nguyên')
    if t == 'percent' and isinstance(value, (int, float)):
        if abs(value) > 1000:
            w.append('phần trăm lớn bất thường (đáp án là số phần trăm, vd 12.4 nghĩa là 12,4%)')
        if 0 < abs(value) < 1 and re.search(r'chiem|ti le|ty le|ti trong', q):
            w.append('phần trăm < 1: có thể đang để dạng tỉ lệ (0.124) thay vì phần trăm (12.4)')
    if t == 'list' and isinstance(value, list):
        m = re.search(r'\b(top|nhung|liet ke)?\s*(\d+)\s+(mat hang|san pham|khach|khach hang|mon|nhan vien|thuoc)', q)
        if m and int(m.group(2)) != len(value) and int(m.group(2)) <= 20:
            w.append(f'câu hỏi cần {m.group(2)} phần tử nhưng danh sách có {len(value)}')
    return w
