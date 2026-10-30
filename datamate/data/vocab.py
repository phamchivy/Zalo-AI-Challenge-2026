"""Từ vựng chuẩn hoá giá trị: trạng thái đơn, hình thức thanh toán, kênh bán, dòng không phải hàng."""
from __future__ import annotations

import re
from typing import Any, Optional

from datamate.core.textutil import fold

STATUS = {
    'completed': {'da hoan thanh', 'hoan thanh', 'ok', 'da giao', 'xong', 'da xong', 'da lay', 'thanh cong', 'completed',
                  'complete', 'done', 'success', 'hoan tat', 'da hoan tat', 'da thanh toan', 'finished', 'delivered',
                  'giao thanh cong', 'da nhan', 'khach da nhan', 'paid'},
    'cancelled': {'da huy', 'huy', 'huy don', 'da huy don', 'khach huy', 'cancelled', 'canceled', 'cancel', 'khong giao duoc',
                  'bi huy', 'don huy', 'void', 'huy bo'},
    'returned': {'da hoan hang', 'hoan hang', 'tra hang', 'tra lai', 'da tra hang', 'hoan tra', 'hang tra ve', 'doi tra',
                 'khach tra', 'returned', 'return', 'refund', 'refunded', 'tra hang 1 phan', 'tra mot phan', 'da tra lai',
                 'khach tra lai', 'tra'},
    'shipping': {'dang giao', 'dang giao hang', 'dang giao dich', 'dang xu ly', 'dang van chuyen', 'shipping', 'dang gui',
                 'cho giao', 'dang ship', 'gui hang', 'processing', 'pending', 'cho xu ly', 'dang chuan bi', 'in transit'},
}
_STATUS_MAP = {v: k for k, vs in STATUS.items() for v in vs}


def norm_status(v: Any) -> Optional[str]:
    f = fold(v)
    if not f:
        return None
    if f in _STATUS_MAP:
        return _STATUS_MAP[f]
    if re.search(r'\bhuy\b|cancel', f):
        return 'cancelled'
    if re.search(r'\btra (hang|lai)\b|hoan hang|doi tra|return', f):
        return 'returned'
    if re.search(r'dang (giao|van chuyen|xu ly|ship)|shipping', f):
        return 'shipping'
    if re.search(r'hoan thanh|xong|thanh cong|da giao|complete', f):
        return 'completed'
    return 'unknown'


def status_from_note(v: Any) -> Optional[str]:
    """Shop không có cột trạng thái, ghi ở ghi chú: 'khách huỷ', 'đang giao', 'khách trả lại 1 hộp'..."""
    f = fold(v)
    if not f:
        return None
    if re.search(r'\bhuy\b|bom hang|khong nhan', f):
        return 'cancelled'
    if re.search(r'\btra (lai|hang)\b|hoan hang|doi tra|tra \d', f):
        return 'returned'
    if re.search(r'dang giao|chua giao|dang ship|dang van chuyen', f):
        return 'shipping'
    return None


PAYMENT = {
    'cash': r'^(tien mat|tm|cash|tien mat tai quay)$',
    'transfer': r'^(chuyen khoan|ck|transfer|bank|banking|ngan hang|qr|chuyen khoan ngan hang)$',
    'ewallet': r'^(vi dien tu|ewallet|e wallet|momo|zalopay|vnpay|shopeepay|vi)$',
    'cod': r'^(cod|giao hang thu tien|thu ho|ship cod)$',
    'debt': r'^(ghi no|no|debt|cong no|mua chiu|chiu|tra sau)$',
    'card': r'^(the|quet the|pos|card|the ngan hang)$',
}


def norm_payment(v: Any) -> Optional[str]:
    f = fold(v)
    if not f:
        return None
    for k, rx in PAYMENT.items():
        if re.search(rx, f):
            return k
    return f


def norm_channel(v: Any) -> Optional[str]:
    f = fold(v)
    return f or None


_NONPRODUCT = re.compile(r'^(voucher|ma giam gia|giam gia( don)?|dung \d+ diem|doi \d+ diem|tru diem|phi (ship|giao( hang)?|van chuyen)|ship)\b')
_SERVICE = re.compile(r'^phi phuc vu')


def is_nonproduct(name: Any) -> bool:
    """Dòng voucher / dùng điểm / phí ship nằm lẫn trong dòng hàng (không phải mặt hàng, không tính doanh thu)."""
    return bool(_NONPRODUCT.match(fold(name)))


def is_service_fee(name: Any) -> bool:
    """Phí phục vụ: không phải mặt hàng nhưng vẫn tính vào doanh thu (rulebook §3)."""
    return bool(_SERVICE.match(fold(name)))
