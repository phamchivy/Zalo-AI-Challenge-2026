"""Phát hiện câu KHÔNG trả lời được (rulebook §7) một cách thận trọng (trả lời 'unanswerable' sai cũng bị -1).

Chỉ kết luận khi chắc chắn:
  1. Kỳ hỏi nằm HOÀN TOÀN ngoài khoảng ngày có dữ liệu của shop (kỳ giao một phần thì tính phần có dữ liệu).
  2. Executor báo thiếu hẳn loại thông tin (Unanswerable) — vd hỏi ca sáng mà dữ liệu không ghi giờ.
  3. Câu hỏi chủ quan / ngoài dữ liệu (thời tiết, đối thủ, dự báo...).
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from datamate.core.types import TimeRange

SUBJECTIVE = r'\b(thoi tiet|doi thu|du bao|du doan|nen nhap|nen ban|tuong lai|sang nam|thang sau se|tai sao|vi sao|lanh nghe|dep nhat|ngon nhat|thich nhat|hai long|danh gia|tan tam|nhiet tinh|chu dao|than thien|nen uong|lieu dung|moi lan may (vien|goi|ml|muong|giot|lieu)|uong may|tac dung phu|chong chi dinh|co tot khong|co hieu qua|chinh hang|gia thi truong|gia dien|gia vang|ty gia|luong cua|tien luong)\b'


# v2.3.2: kỳ hỏi là ngày THUỘC TÍNH ghi trên sổ (hạn trả, hạn dùng), không phải ngày giao dịch -> được nằm sau dữ liệu
ATTR_DATE = r'\b(ngay phai tra|han (phai )?tra|han thanh toan|den han|qua han|het han|han dung|han su dung|hsd|cận date|can date)\b'


def out_of_range(periods: list[TimeRange], dmin: Optional[date], dmax: Optional[date], as_of: date,
                 qn: str = '') -> Optional[str]:
    if not periods or dmin is None or dmax is None:
        return None
    if qn and re.search(ATTR_DATE, qn):
        return None
    flows = [p for p in periods if p.granularity != 'snapshot' and p.label != 'all']
    if not flows:
        return None
    hi = min(dmax, as_of)
    for p in flows:
        if p.end >= dmin and p.start <= hi:
            return None
    p = flows[0]
    return f'Kỳ {p.start}..{p.end} nằm ngoài khoảng dữ liệu {dmin}..{dmax}'


def subjective(qn: str) -> Optional[str]:
    m = re.search(SUBJECTIVE, qn)
    if m and m[0] in ('tai sao', 'vi sao') and re.search(r'\b(huy|tra hang|tra lai|tra mon|doi tra)\b', qn):
        return None                    # 'huỷ chủ yếu vì sao' = lý do huỷ ghi trong dữ liệu
    return f'Câu hỏi cần thông tin ngoài dữ liệu ({m[0]})' if m else None
