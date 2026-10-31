"""Hợp đồng kiểu đáp án: suy ra kiểu mong đợi CHỈ từ dạng câu hỏi (độc lập với executor / LLM).

Dùng để chặn đáp án lệch kiểu ở mọi strategy (rule, LLM plan, LLM code): hỏi "ai / khách nào" mà ra số, hỏi "mấy %" mà
ra tiền, hỏi "ngày nào" mà ra số... -> loại candidate đó (sai format = sai, bỏ trống còn hơn).
Trả về None khi câu không có tín hiệu rõ (không ràng buộc).
"""
from __future__ import annotations

import re
from typing import Optional

from datamate.core.types import AnswerType as T

_PCT = re.compile(r'(?<![\d.,])%|\bphan tram\b|\b(ti|ty) le\b|\b(ti|ty) trong\b|\btang truong\b')
_DATE = re.compile(r'\b(ngay nao|hom nao|thang nao|ngay may|khi nao|luc nao|vao ngay nao|han dung (la )?ngay nao|co han dung (la )?ngay)\b')
_HOUR = re.compile(r'\b(may gio|gio nao|khung gio nao|luc may gio)\b')
_NAME = re.compile(r'\b(ai|khach nao|khach hang nao|nhan vien nao|tho nao|mat hang nao|san pham nao|mon nao|hang nao|thuoc nao|'
                   r'nha cung cap nao|ncc nao|nhom nao|nhom hang nao|danh muc nao|kenh nao|xe nao|loai nao|chi nhanh nao|'
                   r'la gi|ten (la )?gi)\b')
_LIST = re.compile(r'\b(nhung|liet ke|top \d+|\d+ (mat hang|san pham|khach|khach hang|mon|nhan vien|thuoc))\b')
_MULTI = re.compile(r'\b(lan luot|tung (thang|quy|ky)|moi (thang|quy|ky|nhom)|ghi (hai|2|ba|3) so|theo thu tu|va .* bao nhieu)\b')
_NUM = re.compile(r'\b(bao nhieu|bn|may|tong cong|tong so|so luong|dem|chenh lech|hon kem|it hon|nhieu hon)\b')
_BOOL = re.compile(r'\b(co|phai|dung)\b.*\b(khong|ko|hay khong|phai khong|dung khong)\s*$|\bhay khong\s*$|\b(tang|giam) lien tuc\b|\bluon cao hon\b')


_ASK = re.compile(r'\b(bao nhieu|bn|bnhiu|bao nhiu)\b')
_STRONG_MULTI = re.compile(r'\blan luot\b')


def expected_types(qn: str) -> Optional[set[str]]:
    # hỏi nhiều giá trị một lúc ("lần lượt", hai lần "bao nhiêu" nối bằng ",/và") -> chỉ nhận danh sách / bảng
    if _STRONG_MULTI.search(qn) or len(_ASK.findall(qn)) >= 2:
        return {T.LIST.value, T.TABLE.value}
    out: set[str] = set()
    pct = bool(_PCT.search(qn))
    if pct:
        out.add(T.PERCENT.value)
    if _DATE.search(qn):
        out.add(T.DATE.value)
    if _HOUR.search(qn):
        out.add(T.NUMBER.value)
    if _NAME.search(qn):
        out |= {T.STRING.value, T.LIST.value, T.TABLE.value}
    if _LIST.search(qn):
        out |= {T.LIST.value, T.TABLE.value}
    if _MULTI.search(qn):
        out |= {T.LIST.value, T.TABLE.value}
    if _BOOL.search(qn) and not re.search(r'\b(bao nhieu|bn|may|nao|ai)\b', qn):
        out.add(T.BOOLEAN.value)
    if _NUM.search(qn) and not pct:
        out |= {T.NUMBER.value, T.LIST.value}
    if not out:
        return None
    return out


def compatible(atype: str, expected: Optional[set[str]]) -> bool:
    if expected is None or atype in (T.UNANSWERABLE.value, T.ABSTAIN.value):
        return True
    return atype in expected
