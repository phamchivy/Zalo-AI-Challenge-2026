"""Chuẩn hoá câu hỏi: bỏ dấu (fold), sửa gõ Telex lỗi ('dduwowcj' -> 'duoc'), mở teencode ('dthu' -> 'doanh thu').

Đầu ra luôn là chuỗi không dấu chữ thường, giữ ký tự số/ngày (/ - . , % :) để time_resolver đọc.
Chỉ dùng cho nhận diện ý định + thời gian; khớp tên hàng dùng fold gốc (entity_link).
"""
from __future__ import annotations

import re

from datamate.core.textutil import fold

# token (đã fold) -> thay thế
TEEN = {
    'dthu': 'doanh thu', 'doanhthu': 'doanh thu', 'revenue': 'doanh thu', 'doanh so': 'doanh thu',
    'hqua': 'hom qua', 'homqua': 'hom qua', 'hwa': 'hom qua', 'hnay': 'hom nay', 'homnay': 'hom nay', 'hum nay': 'hom nay',
    'bn': 'bao nhieu', 'bnhiu': 'bao nhieu', 'bnh': 'bao nhieu', 'bao nhiu': 'bao nhieu', 'bao nhieeu': 'bao nhieu',
    'nhiu': 'nhieu', 'nhiue': 'nhieu', 'baonhieu': 'bao nhieu',
    'dc': 'duoc', 'đc': 'duoc', 'dk': 'duoc', 'duoc': 'duoc',
    'ko': 'khong', 'k': 'khong', 'kh': 'khach', 'hok': 'khong', 'khg': 'khong',
    'sp': 'san pham', 'mh': 'mat hang', 'kh hang': 'khach hang',
    'trc': 'truoc', 'vs': 'voi', 'w': 'voi',
    'tm': 'tien mat', 'ck': 'chuyen khoan',
    'bill': 'don', 'bills': 'don', 'order': 'don', 'orders': 'don', 'hd': 'hoa don',
    'ncc': 'nha cung cap', 'nv': 'nhan vien', 'sl': 'so luong', 'gtri': 'gia tri', 'gt': 'gia tri',
    'thg': 'thang', 'th': 'thang', 'q': 'quy',
}
# từ đệm chỉ bỏ ở đầu / cuối câu ('e' giữa câu có thể là 'ế')
START_FILLERS = {'e', 'ee', 'oi', 'a', 'z', 'v', 'ei', 'ey', 'alo'}
END_FILLERS = {'z', 'v', 'vay', 'a', 'ha', 'nhi', 'nha', 'ta', 'ne', 'hen', 'ak', 'ah', 'e', 'ee', 'oi', 'do', 'the', 'ho', 'nhe',
               'chua', 'day', 'ban'}
# bảo vệ: không phá các từ thật
TELEX_WHITELIST = {'pos', 'cod', 'vs', 'ios', 'gas', 'plus', 'max', 'box', 'tax', 'mix', 'kids', 'ps', 'top', 'its', 'order', 'orders',
                   'bill', 'bills', 'revenue', 'sales', 'size', 'store', 'shopee', 'grab', 'tiktok', 'online', 'offline', 'combo',
                   'voucher', 'ship', 'shipper', 'check', 'list', 'best', 'seller', 'items', 'total', 'sum', 'count', 'sku',
                   'phone', 'zalo', 'facebook', 'lazada', 'tiki', 'web', 'website', 'app', 'momo', 'cash', 'transfer', 'debt'}


def telex_fix(tok: str) -> str:
    """Gõ Telex thiếu bộ gõ: dấu thanh s f r x j cuối từ, nguyên âm đôi aa/ee/oo, dd, w."""
    if not tok.isalpha() or len(tok) < 3 or tok in TELEX_WHITELIST:
        return tok
    t = tok
    changed = False
    if re.search(r'(aa|ee|oo|dd|uw|ow|aw)', t) or 'w' in t[1:]:
        t = t.replace('dd', 'd').replace('aa', 'a').replace('ee', 'e').replace('oo', 'o')
        t = t.replace('uw', 'u').replace('ow', 'o').replace('aw', 'a').replace('w', '')
        changed = True
    if t[-1] in 'sfrxj' and len(t) >= 3 and re.search(r'[aeiouy]', t[:-1]):
        # âm tiết tiếng Việt không bao giờ kết thúc bằng s/f/r/x/j
        t = t[:-1]
        changed = True
    if changed:
        # dấu thanh đặt giữa từ ('thangs', 'ddown' đã xử lý); 'nguwowif' -> 'nguoii'
        t = re.sub(r'([aeiou])\1', r'\1', t)
    return t


def qnorm(text: str) -> str:
    t = fold(text)
    t = t.replace('–', '-').replace('—', '-').replace('≥', ' tu ').replace('>=', ' tu ')
    # tách dấu câu (giữ / - . , % : trong số)
    t = re.sub(r'(?<!\d)[.,;!?](?!\d)', ' ', t)
    t = re.sub(r'[\"\'“”()\[\]{}]', ' ', t)
    # 'tang/giam', 'cao/thap' -> 'tang hoac giam' (dấu / giữa hai chữ; '8/2025' giữ nguyên)
    t = re.sub(r'(?<=[a-z]{2})\s*/\s*(?=[a-z]{2})', ' hoac ', t)
    toks = t.split()
    toks = [x if x in TEEN else telex_fix(x) for x in toks]
    out = []
    i = 0
    while i < len(toks):
        two = ' '.join(toks[i:i + 2])
        if two in TEEN:
            out.append(TEEN[two])
            i += 2
            continue
        tok = toks[i]
        out.append(TEEN.get(tok, tok))
        i += 1
    out = [x for x in out if x]
    while out and out[0] in START_FILLERS:
        out = out[1:]
    while len(out) > 1 and out[-1] in END_FILLERS and not (out[-1] == 'e' and out[-2] in ('bi', 'hang', 'mat', 'la', 'do')) \
            and not (out[-1] == 'ban' and out[-2] in ('khong', 'ko', 'duoc')):
        out = out[:-1]
    t = ' '.join(out)
    # cụm hay gặp
    t = re.sub(r'\bthang truoc do\b', 'thang lien truoc', t)
    t = re.sub(r'\bbua (nay|ni)\b', 'hom nay', t)
    t = re.sub(r'\bbua (qua|wa|truoc)\b', 'hom qua', t)
    t = re.sub(r'\b(hom|bua) kia\b', 'hom kia', t)
    t = re.sub(r'\btuan (rui|roi|vua roi|qua|truoc do)\b', 'tuan truoc', t)
    t = re.sub(r'\bthang (rui|roi|vua roi|vua qua)\b', 'thang truoc', t)
    t = re.sub(r'\bnam (ngoai|roi|vua roi)\b', 'nam truoc', t)
    t = re.sub(r'\bnhieu tien nhat\b', 'nhieu tien nhat', t)
    t = re.sub(r'\bphan tram\b', '%', t)
    return ' '.join(t.split())
