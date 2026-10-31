"""Khớp thực thể trong câu hỏi với giá trị có trong dữ liệu của shop: mặt hàng, nhóm hàng, nhóm khách, khách, nhân viên,
kênh bán, hình thức thanh toán, hoạt chất, đơn vị, size, ca. Không có dữ liệu thì không khớp (không bịa)."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

from datamate.core.textutil import fold

UNIT_WORDS = ['kg', 'chai', 'lon', 'ly', 'coc', 'goi', 'hop', 'vien', 'cai', 'bao', 'cay', 'thung', 'loc', 'tui', 'vi', 'tuyp',
              'lo', 'binh', 'qua', 'trai', 'met', 'm', 'm2', 'm3', 'cuon', 'cuc', 'dia', 'phan', 'to', 'chen', 'suat', 've',
              'chiec', 'doi', 'bo', 'bich', 'hu', 'can', 'lit', 'ml', 'g', 'tam', 'thanh', 'mieng', 'ong', 'xe', 'tan',
              'thien', 'banh', 'mon', 'nam', 'bom', 'tuyp', 'set', 'combo', 'cap', 'cuon', 'hop', 'kien', 'ram', 'xap', 'cuc',
              'que', 'soi', 'bo', 'canh', 'phan', 'ky', 'ket', 'mieng', 'cai']

CHANNEL_SYN = {
    'store': ['tai quay', 'tai cua hang', 'tai tiem', 'truc tiep', 'offline', 'tai shop', 'tai cho', 'store', 'mua tai quan', 'tai quan'],
    'phone': ['dien thoai', 'goi dien', 'phone', 'hotline'],
    'online': ['online', 'website', 'web'],
    'shopee': ['shopee'], 'lazada': ['lazada'], 'tiktok': ['tiktok', 'tik tok'], 'facebook': ['facebook', 'fb', 'fanpage'],
    'zalo': ['zalo'], 'grab': ['grab', 'grabfood'], 'shopeefood': ['shopeefood', 'shopee food'], 'befood': ['befood', 'be food'],
    'san tmdt': ['san tmdt', 'san thuong mai dien tu', 'tmdt', 'san'], 'giao hang': ['giao hang', 'ship', 'delivery', 'giao tan noi'],
    'app': ['app', 'ung dung'],
}
PAYMENT_SYN = {
    'cash': ['tien mat', 'cash'], 'transfer': ['chuyen khoan', 'transfer', 'ngan hang', 'banking'],
    'ewallet': ['vi dien tu', 'vi momo', 'momo', 'zalopay', 'ewallet', 'e wallet', 'vi'], 'cod': ['cod', 'giao hang thu tien', 'thu ho'],
    'debt': ['ghi no', 'mua chiu', 'tra sau'], 'card': ['quet the', 'the ngan hang', 'card', 'pos'],
}


@dataclass
class EntityIndex:
    products: list[tuple[str, str, str, str]] = field(default_factory=list)   # (key, display, name_f, code_f)
    categories: dict[str, str] = field(default_factory=dict)                 # fold -> display
    cgroups: dict[str, str] = field(default_factory=dict)
    customers: list[tuple[str, str, str]] = field(default_factory=list)      # (key, display, name_f)
    staff: dict[str, str] = field(default_factory=dict)                      # fold(code|name) -> staff code
    channels: list[str] = field(default_factory=list)                        # channel_f values
    payments: list[str] = field(default_factory=list)
    ingredients: dict[str, str] = field(default_factory=dict)
    base_units: set[str] = field(default_factory=set)

    def name_bigrams(self) -> set:
        """Cặp từ liền nhau trong tên hàng / nhóm hàng của shop (để nhận ra tên hàng nhắc trong câu mà chưa nối được)."""
        if getattr(self, '_pb', None) is None:
            pb = set()
            for _, _, nf, _ in self.products:
                t = re.sub(r'[^a-z0-9 ]', ' ', nf).split()
                pb.update(zip(t, t[1:]))
            for f in self.categories:
                t = re.sub(r'[^a-z0-9 ]', ' ', f).split()
                pb.update(zip(t, t[1:]))
            object.__setattr__(self, '_pb', pb)
        return self._pb

    def customer_bigrams(self) -> set:
        if getattr(self, '_cb', None) is None:
            cb = set()
            for _, _, nf in self.customers:
                t = re.sub(r'[^a-z0-9 ]', ' ', nf or '').split()
                cb.update(zip(t, t[1:]))
            object.__setattr__(self, '_cb', cb)
        return self._cb

    @classmethod
    def from_catalog(cls, cat) -> 'EntityIndex':
        staff = None
        for m in cat.mapping.of_type('staff'):
            if 'staff' in m.roles and 'staff_name' in m.roles:
                df = cat.raw[m.table].df
                staff = df[[m.roles['staff'], m.roles['staff_name']]]
                break
        return cls.build(cat.canon, staff)

    @classmethod
    def build(cls, canon, raw_staff: Optional[pd.DataFrame] = None) -> 'EntityIndex':
        L = canon.lines
        ix = cls()
        if L is None or L.empty:
            return ix
        if 'product_key' in L.columns:
            pl = L[~L['is_nonproduct']] if 'is_nonproduct' in L.columns else L
            names = pl.groupby('product_key')['product_name'].agg(lambda s: s.value_counts().index[0] if s.notna().any() else None) \
                if 'product_name' in pl.columns else pd.Series(dtype=object)
            spellings = pl.groupby('product_key')['product_name'].agg(lambda s: set(s.dropna())) if 'product_name' in pl.columns else {}
            codes = pl.groupby('product_key')['product_code'].first() if 'product_code' in pl.columns else {}
            for k, disp in names.items():
                if not k or not disp:
                    continue
                cf = fold(codes.get(k)) if hasattr(codes, 'get') and codes.get(k) else ''
                for sp in spellings.get(k, {disp}):
                    ix.products.append((k, disp, fold(sp), cf))
        for col, target in (('category', ix.categories), ('customer_group', ix.cgroups), ('ingredient', ix.ingredients)):
            if col in L.columns:
                for v in L[col].dropna().unique():
                    target.setdefault(fold(v), str(v))
        if 'customer_key' in L.columns:
            g = L[L['customer_key'].notna()].groupby('customer_key')['customer_display'].agg(lambda s: s.value_counts().index[0])
            ix.customers = [(k, d, fold(d)) for k, d in g.items() if d]
        if 'staff' in L.columns:
            for v in L['staff'].dropna().unique():
                ix.staff[fold(v)] = v
        if raw_staff is not None:
            for code, name in raw_staff.itertuples(index=False):
                if code and name:
                    ix.staff[fold(name)] = code
                    ix.staff[fold(code)] = code
        if 'channel_f' in L.columns:
            ix.channels = sorted(v for v in L['channel_f'].dropna().unique() if v)
        if 'payment_n' in L.columns:
            ix.payments = sorted(v for v in L['payment_n'].dropna().unique() if v)
        if 'base_unit_f' in L.columns:
            ix.base_units = set(L['base_unit_f'].dropna().unique())
        return ix


def _find(term: str, text: str) -> list[int]:
    return [m.start() for m in re.finditer(r'(?<![a-z0-9])' + re.escape(term) + r'(?![a-z0-9])', text)] if term else []


def link_products(ix: EntityIndex, qf: str, max_items: int = 4) -> list[tuple[str, str, int]]:
    """Trả (key, display, vị trí) các mặt hàng nhắc trong câu; ưu tiên khớp dài nhất, không chồng lấn."""
    hits = []
    for k, disp, nf, cf in ix.products:
        if len(nf) >= 3:
            for p in _find(nf, qf):
                hits.append((len(nf), p, k, disp))
        if cf and len(cf) >= 4:
            for p in _find(cf, qf):
                hits.append((len(cf) + 100, p, k, disp))
    hits.sort(key=lambda x: -x[0])
    used: list[tuple[int, int]] = []
    out = []
    for ln, p, k, disp in hits:
        span = (p, p + (ln if ln < 100 else ln - 100))
        if any(not (span[1] <= a or span[0] >= b) for a, b in used) or any(k == o[0] for o in out):
            continue
        used.append(span)
        out.append((k, disp, p))
        if len(out) >= max_items:
            break
    if out:
        return sorted(out, key=lambda x: x[2])
    # mờ: tên dài >= 8 ký tự, partial_ratio >= 92
    choices = {i: nf for i, (_, _, nf, _) in enumerate(ix.products) if len(nf) >= 8}
    if not choices:
        return []
    best = process.extractOne(qf, choices, scorer=fuzz.partial_ratio, score_cutoff=92)
    if best:
        k, disp, nf, _ = ix.products[best[2]]
        return [(k, disp, qf.find(nf[:4]) if nf[:4] in qf else 0)]
    return []


def link_category(ix: EntityIndex, qf: str) -> Optional[str]:
    best = None
    for cf, disp in ix.categories.items():
        if len(cf) >= 2 and _find(cf, qf):
            if re.search(r'(nhom|danh muc|loai|nganh|mang|dong)( hang| san pham| mon)?\s+' + re.escape(cf), qf) or len(cf) >= 6:
                if best is None or len(cf) > len(best[0]):
                    best = (cf, disp)
    return best[1] if best else None


def link_cgroup(ix: EntityIndex, qf: str) -> Optional[str]:
    for cf, disp in sorted(ix.cgroups.items(), key=lambda x: -len(x[0])):
        if cf and re.search(r'(khach|nhom)( hang)?( thuoc)?( nhom)?( khach)?\s+' + re.escape(cf) + r'(?![a-z0-9])', qf):
            return disp
    return None


def link_customer(ix: EntityIndex, qf: str) -> Optional[tuple[str, str]]:
    best = None
    for k, disp, nf in ix.customers:
        if len(nf) >= 5 and _find(nf, qf):
            if best is None or len(nf) > len(best[2]):
                best = (k, disp, nf)
    return (best[0], best[1]) if best else None


def link_staff(ix: EntityIndex, qf: str) -> Optional[str]:
    m = re.search(r'nhan vien (?:ma )?([a-z]{1,4}\d{1,4})\b', qf) or re.search(r'\b(nv\d{1,4})\b', qf)
    if m and fold(m[1]) in ix.staff:
        return ix.staff[fold(m[1])]
    for nf, code in sorted(ix.staff.items(), key=lambda x: -len(x[0])):
        if len(nf) >= 5 and _find(nf, qf):
            return code
    return None


def link_channels(ix: EntityIndex, qf: str) -> list[str]:
    out = []
    for v in ix.channels:
        aliases = {v} | set(CHANNEL_SYN.get(v, []))
        for key, syn in CHANNEL_SYN.items():
            if key in v or v in key:
                aliases |= set(syn)
        pos = min((p for a in aliases if len(a) >= 2 for p in _find(a, qf)), default=None)
        if pos is not None:
            out.append((pos, v))
    return [v for _, v in sorted(out)]


def link_payments(ix: EntityIndex, qf: str) -> list[str]:
    out = []
    for v in ix.payments:
        aliases = {v} | set(PAYMENT_SYN.get(v, []))
        pos = min((p for a in aliases if len(a) >= 2 for p in _find(a, qf)), default=None)
        if pos is not None:
            out.append((pos, v))
    out.sort()
    # 'vi' trong 'vi dien tu' và 'vi' riêng: bỏ trùng
    return list(dict.fromkeys(v for _, v in out))


def link_ingredient(ix: EntityIndex, qf: str) -> Optional[str]:
    m = re.search(r'hoat chat\s+(.+?)(?:\s+(?:thang|trong|tuan|nam|quy|ban|duoc|tong|hom|tu|den|la|co)\b|$)', qf)
    if not m:
        return None
    frag = m[1].strip()
    best = None
    for f, disp in ix.ingredients.items():
        if f == frag or (frag.startswith(f) and (best is None or len(f) > len(best[0]))):
            best = (f, disp)
    return best[1] if best else None


def asked_unit(qf: str) -> Optional[str]:
    units = '|'.join(sorted(set(UNIT_WORDS), key=len, reverse=True))
    m = re.search(r'(?:tong cong bao nhieu|tong so|bao nhieu|may|so|tinh theo)\s+(' + units +
                  r')\b(?!\s*(?:nhieu|hang|khach|don|ngay|thang|tien|phan tram|%))', qf)
    if m:
        return m[1]
    return None


def size_filter(qf: str) -> Optional[str]:
    m = re.search(r'\b(?:size|co|ly|loai)\s+(m|l|s|xl)\b', qf)
    return m[1].upper() if m else None


def shift_filter(qf: str) -> Optional[str]:
    if re.search(r'\bca sang\b', qf):
        return 'morning'
    if re.search(r'\bca chieu\b', qf):
        return 'afternoon'
    if re.search(r'\bca toi\b', qf):
        return 'evening'
    return None
