"""Kiểm tra & làm sạch dữ liệu theo CẤU TRÚC (không theo tên shop / tên file cụ thể).

  - dòng tổng cộng ("Tổng tháng 10", "Cộng", "Total"...) lẫn trong sổ bán hàng
  - dòng tiêu đề nhóm / dòng tổng không nhãn: có tiền nhưng không có hàng và không có số lượng
  - nhiều bảng bán hàng chứa cùng một tập đơn (bảng tổng hợp + bảng con) -> chỉ giữ một
  - đối chiếu tổng thành tiền dòng với tiền hàng ghi trên đơn -> cảnh báo độ tin cậy
Kết quả ghi vào `health` của Canonical; executor / strategy dùng để hạ độ tin cậy hoặc bỏ trống.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from datamate.core.textutil import fold, to_num

# so khớp CÓ DẤU: 'Tổng'/'Cộng' khác họ 'Tống' (fold() làm cả hai thành 'tong').
_SUMMARY_VI = re.compile(r'^(tổng|cộng|lũy kế|luỹ kế)(\s|:|$)')
_SUMMARY_EN = re.compile(r'^(total|subtotal|grand total|sum)\b')
# không dấu: chỉ nhận khi theo sau là ngữ cảnh tổng (tránh 'Tong Chi Ngoc')
_SUMMARY_ASCII = re.compile(r'^(tong|cong)\s*(:|-|$|\d|thang|ngay|nam|quy|tuan|ca\b|cong\b|ket\b|so\b|tien\b|doanh|hoa don|don\b|cuoi)')
LABEL_COLS = ('order_id', 'product_name', 'product_code', 'customer_name', 'customer_code', 'date', 'datetime', 'note', 'staff')


def is_summary_label(v) -> bool:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    import unicodedata
    raw = unicodedata.normalize('NFC', str(v)).strip().lower()
    if _SUMMARY_VI.match(raw) or _SUMMARY_EN.match(raw):
        return True
    return raw.isascii() and bool(_SUMMARY_ASCII.match(raw))


def _blank(s: pd.Series) -> pd.Series:
    return s.isna() | (s.astype(str).str.strip().isin(['', 'nan', 'None']))


def clean_line_frame(f: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bỏ dòng tổng có nhãn; đánh dấu dòng không có hàng+SL nhưng có tiền (tiêu đề nhóm / tổng không nhãn)."""
    rep = {'summary_label_rows': 0, 'header_rows': 0}
    if f.empty:
        return f, rep
    lab = pd.Series(False, index=f.index)
    for c in LABEL_COLS:
        if c in f.columns:
            lab |= f[c].map(is_summary_label)
    if lab.any():
        rep['summary_label_rows'] = int(lab.sum())
        f = f[~lab]
    has_prod_col = any(c in f.columns for c in ('product_name', 'product_code'))
    if has_prod_col and 'amount' in f.columns:
        noprod = pd.Series(True, index=f.index)
        for c in ('product_name', 'product_code'):
            if c in f.columns:
                noprod &= _blank(f[c])
        noqty = _blank(f['qty']) if 'qty' in f.columns else pd.Series(True, index=f.index)
        amt = ~_blank(f['amount'])
        hdr = noprod & noqty & amt
        if hdr.any():
            rep['header_rows'] = int(hdr.sum())
            f = f.copy()
            f['__header__'] = hdr
    return f, rep


def _keyset(f: pd.DataFrame) -> Optional[set]:
    if 'order_id' in f.columns and f['order_id'].notna().mean() > 0.3:
        return set(f['order_id'].dropna().astype(str).str.strip())
    if 'date' in f.columns and 'amount' in f.columns:
        return set(zip(f['date'].astype(str), f['amount'].astype(str)))
    return None


def dedupe_sources(parts: list[pd.DataFrame], names: list[str], threshold: float = 0.8) -> tuple[list[pd.DataFrame], list[str]]:
    """Bảng bán hàng có ≥80% khoá (mã đơn, hoặc (ngày, tiền)) nằm trong một bảng lớn hơn -> trùng nguồn, bỏ."""
    keys = [_keyset(p) for p in parts]
    drop = set()
    order = sorted(range(len(parts)), key=lambda i: (-(len(keys[i]) if keys[i] else 0), -len(parts[i].columns)))
    for a_i, a in enumerate(order):
        if a in drop or not keys[a]:
            continue
        for b in order[a_i + 1:]:
            if b in drop or not keys[b]:
                continue
            inter = len(keys[a] & keys[b])
            if inter / max(len(keys[b]), 1) >= threshold:
                drop.add(b)
    notes = [f'duplicate source dropped: {names[i]} (⊂ another sales table)' for i in sorted(drop)]
    return [p for i, p in enumerate(parts) if i not in drop], notes


def reconcile(L: pd.DataFrame, O: Optional[pd.DataFrame]) -> dict:
    """So tổng thành tiền dòng hàng của mỗi đơn với tiền hàng ghi trên đơn (nếu có)."""
    if O is None or O.empty or 'goods_total' not in O.columns or L.empty or 'amount' not in L.columns:
        return {}
    prod = L[~L['is_nonproduct']] if 'is_nonproduct' in L.columns else L
    s = prod.groupby('order_id')['amount'].sum()
    g = O.set_index('order_id')['goods_total'].dropna()
    both = s.index.intersection(g.index)
    if len(both) < 20:
        return {}
    diff = (s[both] - g[both]).abs()
    bad = (diff > 1.0) & (diff > 0.005 * g[both].abs())
    return {'recon_orders': int(len(both)), 'recon_mismatch': round(float(bad.mean()), 4)}


def amount_is_gross(L: pd.DataFrame) -> bool:
    """Cột thành tiền đang là tiền TRƯỚC giảm dòng (= SL × đơn giá) trong khi có giảm dòng > 0."""
    need = {'amount', 'qty', 'unit_price', 'line_discount'}
    if not need <= set(L.columns):
        return False
    d = L['line_discount'].fillna(0)
    m = (d > 100) & L['qty'].notna() & L['unit_price'].notna() & L['amount'].notna()
    if m.sum() < 20:
        return False
    gross = (L.loc[m, 'qty'] * L.loc[m, 'unit_price'] - L.loc[m, 'amount']).abs() < 1.0
    net = (L.loc[m, 'qty'] * L.loc[m, 'unit_price'] - L.loc[m, 'line_discount'] - L.loc[m, 'amount']).abs() < 1.0
    return gross.mean() > 0.9 and net.mean() < 0.1
