"""Dựng bảng canonical từ bảng gốc đã ánh xạ vai trò.

  lines     : mỗi dòng hàng bán (order_id, date, time, status, customer_*, product_*, qty, qty_base, amount, ...)
  orders    : mỗi đơn (gồm cả đơn không có dòng hàng, cần cho tỉ lệ huỷ)
  products  : danh mục (product_key, product_code, product_name, category, base_unit, cost, price, rx, ingredient)
  units     : (product_key, unit_f, factor)
  customers : (customer_key, customer_code, customer_name, customer_group, phone, address)
  payments  : (order_id, pay_date, amount, method)
  purchases : dòng nhập (receipt_no, p_date, supplier, product_key, p_qty, p_unit, p_unit_cost, p_amount, p_total, p_status)

Áp luật đọc file của rulebook §8 (ô trống ở dòng sau của nhóm = giống dòng đầu, dòng trùng trong cùng đơn tính 1 lần,
ngày không ghi năm lấy năm của sheet) và quy ước chung (số lượng âm = trả hàng, dòng voucher/phí ship không phải hàng).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

import numpy as np
import pandas as pd

from datamate.core.textutil import customer_key, fold, is_walkin, parse_date, parse_time, strip_phone, to_num, year_from_text
from datamate.data.raw_loader import RawTable
from datamate.data.readme import ReadmeInfo, convention_flags
from datamate.data.sanity import (amount_is_gross, clean_line_frame, dedupe_sources, is_summary_label, order_level,
                                  reconcile, stt_groups)
from datamate.data.schema_mapper import BusinessMapping, TableMapping
from datamate.data.vocab import is_nonproduct, is_service_fee, norm_payment, norm_status, status_from_note

ORDER_LEVEL = ['date', 'datetime', 'time', 'status', 'customer_code', 'customer_name', 'customer_group', 'channel', 'payment',
               'staff', 'goods_total', 'order_discount', 'order_total', 'paid', 'debt', 'ship_fee', 'vat', 'note',
               'table_no', 'area', 'n_guests']
NUM_ROLES = {'qty', 'qty_base', 'unit_price', 'line_discount', 'amount', 'cost_unit', 'goods_total', 'order_discount',
             'order_total', 'paid', 'debt', 'ship_fee', 'vat', 'n_guests', 'return_qty'}


@dataclass
class Canonical:
    lines: pd.DataFrame
    orders: pd.DataFrame
    products: pd.DataFrame
    units: pd.DataFrame
    customers: pd.DataFrame
    payments: pd.DataFrame
    purchases: pd.DataFrame
    suppliers: dict = field(default_factory=dict)
    caps: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)
    date_min: Optional[date] = None
    date_max: Optional[date] = None
    # v1.1: bảng cho khối tính chung + sức khoẻ dữ liệu
    stock: pd.DataFrame = field(default_factory=pd.DataFrame)        # product_key, ts, balance, seq (thẻ kho: tồn sau mỗi phát sinh)
    prices: pd.DataFrame = field(default_factory=pd.DataFrame)       # product_key, eff_date, price
    lots: pd.DataFrame = field(default_factory=pd.DataFrame)         # lot, product_key, expiry, received_base, unit_cost_base, recv_date
    lot_sales: pd.DataFrame = field(default_factory=pd.DataFrame)    # order_id, lot, qty_base, return_qty
    deliveries: pd.DataFrame = field(default_factory=pd.DataFrame)   # order_id, deliv_date, deliv_no
    health: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- vector helpers
def _map_unique(s: pd.Series, fn) -> pd.Series:
    vals = s.dropna().unique()
    m = {v: fn(v) for v in vals}
    return s.map(m)


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(_map_unique(s, to_num), errors='coerce')


def _nn(x) -> bool:
    """x có giá trị thật (không None/NaN/chuỗi rỗng)."""
    if x is None:
        return False
    if isinstance(x, float) and np.isnan(x):
        return False
    return str(x).strip() not in ('', 'nan', 'None', 'NaT')


def _str(s: pd.Series) -> pd.Series:
    return s.map(lambda x: None if x is None or (isinstance(x, float) and np.isnan(x)) else (str(x).strip() or None))


def _dates(s: pd.Series, years: Optional[pd.Series] = None) -> pd.Series:
    if years is None:
        return _map_unique(s, parse_date)
    keys = list(zip(s.tolist(), years.tolist()))
    cache: dict = {}
    out = []
    for v, y in keys:
        k = (v if not isinstance(v, (list, dict)) else str(v), y)
        if k not in cache:
            cache[k] = parse_date(v, default_year=y)
        out.append(cache[k])
    return pd.Series(out, index=s.index)


def _frame(t: RawTable, m: TableMapping, roles: Optional[set] = None) -> pd.DataFrame:
    df = pd.DataFrame(index=t.df.index)
    for role, col in m.roles.items():
        if roles is None or role in roles:
            df[role] = t.df[col]
    for c in ('__sheet__', '__file__'):
        if c in t.df.columns:
            df[c] = t.df[c]
    df['__table__'] = t.name
    return df


def _sheet_year(df: pd.DataFrame) -> Optional[pd.Series]:
    src = None
    if '__sheet__' in df.columns:
        src = df['__sheet__']
    if '__file__' in df.columns:
        src = df['__file__'] if src is None else src.fillna(df['__file__']).where(src.map(year_from_text).notna(), df['__file__'])
    if src is None:
        return None
    return _map_unique(src, year_from_text)


def _overlap(a: pd.Series, b: pd.Series) -> float:
    sa = set(_str(a.dropna().head(5000)).dropna())
    sb = set(_str(b.dropna()).dropna())
    return len(sa & sb) / max(len(sa), 1)


def _fix_line_discount(f: pd.DataFrame, t: RawTable, m: TableMapping, notes: list) -> pd.DataFrame:
    """Cột được hiểu là "giảm dòng" nhưng lặp lại y nguyên trên mọi dòng của đơn -> là giảm của cả đơn.
    Chuyển sang order_discount; nếu bảng còn cột ứng viên giảm dòng khác (không lặp theo đơn) thì dùng cột đó."""
    if 'line_discount' not in f.columns or 'order_id' not in f.columns or not order_level(f, 'line_discount'):
        return f
    f = f.copy()
    src = m.roles.get('line_discount')
    if 'order_discount' not in f.columns:
        f['order_discount'] = f['line_discount']
    f = f.drop(columns=['line_discount'])
    for c in m.alts.get('line_discount', []):
        if c in t.df.columns:
            tmp = pd.DataFrame({'order_id': f['order_id'], 'x': t.df.loc[f.index, c]})
            if not order_level(tmp, 'x'):
                f['line_discount'] = t.df.loc[f.index, c]
                notes.append(f'{m.table}: "{src}" is order-level discount; line discount = "{c}"')
                return f
    notes.append(f'{m.table}: "{src}" is order-level discount (moved to order_discount)')
    return f


# ----------------------------------------------------------------------------- main
def build_canonical(raw: dict[str, RawTable], bm: BusinessMapping, info: ReadmeInfo) -> Canonical:
    notes: list[str] = []
    flags = convention_flags(info)
    line_maps = bm.of_type('sales_flat', 'sales_lines')
    order_maps = bm.of_type('sales_orders')

    health: dict = {'summary_label_rows': 0, 'header_rows': 0, 'uncertain_roles': []}
    # ---- 1. dòng hàng (bỏ dòng tổng, đánh dấu dòng tiêu đề nhóm, bỏ bảng trùng nguồn)
    parts, pnames = [], []
    for m in line_maps:
        f = _frame(raw[m.table], m)
        if f.empty:
            continue
        f, rep = clean_line_frame(f)
        for k_, v_ in rep.items():
            health[k_] += v_
        f = _fix_line_discount(f, raw[m.table], m, notes)
        if 'date' in f.columns or 'datetime' in f.columns:
            d_ = _dates(f['date'] if 'date' in f.columns else f['datetime'], _sheet_year(f))
            f['__dkey__'] = d_.map(lambda d: f'{d.year}-{d.month:02d}' if d is not None and d == d else None)
        for r_ in getattr(m, 'uncertain', []):
            if r_ not in health['uncertain_roles']:
                health['uncertain_roles'].append(r_)
        parts.append(f)
        pnames.append(m.table)
    if len(parts) > 1:
        parts, dn = dedupe_sources(parts, pnames)
        notes += dn
        health['duplicate_sources'] = len(dn)
    if health['summary_label_rows'] or health['header_rows']:
        notes.append(f"summary rows dropped: {health['summary_label_rows']}, header rows (no product, no qty): {health['header_rows']}")
    L = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if '__dkey__' in L.columns:
        L = L.drop(columns=['__dkey__'])
    # bảng có tên là sổ thu / phiếu thu nhưng không nhận được cột -> không dùng cột "đã trả" (ảnh chụp cuối kỳ) thay thế
    health['payments_unmapped'] = [m.table for m in bm.tables.values()
                                   if getattr(m, 'name_hint', None) == 'payments'
                                   and not (m.ttype == 'payments' and {'order_id', 'pay_amount'} <= set(m.roles))]
    O_raw = None
    if order_maps:
        # bảng đơn khớp mã đơn với dòng hàng nhiều nhất
        best, best_ov = None, -1.0
        for m in order_maps:
            t = raw[m.table]
            if 'order_id' not in m.roles:
                continue
            ov = _overlap(L['order_id'], t.df[m.roles['order_id']]) if 'order_id' in L.columns and len(L) else 0.0
            score = ov * 10 + len(m.roles) * 0.1 + (1 if 'status' in m.roles else 0)
            if score > best_ov:
                best, best_ov = m, score
        if best is not None:
            O_raw = _frame(raw[best.table], best)
            lab = pd.Series(False, index=O_raw.index)
            for c_ in ('order_id', 'customer_name', 'date', 'note'):
                if c_ in O_raw.columns:
                    lab |= O_raw[c_].map(is_summary_label)
            if lab.any():
                O_raw = O_raw[~lab]
                notes.append(f'orders table summary rows dropped: {int(lab.sum())}')
            O_raw = O_raw[O_raw['order_id'].notna()] if O_raw['order_id'].notna().mean() > 0.3 else O_raw
            notes.append(f'orders table = {best.table}')
    if L.empty and O_raw is None:
        return _empty(notes + ['no sales table'])
    if L.empty and O_raw is not None:
        # chỉ có bảng đơn (vd danh_sach_hoa_don): mỗi đơn 1 dòng giả, doanh thu = tiền hàng của đơn
        L = O_raw.copy()
        amt = 'goods_total' if 'goods_total' in L.columns else ('order_total' if 'order_total' in L.columns else None)
        if amt:
            L['amount'] = L[amt]
        notes.append('no line table: pseudo lines from orders')
        caps_no_lines = True
    else:
        caps_no_lines = False

    years = _sheet_year(L)
    # ---- 2. chuẩn kiểu
    for c in list(L.columns):
        if c in NUM_ROLES:
            L[c] = _num(L[c])
    for c in ('order_id', 'product_code', 'customer_code', 'staff', 'product_name', 'customer_name', 'unit', 'category',
              'channel', 'payment', 'note', 'status', 'customer_group', 'base_unit'):
        if c in L.columns:
            L[c] = _str(L[c])
    real_order_id = 'order_id' in L.columns and L['order_id'].notna().mean() > 0.3
    if real_order_id:
        # ô mã đơn để trống ở dòng sau của nhóm (sổ tay, MISA) -> giống dòng trên
        if L['order_id'].isna().any():
            L['order_id'] = L.groupby('__table__')['order_id'].ffill()
            notes.append('ffill order_id')
    else:
        grp = stt_groups(L)
        if grp is not None:
            L['order_id'] = grp.values
            real_order_id = True
            notes.append(f'order key from row-number groups (STT on first line only): {L["order_id"].nunique()} orders')
        else:
            L['order_id'] = [f'R{i}' for i in range(len(L))]
            notes.append('no order id: one order per row')
    # ngày / giờ
    if 'date' in L.columns:
        L['date'] = _dates(L['date'], years)
    if 'datetime' in L.columns:
        dt = L['datetime']
        if 'date' not in L.columns or L['date'].isna().all():
            L['date'] = _dates(dt, years)
        if 'time' not in L.columns:
            L['time'] = dt
    if 'time' in L.columns:
        L['time'] = _map_unique(L['time'], parse_time)

    # ---- 3. ghép bảng đơn
    if O_raw is not None and not caps_no_lines:
        O = O_raw.copy()
        oy = _sheet_year(O)
        for c in list(O.columns):
            if c in NUM_ROLES:
                O[c] = _num(O[c])
        for c in O.columns:
            if c not in NUM_ROLES and not c.startswith('__'):
                O[c] = _str(O[c])
        if 'date' in O.columns:
            O['date'] = _dates(O['date'], oy)
        if 'datetime' in O.columns:
            if 'date' not in O.columns or O['date'].isna().all():
                O['date'] = _dates(O['datetime'], oy)
            if 'time' not in O.columns:
                O['time'] = O['datetime']
        if 'time' in O.columns:
            O['time'] = _map_unique(O['time'], parse_time)
        O = O.drop_duplicates('order_id')
        keep = [c for c in ORDER_LEVEL if c in O.columns]
        L = L.drop(columns=[c for c in keep if c in L.columns])
        L = L.merge(O[['order_id'] + keep], on='order_id', how='left')
        O_all = O
    else:
        O_all = None

    # ---- 4. ô trống ở dòng sau cùng nhóm = dòng đầu nhóm (§8)
    if real_order_id:
        lvl = [c for c in ORDER_LEVEL if c in L.columns and L[c].isna().any()]
        if lvl:
            g = L.groupby('order_id', sort=False)
            for c in lvl:
                L[c] = g[c].transform('first')

    # ---- 5. trạng thái
    if 'status' in L.columns:
        L['status_raw'] = L['status']
        L['status'] = _map_unique(L['status'], norm_status)
    else:
        L['status'] = 'completed'
        notes.append('no status column: default completed')
    if 'note' in L.columns:
        ns = _map_unique(L['note'], status_from_note)
        use = ns.notna() & (L['status'].isin(['completed', 'unknown']) | L['status'].isna())
        if use.any() and (flags.get('status_in_note') or 'status_raw' not in L.columns):
            L.loc[use, 'status'] = ns[use]
            notes.append(f'status from note: {int(use.sum())} rows')
    L['status'] = L['status'].fillna('completed').replace({'unknown': 'completed'})

    # ---- 6. sản phẩm, đơn vị, quy đổi
    products = _build_products(raw, bm, L)
    if 'product_code' in L.columns and L['product_code'].notna().mean() > 0.5:
        L['product_key'] = L['product_code'].map(lambda x: fold(x) if _nn(x) else None)
    elif 'product_name' in L.columns:
        L['product_key'] = L['product_name'].map(lambda x: fold(x) if _nn(x) else None)
        # danh mục có mã -> đổi key theo tên sang key theo mã
        if not products.empty and products['product_code'].notna().any():
            name2key = dict(zip(products['name_f'], products['product_key']))
            L['product_key'] = L['product_key'].map(lambda k: name2key.get(k, k))
    else:
        L['product_key'] = None
    if not products.empty:
        pinfo = products.set_index('product_key')
        for c in ('product_name', 'category', 'base_unit', 'cost', 'rx', 'ingredient', 'product_code'):
            if c in pinfo.columns:
                mapped = L['product_key'].map(pinfo[c].to_dict())
                if c not in L.columns:
                    L[c] = mapped
                elif c in ('category', 'base_unit', 'product_code'):
                    L[c] = L[c].fillna(mapped)
                elif c == 'product_name':
                    L['product_name_cat'] = mapped
    units = _build_units(raw, bm, info, products)
    L['unit_f'] = L['unit'].map(fold) if 'unit' in L.columns else None
    if 'qty' not in L.columns:
        L['qty'] = 1.0 if caps_no_lines else np.nan
    if 'qty_base' not in L.columns or L['qty_base'].isna().all():
        fac = {(k, u): f for k, u, f in units[['product_key', 'unit_f', 'factor']].itertuples(index=False)} if not units.empty else {}
        L['factor'] = [fac.get((k, u), 1.0) for k, u in zip(L['product_key'], L['unit_f'])]
        L['qty_base'] = L['qty'] * L['factor']
    else:
        L['factor'] = (L['qty_base'] / L['qty']).where(L['qty'].abs() > 0)
    if 'base_unit' not in L.columns or L['base_unit'].isna().all():
        L['base_unit'] = _infer_base_unit(L, units)
    else:
        bu = _infer_base_unit(L, units)
        L['base_unit'] = L['base_unit'].fillna(bu)
    L['base_unit_f'] = L['base_unit'].map(fold)
    # tiền dòng
    if 'amount' not in L.columns or L['amount'].isna().all():
        if 'unit_price' in L.columns:
            L['amount'] = L['qty'] * L['unit_price'] - L.get('line_discount', 0).fillna(0) if 'line_discount' in L.columns else L['qty'] * L['unit_price']
            notes.append('amount = qty*price')
    L['amount'] = L['amount'].fillna(0.0) if 'amount' in L.columns else 0.0
    if amount_is_gross(L):
        # cột "thành tiền" là tiền trước giảm dòng -> canonical amount = sau giảm dòng (giữ bản gốc ở amount_gross)
        L['amount_gross'] = L['amount']
        L['amount'] = L['amount'] - L['line_discount'].fillna(0)
        notes.append('amount column was gross: subtract line discount')

    # ---- 7. dòng không phải hàng / phí phục vụ
    pn = L['product_name'] if 'product_name' in L.columns else pd.Series([None] * len(L), index=L.index, dtype=object)
    L['is_service'] = _map_unique(pn, is_service_fee).fillna(False).astype(bool) if len(L) else False
    L['is_nonproduct'] = _map_unique(pn, is_nonproduct).fillna(False).astype(bool) if len(L) else False
    if '__header__' in L.columns:
        L['is_nonproduct'] = L['is_nonproduct'] | L['__header__'].fillna(False).astype(bool)
        L = L.drop(columns=['__header__'])

    # ---- 8. trả hàng: số lượng âm / bảng trả / cột SL trả -> cả đơn là đơn trả
    returned_ids: set = set()
    neg = (L['qty'] < 0) & ~L['is_nonproduct']
    if neg.any():
        returned_ids |= set(L.loc[neg, 'order_id'])
        notes.append(f'negative qty lines: {int(neg.sum())}')
    for m in bm.of_type('returns'):
        t = raw[m.table]
        if 'order_id' in m.roles:
            returned_ids |= set(_str(t.df[m.roles['order_id']]).dropna())
    for m in bm.of_type('lots', 'sales_lines', 'sales_flat'):
        if 'return_qty' in m.roles and 'order_id' in m.roles:
            t = raw[m.table]
            rq = _num(t.df[m.roles['return_qty']])
            returned_ids |= set(_str(t.df.loc[rq > 0, m.roles['order_id']]).dropna())
    if returned_ids:
        mark = L['order_id'].isin(returned_ids) & (L['status'] != 'cancelled')
        L.loc[mark, 'status'] = 'returned'

    # ---- 9. dòng trùng trong cùng một đơn (§8)
    if real_order_id:
        key_cols = [c for c in L.columns if c not in ('line_no', '__sheet__', '__file__', '__table__', 'status_raw')
                    and not c.startswith('__')]
        before = len(L)
        L = L.drop_duplicates(subset=key_cols).reset_index(drop=True)
        if len(L) < before:
            notes.append(f'dedup identical lines: {before - len(L)}')

    # ---- 10. khách
    customers = _build_customers(raw, bm)
    has_code = 'customer_code' in L.columns and L['customer_code'].notna().any()
    if 'customer_name' not in L.columns:
        L['customer_name'] = None
    if has_code:
        L['customer_key'] = L['customer_code'].map(lambda x: fold(x) if _nn(x) else None)
        if not customers.empty:
            cmap = customers.set_index('customer_key')
            L['customer_name'] = L['customer_name'].fillna(L['customer_key'].map(cmap['customer_name'].to_dict()))
            if 'customer_group' in cmap.columns and ('customer_group' not in L.columns or L['customer_group'].isna().all()):
                L['customer_group'] = L['customer_key'].map(cmap['customer_group'].to_dict())
    else:
        L['customer_key'] = L['customer_name'].map(lambda x: None if (not _nn(x) or is_walkin(x)) else (customer_key(x) or None))
    L['is_walkin'] = L['customer_key'].isna()
    if 'customer_name' in L.columns:
        L['customer_display'] = L['customer_name'].map(lambda x: strip_phone(x) if _nn(x) else None)
    if 'payment' in L.columns:
        L['payment_n'] = _map_unique(L['payment'], norm_payment)
    if 'channel' in L.columns:
        L['channel_f'] = L['channel'].map(fold)
    if 'product_name' in L.columns:
        L['product_f'] = L['product_name'].map(fold)
    if 'category' in L.columns:
        L['category_f'] = L['category'].map(fold)
    if 'customer_group' in L.columns:
        L['customer_group_f'] = L['customer_group'].map(fold)
    if 'cost_unit' in L.columns:
        L['cogs'] = L['qty'] * L['cost_unit']

    # ---- 11. bảng đơn canonical
    orders = _build_orders(L, O_all)
    payments = _build_payments(raw, bm)
    purchases = _build_purchases(raw, bm, products)
    suppliers = _build_suppliers(raw, bm)
    if not purchases.empty and suppliers and 'supplier_code' in purchases.columns:
        sc = purchases['supplier_code'].map(lambda x: suppliers.get(x) if _nn(x) else None)
        purchases['supplier'] = purchases['supplier'].fillna(sc) if 'supplier' in purchases.columns else sc

    # ---- 11b. bảng cho khối tính chung (thẻ kho, lịch sử giá, lô, giao hàng)
    stock = _build_stock(raw, bm, products)
    prices = _build_prices(raw, bm, products)
    lots, lot_sales = _build_lots(raw, bm, products, purchases, units)
    deliveries = _build_deliveries(raw, bm)
    health.update(reconcile(L, orders if O_all is not None else None))

    # ---- 12. capabilities
    caps = set()
    for nm, df_ in (('stock', stock), ('prices', prices), ('lots', lots), ('deliveries', deliveries)):
        if df_ is not None and not df_.empty:
            caps.add(nm)
    if real_order_id:
        caps.add('orders')
    if 'status_raw' in L.columns or 'status from note' in ' '.join(notes):
        caps.add('status')
    if has_code:
        caps.add('customer_code')
    if L['customer_key'].notna().any():
        caps.add('customers')
    for c, cap in (('customer_group', 'customer_group'), ('category', 'category'), ('time', 'time'), ('channel', 'channel'),
                   ('payment', 'payment'), ('staff', 'staff'), ('rx', 'rx'), ('ingredient', 'ingredient')):
        if c in L.columns and L[c].notna().any():
            caps.add(cap)
    if 'cost_unit' in L.columns and L['cost_unit'].notna().any():
        caps.add('cost_line')
    if not products.empty and 'cost' in products.columns and products['cost'].notna().any():
        caps.add('cost_catalog')
    if not payments.empty:
        caps.add('payments')
    if not purchases.empty:
        caps.add('purchases')
    if {'order_total', 'paid'} <= set(orders.columns) or 'debt' in orders.columns:
        caps.add('debt')
    if caps_no_lines:
        caps.add('no_lines')

    dmin = orders['date'].dropna().min() if 'date' in orders.columns and orders['date'].notna().any() else None
    dmax = orders['date'].dropna().max() if 'date' in orders.columns and orders['date'].notna().any() else None
    if info.data_end and (dmax is None or info.data_end > dmax):
        dmax = info.data_end
    return Canonical(lines=L, orders=orders, products=products, units=units, customers=customers, payments=payments,
                     purchases=purchases, suppliers=suppliers, caps=caps, notes=notes, date_min=dmin, date_max=dmax,
                     stock=stock, prices=prices, lots=lots, lot_sales=lot_sales, deliveries=deliveries, health=health)


def _empty(notes):
    e = pd.DataFrame()
    return Canonical(lines=e, orders=e, products=e, units=e, customers=e, payments=e, purchases=e, notes=notes)


# ----------------------------------------------------------------------------- sub builders
def _build_products(raw, bm, L) -> pd.DataFrame:
    rows = []
    maps = bm.of_type('products')
    # chọn bảng danh mục khớp mã/tên với dòng hàng
    best, best_score = None, 0.0
    for m in maps:
        t = raw[m.table]
        sc = 0.0
        if 'product_code' in m.roles and 'product_code' in L.columns:
            sc = max(sc, _overlap(L['product_code'], t.df[m.roles['product_code']]))
        if 'product_name' in m.roles and 'product_name' in L.columns:
            a = L['product_name'].dropna().head(5000).map(fold)
            b = t.df[m.roles['product_name']].dropna().map(fold)
            sc = max(sc, len(set(a) & set(b)) / max(len(set(a)), 1))
        if sc > best_score:
            best, best_score = m, sc
    if best is None:
        return pd.DataFrame(columns=['product_key', 'product_code', 'product_name', 'name_f'])
    t = raw[best.table]
    P = _frame(t, best, {'product_code', 'product_name', 'category', 'base_unit', 'cost', 'price', 'rx', 'ingredient'})
    for c in ('product_code', 'product_name', 'category', 'base_unit'):
        if c in P.columns:
            P[c] = _str(P[c])
    for c in ('cost', 'price'):
        if c in P.columns:
            P[c] = _num(P[c])
    if 'product_code' not in P.columns:
        P['product_code'] = None
    if 'product_name' not in P.columns:
        P['product_name'] = None
    P['name_f'] = P['product_name'].map(fold)
    P['product_key'] = [fold(c) if c else n for c, n in zip(P['product_code'], P['name_f'])]
    # bảng thuốc (kê đơn, hoạt chất) theo mã
    for m in bm.of_type('drugs'):
        D = _frame(raw[m.table], m, {'product_code', 'product_name', 'rx', 'ingredient'})
        key = 'product_code' if 'product_code' in D.columns else ('product_name' if 'product_name' in D.columns else None)
        if not key:
            continue
        D['k'] = _str(D[key]).map(fold)
        for c in ('rx', 'ingredient'):
            if c in D.columns:
                P[c] = P['product_key'].map(dict(zip(D['k'], D[c])))
    if 'rx' in P.columns:
        P['rx'] = P['rx'].map(lambda v: fold(v) in ('true', '1', 'x', 'co', 'yes', 'ke don', 'rx', 'c') if v is not None else False)
    P = P.drop(columns=[c for c in P.columns if c.startswith('__')])
    return P.drop_duplicates('product_key').reset_index(drop=True)


def _build_units(raw, bm, info, products) -> pd.DataFrame:
    rows = []
    name2key = dict(zip(products['name_f'], products['product_key'])) if not products.empty else {}
    for m in bm.of_type('units'):
        U = _frame(raw[m.table], m)
        if 'factor' not in U.columns:
            continue
        if 'product_code' in U.columns:
            k = _str(U['product_code']).map(fold)
        elif 'product_name' in U.columns:
            k = _str(U['product_name']).map(fold).map(lambda x: name2key.get(x, x))
        else:
            continue
        f = _num(U['factor'])
        u = _str(U['unit']).map(fold) if 'unit' in U.columns else None
        b = _str(U['base_unit']) if 'base_unit' in U.columns else pd.Series([None] * len(U))
        for kk, uu, ff, bb in zip(k, u if u is not None else [None] * len(U), f, b):
            if kk and uu and ff == ff:
                rows.append((kk, uu, float(ff), bb))
            if kk and bb:
                rows.append((kk, fold(bb), 1.0, bb))
    for prod, big, fac, base in info.unit_conversions:
        k = fold(prod)
        k = name2key.get(k, k)
        rows.append((k, fold(big), float(fac), base))
        rows.append((k, fold(base), 1.0, base))
    U = pd.DataFrame(rows, columns=['product_key', 'unit_f', 'factor', 'base_unit'])
    return U.drop_duplicates(['product_key', 'unit_f']).reset_index(drop=True)


def _infer_base_unit(L: pd.DataFrame, units: pd.DataFrame) -> pd.Series:
    bmap = {}
    if not units.empty:
        for k, b in units.dropna(subset=['base_unit'])[['product_key', 'base_unit']].itertuples(index=False):
            bmap.setdefault(k, b)
    if 'unit' in L.columns:
        one = L[(L['factor'].fillna(1) == 1) & L['unit'].notna()]
        mode = one.groupby('product_key')['unit'].agg(lambda s: s.value_counts().index[0]) if len(one) else pd.Series(dtype=object)
        for k, u in mode.items():
            bmap.setdefault(k, u)
    return L['product_key'].map(bmap)


def _build_customers(raw, bm) -> pd.DataFrame:
    for m in bm.of_type('customers'):
        C = _frame(raw[m.table], m, {'customer_code', 'customer_name', 'customer_group', 'phone', 'address'})
        if 'customer_code' not in C.columns:
            continue
        for c in C.columns:
            if not c.startswith('__'):
                C[c] = _str(C[c])
        if 'customer_name' not in C.columns:
            C['customer_name'] = None
        C['customer_key'] = C['customer_code'].map(lambda x: fold(x) if _nn(x) else None)
        return C.drop(columns=[c for c in C.columns if c.startswith('__')]).drop_duplicates('customer_key')
    return pd.DataFrame(columns=['customer_key', 'customer_code', 'customer_name'])


def _build_orders(L: pd.DataFrame, O_all: Optional[pd.DataFrame]) -> pd.DataFrame:
    prod = L[~L['is_nonproduct']]
    agg = prod.groupby('order_id', sort=False).agg(revenue=('amount', 'sum'), n_lines=('amount', 'size'))
    firsts = [c for c in ORDER_LEVEL + ['status', 'customer_key', 'is_walkin', 'customer_display', 'payment_n', 'channel_f',
                                        'customer_group_f'] if c in L.columns]
    firsts = list(dict.fromkeys(firsts))
    F = L.groupby('order_id', sort=False)[firsts].first()
    O = F.join(agg, how='left')
    if O_all is not None:
        # đơn không có dòng hàng (vd đơn huỷ chưa xuất chi tiết) vẫn được đếm trong mẫu số tỉ lệ huỷ
        miss = O_all[~O_all['order_id'].isin(O.index)].copy()
        if len(miss):
            if 'status' in miss.columns:
                miss['status'] = _map_unique(miss['status'], norm_status).fillna('completed').replace({'unknown': 'completed'})
            miss = miss.set_index('order_id')
            miss = miss[[c for c in O.columns if c in miss.columns]]
            O = pd.concat([O, miss])
    O['revenue'] = O['revenue'].fillna(0.0)
    O['n_lines'] = O['n_lines'].fillna(0)
    return O.reset_index().rename(columns={'index': 'order_id'})


def _build_payments(raw, bm) -> pd.DataFrame:
    parts = []
    for m in bm.of_type('payments'):
        P = _frame(raw[m.table], m, {'order_id', 'pay_datetime', 'pay_amount', 'payment', 'date', 'datetime'})
        if 'order_id' not in P.columns or 'pay_amount' not in P.columns:
            continue
        dcol = next((c for c in ('pay_datetime', 'datetime', 'date') if c in P.columns), None)
        out = pd.DataFrame({'order_id': _str(P['order_id']), 'amount': _num(P['pay_amount'])})
        out['pay_date'] = _dates(P[dcol]) if dcol else None
        out['method'] = _map_unique(P['payment'], norm_payment) if 'payment' in P.columns else None
        parts.append(out)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['order_id', 'amount', 'pay_date', 'method'])


def _build_suppliers(raw, bm) -> dict:
    out = {}
    for m in bm.of_type('suppliers'):
        if 'supplier_code' in m.roles and 'supplier' in m.roles:
            df = raw[m.table].df
            for c, n in zip(_str(df[m.roles['supplier_code']]), _str(df[m.roles['supplier']])):
                if c and n:
                    out[c] = n
    return out


def _build_purchases(raw, bm, products) -> pd.DataFrame:
    maps = bm.of_type('purchases')
    if not maps:
        return pd.DataFrame()
    name2key = dict(zip(products['name_f'], products['product_key'])) if not products.empty else {}
    lines, heads = [], []
    for m in maps:
        P = _frame(raw[m.table], m)
        has_prod = bool({'product_code', 'product_name'} & set(P.columns))
        (lines if has_prod else heads).append(P)
    if not lines:
        return pd.DataFrame()
    Lp = pd.concat(lines, ignore_index=True)
    for c in ('p_qty', 'p_unit_cost', 'p_amount', 'p_total'):
        if c in Lp.columns:
            Lp[c] = _num(Lp[c])
    for c in ('receipt_no', 'product_code', 'product_name', 'supplier', 'supplier_code', 'p_unit', 'p_status', 'note'):
        if c in Lp.columns:
            Lp[c] = _str(Lp[c])
    if 'receipt_no' in Lp.columns and Lp['receipt_no'].isna().any():
        Lp['receipt_no'] = Lp['receipt_no'].ffill()
    if heads and 'receipt_no' in Lp.columns:
        H = pd.concat(heads, ignore_index=True)
        if 'receipt_no' in H.columns:
            H['receipt_no'] = _str(H['receipt_no'])
            if 'p_total' in H.columns:
                H['p_total'] = _num(H['p_total'])
            keep = [c for c in ('p_date', 'supplier', 'supplier_code', 'p_status', 'p_total', 'note') if c in H.columns]
            Lp = Lp.drop(columns=[c for c in keep if c in Lp.columns]).merge(H[['receipt_no'] + keep].drop_duplicates('receipt_no'), on='receipt_no', how='left')
    if 'p_date' in Lp.columns:
        Lp['p_date'] = _dates(Lp['p_date'], _sheet_year(Lp))
    if 'product_code' in Lp.columns:
        Lp['product_key'] = Lp['product_code'].map(lambda x: fold(x) if x else None)
    elif 'product_name' in Lp.columns:
        Lp['product_key'] = Lp['product_name'].map(lambda x: name2key.get(fold(x), fold(x)) if x else None)
    if 'p_amount' not in Lp.columns and {'p_qty', 'p_unit_cost'} <= set(Lp.columns):
        Lp['p_amount'] = Lp['p_qty'] * Lp['p_unit_cost']
    if 'p_status' in Lp.columns:
        Lp['p_cancelled'] = Lp['p_status'].map(lambda v: bool(re.search(r'huy|cancel', fold(v))) if v else False)
    else:
        Lp['p_cancelled'] = False
    return Lp.drop(columns=[c for c in Lp.columns if c.startswith('__')])


# ----------------------------------------------------------------------------- v1.1: kho, giá, lô, giao hàng
def _pkey(df: pd.DataFrame, products: pd.DataFrame) -> Optional[pd.Series]:
    name2key = dict(zip(products['name_f'], products['product_key'])) if products is not None and not products.empty else {}
    if 'product_code' in df.columns and df['product_code'].notna().any():
        return _str(df['product_code']).map(lambda x: fold(x) if x else None)
    if 'product_name' in df.columns:
        return _str(df['product_name']).map(lambda x: name2key.get(fold(x), fold(x)) if x else None)
    return None


def _build_stock(raw, bm, products) -> pd.DataFrame:
    """Thẻ kho dạng 'tồn sau mỗi phát sinh' (cột tồn cuối + thời điểm). Bảng không có thời điểm -> bỏ (không đoán)."""
    parts = []
    for m in bm.of_type('inventory'):
        f = _frame(raw[m.table], m)
        tcol = next((c for c in ('inv_datetime', 'datetime', 'date') if c in f.columns), None)
        if 'stock' not in f.columns or tcol is None:
            continue
        pk = _pkey(f, products)
        if pk is None:
            continue
        ts = pd.to_datetime(_str(f[tcol]), errors='coerce', dayfirst=False)
        out = pd.DataFrame({'product_key': pk, 'ts': ts, 'balance': _num(f['stock']), 'seq': range(len(f))})
        parts.append(out.dropna(subset=['product_key', 'ts', 'balance']))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_prices(raw, bm, products) -> pd.DataFrame:
    parts = []
    for m in bm.of_type('price_history'):
        f = _frame(raw[m.table], m)
        if 'eff_date' not in f.columns or 'new_price' not in f.columns:
            continue
        pk = _pkey(f, products)
        if pk is None:
            continue
        parts.append(pd.DataFrame({'product_key': pk, 'eff_date': _dates(f['eff_date']), 'price': _num(f['new_price'])})
                     .dropna(subset=['product_key', 'eff_date', 'price']))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _build_lots(raw, bm, products, purchases, units) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lô: bảng có (mã lô, hạn dùng, hàng); phân bổ bán theo lô: bảng có (mã đơn, mã lô, SL)."""
    master, sales = [], []
    for m in bm.of_type('lots', 'purchases'):
        f = _frame(raw[m.table], m)
        if 'lot' not in f.columns:
            continue
        if 'expiry' in f.columns and ({'product_code', 'product_name'} & set(f.columns)) and 'order_id' not in f.columns:
            pk = _pkey(f, products)
            qcol = next((c for c in ('p_qty_base', 'qty_base') if c in f.columns), None)
            if m.ttype == 'purchases' or qcol is None:
                continue                       # dòng nhập có mã lô: dùng để tra đơn vị nhập, không làm bảng lô chính
            d = pd.DataFrame({'lot': _str(f['lot']), 'product_key': pk, 'expiry': _dates(f['expiry']),
                              'received_base': _num(f[qcol]),
                              'unit_cost': _num(f['p_unit_cost']) if 'p_unit_cost' in f.columns else np.nan,
                              'receipt_no': _str(f['receipt_no']) if 'receipt_no' in f.columns else None})
            master.append(d)
        elif 'order_id' in f.columns and 'qty_base' in f.columns:
            sales.append(pd.DataFrame({'order_id': _str(f['order_id']), 'lot': _str(f['lot']), 'qty_base': _num(f['qty_base']),
                                       'return_qty': _num(f['return_qty']) if 'return_qty' in f.columns else 0.0}))
    if not master:
        return pd.DataFrame(), pd.DataFrame()
    Lt = pd.concat(master, ignore_index=True).dropna(subset=['lot']).drop_duplicates('lot')
    Lt['recv_date'] = None
    Lt['unit_cost_base'] = np.nan
    if purchases is not None and not purchases.empty:
        pu = purchases
        if 'receipt_no' in pu.columns and 'p_date' in pu.columns:
            rd = pu.dropna(subset=['receipt_no']).drop_duplicates('receipt_no').set_index('receipt_no')['p_date']
            Lt['recv_date'] = Lt['receipt_no'].map(rd)
        # giá nhập của lô theo đơn vị nào? đối chiếu với dòng phiếu nhập cùng mã lô (giá theo đơn vị nhập):
        #   lot_cost ≈ line_cost / hệ số  -> đã theo đơn vị cơ bản;  lot_cost ≈ line_cost -> theo đơn vị nhập, chia hệ số.
        fac = {(k, u): f for k, u, f in units[['product_key', 'unit_f', 'factor']].itertuples(index=False)} if not units.empty else {}
        link = {}
        if 'lot' in pu.columns and 'p_unit' in pu.columns and 'p_unit_cost' in pu.columns:
            q = pu.assign(lot=_str(pu['lot'])).dropna(subset=['lot']).drop_duplicates('lot')
            link = {r.lot: (r.p_unit, r.p_unit_cost) for r in q[['lot', 'p_unit', 'p_unit_cost']].itertuples(index=False)}

        def per_base(row):
            c = row['unit_cost']
            if c != c:
                return np.nan
            u, lc = link.get(row['lot'], (None, None))
            f = fac.get((row['product_key'], fold(u))) if u else None
            if f is None and u is not None:
                f = 1.0
            if f and lc == lc and lc is not None:
                if abs(lc / f - c) <= max(0.5, 0.005 * c):
                    return c
                if abs(lc - c) <= max(0.5, 0.005 * c):
                    return c / f
                return np.nan                       # không nhất quán -> không đoán
            return c                                # bảng lô ghi SL theo đơn vị cơ bản, không có dòng nhập đối chiếu
        Lt['unit_cost_base'] = Lt.apply(per_base, axis=1)
    LS = pd.concat(sales, ignore_index=True) if sales else pd.DataFrame()
    return Lt.reset_index(drop=True), LS


def _build_deliveries(raw, bm) -> pd.DataFrame:
    parts = []
    for m in bm.of_type('deliveries'):
        f = _frame(raw[m.table], m)
        if 'deliv_date' not in f.columns:
            continue
        parts.append(pd.DataFrame({'order_id': _str(f['order_id']) if 'order_id' in f.columns else None,
                                   'deliv_date': _dates(f['deliv_date']),
                                   'deliv_no': _str(f['deliv_no']) if 'deliv_no' in f.columns else None}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
