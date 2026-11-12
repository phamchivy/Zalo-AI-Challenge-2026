"""v3.0-a — Suy ra cấu trúc dữ liệu từ GIÁ TRỊ, không từ tên cột.

Bằng chứng dùng (đều bất biến khi đổi tên cột / đổi ngôn ngữ):
  1. Khoá chính: cột có giá trị gần như duy nhất trong bảng và được NHIỀU bảng khác tham chiếu (giá trị cột con ⊂ cột cha).
  2. Khoá ngoại: tập giá trị khác nhau của cột con nằm trong tập giá trị khoá chính của bảng cha (>= 95%).
  3. Bảng con của bảng đơn có dòng hàng: tồn tại bộ ba cột số (SL, đơn giá, thành tiền) thoả thành tiền ≈ SL × đơn giá
     (− giảm dòng) trên phần lớn dòng.
  4. Tiền hàng của đơn: cột số ở bảng đơn bằng Σ thành tiền các dòng của đơn.

Chỉ BỔ SUNG role còn thiếu (không ghi đè role regex đã gán), đánh dấu evidence='structure'.

Kèm kiểm tra độ phủ (invariant COVERAGE_child): bảng con của bảng đơn (khoá ⊂ mã đơn, nhiều dòng/mã) mà không được hiểu
là dòng hàng / thanh toán / trả hàng / lô / giao hàng -> dữ liệu bị bỏ sót -> câu dựa trên đơn hàng không đáng tin.
"Không kiểm được" không còn bị coi là "an toàn".
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from datamate.core.textutil import to_num
from datamate.data.schema_mapper import ALLOWED, BusinessMapping, TableMapping

log = logging.getLogger(__name__)

# loại bảng "chưa hiểu": phân loại theo tên chung chung hoặc không phân loại được
GENERIC = {'sales', 'other', None}
KEY_MIN_OVERLAP = 0.95          # tỉ lệ giá trị khác nhau của cột con nằm trong khoá cha
PK_MIN_UNIQUE = 0.98            # tỉ lệ duy nhất của khoá chính
TRIPLE_MIN_SHARE = 0.8          # tỉ lệ dòng thoả thành tiền ≈ SL × giá (− giảm)
SAMPLE = 5000


@dataclass
class StructureReport:
    notes: list[str] = field(default_factory=list)
    added: dict[str, dict[str, str]] = field(default_factory=dict)       # bảng -> {role: cột}
    retyped: dict[str, tuple[str, str]] = field(default_factory=dict)    # bảng -> (cũ, mới)
    uncovered_children: list[str] = field(default_factory=list)          # bảng con của đơn không hiểu được

    def coverage(self) -> dict:
        n = len(self.uncovered_children)
        return {'name': 'COVERAGE_child', 'n': n + 1, 'bad': n, 'level': 'red' if n else 'green',
                'rate': float(n > 0), 'note': f'bảng con của đơn không hiểu được: {self.uncovered_children}' if n else '',
                'examples': [{'bang': t} for t in self.uncovered_children[:3]]}


def _norm_vals(s: pd.Series) -> pd.Series:
    s = s.dropna()
    if s.empty:
        return s.astype(str)
    if pd.api.types.is_float_dtype(s):
        s = s.map(lambda v: str(int(v)) if float(v).is_integer() else str(v))
    else:
        s = s.astype(str)
    s = s.str.strip()
    return s[s != '']


class _Vals:
    """Tập giá trị khác nhau của từng cột (tính lười, có cache)."""

    def __init__(self, raw):
        self.raw = raw
        self._c: dict[tuple[str, str], set] = {}

    def get(self, t: str, c: str) -> set:
        k = (t, c)
        if k not in self._c:
            df = self.raw[t].df
            self._c[k] = set(_norm_vals(df[c]).unique()) if c in df.columns else set()
        return self._c[k]


def _overlap(child: set, parent: set) -> float:
    if not child:
        return 0.0
    return len(child & parent) / len(child)


def _is_unique(df: pd.DataFrame, c: str) -> bool:
    s = _norm_vals(df[c])
    return len(s) >= 10 and s.nunique() >= PK_MIN_UNIQUE * len(s) and len(s) >= 0.9 * len(df)


def _key_cols(t: str, raw, vals: _Vals, profiles) -> list[str]:
    """Cột có thể là khoá: giá trị ngắn, không phải ngày, không phải số tiền lớn liên tục."""
    out = []
    for c, p in profiles[t].columns.items():
        if c.startswith('__') or p.null_rate > 0.5 or p.mean_len > 40 or p.date_rate >= 0.8:
            continue
        out.append(c)
    return out


def _primary_key(t: str, raw, vals: _Vals, profiles, others: list[str], mapped: Optional[str]) -> Optional[str]:
    """Khoá chính của bảng cha: role đã gán (nếu duy nhất) hoặc cột duy nhất được nhiều bảng khác tham chiếu nhất."""
    df = raw[t].df
    if mapped and mapped in df.columns and _is_unique(df, mapped):
        return mapped
    best, best_refs = None, 0
    for c in _key_cols(t, raw, vals, profiles):
        if not _is_unique(df, c):
            continue
        pv = vals.get(t, c)
        refs = 0
        for o in others:
            for oc in _key_cols(o, raw, vals, profiles):
                ov = vals.get(o, oc)
                if len(ov) >= 5 and _overlap(ov, pv) >= KEY_MIN_OVERLAP:
                    refs += 1
                    break
        if refs > best_refs:
            best, best_refs = c, refs
    return best if best_refs >= 1 else None


def _num(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors='coerce')
    miss = x.isna() & s.notna()
    if miss.any():
        x[miss] = s[miss].map(to_num)
    return pd.to_numeric(x, errors='coerce')


def _find_triple(df: pd.DataFrame, profiles_t, used: set) -> Optional[dict]:
    """Tìm (qty, unit_price, amount[, line_discount]) với amount ≈ qty × price − discount trên >= 80% dòng."""
    cols = [c for c, p in profiles_t.columns.items()
            if not c.startswith('__') and p.num_rate >= 0.9 and p.date_rate < 0.5 and c in df.columns]
    if len(cols) < 3:
        return None
    sub = df[cols].head(SAMPLE)
    X = {c: _num(sub[c]) for c in cols}
    best = None
    for a in cols:
        A = X[a]
        okA = A.notna() & (A != 0)
        if okA.sum() < 20 or okA.mean() < 0.8:          # thành tiền: khác 0 ở hầu hết các dòng (giảm giá thì không)
            continue
        for i, q in enumerate(cols):
            for pcol in cols[i + 1:]:
                if len({a, q, pcol}) < 3:
                    continue
                Q, P = X[q], X[pcol]
                m = okA & Q.notna() & P.notna()
                if m.sum() < 20:
                    continue
                tol = np.maximum(1, 0.005 * A[m].abs())
                prod = Q[m] * P[m]
                exact = ((A[m] - prod).abs() <= tol)
                share = float(exact.mean())
                disc = None
                if share < TRIPLE_MIN_SHARE:
                    for d in cols:
                        if d in (a, q, pcol):
                            continue
                        D = X[d][m].fillna(0)
                        if float((D != 0).mean()) > 0.5 or float(D.abs().median()) > float(A[m].abs().median()):
                            continue                          # giảm dòng: phần lớn bằng 0 và nhỏ hơn thành tiền
                        sh = float(((A[m] - (prod - D)).abs() <= tol).mean())
                        if sh > share:
                            share, disc = sh, d
                if share >= TRIPLE_MIN_SHARE and (best is None or (share, disc is None) > (best['share'], best['line_discount'] is None)):
                    # SL: cột có trung vị nhỏ hơn (SL thường 1–50, đơn giá lớn hơn nhiều)
                    mq, mp = float(Q[m].median()), float(P[m].median())
                    qq, pp = (q, pcol) if mq <= mp else (pcol, q)
                    best = {'qty': qq, 'unit_price': pp, 'amount': a, 'line_discount': disc, 'share': share}
    if best and (best['qty'] in used and best['unit_price'] in used and best['amount'] in used):
        return None
    return best


def _assign(tm: TableMapping, role: str, col: Optional[str], rep: StructureReport) -> bool:
    if not col or role in tm.roles or col in tm.col_roles:
        return False
    allowed = ALLOWED.get('sales' if tm.ttype.startswith('sales') else tm.ttype)
    if allowed is not None and role not in allowed:
        return False
    tm.roles[role] = col
    tm.col_roles[col] = role
    tm.evidence[role] = 'structure'
    rep.added.setdefault(tm.table, {})[role] = col
    return True


def _name_col(t: str, raw, profiles, exclude: set) -> Optional[str]:
    """Cột tên trong bảng danh mục: chữ, gần như duy nhất, ít trống, dài nhất."""
    df = raw[t].df
    best, key = None, None
    for c, p in profiles[t].columns.items():
        if c in exclude or c.startswith('__') or p.null_rate > 0.1 or p.num_rate > 0.2 or p.date_rate > 0.2:
            continue
        if p.n_distinct < 0.8 * max(1, len(df) * (1 - p.null_rate)) or p.mean_len < 4:
            continue
        k = (p.n_distinct, p.mean_len)
        if key is None or k > key:
            best, key = c, k
    return best


def infer_structure(raw, profiles, bm: BusinessMapping) -> StructureReport:
    rep = StructureReport()
    try:
        _infer(raw, profiles, bm, rep)
    except Exception as e:  # noqa: BLE001 - suy cấu trúc lỗi không được làm hỏng shop: giữ mapping regex
        rep.notes.append(f'structure failed: {e!r}'[:200])
        log.warning('structure failed: %r', e)
    return rep


def _infer(raw, profiles, bm: BusinessMapping, rep: StructureReport) -> None:
    vals = _Vals(raw)
    names = [n for n in bm.tables if n in raw and n in profiles]
    # ---- 1. danh mục hàng / khách: khoá chính + tên, rồi khoá ngoại ở bảng khác
    for ptype, code_role, name_role in (('products', 'product_code', 'product_name'),
                                        ('customers', 'customer_code', 'customer_name')):
        for pm in bm.of_type(ptype):
            t = pm.table
            if t not in raw:
                continue
            others = [n for n in names if n != t]
            pk = _primary_key(t, raw, vals, profiles, others, pm.roles.get(code_role))
            if not pk:
                continue
            _assign(pm, code_role, pk, rep)
            if name_role not in pm.roles:
                _assign(pm, name_role, _name_col(t, raw, profiles, set(pm.col_roles)), rep)
            pv = vals.get(t, pk)
            for o in others:
                om = bm.tables[o]
                if code_role in om.roles:
                    continue
                cands = []
                for oc in _key_cols(o, raw, vals, profiles):
                    if oc in om.col_roles:
                        continue
                    ov = vals.get(o, oc)
                    if len(ov) >= 5 and _overlap(ov, pv) >= KEY_MIN_OVERLAP:
                        cands.append((len(ov), oc))
                if cands:
                    _assign(om, code_role, max(cands)[1], rep)
    # ---- 2. bảng đơn: khoá mã đơn; bảng con của đơn
    orders = [m for m in bm.of_type('sales_orders') if m.roles.get('order_id') in raw[m.table].df.columns]
    for om in orders:
        O = om.table
        ok_col = om.roles['order_id']
        ov = vals.get(O, ok_col)
        if not ov:
            continue
        for t in names:
            if t == O:
                continue
            tm = bm.tables[t]
            df = raw[t].df
            # cột tham chiếu mã đơn
            ref = tm.roles.get('order_id')
            if not (ref and _overlap(vals.get(t, ref), ov) >= KEY_MIN_OVERLAP):
                ref = None
                for c in _key_cols(t, raw, vals, profiles):
                    cv = vals.get(t, c)
                    if len(cv) >= 5 and _overlap(cv, ov) >= KEY_MIN_OVERLAP:
                        ref = c
                        break
            if not ref:
                continue
            n_keys = len(vals.get(t, ref))
            if tm.ttype not in GENERIC and tm.ttype != 'sales_lines':
                continue                                   # bảng đã hiểu là loại khác (thanh toán, lô, kho…)
            if tm.ttype == 'sales_lines' and {'qty', 'amount'} <= set(tm.roles):
                continue
            if 'order_id' not in tm.roles and ref not in tm.col_roles:
                _assign(tm, 'order_id', ref, rep)
            tri = _find_triple(df, profiles[t], set(tm.col_roles))
            if tri:
                for role in ('qty', 'unit_price', 'amount', 'line_discount'):
                    _assign(tm, role, tri.get(role), rep)
                has_prod = bool({'product_code', 'product_name'} & set(tm.roles))
                if has_prod and {'qty', 'amount'} & set(tm.roles) and tm.ttype != 'sales_lines' and len(df) >= n_keys:
                    rep.retyped[t] = (tm.ttype, 'sales_lines')
                    tm.ttype = 'sales_lines'
                # tiền hàng của đơn = Σ thành tiền các dòng
                if 'goods_total' not in om.roles and 'amount' in tm.roles:
                    _goods_total(raw, profiles, om, tm, rep)
    # ---- 3. độ phủ: bảng con của đơn còn chưa hiểu được
    rep.uncovered_children = uncovered_children(raw, profiles, bm, vals)
    for t, rs in rep.added.items():
        rep.notes.append(f'structure: {t} += {rs}')
    for t, (a, b) in rep.retyped.items():
        rep.notes.append(f'structure: {t} {a} -> {b}')


def uncovered_children(raw, profiles, bm: BusinessMapping, vals: Optional['_Vals'] = None) -> list[str]:
    """Bảng con của bảng đơn (khoá ⊂ mã đơn, nhiều dòng/mã) mà chưa được hiểu (loại chung chung, hoặc 'dòng hàng'
    không có SL/thành tiền). Dùng lại sau khi LLM bổ sung vai trò để chấm lại độ phủ."""
    vals = vals or _Vals(raw)
    names = [n for n in bm.tables if n in raw and n in profiles]
    out: list[str] = []
    for om in bm.of_type('sales_orders'):
        if om.roles.get('order_id') not in raw[om.table].df.columns:
            continue
        ov = vals.get(om.table, om.roles['order_id'])
        for t in names:
            tm = bm.tables[t]
            broken_lines = tm.ttype == 'sales_lines' and not ({'qty', 'amount'} & set(tm.roles))
            if t == om.table or t in out or not (tm.ttype in GENERIC or broken_lines):
                continue
            ref = tm.roles.get('order_id')
            cands = [ref] if ref else _key_cols(t, raw, vals, profiles)
            for c in cands:
                cv = vals.get(t, c)
                if len(cv) >= 5 and _overlap(cv, ov) >= KEY_MIN_OVERLAP and len(raw[t].df) > 1.05 * len(cv):
                    out.append(t)
                    break
    return out


def _goods_total(raw, profiles, om: TableMapping, lm: TableMapping, rep: StructureReport) -> None:
    L, O = raw[lm.table].df, raw[om.table].df
    amt = _num(L[lm.roles['amount']])
    key = _norm_vals(L[lm.roles['order_id']]).reindex(L.index)
    s = amt.groupby(key).sum()
    okey = _norm_vals(O[om.roles['order_id']]).reindex(O.index)
    for c, p in profiles[om.table].columns.items():
        if c in om.col_roles or c.startswith('__') or p.num_rate < 0.9:
            continue
        v = _num(O[c])
        j = pd.DataFrame({'v': v.values, 's': okey.map(s).values}).dropna()
        j = j[j['s'] != 0]
        if len(j) >= 20 and float(((j['v'] - j['s']).abs() <= np.maximum(1, 0.005 * j['s'].abs())).mean()) >= 0.9:
            _assign(om, 'goods_total', c, rep)
            return
