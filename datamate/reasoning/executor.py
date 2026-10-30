"""Thư viện metric deterministic chạy Plan trên bảng canonical (tầng A). LLM không bao giờ tự tính số.

Luật áp dụng (rulebook §1-§3): doanh thu = tổng thành tiền dòng (sau giảm dòng) của đơn không huỷ/không trả (đơn đang
giao vẫn tính); số đơn chỉ tính đơn hợp lệ; tỉ lệ huỷ = huỷ / mọi đơn trong kỳ; bán chạy theo SL quy về đơn vị cơ bản;
phí phục vụ tính doanh thu nhưng không phải mặt hàng; khách lẻ không xếp hạng; hoà điểm xếp theo mã tăng dần.
"""
from __future__ import annotations

import math
from datetime import date
from typing import Any, Optional

import numpy as np
import pandas as pd

from datamate.core.textutil import fold
from datamate.core.types import AnswerType, Plan, TimeRange
from datamate.knowledge.rulebook import RuleParams

VALID = ('completed', 'shipping')


class PlanError(Exception):
    """Plan không chạy được trên dữ liệu này (thiếu cột, thực thể không có...)."""


class Unanswerable(Exception):
    """Dữ liệu không có loại thông tin câu hỏi cần."""


def round_money(x: float) -> int:
    """Tiền làm tròn về đồng, nửa đồng làm tròn lên."""
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def round_pct(x: float) -> float:
    return float(round(x + 0.0, 2))


def as_number(x: float) -> Any:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return float(round(x, 2))


class Executor:
    def __init__(self, canon, params: Optional[RuleParams] = None):
        self.c = canon
        self.p = params or RuleParams()
        L = canon.lines
        self.L = L
        self.O = canon.orders
        self._has = set(L.columns) if L is not None and not L.empty else set()

    # ------------------------------------------------------------------ filters
    def _period_mask(self, df: pd.DataFrame, tr: Optional[TimeRange], col: str = 'date') -> pd.Series:
        if tr is None or col not in df.columns:
            return pd.Series(True, index=df.index)
        d = df[col]
        return d.notna() & (d >= tr.start) & (d <= tr.end)

    def _filter_mask(self, df: pd.DataFrame, filters: dict) -> pd.Series:
        m = pd.Series(True, index=df.index)
        for k, v in (filters or {}).items():
            if v is None or v == [] or v == '':
                continue
            if k == 'product':
                keys = v if isinstance(v, list) else [v]
                m &= df['product_key'].isin([fold(x) if x else x for x in keys]) | df['product_key'].isin(keys)
            elif k == 'category':
                self._need('category_f', 'category')
                vals = [fold(x) for x in (v if isinstance(v, list) else [v])]
                m &= df['category_f'].isin(vals)
            elif k == 'channel':
                self._need('channel_f', 'channel')
                m &= df['channel_f'].isin(v if isinstance(v, list) else [v])
            elif k == 'payment':
                self._need('payment_n', 'payment')
                m &= df['payment_n'].isin(v if isinstance(v, list) else [v])
            elif k == 'customer_group':
                self._need('customer_group_f', 'customer group')
                m &= (df['customer_group_f'] == fold(v)) & ~df['is_walkin']
            elif k == 'shift':
                self._need('time', 'time of order')
                a, b = {'morning': self.p.morning, 'afternoon': self.p.afternoon, 'evening': ('18:00', '23:59')}[v]
                t = df['time'].fillna('')
                m &= (t >= a + ':00') & (t <= b + ':59')
            elif k == 'staff':
                self._need('staff', 'staff')
                m &= df['staff'].map(fold) == fold(v)
            elif k == 'customer':
                m &= df['customer_key'] == v
            elif k == 'rx':
                self._need('rx', 'prescription flag')
                m &= df['rx'].fillna(False).astype(bool) == bool(v)
            elif k == 'ingredient':
                self._need('ingredient', 'active ingredient')
                m &= df['ingredient'].map(fold) == fold(v)
            elif k == 'size':
                m &= df['product_f'].fillna('').str.endswith(f'({str(v).lower()})')
            elif k == 'base_unit':
                m &= df['base_unit_f'] == fold(v)
            elif k == 'walkin':
                m &= df['is_walkin'] == bool(v)
            else:
                raise PlanError(f'unknown filter {k}')
        return m

    def _need(self, col: str, what: str):
        if col not in self._has or self.L[col].isna().all():
            raise Unanswerable(f'Dữ liệu không ghi {what}')

    def lines(self, tr: Optional[TimeRange], filters: dict, valid: bool = True, products_only: bool = False) -> pd.DataFrame:
        L = self.L
        if 'no_lines' in self.c.caps and (products_only or (filters or {}).keys() & {'product', 'category', 'size', 'base_unit', 'rx', 'ingredient'}):
            raise PlanError('shop không có dòng hàng chi tiết')
        m = self._period_mask(L, tr) & self._filter_mask(L, filters) & ~L['is_nonproduct']
        if valid:
            m &= L['status'].isin(VALID)
        if products_only:
            m &= ~L['is_service']
        return L[m]

    # ------------------------------------------------------------------ metrics
    def metric(self, name: str, tr: Optional[TimeRange], filters: dict) -> float:
        if name == 'revenue':
            return float(self.lines(tr, filters)['amount'].sum())
        if name == 'orders':
            return float(self.lines(tr, filters)['order_id'].nunique())
        if name == 'qty':
            return float(self.lines(tr, filters, products_only=True)['qty_base'].sum())
        if name == 'aov':
            x = self.lines(tr, filters)
            n = x['order_id'].nunique()
            if n == 0:
                raise PlanError('no orders')
            return x['amount'].sum() / n
        if name == 'customers':
            x = self.lines(tr, filters)
            return float(x.loc[~x['is_walkin'], 'customer_key'].nunique())
        if name == 'gross_profit':
            self._need('cogs', 'giá vốn trên dòng bán')
            x = self.lines(tr, filters)
            return float(x['amount'].sum() - x['cogs'].fillna(0).sum())
        if name in ('cancelled_orders', 'cancel_rate', 'all_orders'):
            x = self.lines(tr, filters, valid=False) if filters else None
            if x is not None:
                st = x.groupby('order_id')['status'].first()
            else:
                O = self.O
                st = O.loc[self._period_mask(O, tr), 'status']
            if name == 'all_orders':
                return float(len(st))
            canc = float((st == 'cancelled').sum())
            if name == 'cancelled_orders':
                return canc
            if len(st) == 0:
                raise PlanError('no orders in period')
            return 100.0 * canc / len(st)
        if name == 'lines':
            return float(len(self.lines(tr, filters)))
        raise PlanError(f'unknown metric {name}')

    def fmt_metric(self, name: str, v: float) -> tuple[AnswerType, Any]:
        if name in ('cancel_rate',):
            return AnswerType.PERCENT, round_pct(v)
        if name in ('revenue', 'aov', 'gross_profit'):
            return AnswerType.NUMBER, round_money(v)
        return AnswerType.NUMBER, as_number(v)

    # ------------------------------------------------------------------ grouping
    def group_metric(self, name: str, tr: Optional[TimeRange], filters: dict, by: str) -> pd.DataFrame:
        """-> DataFrame[key, display, code, value] cho xếp hạng."""
        if name in ('qty',):
            x = self.lines(tr, filters, products_only=True)
        else:
            x = self.lines(tr, filters)
        if by == 'product':
            if 'no_lines' in self.c.caps:
                raise PlanError('shop không có dòng hàng chi tiết')
            x = x[~x['is_service']]
            key, disp = 'product_key', 'product_name'
            code = 'product_code' if 'product_code' in x.columns else 'product_key'
        elif by == 'customer':
            x = x[~x['is_walkin']]
            key, disp = 'customer_key', 'customer_display'
            code = 'customer_code' if 'customer_code' in x.columns else 'customer_display'
        elif by == 'staff':
            self._need('staff', 'nhân viên')
            key, disp, code = 'staff', 'staff', 'staff'
        elif by == 'category':
            self._need('category', 'nhóm hàng')
            key, disp, code = 'category_f', 'category', 'category_f'
        elif by == 'day':
            key, disp, code = 'date', 'date', 'date'
        elif by == 'month':
            x = x.assign(month=x['date'].map(lambda d: f'{d.year:04d}-{d.month:02d}' if d else None))
            key, disp, code = 'month', 'month', 'month'
        elif by == 'weekday':
            x = x.assign(weekday=x['date'].map(lambda d: d.isoweekday() if d else None))
            key, disp, code = 'weekday', 'weekday', 'weekday'
        else:
            raise PlanError(f'unknown group_by {by}')
        if x.empty:
            return pd.DataFrame(columns=['key', 'display', 'code', 'value'])
        g = x.groupby(key, sort=False)
        if name == 'revenue':
            val = g['amount'].sum()
        elif name == 'qty':
            val = g['qty_base'].sum()
        elif name == 'orders':
            val = g['order_id'].nunique()
        elif name == 'gross_profit':
            self._need('cogs', 'giá vốn trên dòng bán')
            val = g['amount'].sum() - g['cogs'].sum()
        elif name == 'aov':
            val = g['amount'].sum() / g['order_id'].nunique()
        elif name == 'customers':
            val = x[~x['is_walkin']].groupby(key)['customer_key'].nunique()
        else:
            raise PlanError(f'cannot group metric {name}')
        dsp = g[disp].agg(lambda s: s.value_counts().index[0] if s.notna().any() else None)
        cd = g[code].first()
        out = pd.DataFrame({'key': val.index, 'value': val.values})
        out['display'] = out['key'].map(dsp)
        out['code'] = out['key'].map(cd).map(lambda v: str(v) if v is not None else '')
        return out

    def rank(self, df: pd.DataFrame, k: int, ascending: bool = False) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        df['_v'] = df['value'].round(6)
        df = df.sort_values(['_v', 'code'], ascending=[ascending, True], kind='mergesort')
        return df.head(k)

    # ------------------------------------------------------------------ plan
    def run(self, plan: Plan) -> tuple[AnswerType, Any]:
        if self.L is None or self.L.empty:
            raise PlanError('no canonical lines')
        op, met, P, F = plan.op, plan.metric, plan.periods, plan.filters or {}
        p0 = P[0] if P else None
        if met == 'debt':
            return self.run_debt(plan)
        if met.startswith('purchase'):
            return self.run_purchase(plan)
        if op == 'value':
            return self.fmt_metric(met, self.metric(met, p0, F))
        if op == 'series':
            vals = [self.fmt_metric(met, self.metric(met, p, F))[1] for p in P]
            return AnswerType.LIST, vals
        if op == 'values':
            vals = []
            for sv in plan.split_values:
                f2 = dict(F)
                f2[plan.split_by] = sv
                vals.append(self.fmt_metric(met, self.metric(met, p0, f2))[1])
            return AnswerType.LIST, vals
        if op == 'pct_change':
            if len(P) < 2:
                raise PlanError('need 2 periods')
            a, b = self.metric(met, P[0], F), self.metric(met, P[1], F)
            if b == 0:
                raise PlanError('base is zero')
            return AnswerType.PERCENT, round_pct(100.0 * (a - b) / b)
        if op == 'pct_change_filters':
            fa, fb = plan.split_values
            a, b = self.metric(met, p0, {**F, **fa}), self.metric(met, p0, {**F, **fb})
            if b == 0:
                raise PlanError('base is zero')
            return AnswerType.PERCENT, round_pct(100.0 * (a - b) / b)
        if op == 'diff':
            a, b = self.metric(met, P[0], F), self.metric(met, P[1], F)
            return self.fmt_metric(met, a - b)
        if op == 'share':
            base_f = {k: v for k, v in F.items() if k in (plan.note.split(',') if plan.note else [])}
            num = self.metric(met, p0, F)
            den = self.metric(met, p0, base_f)
            if den == 0:
                raise PlanError('zero denominator')
            return AnswerType.PERCENT, round_pct(100.0 * num / den)
        if op == 'trend_up':
            vals = [self.metric(met, p, F) for p in P]
            return AnswerType.BOOLEAN, all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))
        if op in ('top', 'bottom'):
            k = plan.k or 1
            df = self.group_metric(met, p0, F, plan.group_by)
            r = self.rank(df, k, ascending=(op == 'bottom'))
            if r.empty:
                raise PlanError('empty ranking')
            if plan.group_by in ('customer', 'staff', 'product', 'category'):
                names = [d for d in r['display'].tolist()]
            else:
                names = r['key'].tolist()
            if k == 1 and plan.answer_type != AnswerType.LIST:
                return AnswerType.STRING, names[0]
            return AnswerType.LIST, names
        if op == 'argmax':
            df = self.group_metric(met, None if len(P) > 1 else p0, F, plan.group_by)
            if len(P) > 1:
                keys = {f'{p.start.year:04d}-{p.start.month:02d}' for p in P} if plan.group_by == 'month' else None
                if keys:
                    df = df[df['key'].isin(keys)]
            if df.empty:
                raise PlanError('empty')
            df = df.copy()
            df['code'] = df['key'].astype(str)
            asc = plan.note == 'min'
            r = self.rank(df, 1, ascending=asc)
            k0 = r['key'].iloc[0]
            if isinstance(k0, date):
                return AnswerType.DATE, k0.isoformat()
            return AnswerType.DATE, str(k0)
        if op in ('count_ge', 'pct_ge'):
            thr = plan.threshold
            strict = plan.note == 'gt'
            by = plan.group_by
            tm = plan.threshold_metric or met
            if by == 'order':
                x = self.lines(p0, F)
                vals = x.groupby('order_id')['amount'].sum() if tm == 'revenue' else x.groupby('order_id')['qty_base'].sum()
            elif by == 'customer':
                x = self.lines(p0, F)
                x = x[~x['is_walkin']]
                vals = x.groupby('customer_key')['order_id'].nunique() if tm == 'orders' else x.groupby('customer_key')['amount'].sum()
            elif by == 'product':
                x = self.lines(p0, F, products_only=True)
                vals = x.groupby('product_key')['order_id'].nunique() if tm == 'orders' else x.groupby('product_key')['qty_base'].sum()
            else:
                raise PlanError('bad group for count')
            hit = (vals > thr) if strict else (vals >= thr)
            if op == 'count_ge':
                return AnswerType.NUMBER, int(hit.sum())
            if len(vals) == 0:
                raise PlanError('no groups')
            return AnswerType.PERCENT, round_pct(100.0 * hit.sum() / len(vals))
        if op == 'unit_price':
            x = self.lines(p0, F, valid=False, products_only=True)
            if x.empty:
                raise PlanError('no line for product on that day')
            fac = x['factor'].fillna(1.0).replace(0, 1.0)
            per_base = (x['unit_price'] / fac).round(2)
            v = per_base.mode().iloc[0]
            return AnswerType.NUMBER, as_number(v)
        raise PlanError(f'unknown op {op}')

    # ------------------------------------------------------------------ công nợ (rulebook §4)
    _MARKET = r'shopee|lazada|tiki|tiktok|sendo|san|tmdt|marketplace|ecommerce'

    def _debt_base(self) -> pd.DataFrame:
        O = self.O
        if O is None or O.empty or 'order_total' not in O.columns:
            raise Unanswerable('Dữ liệu không ghi tổng tiền phải trả của đơn')
        E = O[O['status'] != 'cancelled'].copy()
        if 'payment_n' in E.columns:
            E = E[E['payment_n'].fillna('') != 'cod']
        if 'channel_f' in E.columns:
            E = E[~E['channel_f'].fillna('').str.contains(self._MARKET, regex=True)]
        E = E[E['date'].notna()]
        return E

    def owed_at(self, X: date) -> pd.DataFrame:
        E = self._debt_base()
        E = E[E['date'] <= X]
        pays = self.c.payments
        if pays is not None and not pays.empty:
            pp = pays[pays['pay_date'].notna() & (pays['pay_date'] <= X)].groupby('order_id')['amount'].sum()
            paid = E['order_id'].map(pp).fillna(0.0)
        elif 'paid' in E.columns:
            paid = E['paid'].fillna(0.0)
        elif 'debt' in E.columns:
            paid = E['order_total'] - E['debt'].fillna(0.0)
        else:
            raise Unanswerable('Dữ liệu không ghi tiền khách đã trả')
        E = E.assign(owed=E['order_total'].fillna(0.0) - paid)
        E = E[E['owed'] > 0.5]
        E['age'] = E['date'].map(lambda d: (X - d).days)
        return E

    def _pays_after_purchase(self) -> pd.DataFrame:
        pays = self.c.payments
        if pays is None or pays.empty:
            raise Unanswerable('Dữ liệu không có phiếu thu theo từng lần trả')
        E = self._debt_base()
        P = pays.merge(E[['order_id', 'date', 'customer_key', 'order_total']], on='order_id', how='inner')
        return P

    def run_debt(self, plan: Plan) -> tuple[AnswerType, Any]:
        op = plan.op
        p0 = plan.periods[0] if plan.periods else None
        X = p0.end if p0 else None
        if op in ('total', 'customers', 'top', 'aging', 'over_days'):
            E = self.owed_at(X)
            if op == 'total':
                return AnswerType.NUMBER, round_money(E['owed'].sum())
            if op == 'over_days':
                thr = plan.threshold or 0
                sel = E[E['age'] > thr] if plan.note != 'ge' else E[E['age'] >= thr]
                return AnswerType.NUMBER, round_money(sel['owed'].sum())
            if op == 'aging':
                b = [E[E['age'].between(0, 7)]['owed'].sum(), E[E['age'].between(8, 15)]['owed'].sum(), E[E['age'] > 15]['owed'].sum()]
                return AnswerType.LIST, [round_money(x) for x in b]
            C = E[E['customer_key'].notna()]
            if op == 'customers':
                return AnswerType.NUMBER, int(C['customer_key'].nunique())
            g = C.groupby('customer_key').agg(value=('owed', 'sum'), display=('customer_display', 'first'))
            g = g.reset_index().rename(columns={'customer_key': 'key'})
            g['code'] = g['key'].astype(str)
            r = self.rank(g, plan.k or 3)
            rows = [[d, round_money(v)] for d, v in zip(r['display'], r['value'])]
            return AnswerType.TABLE, {'columns': ['khach_hang', 'so_tien_no'], 'rows': rows}
        if op in ('collected', 'flow'):
            P = self._pays_after_purchase()
            later = P[P['pay_date'].notna() & (P['pay_date'] > P['date'])]
            later = later[(later['pay_date'] >= p0.start) & (later['pay_date'] <= p0.end)]
            if op == 'collected':
                return AnswerType.NUMBER, round_money(later['amount'].sum())
            ck = plan.filters.get('customer')
            if not ck:
                raise PlanError('need customer')
            E = self._debt_base()
            E = E[(E['customer_key'] == ck) & (E['date'] >= p0.start) & (E['date'] <= p0.end)]
            same = P[P['pay_date'] == P['date']].groupby('order_id')['amount'].sum()
            new_debt = (E['order_total'].fillna(0) - E['order_id'].map(same).fillna(0)).clip(lower=0).sum()
            repaid = later[later['customer_key'] == ck]['amount'].sum()
            return AnswerType.LIST, [round_money(new_debt), round_money(repaid)]
        raise PlanError(f'unknown debt op {op}')

    # ------------------------------------------------------------------ nhập hàng (rulebook §5)
    def _purch(self, tr: Optional[TimeRange]) -> pd.DataFrame:
        Pu = self.c.purchases
        if Pu is None or Pu.empty:
            raise Unanswerable('Dữ liệu không có phiếu nhập hàng')
        x = Pu[~Pu['p_cancelled'].fillna(False).astype(bool)]
        if tr is not None:
            x = x[x['p_date'].notna() & (x['p_date'] >= tr.start) & (x['p_date'] <= tr.end)]
        return x

    def _receipt_amounts(self, x: pd.DataFrame, by: Optional[str] = None) -> pd.Series:
        if 'p_total' in x.columns and x['p_total'].notna().any() and 'receipt_no' in x.columns:
            r = x.drop_duplicates('receipt_no')
            return r.groupby(by)['p_total'].sum() if by else pd.Series([r['p_total'].sum()])
        col = 'p_amount' if 'p_amount' in x.columns else None
        if col is None:
            raise Unanswerable('Dữ liệu không ghi tiền nhập')
        return x.groupby(by)[col].sum() if by else pd.Series([x[col].sum()])

    def run_purchase(self, plan: Plan) -> tuple[AnswerType, Any]:
        p0 = plan.periods[0] if plan.periods else None
        x = self._purch(p0)
        if plan.metric == 'purchase_total':
            return AnswerType.NUMBER, round_money(float(self._receipt_amounts(x).sum()))
        if plan.metric == 'purchase_supplier':
            by = 'supplier' if 'supplier' in x.columns and x['supplier'].notna().any() else 'supplier_code'
            if by not in x.columns:
                raise Unanswerable('Dữ liệu không ghi nhà cung cấp')
            v = self._receipt_amounts(x, by).sort_index()
            if v.empty:
                raise PlanError('no purchases')
            best = v.sort_values(ascending=False, kind='mergesort').index[0]
            name = self.c.suppliers.get(best, best) if hasattr(self.c, 'suppliers') and isinstance(getattr(self.c, 'suppliers'), dict) else best
            return AnswerType.STRING, name
        if plan.metric == 'purchase_avg_price':
            pk = plan.filters.get('product')
            y = x[x['product_key'] == pk]
            if y.empty or 'p_qty' not in y.columns:
                raise PlanError('no purchase lines for product')
            amt = y['p_amount'] if 'p_amount' in y.columns else y['p_qty'] * y['p_unit_cost']
            q = y['p_qty'].sum()
            if q == 0:
                raise PlanError('zero qty')
            return AnswerType.NUMBER, round_money(float(amt.sum() / q))
        raise PlanError(f'unknown purchase metric {plan.metric}')
