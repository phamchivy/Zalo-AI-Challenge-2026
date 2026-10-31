"""Thư viện metric deterministic chạy Plan trên bảng canonical (tầng A). LLM không bao giờ tự tính số.

v1.1: KHÔNG viết cứng luật nghiệp vụ. Mọi quy ước (trạng thái tính doanh thu, làm tròn, mẫu số tỉ lệ huỷ, tiêu chí bán chạy,
công nợ, nhập hàng...) đọc từ RuleSet (knowledge/rules.py). Mỗi Plan khai báo các nút luật + năng lực dữ liệu nó cần
(`requirements`); thiếu nút hoặc nút mang giá trị chưa cài -> RuleMissing -> strategy bỏ trống.
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
from datamate.knowledge.rules import RuleSet


class PlanError(Exception):
    """Plan không chạy được trên dữ liệu này (thiếu cột, thực thể không có...)."""


class Unanswerable(Exception):
    """Dữ liệu không có loại thông tin câu hỏi cần."""


class RuleMissing(PlanError):
    """Rulebook của split không xác nhận luật mà plan cần (hoặc dữ liệu thiếu năng lực) -> bỏ trống, không đoán."""


def round_money_rule(x: float, mode: str) -> int:
    if mode == 'half_up':
        return round_money(x)
    if mode == 'half_even':
        return int(round(x))
    if mode == 'floor':
        return int(math.floor(x))
    raise RuleMissing(f'aov.rounding={mode!r} chưa cài')


# metric -> nút luật cần có
METRIC_RULES = {
    'revenue': ['revenue.valid_status', 'revenue.order_def', 'revenue.partial_return_whole_order', 'revenue.period_by'],
    'orders': ['revenue.valid_status', 'orders.valid_only', 'revenue.period_by'],
    'aov': ['revenue.valid_status', 'revenue.order_def', 'orders.valid_only', 'aov.rounding'],
    'customers': ['revenue.valid_status', 'customer.identity'],
    'qty': ['revenue.valid_status', 'revenue.partial_return_whole_order'],
    'gross_profit': ['revenue.valid_status', 'revenue.order_def', 'cogs.method'],
    'cancel_rate': ['cancel.def', 'cancel_rate.denominator'],
    'cancelled_orders': ['cancel.def'],
    'active_days': ['revenue.valid_status', 'orders.valid_only'],
    'lines': ['revenue.valid_status'],
}
METRIC_CAPS = {'orders': ['orders'], 'aov': ['orders'], 'customers': ['customers'], 'gross_profit': ['cost_line'],
               'cancel_rate': ['orders', 'status'], 'cancelled_orders': ['orders', 'status'], 'active_days': ['orders']}
OP_RULES = {'pct_change': ['pct_change.base'], 'pct_change_filters': ['pct_change.base'], 'diff': ['diff.order'],
            'count_ge': ['threshold.strict'], 'pct_ge': ['threshold.strict'], 'unit_price': ['unit_price.def'],
            'weekday_avg': ['weekday_avg.denominator']}
FILTER_RULES = {'payment': ['payment.by_order_method'], 'shift': ['time.shift'], 'size': ['size.suffix'],
                'rx': ['rx.def'], 'ingredient': ['ingredient.exact'], 'base_unit': ['qty_by_unit'],
                'customer_group': ['customer_group.walkin_none'], 'category': ['category.exact']}
SPECIAL_RULES = {
    'debt': ['debt.include_returned', 'debt.order_owed', 'debt.exclude_cod_marketplace'],
    'purchase_total': ['purchase.total'], 'purchase_avg_price': ['purchase.avg_price'],
    'purchase_minus_sales': ['purchase.minus_sales', 'revenue.valid_status'],
    'price_effect': ['price_effect.def', 'revenue.valid_status'], 'dead_stock': ['dead_stock.def', 'revenue.valid_status'],
    'delivery_count': ['delivery.count'], 'inventory_value': ['inventory.value', 'time.snapshot'],
    'stock_qty': ['time.snapshot'], 'lot_nearest': ['lot.nearest'], 'expiry': ['expiry.window'],
}
SPECIAL_CAPS = {'debt': ['debt'], 'purchase_total': ['purchases'], 'purchase_avg_price': ['purchases'],
                'purchase_supplier': ['purchases'], 'purchase_minus_sales': ['purchases'], 'price_effect': [],
                'delivery_count': ['deliveries'], 'inventory_value': ['stock', 'cost_catalog'], 'stock_qty': ['stock'],
                'lot_nearest': ['lots'], 'expiry': ['lots']}


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
    def __init__(self, canon, params: Optional[RuleParams] = None, rules: Optional[RuleSet] = None):
        self.c = canon
        self.p = params or RuleParams()
        self.r = rules if rules is not None else RuleSet()
        L = canon.lines
        self.L = L
        self.O = canon.orders
        self._has = set(L.columns) if L is not None and not L.empty else set()

    # ------------------------------------------------------------------ luật & năng lực
    def requirements(self, plan: Plan) -> tuple[list[str], list[str]]:
        met, op, F = plan.metric, plan.op, plan.filters or {}
        knobs: list[str] = []
        caps: list[str] = []
        if met in SPECIAL_RULES or met in SPECIAL_CAPS or met.startswith('purchase'):
            knobs += SPECIAL_RULES.get(met, [])
            caps += SPECIAL_CAPS.get(met, [])
            if met == 'debt':
                knobs += {'aging': ['debt.age_from'], 'over_days': ['debt.age_from'], 'collected': ['debt.flow'],
                          'flow': ['debt.flow'], 'top': ['debt.by_customer', 'customer.identity'],
                          'customers': ['debt.by_customer', 'customer.identity']}.get(op, [])
            if met == 'purchase_supplier':
                knobs += ['purchase.top_supplier_by'] if plan.note in ('', 'purchase_amount') else []
            if met == 'expiry' and op == 'capital':
                knobs += ['expiry.capital']
        else:
            knobs += METRIC_RULES.get(met, [])
            caps += METRIC_CAPS.get(met, [])
            if plan.threshold_metric and plan.threshold_metric in METRIC_RULES:
                knobs += METRIC_RULES[plan.threshold_metric]
        knobs += OP_RULES.get(op, [])
        for f in F:
            knobs += FILTER_RULES.get(f, [])
        if F.get('customer_group') and op == 'share':
            knobs.append('share.group_denominator')
        gb = plan.group_by
        if gb == 'product' or 'product' in F or op == 'top' and gb == 'product':
            knobs.append('product.service_fee_excluded')
        if gb == 'customer' or 'customer' in F:
            knobs.append('customer.identity')
            caps.append('customers')
        if gb == 'hour' or 'shift' in F:
            caps.append('time')
        if gb == 'staff' or 'staff' in F:
            caps.append('staff')
        if op == 'top' and gb == 'product' and met == 'qty':
            knobs.append('best_seller.by')
        if op == 'top' and gb == 'product' and met == 'gross_profit':
            knobs.append('top_profit.by')
        if op == 'top' and gb == 'customer' and met == 'revenue':
            knobs.append('customer.top_money_by')
        if F.get('payment') == 'cash':
            knobs.append('payment.cod_separate')
        return list(dict.fromkeys(knobs)), list(dict.fromkeys(caps))

    def check(self, plan: Plan) -> None:
        knobs, caps = self.requirements(plan)
        miss = self.r.missing(knobs)
        if miss:
            raise RuleMissing('rulebook không xác nhận: ' + ', '.join(miss))
        lack = [c for c in caps if c not in self.c.caps]
        if lack:
            raise RuleMissing('dữ liệu thiếu năng lực: ' + ', '.join(lack))
        unc = getattr(self.c, 'health', {}).get('uncertain_roles', [])
        need_roles = {'revenue': 'amount', 'aov': 'amount', 'gross_profit': 'amount', 'qty': 'qty', 'orders': 'order_id'}
        if need_roles.get(plan.metric) in unc:
            raise RuleMissing(f'ánh xạ cột không chắc: {need_roles[plan.metric]}')

    @property
    def VALID(self) -> tuple:
        v = self.r.get('revenue.valid_status')
        if not v:
            raise RuleMissing('revenue.valid_status')
        return tuple(v)

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
                if v not in ('morning', 'afternoon'):
                    raise RuleMissing(f'rulebook không định nghĩa ca {v!r}')       # không tự đặt giờ ca tối
                a, b = {'morning': self.p.morning, 'afternoon': self.p.afternoon}[v]
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
            m &= L['status'].isin(self.VALID)
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
            if self.r.get('cogs.method') != 'line_cost':
                raise RuleMissing('cogs.method')
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
            den = self.r.get('cancel_rate.denominator')
            if den == 'valid_orders':
                st = st[st.isin(self.VALID) | (st == 'cancelled')]
            elif den != 'all_orders':
                raise RuleMissing(f'cancel_rate.denominator={den!r}')
            if len(st) == 0:
                raise PlanError('no orders in period')
            return 100.0 * canc / len(st)
        if name == 'active_days':
            return float(self.lines(tr, filters)['date'].nunique())
        if name == 'lines':
            return float(len(self.lines(tr, filters)))
        raise PlanError(f'unknown metric {name}')

    def fmt_metric(self, name: str, v: float) -> tuple[AnswerType, Any]:
        if name in ('cancel_rate',):
            return AnswerType.PERCENT, round_pct(v)
        if name == 'aov':
            return AnswerType.NUMBER, round_money_rule(v, self.r.get('aov.rounding') or 'half_up')
        if name in ('revenue', 'gross_profit'):
            return AnswerType.NUMBER, round_money(v)
        return AnswerType.NUMBER, as_number(v)

    # ------------------------------------------------------------------ grouping
    def group_metric(self, name: str, tr: Optional[TimeRange], filters: dict, by: str) -> pd.DataFrame:
        """-> DataFrame[key, display, code, value] cho xếp hạng."""
        if name in ('qty',):
            x = self.lines(tr, filters, products_only=True)
        elif name == 'cancelled_orders':
            x = self.lines(tr, filters, valid=False)
            x = x[x['status'] == 'cancelled']
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
        elif by == 'hour':
            self._need('time', 'giờ tạo đơn')
            x = x.assign(hour=x['time'].map(lambda t: int(str(t)[:2]) if t and str(t)[:2].isdigit() else None)).dropna(subset=['hour'])
            key, disp, code = 'hour', 'hour', 'hour'
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
        elif name == 'active_days':
            val = g['date'].nunique()
        elif name == 'cancelled_orders':
            val = g['order_id'].nunique()
        else:
            raise PlanError(f'cannot group metric {name}')
        dsp = g[disp].agg(lambda s: s.value_counts().index[0] if s.notna().any() else None)
        cd = g[code].first()
        out = pd.DataFrame({'key': val.index, 'value': val.values})
        out['display'] = out['key'].map(dsp)
        out['code'] = out['key'].map(cd).map(lambda v: str(v) if v is not None else '')
        out['n_orders'] = out['key'].map(g['order_id'].nunique())
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
        self.check(plan)
        op, met, P, F = plan.op, plan.metric, plan.periods, plan.filters or {}
        p0 = P[0] if P else None
        if met == 'debt':
            return self.run_debt(plan)
        if met.startswith('purchase'):
            return self.run_purchase(plan)
        if met in SPECIAL_RULES:
            return getattr(self, f'run_{met}')(plan)
        if op == 'weekday_avg':
            wd = int(plan.threshold)
            days = pd.date_range(p0.start, p0.end).map(lambda d: d.isoweekday())
            n = int((days == wd).sum())
            if n == 0:
                raise PlanError('no such weekday in period')
            x = self.lines(p0, F, products_only=(met == 'qty'))
            x = x[x['date'].map(lambda d: d.isoweekday() if d else None) == wd]
            tot = {'revenue': lambda: x['amount'].sum(), 'orders': lambda: x['order_id'].nunique(),
                   'qty': lambda: x['qty_base'].sum()}[met]()
            v = tot / n
            return AnswerType.NUMBER, round_money(v) if met == 'revenue' else float(round(v, 2))
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
            if plan.threshold_metric == 'min_orders' and plan.threshold:
                df = df[df['n_orders'] >= plan.threshold]
            if plan.group_by in ('day', 'hour'):
                df = df.assign(code=df['key'].astype(str))
            r = self.rank(df, k, ascending=(op == 'bottom'))
            if r.empty:
                raise PlanError('empty ranking')
            if plan.group_by in ('customer', 'staff', 'product', 'category'):
                names = [d for d in r['display'].tolist()]
            elif plan.group_by == 'day':
                names = [d.isoformat() for d in r['key'].tolist()]
                if k == 1:
                    return AnswerType.DATE, names[0]
            elif plan.group_by == 'hour':
                names = [int(h) for h in r['key'].tolist()]
                if k == 1:
                    return AnswerType.NUMBER, names[0]
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
            elif by == 'day':
                x = self.lines(p0, F)
                vals = x.groupby('date')['amount'].sum() if tm == 'revenue' else x.groupby('date')['order_id'].nunique()
            else:
                raise PlanError('bad group for count')
            hit = (vals > thr) if strict else (vals >= thr)
            if op == 'count_ge':
                return AnswerType.NUMBER, int(hit.sum())
            if len(vals) == 0:
                raise PlanError('no groups')
            return AnswerType.PERCENT, round_pct(100.0 * hit.sum() / len(vals))
        if op == 'unit_price':
            if self.r.get('unit_price.def') != 'line_base_before_disc_all':
                raise RuleMissing('unit_price.def')
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
        if not self.r.get('debt.include_returned'):
            E = E[E['status'] != 'returned']
        if self.r.get('debt.exclude_cod_marketplace'):
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
        if (pays is None or pays.empty) and (getattr(self.c, 'health', None) or {}).get('payments_unmapped'):
            # shop có sổ thu nhưng không đọc được cột -> cột "đã trả" trên đơn là số cuối kỳ, không dùng cho ngày X
            raise RuleMissing('payments ledger present but unmapped')
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
                if not plan.split_values:
                    raise PlanError('aging buckets not given')
                b = [E[(E['age'] >= lo) & (E['age'] <= hi)]['owed'].sum() for lo, hi in plan.split_values]
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
        if plan.metric == 'purchase_minus_sales':
            if self.r.get('purchase.minus_sales') != 'valid_sales_same_unit':
                raise RuleMissing('purchase.minus_sales')
            pk = plan.filters.get('product')
            y = x[x['product_key'] == pk]
            if 'p_qty' not in y.columns:
                raise Unanswerable('Dữ liệu không ghi số lượng nhập')
            units = y['p_unit'] if 'p_unit' in y.columns else pd.Series([None] * len(y), index=y.index)
            inb = float(sum(q * self._factor(pk, u) for q, u in zip(y['p_qty'].fillna(0), units)))
            sold = self._sold_base(pk, p0.start, p0.end)
            f = self._factor(pk, plan.note or None)
            return AnswerType.NUMBER, as_number((inb - sold) / f)
        if plan.metric == 'purchase_total':
            return AnswerType.NUMBER, round_money(float(self._receipt_amounts(x).sum()))
        if plan.metric == 'purchase_supplier':
            by = 'supplier' if 'supplier' in x.columns and x['supplier'].notna().any() else 'supplier_code'
            if by not in x.columns:
                raise Unanswerable('Dữ liệu không ghi nhà cung cấp')
            how = plan.note or 'purchase_amount'
            if how == 'receipts':
                if 'receipt_no' not in x.columns:
                    raise Unanswerable('Dữ liệu không ghi số phiếu nhập')
                v = x.drop_duplicates('receipt_no').groupby(by)['receipt_no'].nunique().sort_index()
            elif how == 'purchase_amount':
                if self.r.get('purchase.top_supplier_by') != 'amount':
                    raise RuleMissing('purchase.top_supplier_by')
                v = self._receipt_amounts(x, by).sort_index()
            else:
                raise RuleMissing(f'supplier ranking {how!r}')
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

    # ------------------------------------------------------------------ khối tính chung v1.1: đơn vị
    def _base_unit_of(self, pk: str) -> Optional[str]:
        P = self.c.products
        if P is not None and not P.empty and 'base_unit' in P.columns:
            v = P.loc[P['product_key'] == pk, 'base_unit']
            if len(v) and v.iloc[0]:
                return fold(v.iloc[0])
        U = self.c.units
        if U is not None and not U.empty:
            v = U[(U['product_key'] == pk) & (U['factor'] == 1.0)]['unit_f']
            if len(v):
                return v.iloc[0]
        if 'base_unit_f' in self._has:
            v = self.L.loc[self.L['product_key'] == pk, 'base_unit_f'].dropna()
            if len(v):
                return v.mode().iloc[0]
        return None

    def _factor(self, pk: str, unit: Optional[str]) -> float:
        """Hệ số đổi 1 <unit> ra đơn vị cơ bản. unit None/đơn vị cơ bản -> 1."""
        if not unit:
            return 1.0
        u = fold(unit)
        if u == self._base_unit_of(pk):
            return 1.0
        U = self.c.units
        if U is not None and not U.empty:
            v = U[(U['product_key'] == pk) & (U['unit_f'] == u)]['factor']
            if len(v) and v.iloc[0] == v.iloc[0]:
                return float(v.iloc[0])
        raise PlanError(f'không có hệ số quy đổi {unit} cho {pk}')

    def _sold_base(self, pk: str, start: date, end: date) -> float:
        x = self.lines(TimeRange(start=start, end=end), {'product': pk}, products_only=True)
        return float(x['qty_base'].sum())

    # ------------------------------------------------------------------ hiệu ứng tăng giá
    def run_price_effect(self, plan: Plan) -> tuple[AnswerType, Any]:
        if self.r.get('price_effect.def') != 'n_from_d_vs_n_before':
            raise RuleMissing('price_effect.def')
        pk = plan.filters.get('product')
        n = int(plan.threshold or 0)
        if not pk or n <= 0:
            raise PlanError('need product and N')
        D = plan.periods[0].start if plan.periods else None
        if D is None:
            pr = getattr(self.c, 'prices', None)
            if pr is None or pr.empty:
                raise Unanswerable('Dữ liệu không có lịch sử giá')
            h = pr[pr['product_key'] == pk].sort_values('eff_date')
            ups = h[h['price'] > h['price'].shift(1)]
            target = plan.split_values[0] if plan.split_values else None
            if target is not None:
                ups = ups[(ups['price'] - target).abs() < 0.5]
            if ups.empty:
                raise PlanError('không tìm thấy lần tăng giá')
            D = ups['eff_date'].iloc[0]         # tăng lên cùng một giá nhiều lần -> lần đầu
        from datetime import timedelta as _td
        a = self._sold_base(pk, D, D + _td(days=n - 1))
        b = self._sold_base(pk, D - _td(days=n), D - _td(days=1))
        if b == 0:
            raise PlanError('base is zero')
        return AnswerType.PERCENT, round_pct(100.0 * (a - b) / b)

    # ------------------------------------------------------------------ hàng ế
    def run_dead_stock(self, plan: Plan) -> tuple[AnswerType, Any]:
        p0 = plan.periods[0]
        L = self.L
        base = L[L['status'].isin(self.VALID) & ~L['is_nonproduct'] & ~L['is_service'] & L['date'].notna()]
        before = set(base.loc[base['date'] < p0.start, 'product_key'].dropna())
        inside = set(base.loc[(base['date'] >= p0.start) & (base['date'] <= p0.end), 'product_key'].dropna())
        dead = sorted(before - inside)
        if plan.op == 'count':
            return AnswerType.NUMBER, len(dead)
        info = base.drop_duplicates('product_key').set_index('product_key')
        code = info['product_code'] if 'product_code' in info.columns else pd.Series(dtype=object)
        dead.sort(key=lambda k: (str(code.get(k) or k)))
        return AnswerType.LIST, [info.at[k, 'product_name'] for k in dead]

    # ------------------------------------------------------------------ đợt giao
    def run_delivery_count(self, plan: Plan) -> tuple[AnswerType, Any]:
        Dl = getattr(self.c, 'deliveries', None)
        if Dl is None or Dl.empty:
            raise Unanswerable('Dữ liệu không có sổ giao hàng')
        p0 = plan.periods[0]
        x = Dl[Dl['deliv_date'].notna() & (Dl['deliv_date'] >= p0.start) & (Dl['deliv_date'] <= p0.end)]
        keys = [c for c in ('order_id', 'deliv_no') if c in x.columns and x[c].notna().any()]
        return AnswerType.NUMBER, int(len(x.drop_duplicates(keys)) if len(keys) == 2 else len(x))

    # ------------------------------------------------------------------ tồn kho theo thời điểm
    def _stock_at(self, X: date) -> pd.Series:
        S = getattr(self.c, 'stock', None)
        if S is None or S.empty:
            raise Unanswerable('Dữ liệu không có thẻ kho')
        end = pd.Timestamp(X) + pd.Timedelta(hours=23, minutes=59, seconds=59)
        x = S[S['ts'] <= end].sort_values(['product_key', 'ts', 'seq'], kind='mergesort')
        return x.groupby('product_key')['balance'].last()

    def run_inventory_value(self, plan: Plan) -> tuple[AnswerType, Any]:
        if self.r.get('inventory.value') != 'stock_x_catalog_cost_base':
            raise RuleMissing('inventory.value')
        st = self._stock_at(plan.periods[0].end)
        st = st[st > 0]
        P = self.c.products
        if P is None or P.empty or 'cost' not in P.columns:
            raise Unanswerable('Danh mục không ghi giá vốn')
        cost = P.set_index('product_key')['cost']
        miss = [k for k in st.index if pd.isna(cost.get(k))]
        if miss:
            raise PlanError(f'thiếu giá vốn danh mục cho {len(miss)} mặt hàng')
        return AnswerType.NUMBER, round_money(float(sum(st[k] * cost[k] for k in st.index)))

    def run_stock_qty(self, plan: Plan) -> tuple[AnswerType, Any]:
        pk = plan.filters.get('product')
        st = self._stock_at(plan.periods[0].end)
        q = float(st.get(pk, 0.0))
        note = plan.note or ''
        if note.startswith('pair'):
            if self.r.get('stock.big_unit_format') != 'pair':
                raise RuleMissing('stock.big_unit_format')
            big = note.split(':', 1)[1] if ':' in note else ''
            f = self._factor(pk, big)
            if f <= 1:
                raise PlanError('need big unit')
            return AnswerType.LIST, [int(q // f), as_number(q - (q // f) * f)]
        f = self._factor(pk, note or None)
        return AnswerType.NUMBER, as_number(q / f)

    # ------------------------------------------------------------------ lô & hạn dùng
    def _lots_at(self, X: date) -> pd.DataFrame:
        Lt = getattr(self.c, 'lots', None)
        if Lt is None or Lt.empty:
            raise Unanswerable('Dữ liệu không có lô hàng')
        lt = Lt[Lt['recv_date'].isna() | (Lt['recv_date'] <= X)].copy()
        LS = getattr(self.c, 'lot_sales', None)
        out = pd.Series(0.0, index=lt['lot'])
        if LS is not None and not LS.empty:
            O = self.O[['order_id', 'date', 'status']] if self.O is not None and not self.O.empty else None
            s = LS.merge(O, on='order_id', how='left') if O is not None else LS.assign(date=None, status=None)
            s = s[s['date'].notna() & (s['date'] <= X) & (s['status'] != 'cancelled')]
            used = (s['qty_base'].fillna(0) - s['return_qty'].fillna(0)).groupby(s['lot']).sum()
            out = out.add(used, fill_value=0.0).reindex(lt['lot']).fillna(0.0)
        lt['remain'] = lt['received_base'].values - out.values
        return lt

    def run_lot_nearest(self, plan: Plan) -> tuple[AnswerType, Any]:
        X = plan.periods[0].end
        lt = self._lots_at(X)
        x = lt[(lt['product_key'] == plan.filters.get('product')) & (lt['remain'] > 0) & lt['expiry'].notna() & (lt['expiry'] >= X)]
        if x.empty:
            raise PlanError('không còn lô tồn chưa hết hạn')
        return AnswerType.DATE, x['expiry'].min().isoformat()

    def run_expiry(self, plan: Plan) -> tuple[AnswerType, Any]:
        X = plan.periods[0].end
        w = plan.periods[1]
        lt = self._lots_at(X)
        x = lt[(lt['remain'] > 0) & lt['expiry'].notna() & (lt['expiry'] >= w.start) & (lt['expiry'] <= w.end)]
        if plan.op == 'capital':
            if self.r.get('expiry.capital') != 'lot_cost_base':
                raise RuleMissing('expiry.capital')
            if x['unit_cost_base'].isna().any():
                raise PlanError('thiếu giá nhập lô')
            return AnswerType.NUMBER, round_money(float((x['remain'] * x['unit_cost_base']).sum()))
        if plan.op == 'count':
            return AnswerType.NUMBER, int(x['product_key'].nunique())
        first = x.groupby('product_key')['expiry'].min()
        P = self.c.products.set_index('product_key') if self.c.products is not None and not self.c.products.empty else None
        def code(k):
            return str(P.at[k, 'product_code']) if P is not None and k in P.index and P.at[k, 'product_code'] else str(k)
        def name(k):
            return P.at[k, 'product_name'] if P is not None and k in P.index else k
        keys = sorted(first.index, key=lambda k: (first[k], code(k)))
        return AnswerType.LIST, [name(k) for k in keys]
