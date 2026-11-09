"""v2.4.0 — Invariants: kiểm tra canonical KHÔNG cần đáp án (chạy được trên mọi shop: train, public, private).

Mỗi invariant trả về: n (số phần tử kiểm được), bad (số vi phạm), rate, level (green/yellow/red/na) và tối đa 3 ví dụ.
Chỉ chẩn đoán; không thay đổi dữ liệu hay đáp án. Dùng để:
  - tìm lỗi dựng bảng (cột tiền map nhầm, giảm cấp đơn bị trừ vào dòng, trạng thái lạ chưa map, thang tiền ×1000…)
  - sau này: chọn mapping (regex vs LLM) và tính điểm rủi ro shop.

Tên trường theo canon hiện tại (canonical.py); bảng nối với canonical v2 (02_du_lieu.md):
  lines.amount = order_lines.line_total · orders.goods_total = total_goods · orders.order_total = total_payable
  orders.paid = paid_amount · orders.ship_fee = delivery_fee · lines.is_nonproduct ~ line_kind != product
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from datamate.data.vocab import norm_status

# ngưỡng tỉ lệ vi phạm: (vàng, đỏ). Ngưỡng thấp cho kiểm số học (dữ liệu sinh có quy luật chặt).
LEVELS = {
    'I1_line_arith': (0.03, 0.08),       # v2.4.1: giảm nhỏ không ghi cột (~1–2,5%) là dữ liệu thật
    'I2_goods_total': (0.01, 0.05),
    'I3_payable': (0.01, 0.05),
    'I6_one_status': (0.001, 0.01),
    'I15_status_mapped': (0.001, 0.01),
    'I15_payment_mapped': (0.01, 0.05),
    'I9_customer_join': (0.02, 0.10),
    'I12_purchase_total': (0.01, 0.05),
    'MONEY_scale': (0.001, 0.005),
    'DATE_parsed': (0.005, 0.02),
    'DATE_range': (0.005, 0.02),
    'STOCK_card': (0.01, 0.05),
    'I5_unit_factor': (0.02, 0.10),
}
PAYMENT_KEYS = {'cash', 'transfer', 'ewallet', 'cod', 'debt', 'card'}


@dataclass
class InvResult:
    name: str
    n: int = 0
    bad: int = 0
    level: str = 'na'
    note: str = ''
    examples: list = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.bad / self.n if self.n else 0.0

    def finish(self) -> 'InvResult':
        if self.n == 0:
            self.level = 'na'
            return self
        y, r = LEVELS.get(self.name, (0.01, 0.05))
        self.level = 'red' if self.rate > r else 'yellow' if self.rate > y else 'green'
        return self

    def to_dict(self) -> dict:
        return {'name': self.name, 'n': self.n, 'bad': self.bad, 'rate': round(self.rate, 5), 'level': self.level,
                'note': self.note, 'examples': self.examples}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors='coerce')


def _close(a: pd.Series, b: pd.Series, rel: float = 0.005, abs_: float = 1.0) -> pd.Series:
    return (a - b).abs() <= np.maximum(abs_, rel * b.abs())


def _ex(df: pd.DataFrame, mask: pd.Series, cols: list[str], k: int = 3) -> list:
    cols = [c for c in cols if c in df.columns]
    sub = df.loc[mask, cols].head(k)
    return [{c: (None if pd.isna(v) else (v.isoformat() if hasattr(v, 'isoformat') else (float(v) if isinstance(v, (np.floating, np.integer)) else str(v))))
             for c, v in r.items()} for r in sub.to_dict('records')]


# ----------------------------------------------------------------------------- invariants
def inv_line_arith(L: pd.DataFrame) -> InvResult:
    """I1: thành tiền dòng = SL × đơn giá − giảm dòng. Báo kèm tỉ lệ khớp nếu KHÔNG trừ giảm dòng (gợi ý giảm cấp đơn)."""
    r = InvResult('I1_line_arith')
    need = {'qty', 'unit_price', 'amount'}
    if not need <= set(L.columns):
        r.note = 'thiếu qty/unit_price/amount'
        return r.finish()
    q, p, a = _num(L['qty']), _num(L['unit_price']), _num(L['amount'])
    d = _num(L['line_discount']).fillna(0) if 'line_discount' in L.columns else pd.Series(0.0, index=L.index)
    m = q.notna() & p.notna() & a.notna() & (p != 0)
    if 'is_nonproduct' in L.columns:
        m &= ~L['is_nonproduct'].fillna(False).astype(bool)
    net_ok = _close(a, q * p - d)
    gross_ok = _close(a, q * p)
    # đơn giá có thể theo đơn vị cơ bản còn SL theo đơn vị dòng (hoặc ngược lại): chấp nhận khớp qua hệ số quy đổi
    if 'factor' in L.columns:
        f = _num(L['factor']).fillna(1)
        net_ok |= _close(a, q * f * p - d) | _close(a, q * p / f.replace(0, np.nan) - d)
    bad = m & ~net_ok
    over = bad & (a > q * p + np.maximum(1, 0.005 * a.abs()))      # thành tiền LỚN hơn SL × giá: gộp thêm thứ khác vào dòng
    r.n, r.bad = int(m.sum()), int(bad.sum())
    if r.n:
        r.note = (f'khớp không trừ giảm dòng: {float((m & gross_ok).sum()) / r.n:.3f}; '
                  f'thành tiền > SL×giá: {int(over.sum())} dòng')
    show = over if over.any() else bad
    r.examples = _ex(L, show, ['order_id', 'product_name', 'qty', 'unit', 'unit_price', 'line_discount', 'amount'])
    r.finish()
    # v2.4.1: chiều 'lớn hơn' (topping/phụ phí gộp ô, đơn giá sai cột) là dấu hiệu lỗi dựng bảng dù tỉ lệ thấp
    if r.n and over.sum() / r.n > 0.005:
        r.level = 'red' if over.sum() / r.n > 0.01 else max(r.level, 'yellow', key=['na', 'green', 'yellow', 'red'].index)
    return r


def inv_goods_total(L: pd.DataFrame, O: pd.DataFrame) -> InvResult:
    """I2: Σ thành tiền các dòng của đơn = tổng tiền hàng ghi trên đơn."""
    r = InvResult('I2_goods_total')
    if 'goods_total' not in O.columns or 'order_id' not in L.columns or 'amount' not in L.columns:
        r.note = 'không có cột tổng tiền hàng của đơn'
        return r.finish()
    s = L.groupby('order_id', sort=False)['amount'].apply(lambda x: _num(x).sum(min_count=1))
    g = O.set_index('order_id')['goods_total'].pipe(_num)
    j = pd.DataFrame({'lines': s}).join(g.rename('goods'), how='inner').dropna()
    ok = _close(j['lines'], j['goods'])
    r.n, r.bad = len(j), int((~ok).sum())
    if 'order_discount' in O.columns and r.bad:
        od = O.set_index('order_id')['order_discount'].pipe(_num).reindex(j.index).fillna(0)
        r.note = (f'Σ dòng + giảm đơn = tiền hàng (giảm cả đơn bị trừ vào dòng): {float(_close(j["lines"] + od, j["goods"]).mean()):.3f}; '
                  f'Σ dòng = tiền hàng − giảm đơn: {float(_close(j["lines"], j["goods"] - od).mean()):.3f}')
    jj = j.reset_index().rename(columns={'index': 'order_id'})
    r.examples = _ex(jj, ~ok.values, ['order_id', 'lines', 'goods'])
    return r.finish()


def inv_payable(O: pd.DataFrame) -> InvResult:
    """I3: khách phải trả = tiền hàng − giảm đơn + phí giao (+ VAT nếu có)."""
    r = InvResult('I3_payable')
    if 'order_total' not in O.columns or 'goods_total' not in O.columns:
        r.note = 'không có cột khách phải trả / tiền hàng'
        return r.finish()
    g, t = _num(O['goods_total']), _num(O['order_total'])
    od = _num(O['order_discount']).fillna(0) if 'order_discount' in O.columns else 0
    fee = _num(O['ship_fee']).fillna(0) if 'ship_fee' in O.columns else 0
    vat = _num(O['vat']).fillna(0) if 'vat' in O.columns else 0
    m = g.notna() & t.notna()
    exp = g - od + fee
    ok = _close(t, exp) | _close(t, exp + vat)
    bad = m & ~ok
    r.n, r.bad = int(m.sum()), int(bad.sum())
    r.examples = _ex(O, bad, ['order_id', 'goods_total', 'order_discount', 'ship_fee', 'vat', 'order_total'])
    r.finish()
    # v2.4.1: lệch là vài hằng số (phí giao 15k, voucher 10k… không có cột riêng) -> khoản chưa gán, không phải map sai
    if r.bad:
        off = (t - exp)[bad].round(-2).value_counts()
        top = off.head(2)
        if top.sum() >= 0.9 * r.bad:
            r.level = 'yellow'
            r.note = 'lệch hằng số: ' + ', '.join(f'{v:+,.0f}đ ({c} đơn)' for v, c in top.items()) + \
                     ' -> phí giao / voucher chưa có cột hoặc chưa gán'
    return r


def inv_one_status(L: pd.DataFrame) -> InvResult:
    """I6: mỗi đơn một trạng thái (dòng của cùng đơn không mâu thuẫn)."""
    r = InvResult('I6_one_status')
    if 'status' not in L.columns or 'order_id' not in L.columns:
        r.note = 'không có trạng thái'
        return r.finish()
    nu = L.dropna(subset=['order_id']).groupby('order_id', sort=False)['status'].nunique()
    bad_ids = set(nu[nu > 1].index)
    r.n, r.bad = len(nu), len(bad_ids)
    r.examples = _ex(L, L['order_id'].isin(list(bad_ids)[:3]), ['order_id', 'status_raw', 'status'], k=6)
    return r.finish()


def inv_status_mapped(L: pd.DataFrame) -> InvResult:
    """I15: mọi nhãn trạng thái gốc đều gán được vào {completed, cancelled, returned, shipping}.
    canonical hiện coi nhãn lạ là 'completed' (âm thầm) -> invariant này lộ ra các nhãn đó."""
    r = InvResult('I15_status_mapped')
    if 'status_raw' not in L.columns:
        # v2.4.1: trạng thái lấy từ Ghi chú -> ghi chú có từ khoá trạng thái mà không gán được
        if 'note' not in L.columns:
            r.note = 'không có cột trạng thái gốc'
            return r.finish()
        from datamate.data.vocab import status_from_note
        from datamate.core.textutil import fold
        notes = L['note'].dropna().astype(str)
        notes = notes[notes.str.strip() != '']
        vals = notes.value_counts()
        kw = r'\b(huy|tra hang|tra lai|khach tra|hoan hang|hoan tra|doi tra|dang giao|chua giao|dang ship|cancel\w*|return\w*|bom hang|refund\w*)\b'
        unk = {v: c for v, c in vals.items() if re.search(kw, fold(v) or '') and status_from_note(v) is None}
        r.n, r.bad = int(vals.sum()), int(sum(unk.values()))
        r.note = f'trạng thái từ Ghi chú: {len(vals)} ghi chú khác nhau'
        r.examples = [{'ghi_chu': k, 'so_dong': int(c)} for k, c in sorted(unk.items(), key=lambda x: -x[1])[:5]]
        return r.finish()
    raw = L['status_raw'].dropna().astype(str)
    raw = raw[raw.str.strip() != '']
    if raw.empty:
        return r.finish()
    vals = raw.value_counts()
    unk = {v: c for v, c in vals.items() if norm_status(v) in (None, 'unknown')}
    r.n, r.bad = int(vals.sum()), int(sum(unk.values()))
    r.note = f'{len(vals)} nhãn khác nhau'
    r.examples = [{'nhan': k, 'so_dong': int(c)} for k, c in sorted(unk.items(), key=lambda x: -x[1])[:5]]
    return r.finish()


def inv_payment_mapped(O: pd.DataFrame) -> InvResult:
    """I15b: hình thức thanh toán gán được vào tập chuẩn (cash/transfer/ewallet/cod/debt/card)."""
    r = InvResult('I15_payment_mapped')
    col = 'payment_n' if 'payment_n' in O.columns else None
    if col is None or 'payment' not in O.columns:
        r.note = 'không có hình thức thanh toán'
        return r.finish()
    m = O['payment'].notna() & (O['payment'].astype(str).str.strip() != '')
    bad = m & ~O[col].isin(PAYMENT_KEYS)
    r.n, r.bad = int(m.sum()), int(bad.sum())
    vc = O.loc[bad, 'payment'].astype(str).value_counts().head(5)
    r.examples = [{'nhan': k, 'so_don': int(v)} for k, v in vc.items()]
    return r.finish()


def inv_customer_join(O: pd.DataFrame, C: pd.DataFrame) -> InvResult:
    """I9: đơn có mã khách thì mã có trong bảng khách; khách vãng lai không mang nhóm khách."""
    r = InvResult('I9_customer_join')
    if 'customer_code' not in O.columns or C is None or C.empty or 'customer_code' not in C.columns:
        r.note = 'không có mã khách hoặc bảng khách'
        return r.finish()
    codes = set(C['customer_code'].dropna().astype(str).str.strip())
    oc = O['customer_code'].dropna().astype(str).str.strip()
    oc = oc[oc != '']
    walk = O['is_walkin'].fillna(False).astype(bool) if 'is_walkin' in O.columns else pd.Series(False, index=O.index)
    miss = oc[~oc.isin(codes) & ~walk.reindex(oc.index).fillna(False)]
    grp_walk = pd.Series(False, index=O.index)
    if 'customer_group' in O.columns:
        grp_walk = walk & O['customer_group'].notna()
    r.n = int(len(oc) + walk.sum())
    r.bad = int(len(miss) + grp_walk.sum())
    r.note = f'mã không có trong bảng khách: {len(miss)}; vãng lai có nhóm: {int(grp_walk.sum())}'
    r.examples = [{'customer_code': v} for v in miss.head(3)]
    return r.finish()


def inv_purchase_total(P: pd.DataFrame) -> InvResult:
    """I12: tổng phiếu nhập = Σ thành tiền các dòng nhập của phiếu."""
    r = InvResult('I12_purchase_total')
    if P is None or P.empty or not {'receipt_no', 'p_total', 'p_amount'} <= set(P.columns):
        r.note = 'không có tổng phiếu / thành tiền dòng nhập'
        return r.finish()
    g = P.groupby('receipt_no', sort=False).agg(lines=('p_amount', lambda x: _num(x).sum(min_count=1)),
                                                 total=('p_total', lambda x: _num(x).dropna().iloc[0] if _num(x).notna().any() else np.nan)).dropna()
    ok = _close(g['lines'], g['total'])
    r.n, r.bad = len(g), int((~ok).sum())
    gg = g.reset_index()
    r.examples = _ex(gg, ~ok.values, ['receipt_no', 'lines', 'total'])
    return r.finish()


def inv_money_scale(L: pd.DataFrame) -> InvResult:
    """Thang tiền: cùng mặt hàng, đơn giá (quy về đơn vị cơ bản) không lệch bậc ×≥300 so với trung vị
    -> phát hiện '45' bị đọc là 45đ trong khi phần lớn là '45k' = 45 000đ."""
    r = InvResult('MONEY_scale')
    if not {'product_key', 'unit_price'} <= set(L.columns):
        r.note = 'thiếu mặt hàng / đơn giá'
        return r.finish()
    p = _num(L['unit_price'])
    if 'factor' in L.columns:
        p = p / _num(L['factor']).replace(0, np.nan).fillna(1)
    df = pd.DataFrame({'k': L['product_key'], 'p': p}).dropna()
    df = df[df['p'] > 0]
    if df.empty:
        return r.finish()
    med = df.groupby('k')['p'].transform('median')
    ratio = df['p'] / med
    bad = (ratio >= 300) | (ratio <= 1 / 300)
    r.n, r.bad = len(df), int(bad.sum())
    r.examples = _ex(L.loc[df.index], bad.values, ['order_id', 'product_name', 'unit', 'unit_price'])
    return r.finish()


def inv_dates(L: pd.DataFrame, date_end: Optional[Any]) -> tuple[InvResult, InvResult]:
    """Ngày: tỉ lệ dòng không đọc được ngày; tỉ lệ ngày nằm sau ngày cuối dữ liệu ghi trong README (năm lấy sai)."""
    a, b = InvResult('DATE_parsed'), InvResult('DATE_range')
    if 'date' not in L.columns:
        a.note = b.note = 'không có ngày'
        return a.finish(), b.finish()
    d = pd.to_datetime(L['date'], errors='coerce')
    a.n, a.bad = len(L), int(d.isna().sum())
    a.examples = _ex(L, d.isna(), ['order_id', 'date', '__table__'])
    if date_end is not None:
        try:
            end = pd.Timestamp(date_end)
            ok = d.notna()
            late = ok & (d > end + pd.Timedelta(days=1))
            b.n, b.bad = int(ok.sum()), int(late.sum())
            b.note = f'README: dữ liệu đến {end.date()}; min={d.min().date() if ok.any() else None}, max={d.max().date() if ok.any() else None}'
            b.examples = _ex(L, late, ['order_id', 'date', '__table__'])
        except Exception:  # noqa: BLE001
            b.note = f'không đọc được ngày cuối README: {date_end!r}'
    else:
        b.note = 'README không ghi ngày cuối'
    return a.finish(), b.finish()


def inv_unit_factor(L: pd.DataFrame, units: Optional[pd.DataFrame]) -> InvResult:
    """I5 (v2.5.0): dòng bán theo đơn vị KHÁC đơn vị cơ bản phải có hệ số quy đổi ≠ 1.
    Phát hiện bảng quy đổi không gán được (đổi tên cột 'Tên sản phẩm'…) -> số lượng quy đổi sai im lặng."""
    r = InvResult('I5_unit_factor')
    if not {'unit_f', 'base_unit_f', 'factor'} <= set(L.columns):
        r.note = 'không có đơn vị / đơn vị cơ bản'
        return r.finish()
    m = L['unit_f'].notna() & L['base_unit_f'].notna() & (L['unit_f'] != L['base_unit_f'])
    f = pd.to_numeric(L['factor'], errors='coerce')
    bad = m & (f.isna() | (f == 1))
    r.n, r.bad = int(m.sum()), int(bad.sum())
    r.note = f'bảng quy đổi: {0 if units is None else len(units)} dòng'
    r.examples = _ex(L, bad, ['product_name', 'unit', 'base_unit', 'factor', 'qty'])
    return r.finish()


def inv_stock_card(S: pd.DataFrame, L: pd.DataFrame) -> InvResult:
    """Thẻ kho (tồn sau mỗi phát sinh): tồn không âm. Đặt nền cho tính tồn ngược từ snapshot (v2.4.x)."""
    r = InvResult('STOCK_card')
    if S is None or S.empty or 'balance' not in S.columns:
        r.note = 'không có thẻ kho'
        return r.finish()
    bal = _num(S['balance'])
    bad = bal < 0
    r.n, r.bad = int(bal.notna().sum()), int(bad.sum())
    r.examples = _ex(S, bad, ['product_key', 'ts', 'balance'])
    return r.finish()


# ----------------------------------------------------------------------------- chạy cho 1 shop
def check_canon(canon, readme=None) -> dict:
    L, O = canon.lines, canon.orders
    res: list[InvResult] = []
    for fn in (lambda: inv_line_arith(L), lambda: inv_goods_total(L, O), lambda: inv_payable(O),
               lambda: inv_one_status(L), lambda: inv_status_mapped(L), lambda: inv_payment_mapped(O),
               lambda: inv_customer_join(O, canon.customers), lambda: inv_purchase_total(canon.purchases),
               lambda: inv_money_scale(L), lambda: inv_stock_card(canon.stock, L),
               lambda: inv_unit_factor(L, canon.units)):
        try:
            res.append(fn())
        except Exception as e:  # noqa: BLE001 - một invariant lỗi không làm hỏng cả báo cáo
            res.append(InvResult(getattr(fn, '__name__', 'inv'), note=f'lỗi: {e!r}'[:200]).finish())
    try:
        end = getattr(readme, 'data_end', None) if readme is not None else None
        res.extend(inv_dates(L, end))
    except Exception as e:  # noqa: BLE001
        res.append(InvResult('DATE_parsed', note=f'lỗi: {e!r}'[:200]).finish())
    levels = [x.level for x in res]
    risk = 3 * levels.count('red') + levels.count('yellow')
    return {'lines': len(L), 'orders': len(O), 'risk': risk, 'red': levels.count('red'), 'yellow': levels.count('yellow'),
            'health': {k: v for k, v in (canon.health or {}).items() if v not in (None, [], 0)},
            'invariants': [x.to_dict() for x in res]}


# ----------------------------------------------------------------------------- báo cáo
_ICON = {'red': '🔴', 'yellow': '🟡', 'green': '🟢', 'na': '·'}


def render_markdown(split: str, report: dict, nq: dict) -> str:
    names = list(LEVELS)
    out = [f'# Invariants — {split}', '',
           'Kiểm tra canonical không cần đáp án. 🔴 đỏ / 🟡 vàng / 🟢 xanh / · không kiểm được. '
           'Ô ghi tỉ lệ vi phạm. Ưu tiên = rủi ro × số câu của shop.', '',
           '| shop | câu | rủi ro | ' + ' | '.join(n.replace('_', ' ') for n in names) + ' |',
           '|' + '---|' * (3 + len(names))]
    order = sorted(report, key=lambda b: (-(report[b].get('risk', 0) * max(nq.get(b, 1), 1)), b))
    for b in order:
        rep = report[b]
        if 'error' in rep:
            out.append(f'| {b} | {nq.get(b, 0)} | lỗi build | ' + ' | '.join('' for _ in names) + ' |')
            continue
        by = {x['name']: x for x in rep['invariants']}
        cells = []
        for n in names:
            x = by.get(n)
            cells.append('·' if not x or x['level'] == 'na' else f"{_ICON[x['level']]} {x['rate']:.3f}")
        out.append(f"| {b} | {nq.get(b, 0)} | {rep['risk']} | " + ' | '.join(cells) + ' |')
    out += ['', '## Chi tiết các ô vàng/đỏ', '']
    for b in order:
        rep = report[b]
        if 'error' in rep:
            out.append(f'### {b}\n- lỗi build: {rep["error"]}\n')
            continue
        bad = [x for x in rep['invariants'] if x['level'] in ('red', 'yellow')]
        if not bad:
            continue
        out.append(f'### {b} (rủi ro {rep["risk"]}, {nq.get(b, 0)} câu)')
        for x in bad:
            out.append(f"- {_ICON[x['level']]} **{x['name']}**: {x['bad']}/{x['n']} ({x['rate']:.3%}). {x['note']}")
            for e in x['examples'][:3]:
                out.append(f'  - `{e}`')
        if rep.get('health'):
            out.append(f"- health: `{rep['health']}`")
        out.append('')
    return '\n'.join(out) + '\n'


def run_split(split_dir: str, cache_dir: str, biz: Optional[list[str]] = None, workers: int = 4,
              build: Optional[Callable] = None) -> tuple[dict, dict]:
    import json
    from pathlib import Path
    from datamate.data.catalog import build_all
    data = Path(split_dir) / 'data'
    ids = biz or sorted(p.name for p in data.iterdir() if p.is_dir())
    nq: dict[str, int] = {}
    qf = Path(split_dir) / 'questions.jsonl'
    if qf.exists():
        for line in qf.read_text(encoding='utf-8-sig').splitlines():
            if line.strip():
                b = json.loads(line).get('business_id')
                nq[b] = nq.get(b, 0) + 1
    cats = (build or build_all)(data, ids, cache_dir, workers)
    report = {}
    for b in ids:
        c = cats.get(b)
        if c is None:
            report[b] = {'error': 'không build được catalog'}
            continue
        try:
            report[b] = check_canon(c.canon, c.readme)
        except Exception as e:  # noqa: BLE001
            report[b] = {'error': repr(e)[:300]}
    return report, nq
