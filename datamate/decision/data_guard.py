"""v2.5.0 — Chặn đáp án theo rủi ro dữ liệu (invariants chạy lúc dự đoán) + kiểm tính hợp lý của đáp án.

Ý tưởng: hệ thống không thể biết trước mọi kiểu lỗi dữ liệu của shop mới (private). Nhưng invariants (không cần đáp án)
phát hiện được *phần dữ liệu nào* đáng ngờ. Câu hỏi nào dựa vào phần đó thì bỏ trống (0 điểm) thay vì trả lời sai (−1).
Không chặn cả shop: chỉ chặn câu có loại (intent) phụ thuộc vào invariant báo đỏ.

Kiểm tính hợp lý: đáp án số âm cho đại lượng không thể âm, phần trăm ngoài miền, NaN/inf -> bỏ trống.
"""
from __future__ import annotations

import math
from typing import Optional

from datamate.core.types import AnswerType, Decision

# invariant -> nhóm intent phụ thuộc. 'ALL' = mọi câu dựa trên đơn hàng.
_ORDER_BASED = 'ALL'
MONEY_INTENTS = {'revenue', 'aov', 'share', 'compare_period', 'top_product', 'top_customer', 'top_staff', 'best_day',
                 'best_month', 'series', 'values_channel', 'values_payment', 'customer_thr', 'order_count_thr',
                 'day_count_thr', 'gross_profit', 'net_profit', 'rank_def', 'trend_bool', 'group', 'compare_shift',
                 'unit_price', 'weekday_avg', 'special'}
DEBT_INTENTS = {'debt', 'debt_total', 'debt_top', 'debt_customers', 'debt_collected', 'debt_flow', 'debt_aging',
                'debt_over_days'}
CUSTOMER_INTENTS = {'top_customer', 'customer_thr', 'customers', 'rank_def', 'debt_top', 'debt_customers'}
PURCHASE_INTENTS = {'purchase', 'purchase_total', 'purchase_price', 'purchase_supplier', 'purchase_minus_sales'}
STOCK_INTENTS = {'stock_qty', 'inventory', 'inventory_value', 'dead_stock', 'expiry', 'expiry_capital', 'expiry_list',
                 'expiry_count', 'lot_nearest'}

QTY_INTENTS = {'qty_sold', 'top_product', 'qty_unit', 'product_count', 'stock_qty', 'purchase_minus_sales',
               'price_effect', 'inventory', 'dead_stock'}

DEPENDS: dict[str, object] = {
    'COVERAGE_child': _ORDER_BASED,
    'STATUS_unresolved': _ORDER_BASED,          # v3.0.1: có cột trạng thái mà không hiểu -> số đơn/doanh thu không tin được
    'I5_unit_factor': QTY_INTENTS,
    'DATE_parsed': _ORDER_BASED,
    'DATE_range': _ORDER_BASED,
    'I6_one_status': _ORDER_BASED,
    'I15_status_mapped': _ORDER_BASED,
    'I1_line_arith': MONEY_INTENTS,
    'MONEY_scale': MONEY_INTENTS | {'qty_sold'},
    'I2_goods_total': DEBT_INTENTS,
    'I3_payable': DEBT_INTENTS,
    'I15_payment_mapped': DEBT_INTENTS | {'values_payment'},
    'I9_customer_join': CUSTOMER_INTENTS,
    'I12_purchase_total': PURCHASE_INTENTS,
    'STOCK_card': STOCK_INTENTS,
}
NON_ORDER = PURCHASE_INTENTS | STOCK_INTENTS        # câu không dựa trên bảng đơn: không bị chặn bởi nhóm 'ALL'

# đại lượng không thể âm (đáp án số âm = tính sai) — trừ câu hỏi chênh lệch / so sánh
NON_NEGATIVE = {'revenue', 'aov', 'order_count', 'qty_sold', 'cancel_count', 'customers', 'customer_thr',
                'order_count_thr', 'day_count_thr', 'product_count', 'debt_total', 'debt_collected', 'purchase_total',
                'stock_qty', 'inventory_value', 'expiry_capital', 'unit_price', 'purchase_price', 'delivery_count',
                'qty_unit'}
PERCENT_0_100 = {'share', 'cancel_rate'}


def blocked_by(intent: str, levels: dict[str, str], block_level: str = 'red') -> Optional[str]:
    """Trả về tên invariant (mức block_level) mà intent phụ thuộc, hoặc None."""
    if not intent or not levels:
        return None
    for name, lvl in levels.items():
        if lvl != block_level:
            continue
        dep = DEPENDS.get(name)
        if dep is None:
            continue
        if dep == _ORDER_BASED:
            if intent not in NON_ORDER:
                return name
        elif intent in dep:
            return name
    return None


def implausible(intent: str, d: Decision) -> Optional[str]:
    if d.type not in (AnswerType.NUMBER, AnswerType.PERCENT):
        return None
    try:
        v = float(d.answer)
    except (TypeError, ValueError):
        return 'không phải số'
    if math.isnan(v) or math.isinf(v):
        return 'NaN/inf'
    if intent in NON_NEGATIVE and v < 0:
        return f'{intent} âm ({v:g})'
    if d.type == AnswerType.PERCENT and intent in PERCENT_0_100 and not (-1e-9 <= v <= 100 + 1e-9):
        return f'{intent} ngoài 0–100% ({v:g})'
    return None


def guard(d: Decision, intent: str, levels: Optional[dict[str, str]], enabled: bool = True) -> Decision:
    """Áp lên quyết định cuối. Chỉ chặn đáp án có giá trị (không chặn 'unanswerable' / 'abstain')."""
    if not enabled or d.type in (AnswerType.ABSTAIN, AnswerType.UNANSWERABLE):
        return d
    why = blocked_by(intent, levels or {})
    if why:
        return Decision(qid=d.qid, type=AnswerType.ABSTAIN, confidence=d.confidence, support=d.support, total=d.total,
                        policy=f'data_risk({why})')
    bad = implausible(intent, d)
    if bad:
        return Decision(qid=d.qid, type=AnswerType.ABSTAIN, confidence=d.confidence, support=d.support, total=d.total,
                        policy=f'implausible({bad})')
    return d
