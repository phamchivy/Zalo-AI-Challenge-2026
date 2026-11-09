"""v2.4.1: sửa canonical theo invariants (COD, tiền cấp đơn theo nghìn, topping chung ô) + chỉnh bộ đo."""
import numpy as np
import pandas as pd

from datamate.data.invariants import inv_line_arith, inv_payable
from datamate.data.sanity import fix_order_money_scale, split_toppings
from datamate.data.vocab import norm_payment


def test_payment_keywords():
    assert norm_payment('Thu hộ (COD)') == 'cod'
    assert norm_payment('Tiền mặt') == 'cash'
    assert norm_payment('Chuyển khoản NH') == 'transfer'
    assert norm_payment('Ví MoMo') == 'ewallet'
    assert norm_payment('Ghi nợ') == 'debt'


def test_order_money_in_thousands():
    rows = []
    for o in range(50):
        for i in range(2):
            rows.append({'order_id': f'o{o}', 'amount': 60_000.0, 'goods_total': 120.0, 'order_total': 135.0,
                         'paid': 135.0, 'ship_fee': 15.0})
    L = fix_order_money_scale(pd.DataFrame(rows), notes := [])
    assert L['goods_total'].eq(120_000).all() and L['order_total'].eq(135_000).all()
    assert L['paid'].eq(135_000).all() and L['ship_fee'].eq(15_000).all() and notes
    # tiền đơn đã đúng đồng -> không đụng
    ok = pd.DataFrame([{'order_id': f'o{o}', 'amount': 60_000.0, 'goods_total': 60_000.0} for o in range(50)])
    assert fix_order_money_scale(ok.copy(), []).equals(ok)


def _cafe(n=60):
    rows = []
    for o in range(n):
        rows.append({'order_id': f'o{o}', 'product_name': 'Trà đào cam sả (M) + Trân châu đen (7.000đ)', 'product_key': None,
                     'qty': 2.0, 'unit': 'ly', 'unit_price': 39_000.0, 'amount': 92_000.0})
        rows.append({'order_id': f'o{o}', 'product_name': 'Trân châu đen', 'product_key': 'tc_den', 'qty': 1.0, 'unit': 'phần',
                     'unit_price': 7_000.0, 'amount': 7_000.0})
        rows.append({'order_id': f'o{o}', 'product_name': 'Trà đào cam sả (M)', 'product_key': 'tra_dao_m', 'qty': 1.0,
                     'unit': 'ly', 'unit_price': 39_000.0, 'amount': 39_000.0})
    return pd.DataFrame(rows)


def test_split_topping_two_products():
    L = _cafe()
    assert inv_line_arith(L).level == 'red'                      # thành tiền > SL×giá: bộ đo phải bắt được
    L2 = split_toppings(L.copy(), notes := [])
    assert len(L2) == len(L) + 60 and notes
    top = L2[L2.get('line_origin').eq('topping')]
    assert top['product_key'].eq('tc_den').all() and top['amount'].eq(14_000).all() and top['qty'].eq(2).all()
    main = L2.iloc[0]
    assert main['product_name'] == 'Trà đào cam sả (M)' and main['product_key'] == 'tra_dao_m' and main['amount'] == 78_000
    assert np.isclose(L2['amount'].sum(), L['amount'].sum())    # tổng doanh thu không đổi
    assert inv_line_arith(L2).level == 'green'


def test_payable_constant_offset_is_yellow():
    O = pd.DataFrame({'order_id': [f'o{i}' for i in range(100)], 'goods_total': 100_000.0, 'order_discount': 0.0,
                      'order_total': [115_000.0 if i % 10 == 0 else 100_000.0 for i in range(100)]})
    r = inv_payable(O)
    assert r.level == 'yellow' and '+15,000đ' in r.note
