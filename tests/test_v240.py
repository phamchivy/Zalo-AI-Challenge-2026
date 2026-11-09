"""v2.4.0: invariants phát hiện được lỗi dựng bảng đã biết (không cần đáp án)."""
import pandas as pd

from datamate.data.invariants import (inv_goods_total, inv_line_arith, inv_money_scale, inv_one_status,
                                      inv_payable, inv_status_mapped)


def _lines(n=200, sub_order_discount=False):
    rows = []
    for o in range(n):
        for i in range(2):
            amt = 100_000
            if sub_order_discount and i == 0:
                amt -= 10_000                      # giảm cả đơn bị trừ nhầm vào dòng đầu (lỗi kiểu "Bớt")
            rows.append({'order_id': f'o{o}', 'product_key': f'p{i}', 'product_name': f'P{i}', 'qty': 1,
                         'unit_price': 100_000, 'amount': amt, 'status': 'completed', 'status_raw': 'Xong'})
    return pd.DataFrame(rows)


def test_green_on_clean_data():
    L = _lines()
    O = pd.DataFrame({'order_id': [f'o{o}' for o in range(200)], 'goods_total': 200_000, 'order_discount': 10_000,
                      'ship_fee': 0, 'order_total': 190_000})
    for r in (inv_line_arith(L), inv_goods_total(L, O), inv_payable(O), inv_one_status(L), inv_status_mapped(L),
              inv_money_scale(L)):
        assert r.level == 'green', (r.name, r.to_dict())


def test_order_discount_subtracted_into_lines_is_red():
    L = _lines(sub_order_discount=True)
    O = pd.DataFrame({'order_id': [f'o{o}' for o in range(200)], 'goods_total': 200_000, 'order_discount': 10_000})
    assert inv_line_arith(L).level == 'red'
    r = inv_goods_total(L, O)
    assert r.level == 'red' and 'bị trừ vào dòng): 1.000' in r.note


def test_unmapped_status_and_mixed_status():
    L = _lines()
    L.loc[L.index[:40], 'status_raw'] = 'khách bùng hàng'      # nhãn lạ: canonical đang âm thầm coi là completed
    assert inv_status_mapped(L).level == 'red'
    L.loc[0, 'status'] = 'cancelled'                          # 2 dòng cùng đơn khác trạng thái
    assert inv_one_status(L).level in ('yellow', 'red')


def test_money_scale_outlier():
    L = _lines()
    L.loc[L.index[:20], 'unit_price'] = 100                   # '100' (nghìn) đọc thành 100đ
    assert inv_money_scale(L).level == 'red'
