"""Làm sạch dữ liệu theo cấu trúc: dòng tổng, dòng tiêu đề nhóm, nguồn trùng, thành tiền trước giảm."""
import pandas as pd

from datamate.data.sanity import amount_is_gross, clean_line_frame, dedupe_sources, is_summary_label


def test_summary_label_not_surname():
    assert is_summary_label('Tổng tháng 10/2025') and is_summary_label('Tổng ngày 11/10') and is_summary_label('TOTAL')
    assert is_summary_label('Tong thang 10') and is_summary_label('Cộng')
    assert not is_summary_label('Tống Chi Ngọc') and not is_summary_label('Tong Chi Ngoc')
    assert not is_summary_label('Công ty ABC') and not is_summary_label('Cong ty TNHH')


def test_clean_line_frame():
    f = pd.DataFrame({'order_id': ['A1', None, None, 'Tổng ngày 01/06'], 'product_name': ['Coca', 'Bánh', None, None],
                      'qty': ['1', '2', None, None], 'amount': ['10', '20', '30', '30']})
    g, rep = clean_line_frame(f)
    assert rep == {'summary_label_rows': 1, 'header_rows': 1}
    assert len(g) == 3 and g['__header__'].tolist() == [False, False, True]


def test_dedupe_sources():
    a = pd.DataFrame({'order_id': [f'O{i}' for i in range(100)]})
    b = pd.DataFrame({'order_id': [f'O{i}' for i in range(50)]})          # bảng con nằm trọn trong a
    c = pd.DataFrame({'order_id': [f'X{i}' for i in range(30)]})          # chi nhánh khác: giữ
    keep, notes = dedupe_sources([a, b, c], ['all', 'online', 'branch2'])
    assert len(keep) == 2 and 'online' in notes[0]


def test_amount_gross():
    n = 30
    L = pd.DataFrame({'qty': [2.0] * n, 'unit_price': [10000.0] * n, 'line_discount': [2000.0] * n, 'amount': [20000.0] * n})
    assert amount_is_gross(L)
    assert not amount_is_gross(L.assign(amount=18000.0))
