"""v3.0-a: suy cấu trúc từ giá trị (không dùng tên cột) + kiểm tra độ phủ bảng con."""
import random

import pandas as pd

from datamate.data.profiler import profile_table
from datamate.data.raw_loader import load_business
from datamate.data.schema_mapper import map_business
from datamate.data.readme import parse_readme
from datamate.data.structure import infer_structure


def _shop(tmp_path, consistent=True):
    random.seed(0)
    prods = pd.DataFrame({'Mã hàng': [f'SP{i:04d}' for i in range(60)], 'Tên hàng': [f'Hàng số {i} loại tốt' for i in range(60)],
                          'Giá bán': [1000 * (i + 5) for i in range(60)]})
    orders, lines = [], []
    for o in range(400):
        oid = f'HD{o:05d}'
        tot = 0
        for j in range(random.randint(1, 3)):
            p = random.randrange(60)
            q = random.randint(1, 4)
            price = 1000 * (p + 5)
            amt = q * price if consistent else random.randint(1, 10) * 777
            tot += amt
            lines.append({'XA': oid, 'XB': f'SP{p:04d}', 'XC': q, 'XD': price, 'XE': amt})
        orders.append({'Mã hóa đơn': oid, 'Ngày bán': f'2025-01-{1 + o % 28:02d}', 'Trạng thái': 'Hoàn thành', 'YY': tot})
    d = tmp_path / 'biz'
    d.mkdir()
    pd.DataFrame(orders).to_csv(d / 'HoaDon.csv', index=False)
    pd.DataFrame(lines).to_csv(d / 'HoaDonChiTiet.csv', index=False)     # tên cột vô nghĩa: regex không hiểu được
    prods.to_csv(d / 'SanPham.csv', index=False)
    raw = load_business(d)
    profiles = {n: profile_table(t) for n, t in raw.items()}
    return raw, profiles, map_business(profiles, parse_readme(''))


def test_structure_recovers_line_table(tmp_path):
    raw, profiles, bm = _shop(tmp_path)
    lm = bm.tables['HoaDonChiTiet']
    assert not ({'qty', 'amount'} & set(lm.roles))                       # regex không hiểu bảng dòng
    rep = infer_structure(raw, profiles, bm)
    assert lm.ttype == 'sales_lines'
    assert lm.roles['order_id'] == 'XA' and lm.roles['product_code'] == 'XB'
    assert (lm.roles['qty'], lm.roles['unit_price'], lm.roles['amount']) == ('XC', 'XD', 'XE')
    assert bm.tables['HoaDon'].roles.get('goods_total') == 'YY'        # tiền hàng = Σ thành tiền dòng
    assert rep.coverage()['level'] == 'green'


def test_uncovered_child_table_is_red(tmp_path):
    raw, profiles, bm = _shop(tmp_path, consistent=False)              # không có quan hệ SL × giá = thành tiền
    rep = infer_structure(raw, profiles, bm)
    assert 'HoaDonChiTiet' in rep.uncovered_children
    assert rep.coverage()['level'] == 'red'
