"""v1.2: tiền viết tắt, tên cột có đơn vị / ngoặc, Hàng≠Hạng, gợi ý theo tên sheet, sổ quỹ, đơn theo STT,
giảm cấp đơn, khoá trùng nguồn theo (mã đơn, tháng), chia dev/holdout."""
import pandas as pd
import pytest
import yaml

from datamate.core.textutil import to_num
from datamate.data.catalog import build_catalog
from datamate.data.sanity import dedupe_sources, order_level, stt_groups
from datamate.data.schema_mapper import _name_score, classify_table, name_hint


def test_money_shorthand():
    cases = {'1tr440': 1_440_000, '1tr5': 1_500_000, '1tr090': 1_090_000, '1,5tr': 1_500_000, '500k': 500_000,
             '1ty2': 1_200_000_000, '3tr450k': 3_450_000, '2tr': 2_000_000, '2,176tr': 2_176_000, '90.000đ': 90_000,
             '1.200.000': 1_200_000, '10 ti vi': 10, '2 túi': 2}
    for s, v in cases.items():
        assert to_num(s) == v, s


def test_column_name_variants_and_diacritics():
    assert _name_score('Thành tiền (đ)', 'amount') > 0
    assert _name_score('S.L', 'qty') == 2.0 and _name_score('SL (ĐV cơ bản)', 'qty_base') == 2.0
    assert _name_score('Giá 1 đv', 'unit_price') > 0 and _name_score('Mã H.hóa', 'product_code') > 0
    assert _name_score('Khách hàng (tên)', 'customer_name') > 0 and _name_score('TongTienVND', 'order_total') > 0
    assert _name_score('Hàng', 'product_name') > 0 and _name_score('Hàng', 'customer_group') == 0
    assert _name_score('Hạng', 'customer_group') > 0 and _name_score('Hạng', 'product_name') == 0


def test_sheet_name_hint_first():
    assert name_hint('BanHang::BanHang_LichSuGia') == 'price_history'
    assert name_hint('BanHang::BanHang') == 'sales'
    assert classify_table('SoQuy', {'order_id', 'datetime', 'payment', 'pay_amount', 'receipt_id'}, 900) == 'payments'


def test_order_level_and_stt_groups():
    f = pd.DataFrame({'order_id': [f'O{i // 2}' for i in range(60)], 'x': [str(1000 * (i // 2 + 1)) for i in range(60)],
                      'y': [str(10 * i + 5) for i in range(60)]})
    assert order_level(f, 'x') and not order_level(f, 'y')
    rows = []
    for o in range(40):                                            # STT chỉ ở dòng đầu, khách trả = tổng nhóm - bớt
        rows.append({'line_no': str(o + 1), 'customer_name': 'kl', 'amount': 580000.0, 'paid': 720000.0})
        rows.append({'line_no': None, 'customer_name': None, 'amount': 150000.0, 'paid': None})
    L = pd.DataFrame(rows)
    L['__table__'] = 'So'
    g = stt_groups(L)
    assert g is not None and g.nunique() == 40 and g.iloc[0] == g.iloc[1] != g.iloc[2]
    L2 = L.copy()
    L2['paid'] = L2['amount'].where(L2['line_no'].notna())          # tiền trả = tiền dòng -> mỗi dòng là một đơn
    assert stt_groups(L2) is None


def test_dedupe_key_uses_month():
    a = pd.DataFrame({'order_id': [f'HD{i}' for i in range(100)], '__dkey__': ['2024-01'] * 100})
    b = pd.DataFrame({'order_id': [f'HD{i}' for i in range(80)], '__dkey__': ['2025-01'] * 80})   # mã đánh lại năm sau
    keep, notes = dedupe_sources([a, b], ['2024', '2025'])
    assert len(keep) == 2 and not notes


def _write_shop(d):
    d.mkdir(parents=True)
    rows = []
    for o in range(60):
        day = f'2025-06-{o % 28 + 1:02d}'
        rows.append({'Ngày': day, 'STT': o + 1, 'Khách': f'Khách {o % 7}', 'Hàng': 'Sạc 25W', 'SL': 2, 'Giá': '290.000',
                     'Thành tiền (đ)': '580.000', 'Bớt': '10.000', 'Khách trả': '1tr140'})
        rows.append({'Ngày': day, 'STT': None, 'Khách': None, 'Hàng': 'Cáp C-L', 'SL': 1, 'Giá': '150.000',
                     'Thành tiền (đ)': '570.000', 'Bớt': None, 'Khách trả': None})
    with pd.ExcelWriter(d / 'SoBanHang.xlsx') as w:
        pd.DataFrame(rows).to_excel(w, sheet_name='So ban hang', index=False)
        pd.DataFrame({'Mã H.hóa': ['SP1'], 'Item': ['Sạc 25W'], 'Ngày áp dụng': ['2025-06-01'], 'Giá cũ': [280000],
                      'Giá mới': [290000]}).to_excel(w, sheet_name='BanHang_LichSuGia', index=False)
    pd.DataFrame({'Mã phiếu': [f'PT{o}' for o in range(60)], 'Thời gian': ['2025-06-01 10:00'] * 60,
                  'Mã hóa đơn': [f'X{o}' for o in range(60)], 'Phương thức thanh toán': ['Tiền mặt'] * 60,
                  'Giá trị': ['90.000đ'] * 60}).to_csv(d / 'SoQuy.csv', index=False)
    (d / 'README.md').write_text('# shop\nDữ liệu có đến hết 30/06/2025.\n', encoding='utf-8')


def test_catalog_structure(tmp_path):
    _write_shop(tmp_path / 'biz_x')
    c = build_catalog(tmp_path / 'biz_x', None)
    types = {n: m.ttype for n, m in c.mapping.tables.items()}
    assert any(t == 'price_history' for n, t in types.items() if 'LichSuGia' in n)
    assert types.get('SoQuy') == 'payments'
    L = c.canon.lines
    assert L['order_id'].nunique() == 60                        # 120 dòng, 60 đơn theo nhóm STT
    assert 'product_name' in L.columns and L['product_name'].notna().all()
    assert L['amount'].sum() == pytest.approx(60 * (580000 + 570000) - 60 * 10000)   # bớt ở dòng đầu trừ vào doanh thu
    o = c.canon.orders.set_index('order_id')
    assert o['paid'].dropna().eq(1_140_000).all()


def test_make_split(tmp_path):
    from datamate.offline.split import load_part, make_split
    data = tmp_path / 'train' / 'data'
    for i in range(10):
        dd = data / f'biz_{i:03d}'
        dd.mkdir(parents=True)
        pd.DataFrame({'Nội dung': ['Panadol 500mg hộp 10 vỉ x 12 viên'] * 5}).to_csv(dd / 'ban_hang.csv', index=False)
    out = tmp_path / 'split.yaml'
    d1 = make_split(str(tmp_path / 'train'), ['biz_000', 'biz_001'], n_holdout=3, out=str(out))
    d2 = make_split(str(tmp_path / 'train'), ['biz_000', 'biz_001'], n_holdout=3)
    assert d1['holdout'] == d2['holdout'] and len(d1['holdout']) == 3
    assert not set(d1['holdout']) & {'biz_000', 'biz_001'}
    assert set(d1['holdout']) | set(d1['dev']) == {f'biz_{i:03d}' for i in range(10)}
    assert load_part(str(out), 'holdout') == set(yaml.safe_load(out.read_text(encoding='utf-8'))['holdout'])
    with pytest.raises(ValueError):
        load_part(str(out), 'public')


def test_unmapped_ledger_flagged(tmp_path):
    d = tmp_path / 'biz_y'
    _write_shop(d)
    pd.DataFrame({'Số CT': [f'C{o}' for o in range(60)], 'Lúc': ['10:00'] * 60, 'Về việc': ['thu'] * 60}) \
        .to_csv(d / 'SoQuy.csv', index=False)                    # sổ quỹ không đọc được cột tiền
    c = build_catalog(d, None)
    assert c.canon.health.get('payments_unmapped') == ['SoQuy']
