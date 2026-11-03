"""v2.3.2: lỗi chung tìm được khi kiểm toán tầng A bằng agent 27B (log v23 public)."""
from datetime import date

import pandas as pd

from datamate.data.sanity import amount_is_gross, first_line_only
from datamate.understand.answerability import out_of_range, subjective
from datamate.understand.intent import _PLURAL, _k
from datamate.understand.normalize import qnorm
from datamate.core.types import TimeRange


def _ledger(first_only=True, n=200):
    rows = []
    for o in range(n):
        for i in range(3 if o % 2 else 1):
            d = (50000 if (i == 0 or not first_only) else None) if o % 3 == 0 else None
            rows.append({'order_id': f'o{o}', 'qty': 1, 'unit_price': 1_000_000, 'amount': 1_000_000, 'line_discount': d})
    return pd.DataFrame(rows)


def test_order_discount_on_first_line_only():
    L = _ledger(True)
    assert first_line_only(L, 'line_discount')
    assert not amount_is_gross(L)              # 'Bớt' cả đơn -> không trừ vào tiền dòng
    L2 = _ledger(False)                        # giảm ghi trên mọi dòng -> không phải mẫu "chỉ dòng đầu"
    assert not first_line_only(L2, 'line_discount')


def test_top_k_not_from_dates():
    assert _k(qnorm('Th 9/25 khách mô mua nhiều đơn nhất rứa?')) is None
    assert _k(qnorm('Tháng 1/26 sp nào bán chạy nhất?')) is None
    assert _k(qnorm('Top năm món bán chạy nhất tuần này')) == 5
    assert _k(qnorm('Th9/2025 3 khach nao chi nhieu tien nhat?')) == 3


def test_scope_phrase_is_not_plural():
    assert not _PLURAL(qnorm('Tháng 3/2025, trong các khách hàng, ai trung thành nhất?'))
    assert _PLURAL(qnorm('Liệt kê các khách mua nhiều nhất'))


def test_guard_attribute_dates_and_reasons():
    p = [TimeRange(start=date(2026, 11, 1), end=date(2026, 11, 30), label='main', granularity='month')]
    q = qnorm('Tính tới hôm nay có mấy đơn bán chịu có ngày phải trả trong tháng 11/2026?')
    assert out_of_range(p, date(2025, 1, 1), date(2026, 6, 10), date(2026, 6, 10), q) is None
    assert out_of_range(p, date(2025, 1, 1), date(2026, 6, 10), date(2026, 6, 10), 'doanh thu') is not None
    assert subjective(qnorm('Th7/2025 bán ra bao nhiêu chiếc điện thoại, cả máy mới lẫn máy cũ?')) is None
    assert subjective(qnorm('Th5/2026 khách huỷ đơn chủ yếu vì sao?')) is None
    assert subjective(qnorm('Sản phẩm nào ngon nhất?')) is not None


def test_colloquial_v232():
    assert 'ngay nao' in qnorm('Tháng 4/2026 bữa nào bán nhiều tiền nhất?')
    assert 'may don tien hang tu' in qnorm('có mấy đơng tiền hàng từ 760.000đ trở lên')
    assert 'may dong' in qnorm('tháng này bán được mấy đồng')
