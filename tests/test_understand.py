from datetime import date

from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time


def _r(q, asof):
    tp = resolve_time(qnorm(q), date.fromisoformat(asof))
    return [(str(r.start), str(r.end)) for r in tp.ranges], tp.flags


def test_qnorm():
    assert qnorm('thangs 8/2025 bán Yakult dduwowcj bao nhiêu tiền') == 'thang 8/2025 ban yakult duoc bao nhieu tien'
    assert 'doanh thu' in qnorm('dthu hqua') and 'hom qua' in qnorm('dthu hqua')
    assert qnorm('bao nhiêu mặt hàng bị ế nhỉ').endswith('bi e')


def test_time_basic():
    assert _r('doanh thu tháng 6/2025', '2026-01-01')[0] == [('2025-06-01', '2025-06-30')]
    assert _r('số đơn quý 4/2025', '2026-01-01')[0] == [('2025-10-01', '2025-12-31')]
    assert _r('dthu hqua', '2025-09-25')[0] == [('2025-09-24', '2025-09-24')]
    assert _r('sp bán chạy tuần rồi', '2026-06-16')[0] == [('2026-06-08', '2026-06-14')]
    assert _r('dthu 30 ngày gần nhất', '2026-06-16')[0] == [('2026-05-18', '2026-06-16')]
    assert _r('doanh thu T1/26', '2026-06-16')[0] == [('2026-01-01', '2026-01-31')]
    assert _r('doanh thu tháng 9', '2026-06-16')[0] == [('2025-09-01', '2025-09-30')]


def test_time_series_and_flags():
    r, _ = _r('dthu 3 tháng liền nhau kết thúc ở tháng 6/2025', '2026-01-01')
    assert r == [('2025-04-01', '2025-04-30'), ('2025-05-01', '2025-05-31'), ('2025-06-01', '2025-06-30')]
    r, _ = _r('Từ tháng 8/2025 tới tháng 10/2025, tháng nào bán nhiều nhất', '2026-01-01')
    assert len(r) == 3
    _, f = _r('Từ đầu tháng này đến hết hôm nay, doanh thu hơn hay kém cùng kỳ tháng trước mấy %', '2026-06-16')
    assert 'mtd_compare' in f
    r, f = _r('Tính đến hết ngày 19/08/2026, khách còn nợ bao nhiêu', '2026-08-24')
    assert 'snapshot' in f and r[0][1] == '2026-08-19'
