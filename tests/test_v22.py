"""v2.2: lỗi đọc thời gian / loại câu tìm thấy trong log public (sửa theo quy tắc chung, không theo từng câu)."""
from datetime import date

from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time


def rng(q, as_of):
    tp = resolve_time(qnorm(q), date.fromisoformat(as_of))
    return [f'{r.start}..{r.end}' for r in tp.ranges], tp.flags


def test_bare_month_short_year():
    assert rng('2/25 bi huy bao nhieu don?', '2026-05-24')[0] == ['2025-02-01..2025-02-28']
    assert rng('So sánh 3/25, 4/25 và 5/25 xem tháng nào bán nhiều tiền nhất', '2026-07-08')[0] == [
        '2025-03-01..2025-03-31', '2025-04-01..2025-04-30', '2025-05-01..2025-05-31']
    assert rng('giá 35.000/ly', '2026-07-08')[0] == []                        # không phải tháng
    assert rng('doanh thu ngày 1/5', '2026-07-08')[0] == ['2026-05-01..2026-05-01']   # ngày/tháng giữ nguyên


def test_month_ranges_enumerate_all_months():
    assert len(rng('T1/2026 tới T3/2026, tháng nào bán nhiều tiền nhất?', '2026-08-21')[0]) == 3
    assert len(rng('3 tháng từ th12/2025 đến th2/2026', '2026-08-21')[0]) == 3
    assert len(rng('Dthu từng tháng từ 02/2025 tới 04/2025', '2026-07-08')[0]) == 3
    assert rng('doanh thu các tháng 10, 11, 12 năm 2025', '2026-07-22')[0][-1] == '2025-12-01..2025-12-31'
    assert rng('Dthu tháng 3, tháng 4, tháng 5 năm 2025', '2026-07-22')[0][0] == '2025-03-01..2025-03-31'


def test_endpoint_today_is_not_the_period():
    assert rng('Tính luôn hnay thì 7 ngày gần nhất có bnhiu đơn', '2026-02-14')[0] == ['2026-02-08..2026-02-14']
    assert rng('Đầu năm tới chừ được mấy đơn?', '2026-05-03')[0] == ['2026-01-01..2026-05-03']
    assert rng('Hôm nay bán được mấy đơn?', '2026-02-20')[0] == ['2026-02-20..2026-02-20']


def test_month_to_date_compare_and_last_year():
    r, f = rng('dthu từ mùng 1 tới hết hôm nay so với cùng kỳ tháng trước', '2025-08-08')
    assert r == ['2025-08-01..2025-08-08'] and 'mtd_compare' in f
    r, f = rng('Tháng này tới giờ dthu so với tháng trước tăng hay giảm mấy %?', '2025-05-13')
    assert r == ['2025-05-01..2025-05-13'] and 'mtd_compare' in f
    r, f = rng('so với cùng số ngày đầu tháng rồi, doanh thu tháng này', '2025-10-22')
    assert 'mtd_compare' in f
    assert rng('thang 5/2026 so voi thang 5 nam ngoai', '2026-07-22')[0] == ['2026-05-01..2026-05-31', '2025-05-01..2025-05-31']
    assert rng('Tháng rồi doanh thu bao nhiêu?', '2026-07-22')[0] == ['2026-06-01..2026-06-30']
    assert rng('3 tháng tháng sau có luôn cao hơn tháng trước không', '2026-07-22')[0] == []
