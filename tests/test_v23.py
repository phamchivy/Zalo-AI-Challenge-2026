"""v2.3: viết tắt th/t, danh sách tháng, kỳ trước tháng tường minh, mốc thời gian chưa đọc, chuẩn hoá khẩu ngữ."""
from datetime import date

from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time


def tp(q, as_of):
    return resolve_time(qnorm(q), date.fromisoformat(as_of))


def rng(q, as_of):
    t = tp(q, as_of)
    return [f'{r.start}..{r.end}' for r in t.ranges]


def test_abbrev_month_list_shared_year():
    assert rng('Th2, th3, th4/2026 tháng nào tiệm bán được nhiều tiền nhất z?', '2026-07-22') == [
        '2026-02-01..2026-02-28', '2026-03-01..2026-03-31', '2026-04-01..2026-04-30']


def test_bare_abbrev_month_uses_rulebook_year_rule():
    assert rng('Th3 trung bình mỗi đơn được bao nhiêu doanh thu?', '2026-07-22') == ['2026-03-01..2026-03-31']
    assert rng('Th9 bán được bao nhiêu?', '2026-07-22') == ['2025-09-01..2025-09-30']   # tháng chưa tới -> năm trước


def test_prev_period_after_explicit_month():
    t = tp('Tháng 5/2025 doanh thu so với tháng trước tăng hay giảm?', '2026-07-22')
    assert 'prev_period' in t.flags


def test_unused_time_tokens_are_reported():
    assert tp('Doanh thu tháng 3/2025 bao nhiêu?', '2026-07-22').unused == []
    # khoảng tuổi nợ '0-7, 8-15 ngày' không phải tháng/năm
    assert tp('Chia nợ theo tuổi 0-7, 8-15, trên 15 ngày', '2026-07-22').unused == []


def test_colloquial_normalization():
    assert qnorm('tiệm chốt được bao nhiêu o đờ hả?').endswith('ban duoc bao nhieu don')
    assert 'tang hay giam' in qnorm('doanh thu lên hay xuống')
    assert 'so voi' in qnorm('so dzới tháng trước')
    assert 'bao nhieu tien' in qnorm('khách mua được nhiều tiềng')
