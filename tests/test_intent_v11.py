"""Parser v1.1: không fallback doanh thu, luật lạ, xếp hạng theo rulebook, mốc tuổi nợ đọc từ câu."""
from datetime import date

from conftest import RULEBOOK
from datamate.knowledge.rules import compile_rules
from datamate.understand.entity_link import EntityIndex
from datamate.understand.intent import age_buckets, parse_question
from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time

RS = compile_rules(RULEBOOK + '\n## 6. Theo ngành\n| Luật | Mặc định |\n|---|---|\n| Trả góp | Đơn trả góp là đơn ghi nợ |\n')
CAPS = {'orders', 'status', 'customers', 'time'}


def P(q, asof='2025-07-31'):
    qn = qnorm(q)
    d = date.fromisoformat(asof)
    return parse_question(qn, qn, d, resolve_time(qn, d), EntityIndex(), CAPS, date(2025, 6, 1), date(2025, 7, 31), rules=RS)


def test_no_revenue_fallback():
    assert P('tháng 6/2025 cửa hàng thế nào').plan is None
    assert P('doanh thu tháng 6/2025').plan.metric == 'revenue'


def test_unknown_rule_blocks():
    r = P('tháng 6/2025 khách trả góp bao nhiêu tiền')
    assert r.plan is None and r.intent in ('rule_unknown', 'special')


def test_ranking_from_rulebook():
    r = P('khách trung thành nhất tháng 6/2025 là ai')
    assert r.plan.op == 'top' and r.plan.metric == 'active_days' and r.plan.group_by == 'customer'


def test_slash_and_day_count():
    assert P('doanh thu tháng 7/2025 tăng/giảm bao nhiêu % so với tháng 6/2025').plan.op == 'pct_change'
    r = P('Tháng 6/2025 có mấy ngày doanh thu đạt 100.000 đồng trở lên?')
    assert r.plan.op == 'count_ge' and r.plan.group_by == 'day' and r.plan.threshold == 100000


def test_age_buckets_from_question():
    b = age_buckets(qnorm('nhóm tuổi nợ 0–7 ngày, 8–15 ngày và trên 15 ngày'))
    assert b[:2] == [[0, 7], [8, 15]] and b[2][0] == 16


# ----------------------------------------------------------------------------- v1.2
def _ix_with(products=(), payments=()):
    from datamate.core.textutil import fold
    ix = EntityIndex()
    ix.products = [(k, d, fold(d), '') for k, d in products]
    ix.payments = list(payments)
    ix.base_units = {'cai', 'bom'}
    return ix


def P2(q, ix, asof='2025-07-31'):
    qn = qnorm(q)
    d = date.fromisoformat(asof)
    return parse_question(qn, qn, d, resolve_time(qn, d), ix, CAPS | {'payment'}, date(2025, 6, 1), date(2025, 7, 31), rules=RS)


def test_payment_filter_not_linked_abstains():
    ix = _ix_with(payments=['cash', 'transfer'])
    assert P2('doanh thu các đơn trả bằng tiền tươi tháng 7/2025', ix).plan is None      # có "trả bằng" mà không nối được
    assert P2('doanh thu các đơn trả bằng COD tháng 7/2025', ix).plan is None             # COD không có trong dữ liệu
    assert P2('doanh thu các đơn trả bằng tiền mặt tháng 7/2025', ix).plan.filters.get('payment') == 'cash'


def test_unsupported_attribute_abstains():
    assert P('Tháng 6/2025 khách nam mua tổng cộng bao nhiêu tiền?').plan is None
    assert P('khách năm 2025 mua bao nhiêu tiền').intent != 'attribute_filter'


def test_cong_lai_is_not_profit():
    r = P('tháng 6/2025 tiền hàng các đơn cộng lại là bao nhiêu')
    assert r.plan is None or r.plan.metric != 'gross_profit'
    assert P('tháng 6/2025 lãi bao nhiêu').plan.metric == 'gross_profit'


def test_nhieu_ngay_count():
    r = P('tháng 6/2025 có nhiêu ngày đạt dthu từ 12.000.000 đồng trở lên')
    assert r.plan.op == 'count_ge' and r.plan.threshold == 12000000


def test_revenue_word_beats_unit_in_product_name():
    ix = _ix_with(products=[('sp1', 'Máy bơm nước Panasonic GP-129JXK 125W')])
    r = P2('dthu của Máy bơm nước Panasonic GP-129JXK 125W trong tháng 7 năm 2025 là bao nhiêu?', ix)
    assert r.plan.metric == 'revenue' and r.plan.filters.get('product') == 'sp1'
