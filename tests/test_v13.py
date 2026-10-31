"""v1.3: bộ phạt "từ lạ" không còn dựa vào từ vựng đóng của train; chuẩn hoá chat / phương ngữ chung."""
from datetime import date

from conftest import RULEBOOK
from datamate.core.textutil import fold
from datamate.knowledge.rules import compile_rules
from datamate.understand.entity_link import EntityIndex
from datamate.understand.intent import _money_threshold, parse_question, unexplained
from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import resolve_time

RS = compile_rules(RULEBOOK)
CAPS = {'orders', 'status', 'customers', 'time'}
AS_OF = date(2025, 10, 15)


def _ix():
    ix = EntityIndex()
    ix.products = [(k, d, fold(d), '') for k, d in [('sp1', 'Cà phê sữa (M)'), ('sp2', 'Nước suối Aquafina 500ml'),
                                                   ('sp3', 'Sạc nhanh Samsung 25W')]]
    ix.customers = [('kh1', 'Tô Dũng Thảo', fold('Tô Dũng Thảo'))]
    return ix


def U(q, linked=(), ix=None):
    qn = qnorm(q)
    return unexplained(qn, list(linked), resolve_time(qn, AS_OF), raw=q, ix=ix)


def P(q, ix=None):
    qn = qnorm(q)
    return parse_question(qn, fold(q), AS_OF, resolve_time(qn, AS_OF), ix or EntityIndex(), CAPS, date(2025, 1, 1),
                          AS_OF, rules=RS, raw=q)


def test_dialect_and_chat_normalized():
    assert qnorm('Trong T5/2026, ngày mô dthu lên cao nhất?') == 'trong t5/2026, ngay nao doanh thu len cao nhat'
    assert qnorm('Hom ni quan duoc may don roi?').startswith('hom nay')
    assert 'bao nhieu ngay' in qnorm('Tháng 3/2026 có bao nhiêu bữa doanh thu đạt từ 11 củ trở lên?')
    assert '11 trieu' in qnorm('Tháng 3/2026 có bao nhiêu bữa doanh thu đạt từ 11 củ trở lên?')
    assert 'may don hang' in qnorm('8/2025 đếm giùm mấy đh dthu từ 180.000đ trở lên')
    assert 'nhap hang' in qnorm('Tháng 8/2026 quán cất hàng hết bao nhiêu tiền?')
    assert 'het hom nay' in qnorm('tháng này tới giờ doanh thu bao nhiêu')
    # không phá tên hàng: 'họng', 'chén', 'mì'
    assert 'suc hong' in qnorm('Betadine gargle súc họng chai 125ml')
    assert 'rua chen' in qnorm('Nước rửa chén Sunlight 750g')
    assert 'mi hao hao' in qnorm('mì hảo hảo bán được mấy gói')


def test_fillers_not_penalized():
    for q in ['Hnay tiệm bán đc bnhiu tiền rồi?', 'ad oi th5/2026 shop mik bn don z ha', 'Th 9/25 khách mô mua nhiều đơn nhất rứa?',
              'cko e coi tổng tiền hàng trog th4/2025 jùm mik nha']:
        assert U(q) == [], q


def test_unlinked_entity_penalized():
    ix = _ix()
    assert len(U('Tháng 1/2026 bán Hoa quả nghiền HiPPiS Organic 100g được bao nhiêu tiền?')) >= 2   # tên riêng + 100g
    assert U('doanh thu ca phe sua thang 9 la bao nhieu', ix=ix)                                     # cụm có trong tên hàng
    assert U('doanh thu ca phe sua thang 9 la bao nhieu', linked=['Cà phê sữa (M)'], ix=ix) == []    # đã nối
    assert U('thang 9 khach to dung thao mua bao nhieu tien', ix=ix)                                 # tên khách chưa nối
    assert U('doanh thu thang 9 la bao nhieu', ix=ix) == []
    assert U('doanh thu tu 3tr7 tro len, 500k, 5h, q1, th3/2025') == []                              # tiền / giờ / kỳ


def test_money_shorthand_threshold():
    assert _money_threshold('doanh thu dat tu 3tr7 tro len') == (3_700_000, False)
    assert _money_threshold('dat tu 11 cu tro len') == (11_000_000, False)


def test_listed_periods_are_series():
    r = P('cho xin doanh thu th7/2025, th8/2025, th9/2025 nhé')
    assert r.plan.op == 'series' and len(r.plan.periods) == 3
    r = P('Năm nay, tính từ đầu năm đến hết hôm nay, doanh thu của cửa hàng là bao nhiêu?')
    assert r.plan.op != 'series'
