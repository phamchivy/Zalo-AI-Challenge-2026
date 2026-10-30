from datetime import date

from datamate.core.textutil import customer_key, fold, norm_answer_str, parse_date, parse_time, to_num


def test_to_num():
    cases = {'560,000': 560000, '38000': 38000, '1.200.000đ': 1200000, '1 gói': 1, '18.6': 18.6, '-12,5': -12.5,
             '407k': 407000, '2,176tr': 2176000, '11.000đ': 11000, '225,000.0000': 225000, '(5.000)': -5000, '5 m': 5}
    for s, v in cases.items():
        assert abs(to_num(s) - v) < 1e-9, s
    assert to_num('') is None and to_num(None) is None and to_num('abc') is None


def test_parse_date():
    assert parse_date('13 thg 9 2025') == date(2025, 9, 13)
    assert parse_date('01/02/2025') == date(2025, 2, 1)
    assert parse_date('2025-06-01 10:00:00') == date(2025, 6, 1)
    assert parse_date('45806') == date(2025, 5, 29)
    assert parse_date('05/03', default_year=2026) == date(2026, 3, 5)
    assert parse_time('2025-01-01 06:03:04') == '06:03:04'


def test_fold_and_norm():
    assert fold('Đã Hủy  đơn') == 'da huy don'
    assert norm_answer_str('Sting dâu-330ml!') == 'sting dâu 330ml'
    assert customer_key('Cao Long Minh (0321806537)') == 'cao long minh'
    assert customer_key('Hoàng Trung Hương - 0971 618 698') == 'hoang trung huong'
