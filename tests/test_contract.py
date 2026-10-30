"""Hợp đồng kiểu đáp án suy từ dạng câu hỏi."""
from datamate.understand.expected_type import compatible, expected_types
from datamate.understand.normalize import qnorm


def E(q):
    return expected_types(qnorm(q))


def test_expected_types():
    assert compatible('string', E('Ai là khách mua nhiều lần nhất tháng 5?'))
    assert not compatible('number', E('Ai là khách mua nhiều lần nhất tháng 5?'))
    assert compatible('percent', E('Doanh thu tháng 5 tăng/giảm bao nhiêu % so với tháng 4?'))
    assert not compatible('number', E('Doanh thu tháng 5 tăng/giảm bao nhiêu % so với tháng 4?'))
    assert compatible('date', E('Ngày nào trong tháng 6 có doanh thu cao nhất?'))
    # '%' trong tên hàng không phải câu hỏi phần trăm
    assert compatible('number', E('Tháng 7/2025 Otrivin 0,1% xịt mũi bán được bao nhiêu chai?'))
    assert not compatible('percent', E('Tháng 7/2025 Otrivin 0,1% xịt mũi bán được bao nhiêu chai?'))
    assert compatible('unanswerable', E('Ai là khách mua nhiều lần nhất tháng 5?'))
    assert E('xin chào') is None
