"""RuleSet: đọc luật từ rulebook, phát hiện luật lạ, định nghĩa xếp hạng, ghi đè bằng rules.json."""
from datamate.knowledge.rules import RuleSet, compile_rules

from conftest import RULEBOOK

EXTRA = """
## 3. Xếp hạng bổ sung
| Luật | Mặc định |
|---|---|
| Ngày đắt khách nhất | ngày có **nhiều đơn nhất** trong kỳ |
| Giờ cao điểm | khung **một giờ tròn** theo giờ tạo đơn có nhiều đơn nhất trong kỳ |
| Khách sộp nhất | khách có **giá trị trung bình một đơn** cao nhất trong kỳ, chỉ xét khách có **từ 3 đơn trở lên** trong kỳ |
## 6. Theo ngành
### Cửa hàng điện thoại
| Luật | Mặc định |
|---|---|
| Thu máy cũ | Khách phải trả là tiền của đơn sau bớt và sau trừ máy cũ |
"""


def test_knobs_from_rulebook():
    rs = compile_rules(RULEBOOK)
    assert rs.get('revenue.valid_status') == ['completed', 'shipping']
    assert rs.get('aov.rounding') == 'half_up'
    assert rs.get('cancel_rate.denominator') == 'all_orders'
    assert rs.get('purchase.avg_price') is None            # rulebook không nói -> không đoán
    assert rs.unknown == []


def test_changed_rule_is_unknown():
    rs = compile_rules(RULEBOOK.replace('nửa đồng làm tròn lên', 'làm tròn xuống'))
    assert rs.get('aov.rounding') is None


def test_rankings_and_unknown():
    rs = compile_rules(RULEBOOK + EXTRA)
    r = rs.ranking_for('ngay dat khach nhat thang 6/2025')
    assert r and r.entity == 'day' and r.metric == 'orders'
    r = rs.ranking_for('gio cao diem thang 6 la may gio')
    assert r and r.entity == 'hour'
    r = rs.ranking_for('khach sop nhat quy 2')
    assert r and r.metric == 'aov' and r.min_orders == 3
    assert rs.ranking_for('khach trung thanh nhat').metric == 'active_days'
    u = rs.touches_unknown('tien thu may cu thang 6 la bao nhieu')
    assert u and u['name'] == 'thu may cu'
    assert rs.touches_unknown('doanh thu thang 6') is None


def test_override_roundtrip():
    rs = compile_rules(RULEBOOK)
    ov = RuleSet(knobs={'aov.rounding': None, 'purchase.avg_price': 'qty_weighted'}, source='llm')
    m = rs.merged(RuleSet.from_json(ov.to_json()))
    assert m.get('aov.rounding') is None and m.get('purchase.avg_price') == 'qty_weighted'
    assert m.get('revenue.valid_status') == ['completed', 'shipping']
