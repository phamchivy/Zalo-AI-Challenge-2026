"""v2.5.0: đọc số nhóm nghìn bằng khoảng trắng, trạng thái từ Ghi chú, chặn theo rủi ro dữ liệu, kiểm hợp lý, I5."""
import pandas as pd

from datamate.core.textutil import to_num
from datamate.core.types import AnswerType, Decision
from datamate.data.invariants import inv_unit_factor
from datamate.data.vocab import status_from_note
from datamate.decision.data_guard import blocked_by, guard


def test_space_grouped_numbers():
    assert to_num('1 851 000') == 1_851_000 and to_num('10 000') == 10_000 and to_num('113 889') == 113_889
    assert to_num('1 851 000đ') == 1_851_000 and to_num('(5 000)') == -5000
    assert to_num('1 gói') == 1 and to_num('2 100ml') == 2          # số lượng kèm đơn vị: không gộp


def test_status_from_note_shared_vocab():
    assert status_from_note('cancel') == 'cancelled' and status_from_note('Refund') == 'returned'
    assert status_from_note('trả tiền mặt') is None and status_from_note('giao gấp') is None


def test_guard_blocks_only_dependent_intents():
    lv = {'I2_goods_total': 'red', 'DATE_parsed': 'green'}
    assert blocked_by('debt_total', lv) == 'I2_goods_total'
    assert blocked_by('revenue', lv) is None
    assert blocked_by('revenue', {'I15_status_mapped': 'red'}) == 'I15_status_mapped'
    assert blocked_by('purchase_total', {'I15_status_mapped': 'red'}) is None     # câu nhập hàng không dựa trên đơn
    d = Decision(qid='q', type=AnswerType.NUMBER, answer=5)
    assert guard(d, 'debt_total', lv).type == AnswerType.ABSTAIN
    assert guard(d, 'revenue', lv) is d
    u = Decision(qid='q', type=AnswerType.UNANSWERABLE)
    assert guard(u, 'debt_total', lv) is u


def test_implausible_answers():
    assert guard(Decision(qid='q', type=AnswerType.NUMBER, answer=-3), 'revenue', {}).type == AnswerType.ABSTAIN
    assert guard(Decision(qid='q', type=AnswerType.PERCENT, answer=130), 'share', {}).type == AnswerType.ABSTAIN
    ok = Decision(qid='q', type=AnswerType.PERCENT, answer=-12.5)
    assert guard(ok, 'compare_period', {}) is ok                                  # tăng trưởng âm là hợp lệ


def test_unit_factor_invariant():
    L = pd.DataFrame({'unit_f': ['thung'] * 50 + ['goi'] * 50, 'base_unit_f': ['goi'] * 100,
                      'factor': [24.0] * 50 + [1.0] * 50, 'product_name': 'X', 'unit': 'u', 'base_unit': 'b', 'qty': 1})
    assert inv_unit_factor(L, pd.DataFrame({'a': [1]})).level == 'green'
    L.loc[:49, 'factor'] = 1.0                                                     # bảng quy đổi không gán được
    assert inv_unit_factor(L, pd.DataFrame()).level == 'red'
