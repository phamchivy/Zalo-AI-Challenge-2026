from datamate.core.config import DecisionConfig
from datamate.core.types import AnswerType, Candidate
from datamate.decision.voter import decide
from datamate.offline.scorer import score_one
from datamate.output.formatter import normalize_answer


def test_scorer_rules():
    assert score_one({'type': 'number', 'answer': 48250000}, {'type': 'number', 'answer': 48250000}) == 1
    assert score_one({'type': 'percent', 'answer': -12.449999}, {'type': 'percent', 'answer': -12.45}) == 1
    assert score_one({'type': 'string', 'answer': 'sting dâu 330ML'}, {'type': 'string', 'answer': 'Sting dâu 330ml'}) == 1
    assert score_one({'type': 'string', 'answer': 'Sting dau 330ml'}, {'type': 'string', 'answer': 'Sting dâu 330ml'}) == -1
    assert score_one({'type': 'string', 'answer': 'SP0042'}, {'type': 'string', 'answer': 'X', 'alias': ['SP0042']}) == 1
    assert score_one({'type': 'abstain'}, {'type': 'number', 'answer': 1}) == 0
    assert score_one({'type': 'number', 'answer': 1}, {'type': 'unanswerable', 'answer': None}) == -1
    t = {'columns': ['khach_hang', 'so_tien_no'], 'rows': [['Chị Lan', 3200000], ['Anh Tú', 1500000]]}
    p = {'columns': ['khach_hang', 'so_tien_no'], 'rows': [['Anh Tú', 1500000], ['chị lan', 3200000]]}
    assert score_one({'type': 'table', 'answer': p}, {'type': 'table', 'answer': t}) == 1
    assert score_one({'type': 'list', 'answer': ['A', 'B']}, {'type': 'list', 'answer': ['B', 'A']}) == -1


def test_formatter_and_voter():
    assert normalize_answer('number', '4,820,000') == ('number', 4820000)
    assert normalize_answer('percent', 12.4567) == ('percent', 12.46)
    assert normalize_answer('date', '15/08/2026') == ('date', '2026-08-15')
    cfg = DecisionConfig(min_score=0.5, min_share=0.6, unanswerable_min_score=0.9)
    c = [Candidate(cand_id='a', strategy='rule_plan', weight=0.9, answer_type=AnswerType.NUMBER, answer=10),
         Candidate(cand_id='b', strategy='llm_code', weight=0.5, answer_type=AnswerType.NUMBER, answer=10.0)]
    d = decide('q', c, cfg)
    assert d.type == AnswerType.NUMBER and d.answer == 10
    c2 = [Candidate(cand_id='a', strategy='rule_plan', weight=0.5, answer_type=AnswerType.UNANSWERABLE)]
    assert decide('q', c2, cfg).type == AnswerType.ABSTAIN
    c3 = [Candidate(cand_id='a', strategy='x', weight=0.5, answer_type=AnswerType.NUMBER, answer=1),
          Candidate(cand_id='b', strategy='y', weight=0.5, answer_type=AnswerType.NUMBER, answer=2)]
    assert decide('q', c3, cfg).type == AnswerType.ABSTAIN
