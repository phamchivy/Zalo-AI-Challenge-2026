"""v3.1.0: LLM cho câu tầng A bỏ trống — hàm hỗ trợ tổng quát, đếm phiếu (tầng A = 1 phiếu thường), khoá API từ env."""
import asyncio
import json
import pickle
from datetime import date
from pathlib import Path

import pandas as pd

from datamate.agent import helpers as h
from datamate.core.config import DecisionConfig, LLMConfig, load_config
from datamate.core.types import AnswerType, Candidate
from datamate.decision.consensus import consensus
from datamate.runtime.llm_client import LLMClient, MockLLM
from datamate.runtime.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


# ----------------------------------------------------------------------------- hàm hỗ trợ
def test_find_matches_without_accents_and_abbrev():
    vals = pd.Series(['Sữa Ông Thọ hộp 380g', 'Sữa đặc Ngôi Sao', 'Bánh Oreo', 'Sữa Ông Thọ hộp 380g', None])
    assert h.find(vals, 'sua ong tho')[0] == 'Sữa Ông Thọ hộp 380g'
    assert h.find(vals, 'oreo') == ['Bánh Oreo']
    assert h.find(vals, 'xe máy') == []
    s = h.find(['Chiếc', 'Hộp', 'Vỉ'], 'chiec', scores=True)
    assert s[0][0] == 'Chiếc' and s[0][1] >= 0.98


def test_period_uses_tier_a_time_reader():
    r = h.period('tháng trước', date(2026, 3, 15))
    assert (r[0]['start'], r[0]['end']) == (pd.Timestamp('2026-02-01'), pd.Timestamp('2026-02-28'))
    r = h.period('từ 03/05/2026 đến 10/05/2026', '2026-06-01')
    assert (r[0]['start'], r[0]['end']) == (pd.Timestamp('2026-05-03'), pd.Timestamp('2026-05-10'))
    assert h.period('món nào ngon', date(2026, 3, 15)) == []


def test_lunar_calendar_helpers(tmp_path):
    p = tmp_path / 'calendar.csv'
    p.write_text('date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n'
                 '2026-02-17,2,1,1,2026,False,Tết Nguyên Đán\n2026-02-18,3,2,1,2026,False,\n', encoding='utf-8')
    assert h.lunar('2026-02-17', str(p))['le'] == 'Tết Nguyên Đán'
    assert h.solar(2026, 1, 2, calendar_path=str(p)) == pd.Timestamp('2026-02-18')
    assert h.holidays('tet', str(p)) == [pd.Timestamp('2026-02-17')]


def test_describe_lists_columns():
    txt = h.describe(pd.DataFrame({'a': [1, 2, None], 'b': ['x', 'x', 'y']}), 'T')
    assert 'T: 3 dòng' in txt and "- b [" in txt and "'x'" in txt


def test_kernel_exposes_helpers(tmp_path):
    from datamate.agent.kernel import KernelPool
    pk = tmp_path / 'p.pkl'
    pickle.dump({'tables': {'Sổ': pd.DataFrame({'Làm gì': ['thay nhớt', 'bảo hành']})},
                 'canon': {'products': pd.DataFrame({'product_name': ['Sữa Ông Thọ', 'Bánh Oreo']})}, 'meta': {}},
                open(pk, 'wb'))

    async def run():
        kp = KernelPool(timeout_sec=20, per_shop=1, max_workers=1, mem_gb=4)
        r = await kp.execute(str(pk), 's', "peek()\nprint(find(canon['products']['product_name'], 'ong tho'))\n"
                                           "print(period('tháng này')[0]['start'])",
                             meta={'as_of_date': '2026-07-20', 'calendar_path': ''})
        await kp.close()
        return r
    r = asyncio.run(run())
    assert r.status == 'ok', r.error
    assert 'Sổ' in r.stdout and "['Sữa Ông Thọ']" in r.stdout and '2026-07-01' in r.stdout


# ----------------------------------------------------------------------------- đếm phiếu
def _c(strategy, t, a, status='ok'):
    return Candidate(cand_id=strategy, strategy=strategy, status=status, answer_type=AnswerType(t) if t else None, answer=a)


LLM = {'agent'}
CFG = DecisionConfig(min_votes=2, unanswerable_min_votes=3)


def test_consensus_two_of_three():
    d = consensus('q', [_c('agent', 'number', 10), _c('agent', 'number', 10.0), _c('agent', 'number', 12)], CFG, LLM)
    assert d.type == AnswerType.NUMBER and d.answer == 10 and d.policy.startswith('consensus(2/3')


def test_consensus_tier_a_is_one_ordinary_vote_without_veto():
    # tầng A khác đáp án nhưng 2 lời giải LLM đồng ý -> vẫn trả lời theo LLM (tầng A không phủ quyết)
    d = consensus('q', [_c('rule_plan', 'number', 5), _c('agent', 'number', 7), _c('agent', 'number', 7),
                        _c('agent', 'number', 9)], CFG, LLM)
    assert d.type == AnswerType.NUMBER and d.answer == 7
    # tầng A + 1 LLM đồng ý = 2 phiếu -> trả lời
    d = consensus('q', [_c('rule_plan', 'number', 5), _c('agent', 'number', 5), _c('agent', 'number', 9),
                        _c('agent', 'number', 8)], CFG, LLM)
    assert d.answer == 5
    # tầng A một mình (LLM lỗi hết) -> không trả lời
    d = consensus('q', [_c('rule_plan', 'number', 5), _c('agent', None, None, status='error')], CFG, LLM)
    assert d.type == AnswerType.ABSTAIN


def test_consensus_tie_and_unanswerable():
    d = consensus('q', [_c('agent', 'number', 1), _c('agent', 'number', 1), _c('agent', 'number', 2),
                        _c('rule_plan', 'number', 2)], CFG, LLM)
    assert d.type == AnswerType.ABSTAIN                                  # hoà 2-2
    d = consensus('q', [_c('agent', 'unanswerable', None), _c('agent', 'unanswerable', None), _c('agent', 'number', 3)],
                  CFG, LLM)
    assert d.type == AnswerType.ABSTAIN                                  # unanswerable cần 3 phiếu
    d = consensus('q', [_c('agent', 'unanswerable', None)] * 3, CFG, LLM)
    assert d.type == AnswerType.UNANSWERABLE


def test_llm_key_from_env(monkeypatch):
    monkeypatch.setenv('DM_TEST_KEY', 'k123')
    assert LLMClient(LLMConfig(api_key_env='DM_TEST_KEY')).key == 'k123'
    assert LLMClient(LLMConfig()).key == 'EMPTY'


def test_qa_configs_load_and_differ_only_in_model_blocks():
    g, q = load_config(str(ROOT / 'configs' / 'qa_glm.yaml')), load_config(str(ROOT / 'configs' / 'qa_qwen14b.yaml'))
    assert g.decision.model_copy(update={'min_votes': 0}) == q.decision.model_copy(update={'min_votes': 0})
    assert [s.name for s in g.strategies] == [s.name for s in q.strategies]
    assert g.llm.api_key_env == 'DM_LLM_KEY' and not g.llm.health_check and q.llm.model == 'qwen3-14b'
    assert 'sk-' not in (ROOT / 'configs' / 'qa_glm.yaml').read_text(encoding='utf-8')


# ----------------------------------------------------------------------------- 2 pha end-to-end (LLM giả)
def _qa_cfg(cache):
    return load_config(str(ROOT / 'configs' / 'qa_glm.yaml'), {
        'cache_dir': str(cache), 'workers': 1, 'question_concurrency': 2, 'mapping_llm': {'enabled': False},
        'llm': {'health_check': True, 'wait_ready_sec': 1},
        'strategies': [{'name': 'guard', 'weight': 1.0}, {'name': 'rule_plan', 'weight': 1.0},
                       {'name': 'agent', 'weight': 1.0, 'samples': 3,
                        'options': {'max_steps': 3, 'shop_notes': False, 'cell_timeout': 20,
                                    'kernel': {'per_shop': 1, 'max_workers': 2, 'mem_gb': 4}}}],
        'decision': {'calibration_path': None}})


def test_only_abstained_questions_go_to_llm_and_consensus_decides(mini_split, tmp_path):
    seen = []

    def respond(messages, schema):
        seen.append(messages[-1]['content'][:80])
        return "Trả lời.\n```python\nprint(peek())\nanswer('number', 4242)\n```"
    qs = [json.loads(l) for l in (mini_split / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    out = tmp_path / 'out'
    s = asyncio.run(Orchestrator(_qa_cfg(tmp_path / 'cache'), str(mini_split), str(out), llm=MockLLM(respond))
                    .run(resume=False))
    traces = [json.loads(p.read_text(encoding='utf-8')) for p in (out / 'logs').glob('*.json')]
    with_llm = [t for t in traces if any(c['strategy'] == 'agent' for c in t['candidates'])]
    without = [t for t in traces if t not in with_llm]
    assert with_llm, 'phải có câu tầng A bỏ trống được chuyển cho LLM'
    assert s['llm_questions'] == len(with_llm)
    # câu tầng A đã trả lời: không gọi LLM, giữ nguyên đáp án tầng A
    assert all(t['decision']['type'] != 'abstain' or 'crash' in (t['decision']['policy'] or '') for t in without)
    # câu gọi LLM: 3 lời giải cùng 4242 -> trả lời bằng đếm phiếu
    for t in with_llm:
        assert t['decision']['policy'].startswith('consensus(') and t['decision']['answer'] == 4242, t['decision']
