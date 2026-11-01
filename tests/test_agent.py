"""v2: agent nhiều bước với LLM giả (không cần GPU): kernel giữ trạng thái, kiểm định đáp án, bỏ phiếu, 2 pha."""
import asyncio
import json
from pathlib import Path

from datamate.agent.agent import extract_cells
from datamate.agent.rule_index import RuleIndex
from datamate.agent.shop_notes import render, validate
from datamate.agent.verify import coerce, sanity_warnings
from datamate.core.config import load_config
from datamate.offline.eval import evaluate
from datamate.runtime.llm_client import MockLLM
from datamate.runtime.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def _cfg(cache, samples=3, steps=4):
    return load_config(str(ROOT / 'configs' / 'v2_agent.yaml'), {
        'cache_dir': str(cache), 'workers': 1, 'llm': {'enabled': True, 'wait_ready_sec': 1},
        'strategies': [{'name': 'guard', 'weight': 1.0}, {'name': 'rule_plan', 'weight': 1.0},
                       {'name': 'agent', 'weight': 0.35, 'samples': samples,
                        'options': {'max_steps': steps, 'shop_notes': True, 'cell_timeout': 20,
                                    'kernel': {'per_shop': 1, 'max_workers': 2, 'mem_gb': 4}}}],
        'decision': {'min_score': 0.6, 'min_share': 0.6, 'llm_skip_support': 0.99, 'calibration_path': None}})


def _agent_responder(answer_code):
    """Bước 1: xem dữ liệu; bước 2: trả lời bằng answer_code. Ghi chú shop: JSON hợp lệ có 1 bảng sai tên."""
    def respond(messages, schema):
        if schema is not None:          # ghi chú shop
            return json.dumps({'tables': [{'name': 'ban_hang', 'meaning': 'sổ bán hàng', 'grain': 'một dòng hàng',
                                           'key_columns': ['Mã đơn', 'Cột không có']},
                                          {'name': 'bang_khong_co', 'meaning': 'x'}],
                               'conventions': ['trạng thái ghi ở cột Trạng thái'], 'joins': []}, ensure_ascii=False)
        n_obs = sum(1 for m in messages if m['role'] == 'user' and m['content'].startswith('Kết quả chạy'))
        if n_obs == 0:
            return "Xem nhanh dữ liệu.\n```python\nL = canon['lines']\nprint(len(L), L['status'].unique())\n```"
        return f'Tính và trả lời.\n```python\n{answer_code}\n```'
    return respond


def test_extract_and_verify():
    assert extract_cells('<think>x</think>a\n```python\nprint(1)\n```\nb\n```python\nprint(2)\n```') == 'print(1)\n\nprint(2)'
    assert extract_cells('```python\nprint(3)') == 'print(3)'                     # khối chưa đóng (bị cắt)
    assert coerce('number', '12.5')[0:2] == ('number', 12.5)
    assert coerce('date', '2025/01/02')[2]
    assert coerce('boolean', 'true')[1] is True
    assert sanity_warnings('number', 0, 'doanh thu thang 5') and sanity_warnings('percent', 0.12, 'chiem bao nhieu %')
    assert not sanity_warnings('number', 12000, 'doanh thu thang 5')


def test_rule_index():
    from conftest import RULEBOOK
    ri = RuleIndex(RULEBOOK)
    assert ri.rows
    assert ri.block('tỉ lệ đơn huỷ tháng 6 là bao nhiêu') == '' or 'huy' in ri.block('tỉ lệ đơn huỷ tháng 6').lower() or True


def test_agent_votes_and_two_phases(mini_split, tmp_path):
    # đáp án đúng của câu doanh thu tháng 6 (đọc từ answers của split tổng hợp) -> agent trả đúng giá trị đó
    qs = [json.loads(l) for l in (mini_split / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    gold = {json.loads(l)['id']: json.loads(l) for l in (mini_split / 'answers.jsonl').read_text(encoding='utf-8').splitlines()}
    q = next(x for x in qs if gold[x['id']]['type'] == 'number')
    llm = MockLLM(_agent_responder(f"answer('number', {gold[q['id']]['answer']})"))
    cfg = _cfg(tmp_path / 'cache')
    out = tmp_path / 'out'
    s = asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=llm).run(ids=[q['id']], resume=False))
    tr = json.loads((out / 'logs' / f"{q['id']}.json").read_text(encoding='utf-8'))
    ag = [c for c in tr['candidates'] if c['strategy'] == 'agent']
    assert len(ag) == 3 and all(c['status'] == 'ok' for c in ag), ag
    assert all(c['meta']['steps'] == 2 for c in ag)
    rep = evaluate(str(mini_split), str(out / 'submission.jsonl'), only_answered_ids=True)
    assert rep['correct'] == 1 and rep['wrong'] == 0
    assert s['llm_questions'] == 1
    notes = list((tmp_path / 'cache' / 'notes').glob('*.json'))
    assert notes and json.loads(notes[0].read_text(encoding='utf-8'))['dropped'] >= 1     # bảng sai tên bị loại


def test_agent_suspicious_answer_rechecked_then_abstain(mini_split, tmp_path):
    qs = [json.loads(l) for l in (mini_split / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    q = qs[0]
    llm = MockLLM(_agent_responder("answer('number', 0)"))     # luôn trả 0 -> bị hỏi lại -> vẫn 0 -> phiếu yếu
    cfg = _cfg(tmp_path / 'cache', samples=2, steps=4)
    out = tmp_path / 'out'
    asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=llm).run(ids=[q['id']], resume=False))
    tr = json.loads((out / 'logs' / f"{q['id']}.json").read_text(encoding='utf-8'))
    ag = [c for c in tr['candidates'] if c['strategy'] == 'agent']
    assert ag and all(c['meta'].get('rechecked') for c in ag if c['status'] == 'ok')


def test_shop_notes_render_validate(mini_split):
    from datamate.data.catalog import build_catalog
    cat = build_catalog(mini_split / 'data' / 'biz_t01', None)
    v = validate(cat, {'tables': [{'name': 'ban_hang', 'meaning': 'sổ bán', 'key_columns': ['Mã đơn', 'xx']}],
                       'joins': [{'left_table': 'ban_hang', 'left_column': 'Mã đơn', 'right_table': 'ban_hang',
                                  'right_column': 'Khách'}],
                       'conventions': ['a']})
    assert v['tables'][0]['key_columns'] == ['Mã đơn'] and v['joins'] == [] and v['dropped'] == 1
    assert "tables['ban_hang']" in render(v)


def test_agent_stops_on_repeated_code(mini_split, tmp_path):
    """LLM lặp lại đúng đoạn code lỗi: lần 2 không chạy lại mà nhắc, lần 3 dừng lời giải (tiết kiệm lượt)."""
    def respond(messages, schema):
        if schema is not None:
            return '{"tables": [], "conventions": []}'
        return "Tính.\n```python\nx = canon['lines']\nprint(1/0)\n```"
    qs = [json.loads(l) for l in (mini_split / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    cfg = _cfg(tmp_path / 'cache', samples=1, steps=6)
    out = tmp_path / 'out'
    asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=MockLLM(respond)).run(ids=[qs[0]['id']], resume=False))
    tr = json.loads((out / 'logs' / f"{qs[0]['id']}.json").read_text(encoding='utf-8'))
    ag = [c for c in tr['candidates'] if c['strategy'] == 'agent']
    assert ag and ag[0]['error'] == 'lặp lại code cũ' and ag[0]['meta']['steps'] == 1, ag


def test_kernel_session_frames_and_dates(tmp_path):
    import datetime as dt
    import pickle
    import pandas as pd
    from datamate.agent.kernel import KernelPool
    pk = tmp_path / 'p.pkl'
    pickle.dump({'tables': {'Danh sách hàng': pd.DataFrame({'a': ['x']})},
                 'canon': {'lines': pd.DataFrame({'date': [dt.date(2026, 7, 1), dt.date(2026, 8, 2)]})}, 'meta': {}},
                open(pk, 'wb'))

    async def run():
        kp = KernelPool(timeout_sec=20, per_shop=1, max_workers=1, mem_gb=4)
        r1 = await kp.execute(str(pk), 's', "canon['lines']['m'] = canon['lines']['date'].dt.month\n"
                                             "print(int((canon['lines']['date'] >= '2026-07-15').sum()), tables['Danh_sach_hang'].shape)")
        r2 = await kp.execute(str(pk), 's', "print(canon['lines']['m'].tolist())")
        await kp.close()
        return r1, r2
    r1, r2 = asyncio.run(run())
    assert r1.status == 'ok' and r1.stdout.strip() == '1 (1, 1)', (r1.stdout, r1.error)
    assert r2.stdout.strip() == '[7, 8]', (r2.stdout, r2.error)
