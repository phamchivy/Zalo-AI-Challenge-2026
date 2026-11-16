"""v3.3.3: LLM kiểm tra kế hoạch tầng A (in ra câu chữ) so với câu hỏi + rulebook; bác -> bỏ trống -> LLM giải."""
import asyncio
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from datamate.core.config import load_config
from datamate.core.types import Plan, TimeRange
from datamate.knowledge.rules import RuleSet
from datamate.runtime.llm_client import MockLLM
from datamate.runtime.orchestrator import Orchestrator
from datamate.understand import plan_check as pc

ROOT = Path(__file__).resolve().parents[1]


def test_render_shows_dates_weekdays_matched_text_and_rules():
    p = Plan(metric='revenue', op='value', periods=[TimeRange(start=date(2026, 2, 1), end=date(2026, 2, 28), expr='2/2026',
                                                              granularity='month')])
    rs = RuleSet(knobs={'revenue.valid_status': ['completed'], 'revenue.period_by': 'order_date', 'time.month': 'full'},
                 evidence={'revenue.period_by': '| Đơn thuộc kỳ nào | theo ngày đặt |'})
    t = pc.render_plan(p, rs, 'number', 123)
    assert 'doanh thu' in t and '01/02/2026 (Chủ nhật) → 28/02/2026 (thứ Bảy)' in t and '"2/2026"' in t
    assert 'revenue.period_by = order_date' in t and 'theo ngày đặt' in t and 'time.month' in t
    t = pc.render_plan(Plan(metric='orders', op='count_ge', group_by='day', threshold=180000, threshold_metric='revenue',
                            periods=[TimeRange(start=date(2026, 3, 1), end=date(2026, 3, 31))]))
    assert 'đếm số ngày có doanh thu (tiền bán hàng) của mỗi ngày ≥ 180000' in t


def test_parse_rejects_only_with_concrete_mismatch():
    bad = json.dumps({'dung': False, 'loi': [{'phan': 'thoi_gian', 'cau_hoi_noi': 'quý 2/2026', 'ke_hoach_lam': 'tháng 2/2026'}]})
    assert pc._parse(bad)['reject']
    assert not pc._parse(json.dumps({'dung': False, 'loi': []}))['reject']                  # không chỉ ra chỗ lệch
    assert not pc._parse(json.dumps({'dung': False, 'loi': [{'phan': 'khac', 'cau_hoi_noi': 'x', 'ke_hoach_lam': 'y'}]}))['reject']
    assert not pc._parse(json.dumps({'dung': True, 'loi': []}))['reject']
    assert pc._parse('không phải json') is None


def test_check_one_needs_min_reject_votes(tmp_path):
    seq = iter([True, False])

    def respond(messages, schema):
        if next(seq):
            return json.dumps({'dung': False, 'loi': [{'phan': 'dai_luong', 'cau_hoi_noi': 'số đơn', 'ke_hoach_lam': 'số lượng'}]})
        return json.dumps({'dung': True, 'loi': []})
    cfg = SimpleNamespace(samples=2, min_reject=2, max_tokens=100, temperature=0.3, rulebook_chars=1000)
    v = asyncio.run(pc.check_one(MockLLM(respond), 'sl đh hôm qua', date(2026, 3, 2), 'x', 'RB', cfg, None))
    assert v['n_reject'] == 1 and not v['reject']
    cfg.min_reject = 1
    v = asyncio.run(pc.check_one(MockLLM(lambda m, s: json.dumps({'dung': False, 'loi': [
        {'phan': 'dai_luong', 'cau_hoi_noi': 'số đơn', 'ke_hoach_lam': 'số lượng'}]})), 'q', date(2026, 3, 2), 'x', 'RB', cfg,
        str(tmp_path)))
    assert v['reject'] and v['errors'][0]['phan'] == 'dai_luong'
    assert any((tmp_path / 'plan_check').iterdir())                                      # có cache


def _run(mini_split, tmp_path, verdict, cfg_file='canon_glm.yaml', agent_answer=None):
    def respond(messages, schema):
        u, s = messages[-1]['content'], messages[0]['content']
        if u.startswith('Câu gốc: '):
            return json.dumps({'q': u[len('Câu gốc: '):]}, ensure_ascii=False)
        if s.startswith('Bạn là người KIỂM TRA'):
            assert 'RULEBOOK' in s and 'KẾ HOẠCH TÍNH' in u
            return json.dumps(verdict, ensure_ascii=False)
        return agent_answer(u) if agent_answer else '```python\nanswer("abstain")\n```'
    over = {'cache_dir': str(tmp_path / 'cache'), 'workers': 1, 'mapping_llm': {'enabled': False},
            'decision': {'calibration_path': None}, 'canon_q': {'concurrency': 4}, 'plan_check': {'enabled': True}}
    if cfg_file == 'qa_glm.yaml':
        over['strategies'] = [{'name': 'guard', 'weight': 1.0}, {'name': 'rule_plan', 'weight': 1.0},
                              {'name': 'agent', 'weight': 1.0, 'samples': 3,
                               'options': {'max_steps': 2, 'shop_notes': False, 'cell_timeout': 20, 'code_lint': False,
                                           'kernel': {'per_shop': 1, 'max_workers': 2, 'mem_gb': 4}}}]
    cfg = load_config(str(ROOT / 'configs' / cfg_file), over)
    out = tmp_path / 'out'
    s = asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=MockLLM(respond)).run(resume=False))
    tr = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (out / 'logs').glob('*.json')}
    return s, tr


def test_accepted_plan_keeps_tier_a(mini_split, tmp_path):
    s, tr = _run(mini_split, tmp_path, {'dung': True, 'loi': []})
    checked = [t for t in tr.values() if t.get('plan_check')]
    assert checked and s['by_type'].get('plan_ok') == len(checked) and not s['by_type'].get('plan_reject')
    assert all(t['decision']['type'] != 'abstain' for t in checked)
    assert all('KẾ HOẠCH' not in t['plan_check']['plan_text'] and t['plan_check']['tier_a'] for t in checked)


def test_rejected_plan_abstains_without_agent(mini_split, tmp_path):
    bad = {'dung': False, 'loi': [{'phan': 'thoi_gian', 'cau_hoi_noi': 'quý 2', 'ke_hoach_lam': 'tháng 2'}]}
    s, tr = _run(mini_split, tmp_path, bad)
    rej = [t for t in tr.values() if (t.get('plan_check') or {}).get('reject')]
    assert rej and all(t['decision']['type'] == 'abstain' and t['decision']['policy'] == 'plan_check_reject' for t in rej)
    assert s['by_type']['plan_reject'] == len(rej)


def test_rejected_plan_goes_to_llm_and_tier_a_still_votes(mini_split, tmp_path):
    bad = {'dung': False, 'loi': [{'phan': 'dai_luong', 'cau_hoi_noi': 'số đơn', 'ke_hoach_lam': 'doanh thu'}]}
    seen = []

    def agent(u):
        seen.append(u)
        return "```python\nanswer('number', 4242)\n```"
    s, tr = _run(mini_split, tmp_path, bad, 'qa_glm.yaml', agent)
    rej = [t for t in tr.values() if (t.get('plan_check') or {}).get('reject')]
    assert rej and s['llm_questions'] >= len(rej)
    assert all(any(c['strategy'] == 'agent' for c in t['candidates']) for t in rej)
    # câu kiểu số: 3 lời giải LLM cùng 4242 -> trả lời theo đếm phiếu (câu kiểu khác bị hợp đồng kiểu đáp án chặn)
    ok = [t for t in rej if t['decision']['policy'].startswith('consensus(')]
    assert ok and all(t['decision']['answer'] == 4242 for t in ok)
    assert any('máy kiểm tra cho rằng' in u for u in seen)                      # lý do bác được đưa cho LLM giải


def test_configs_ship_plan_check_disabled():
    for f in ('qa_glm', 'canon_glm', 'qa_qwen14b', 'qa_qwen14b_or'):
        c = load_config(str(ROOT / 'configs' / f'{f}.yaml'))
        assert not c.plan_check.enabled and c.plan_check.samples == 2 and c.plan_check.min_reject == 2


def test_render_matches_what_executor_computes():
    # v3.3.4: price_effect -> in đúng 2 cửa sổ N ngày executor tính (không phải "1 ngày D")
    p = Plan(metric='price_effect', op='value', filters={'product': 'latte (l)'}, threshold=30,
             periods=[TimeRange(start=date(2026, 2, 5), end=date(2026, 2, 5), expr='05/02/2026')])
    t = pc.render_plan(p, RuleSet(knobs={'price_effect.def': 'n_from_d_vs_n_before'}))
    assert '05/02/2026 (thứ Năm) → 06/03/2026' in t and '06/01/2026 (thứ Ba) → 04/02/2026' in t and 'N = 30' in t
    assert '1 ngày' not in t
    # count_ge: dấu so sánh thật ghi ở dòng phép tính, không liệt kê threshold.strict
    p = Plan(metric='orders', op='count_ge', group_by='product', threshold=187, threshold_metric='orders')
    t = pc.render_plan(p, RuleSet(knobs={'threshold.strict': True, 'revenue.valid_status': ['completed']}))
    assert '≥ 187' in t and 'threshold.strict' not in t
    # đơn giá: lấy cả dòng đơn huỷ / trả -> không liệt kê revenue.valid_status
    p = Plan(metric='revenue', op='unit_price', filters={'product': 'mi hao hao'},
             periods=[TimeRange(start=date(2025, 5, 25), end=date(2025, 5, 25))])
    t = pc.render_plan(p, RuleSet(knobs={'unit_price.def': 'line_base_before_disc_all', 'revenue.valid_status': ['completed']}))
    assert 'revenue.valid_status' not in t and 'unit_price.def' in t and 'đơn huỷ' in t
