"""v3.3.0: chuẩn hoá câu hỏi bằng LLM + kiểm tra đồng thuận với tầng A; max_steps 10."""
import asyncio
import json
import re
from pathlib import Path

from datamate.core.config import load_config
from datamate.runtime.llm_client import MockLLM
from datamate.runtime.orchestrator import Orchestrator
from datamate.understand.canonicalize import numbers_kept, same_text

ROOT = Path(__file__).resolve().parents[1]


def test_numbers_kept():
    assert numbers_kept('dthu T4/26 top 3', 'doanh thu tháng 4/2026 top 3')
    assert not numbers_kept('top 3 tháng 12/2025', 'top 3 tháng 12')
    assert same_text('Doanh thu  hôm qua?', 'doanh thu hom qua?')


def _run(mini_split, tmp_path, rewrite):
    def respond(messages, schema):
        u = messages[-1]['content']
        if u.startswith('Câu gốc: '):
            return json.dumps({'q': rewrite(u[len('Câu gốc: '):])}, ensure_ascii=False)
        return '```python\nanswer("abstain")\n```'
    cfg = load_config(str(ROOT / 'configs' / 'canon_glm.yaml'), {
        'cache_dir': str(tmp_path / 'cache'), 'workers': 1, 'mapping_llm': {'enabled': False},
        'decision': {'calibration_path': None}, 'canon_q': {'concurrency': 4}})
    out = tmp_path / 'out'
    s = asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=MockLLM(respond)).run(resume=False))
    tr = {p.stem: json.loads(p.read_text(encoding='utf-8')) for p in (out / 'logs').glob('*.json')}
    return s, tr


def test_agree_keeps_tier_a_and_logs_canonical(mini_split, tmp_path):
    s, tr = _run(mini_split, tmp_path, lambda q: 'Cho mình hỏi: ' + q)          # thêm lời dẫn: cùng nghĩa
    with_c = [t for t in tr.values() if t.get('canon_question')]
    assert with_c and s['by_type'].get('canon_conflict', 0) == 0
    assert any(c['strategy'] == 'rule_plan_canon' for t in with_c for c in t['candidates'])


def test_conflict_abstains(mini_split, tmp_path):
    # đổi đại lượng (doanh thu -> số đơn): tầng A trên 2 câu ra đáp án khác -> không tin tầng A, bỏ trống
    s, tr = _run(mini_split, tmp_path, lambda q: re.sub(r'(?i)doanh thu', 'số đơn', q))
    conf = [t for t in tr.values() if t['decision']['policy'] == 'canon_conflict']
    assert conf and all(t['decision']['type'] == 'abstain' for t in conf)
    assert s['by_type'].get('canon_conflict', 0) == len(conf)


def test_qa_configs_have_canon_and_10_steps():
    for f in ('qa_glm', 'qa_qwen14b', 'qa_qwen14b_or'):
        c = load_config(str(ROOT / 'configs' / f'{f}.yaml'))
        assert c.canon_q.enabled and c.strategy('agent').options['max_steps'] == 10
