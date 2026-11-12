"""v3.1.2: tối ưu cho model nhỏ — hiện biểu thức cuối (như Jupyter), chặn đọc file, kiểm tra code khi nộp đáp án,
lưu kết quả từng bước; config Qwen3-14B qua OpenRouter."""
import asyncio
import json
import pickle
from pathlib import Path

import pandas as pd

from datamate.agent.verify import code_warnings
from datamate.core.config import load_config
from datamate.runtime.llm_client import MockLLM
from datamate.runtime.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def _exec(tmp_path, *cells):
    from datamate.agent.kernel import KernelPool
    pk = tmp_path / 'p.pkl'
    pickle.dump({'tables': {}, 'canon': {'orders': pd.DataFrame({'status': ['completed', 'cancelled'], 'revenue': [5, 7]})},
                 'meta': {}}, open(pk, 'wb'))

    async def run():
        kp = KernelPool(timeout_sec=20, per_shop=1, max_workers=1, mem_gb=4)
        out = [await kp.execute(str(pk), 's', c) for c in cells]
        await kp.close()
        return out
    return asyncio.run(run())


def test_last_expression_is_displayed_like_jupyter(tmp_path):
    r1, r2, r3 = _exec(tmp_path, "o = canon['orders']\ntotal = o['revenue'].sum()\ntotal",
                       "o[o['status'] == 'completed']", "x = 1\nprint('in')\nNone")
    assert r1.status == 'ok' and r1.stdout.strip() == '12'
    assert 'completed' in r2.stdout and 'cancelled' not in r2.stdout
    assert r3.stdout.strip() == 'in'                                      # None không in


def test_reading_files_is_blocked_with_hint(tmp_path):
    r1, r2 = _exec(tmp_path, "pd.read_csv('don_hang.csv')", "open('x.txt')")
    assert r1.status == 'error' and 'canon[' in r1.error and 'peek()' in r1.error
    assert r2.status == 'error' and 'Không có file' in r2.error


def test_code_warnings():
    w = code_warnings("o = canon['orders']\nx = o[o['channel_f'] == 'zalo_oa']['revenue'].sum()")
    assert len(w) == 2 and 'trạng thái' in w[0] and "'zalo_oa'" in w[1]
    assert code_warnings("L = canon['lines']\nx = L[(L['status'] != 'cancelled') & (L['date'] >= '2025-10-01')]") == []
    assert code_warnings("print(find(canon['orders']['channel_f'], 'zalo'))\nO=canon['orders']\n"
                         "O[(O.status=='completed') & (O.channel_f=='zalo oa')]") == []
    assert code_warnings("x = 'abc'\nprint(x)") == []                     # gán chuỗi thường: không cảnh báo
    assert "'Zalo OA'" in code_warnings('sql("SELECT sum(amount) FROM lines WHERE status = \'completed\' '
                                        'AND channel = \'Zalo OA\'")')[0]


def test_agent_recheck_on_code_warning_and_outputs_logged(mini_split, tmp_path):
    calls = []

    def respond(messages, schema):
        last = messages[-1]['content']
        calls.append(last[:60])
        if 'Điểm đáng ngờ' in last:
            return "Đã kiểm tra.\n```python\nanswer('number', 4242)\n```"
        if last.startswith('Kết quả chạy'):
            return "Xong.\n```python\nanswer('number', 4242)\n```"
        return "Tính.\n```python\nL = canon['lines']\nv = L['amount'].sum()\nv\n```"
    qs = [json.loads(l) for l in (mini_split / 'questions.jsonl').read_text(encoding='utf-8').splitlines()]
    cfg = load_config(str(ROOT / 'configs' / 'qa_qwen14b.yaml'), {
        'cache_dir': str(tmp_path / 'cache'), 'workers': 1, 'mapping_llm': {'enabled': False},
        'llm': {'wait_ready_sec': 1},
        'strategies': [{'name': 'guard'}, {'name': 'rule_plan'},
                       {'name': 'agent', 'samples': 1, 'options': {'max_steps': 4, 'shop_notes': False,
                                                                     'kernel': {'per_shop': 1, 'max_workers': 1, 'mem_gb': 4}}}],
        'decision': {'calibration_path': None, 'llm_only_abstained': False, 'llm_skip_support': 1.01}})
    out = tmp_path / 'out'
    asyncio.run(Orchestrator(cfg, str(mini_split), str(out), llm=MockLLM(respond)).run(ids=[qs[0]['id']], resume=False))
    tr = json.loads((out / 'logs' / f"{qs[0]['id']}.json").read_text(encoding='utf-8'))
    ag = [c for c in tr['candidates'] if c['strategy'] == 'agent'][0]
    assert ag['status'] == 'ok' and ag['meta']['rechecked'], (ag['error'], ag['meta'], ag['program'])                 # code không xét status -> bị hỏi lại 1 lần
    assert ag['meta']['outputs'] and ag['meta']['outputs'][0].strip().replace('.', '').isdigit()   # biểu thức cuối hiện ra
    assert any('chưa xét trạng thái' in c or 'Điểm đáng ngờ' in c for c in calls)


def test_openrouter_config_differs_only_in_provider_blocks():
    a = load_config(str(ROOT / 'configs' / 'qa_qwen14b.yaml'))
    b = load_config(str(ROOT / 'configs' / 'qa_qwen14b_or.yaml'))
    assert a.decision == b.decision and a.strategies == b.strategies
    assert b.llm.api_key_env == 'OPENROUTER_API_KEY' and b.llm.extra_body['reasoning'] == {'enabled': False}
    assert 'sk-or' not in (ROOT / 'configs' / 'qa_qwen14b_or.yaml').read_text(encoding='utf-8')
