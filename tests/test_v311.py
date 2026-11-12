"""v3.1.1: client LLM chịu lỗi tạm (429 / rỗng / tham số không nhận), extra_body, nguyên tắc 'luật áp đúng đại lượng'."""
import asyncio
import json
from pathlib import Path

import httpx

from datamate.agent.prompts import AGENT_SYSTEM
from datamate.core.config import LLMConfig, load_config
from datamate.runtime.llm_client import LLMClient

ROOT = Path(__file__).resolve().parents[1]


def _ok(text):
    return httpx.Response(200, json={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                                     'usage': {'total_tokens': 7}})


def _run(cfg, handler):
    cl = LLMClient(cfg)
    cl._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return asyncio.run(cl.chat([{'role': 'user', 'content': 'x'}])), cl


def test_429_then_success_and_extra_body_sent():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(429, headers={'retry-after': '0'}, text='rate limit')
        return _ok('xin chào')
    r, cl = _run(LLMConfig(extra_body={'reasoning_effort': 'low'}, retries=3), h)
    assert r.text == 'xin chào' and r.error is None and len(seen) == 2
    assert seen[-1]['reasoning_effort'] == 'low'


def test_unsupported_optional_param_is_dropped():
    seen = []

    def h(req):
        body = json.loads(req.content)
        seen.append(body)
        if 'seed' in body:
            return httpx.Response(400, text='Invalid JSON payload received. Unknown name "seed"')
        return _ok('ok')
    r, _ = _run(LLMConfig(disable_thinking=False), h)
    assert r.text == 'ok' and 'seed' not in seen[-1]


def test_empty_reply_is_retried_once():
    n = []

    def h(req):
        n.append(1)
        return _ok('') if len(n) == 1 else _ok('có nội dung')
    r, cl = _run(LLMConfig(), h)
    assert r.text == 'có nội dung' and len(n) == 2


def test_other_400_not_retried():
    n = []

    def h(req):
        n.append(1)
        return httpx.Response(400, text='bad request: messages invalid')
    r, _ = _run(LLMConfig(disable_thinking=False, retries=5), h)
    assert r.text == '' and r.error and len(n) == 1


def test_mapping_client_backs_off_on_429(monkeypatch):
    from datamate.llm import json_chat
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if len(calls) == 1:
            return httpx.Response(429, headers={'retry-after': '0'}, request=httpx.Request('POST', url))
        return httpx.Response(200, json={'choices': [{'message': {'content': '{"a": 1}'}}]},
                              request=httpx.Request('POST', url))
    monkeypatch.setattr(json_chat.httpx, 'post', fake_post)
    c = json_chat.JSONChat({'base_url': 'http://x/v1', 'model': 'm', 'json_mode': 'prompt', 'cache': False,
                            'extra_body': {'reasoning_effort': 'low'}, 'api_key_env': 'NOPE'}, None)
    assert c.ask('s', 'u', None, lambda o: o) == {'a': 1}
    assert len(calls) == 2 and calls[-1]['reasoning_effort'] == 'low'


def test_prompt_rule_scope_principle():
    assert 'chỉ áp cho đại lượng mà nó nêu' in AGENT_SYSTEM and 'đối chiếu chéo' in AGENT_SYSTEM


def test_qa_configs_aligned():
    cs = [load_config(str(ROOT / 'configs' / f)) for f in ('qa_glm.yaml', 'qa_gemini.yaml', 'qa_qwen14b.yaml')]
    assert all(c.decision == cs[0].decision for c in cs)
    assert all(not c.strategy('agent').options['shop_notes'] for c in cs)
    assert cs[1].llm.api_key_env == 'GEMINI_API_KEY' and cs[1].llm.extra_body == {'reasoning_effort': 'low'}
    for f in ('qa_glm.yaml', 'qa_gemini.yaml'):
        assert 'AIza' not in (ROOT / 'configs' / f).read_text(encoding='utf-8')
