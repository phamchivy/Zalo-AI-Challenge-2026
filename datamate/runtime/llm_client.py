"""Client LLM tương thích OpenAI (vLLM serve). Không gọi API ngoài: base_url trỏ vào vLLM trong cùng máy/container.

- Round-robin nhiều replica (mỗi GPU một vLLM) qua cfg.base_urls.
- temperature=0 + seed cố định -> ổn định giữa các lần chạy (|ΔPA| <= 0.03).
- guided JSON: response_format json_schema (vLLM >= 0.6) để ép đúng cấu trúc Plan.
- Qwen3.x: tắt thinking qua chat_template_kwargs.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from datamate.core.config import LLMConfig
from datamate.core.log import get_logger

log = get_logger('llm')


@dataclass
class LLMResult:
    text: str
    tokens: int = 0
    elapsed: float = 0.0
    error: Optional[str] = None


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._rr = itertools.cycle(cfg.base_urls)
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None
        self.calls = 0
        self.tokens = 0

    async def _get(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.cfg.timeout_sec)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()

    async def health(self) -> bool:
        c = await self._get()
        for u in self.cfg.base_urls:
            try:
                r = await c.get(u.rstrip('/') + '/models', headers={'Authorization': f'Bearer {self.cfg.api_key}'})
                if r.status_code != 200:
                    return False
            except Exception:  # noqa: BLE001
                return False
        return True

    async def chat(self, messages: list[dict], json_schema: Optional[dict] = None, max_tokens: Optional[int] = None,
                   temperature: Optional[float] = None, seed: Optional[int] = None, stop: Optional[list[str]] = None) -> LLMResult:
        body: dict[str, Any] = {
            'model': self.cfg.model,
            'messages': messages,
            'temperature': self.cfg.temperature if temperature is None else temperature,
            'max_tokens': max_tokens or self.cfg.max_tokens,
            'seed': self.cfg.seed if seed is None else seed,
        }
        if stop:
            body['stop'] = stop
        if self.cfg.disable_thinking:
            body['chat_template_kwargs'] = {'enable_thinking': False}
        if json_schema is not None and self.cfg.guided_json:
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'out', 'schema': json_schema}}
        t0 = time.time()
        last_err = None
        async with self._sem:
            c = await self._get()
            for attempt in range(3):
                url = next(self._rr).rstrip('/') + '/chat/completions'
                try:
                    r = await c.post(url, json=body, headers={'Authorization': f'Bearer {self.cfg.api_key}'})
                    if r.status_code != 200:
                        last_err = f'HTTP {r.status_code}: {r.text[:300]}'
                        # server không hỗ trợ response_format -> bỏ ràng buộc, parse JSON thủ công
                        if r.status_code == 400 and 'response_format' in body:
                            body.pop('response_format', None)
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    data = r.json()
                    text = data['choices'][0]['message'].get('content') or ''
                    usage = data.get('usage') or {}
                    ntok = int(usage.get('total_tokens', 0))
                    self.calls += 1
                    self.tokens += ntok
                    return LLMResult(text=text, tokens=ntok, elapsed=time.time() - t0)
                except Exception as e:  # noqa: BLE001
                    last_err = repr(e)[:300]
                    await asyncio.sleep(0.5 * (attempt + 1))
        log.warning('llm failed: %s', last_err)
        return LLMResult(text='', elapsed=time.time() - t0, error=last_err)


class MockLLM:
    """LLM giả cho test/không GPU: trả rỗng (strategy LLM sẽ 'empty') hoặc theo hàm `responder`."""

    def __init__(self, responder=None):
        self.responder = responder
        self.calls = 0
        self.tokens = 0

    async def health(self) -> bool:
        return True

    async def close(self):
        pass

    async def chat(self, messages, json_schema=None, **kw) -> LLMResult:
        self.calls += 1
        if self.responder:
            return LLMResult(text=self.responder(messages, json_schema))
        return LLMResult(text='', error='mock')


def extract_json(text: str) -> Optional[Any]:
    """Lấy JSON đầu tiên trong text (khi server không ép guided JSON)."""
    if not text:
        return None
    t = text.strip()
    if t.startswith('```'):
        t = t.strip('`')
        t = t[t.find('\n') + 1:] if '\n' in t else t
    for opener, closer in (('{', '}'), ('[', ']')):
        i = t.find(opener)
        while i != -1:
            depth = 0
            for j in range(i, len(t)):
                if t[j] == opener:
                    depth += 1
                elif t[j] == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(t[i:j + 1])
                        except json.JSONDecodeError:
                            break
            i = t.find(opener, i + 1)
    return None


def extract_code(text: str) -> str:
    """Lấy khối ```python ...``` (hoặc toàn bộ text nếu không có rào)."""
    if '```' in text:
        parts = text.split('```')
        for k in range(1, len(parts), 2):
            block = parts[k]
            if block.startswith(('python', 'py')):
                block = block.split('\n', 1)[1] if '\n' in block else ''
            if 'result' in block or 'def ' in block or 'import' in block:
                return block.strip()
        return parts[1].split('\n', 1)[-1].strip()
    return text.strip()
