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


OPTIONAL_KEYS = ('chat_template_kwargs', 'seed', 'stop', 'reasoning_effort', 'response_format')


def _backoff(attempt: int, retry_after: Optional[str]) -> float:
    """v3.1.1: chờ trước khi thử lại — theo Retry-After nếu có, không thì 2, 4, 8... giây (tối đa 60) + nhiễu."""
    import random
    try:
        if retry_after:
            return min(120.0, float(retry_after))
    except ValueError:
        pass
    return min(60.0, 2.0 ** attempt) * (0.75 + 0.5 * random.random())


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._rr = itertools.cycle(cfg.base_urls)
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._client: Optional[httpx.AsyncClient] = None
        self.calls = 0
        self.tokens = 0
        # v3.1.0: khoá đọc từ biến môi trường (cfg.api_key_env) nếu có, không ghi trong config
        import os
        self.key = (os.environ.get(cfg.api_key_env) if cfg.api_key_env else None) or cfg.api_key

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
                r = await c.get(u.rstrip('/') + '/models', headers={'Authorization': f'Bearer {self.key}'})
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
        body.update(self.cfg.extra_body or {})          # v3.1.1: vd Gemini {reasoning_effort: low}
        t0 = time.time()
        last_err = None
        empties = 0
        async with self._sem:
            c = await self._get()
            attempt = 0
            while attempt < max(1, self.cfg.retries):
                url = next(self._rr).rstrip('/') + '/chat/completions'
                try:
                    r = await c.post(url, json=body, headers={'Authorization': f'Bearer {self.key}'})
                    if r.status_code != 200:
                        last_err = f'HTTP {r.status_code}: {r.text[:300]}'
                        if r.status_code == 400 and 'context length' in r.text:
                            break               # vượt ngữ cảnh: gửi lại y nguyên vô ích, để agent rút gọn rồi gửi lại
                        if r.status_code in (400, 422):
                            # v3.1.1: tham số tuỳ chọn mà nhà cung cấp không nhận -> bỏ rồi gửi lại (không tính lượt)
                            bad = [k for k in OPTIONAL_KEYS if k in body and k in r.text]
                            if not bad and 'response_format' in body:
                                bad = ['response_format']     # không nhận json_schema -> parse JSON thủ công
                            if bad:
                                for k in bad:
                                    body.pop(k, None)
                                continue
                            break                   # lỗi yêu cầu khác: gửi lại y nguyên vô ích
                        attempt += 1
                        await asyncio.sleep(_backoff(attempt, r.headers.get('retry-after')))
                        continue
                    data = r.json()
                    ch = data['choices'][0]
                    text = (ch.get('message') or {}).get('content') or ''
                    usage = data.get('usage') or {}
                    ntok = int(usage.get('total_tokens', 0))
                    self.calls += 1
                    self.tokens += ntok
                    if not text.strip() and empties < self.cfg.empty_retries:
                        empties += 1                # v3.1.1: rỗng (suy luận ăn hết max_tokens / lỗi tạm) -> gọi lại
                        last_err = f"rỗng (finish_reason={ch.get('finish_reason')})"
                        continue
                    return LLMResult(text=text, tokens=ntok, elapsed=time.time() - t0)
                except Exception as e:  # noqa: BLE001
                    last_err = repr(e)[:300]
                    attempt += 1
                    await asyncio.sleep(_backoff(attempt, None))
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
