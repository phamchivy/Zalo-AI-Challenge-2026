"""v3.0.0 — Client LLM đồng bộ, trả JSON đã kiểm, dùng cho bước dựng dữ liệu (gán nhãn, gán vai trò cột).

Không gắn với nhà cung cấp: mọi endpoint chuẩn OpenAI (/v1/chat/completions). Đổi model = đổi config:
  - phát triển: API nội bộ (GLM-4.7…), chỉ dùng dữ liệu train
  - nộp bài:    Qwen3-14B chạy vLLM trong container (json_mode=schema, ép đúng JSON schema)

json_mode:
  schema  -> response_format json_schema (vLLM/xgrammar ép đúng cấu trúc)
  object  -> response_format json_object
  prompt  -> không gửi response_format; schema + ví dụ nằm trong prompt
  auto    -> thử schema; endpoint từ chối (4xx) thì chuyển hẳn sang prompt
Mọi chế độ đều: bóc ```json```, parse, kiểm bằng hàm validate; sai thì gửi lỗi lại cho model, tối đa `retries` lần;
vẫn sai -> trả None (người gọi quay về regex). Kết quả được cache theo hash (model + prompt) để chạy lại không tốn lượt gọi.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional

import httpx

log = logging.getLogger(__name__)

_FENCE = re.compile(r'```(?:json)?\s*(.*?)```', re.S)


def extract_json(text: Optional[str]) -> Any:
    """Lấy đối tượng JSON đầu tiên trong câu trả lời (bỏ ```json```, chữ thừa trước/sau)."""
    if not text:
        raise ValueError('câu trả lời rỗng')
    m = _FENCE.search(text)
    s = m.group(1) if m else text
    s = s.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    i = s.find('{')
    if i < 0:
        raise ValueError('không thấy JSON')
    depth, j = 0, i
    in_str = esc = False
    for j in range(i, len(s)):
        ch = s[j]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return json.loads(s[i:j + 1])
    raise ValueError('JSON không đóng ngoặc')


class JSONChat:
    def __init__(self, cfg: dict, cache_dir: Optional[str] = None):
        self.cfg = cfg
        self.base = cfg.get('base_url', '').rstrip('/')
        self.model = cfg.get('model', '')
        key = os.environ.get(cfg.get('api_key_env') or '', '') or cfg.get('api_key') or 'EMPTY'
        self.headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
        self.mode = cfg.get('json_mode', 'auto')
        self.cache = Path(cache_dir) / 'llm_map' if cache_dir and cfg.get('cache', True) else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.calls = 0
        self.failures = 0

    def _key(self, messages: list[dict], schema: Optional[dict]) -> str:
        h = hashlib.sha1(json.dumps([self.model, messages, schema], ensure_ascii=False, sort_keys=True).encode())
        return h.hexdigest()[:24]

    def _send(self, body: dict):
        """v3.1.1: 429 / 5xx -> chờ tăng dần (theo Retry-After) rồi gửi lại."""
        import time as _time

        from datamate.runtime.llm_client import _backoff
        r = None
        for attempt in range(1, 7):
            r = httpx.post(self.base + '/chat/completions', headers=self.headers, json=body,
                           timeout=float(self.cfg.get('timeout_sec', 120)))
            if r.status_code != 429 and r.status_code < 500:
                return r
            _time.sleep(_backoff(attempt, r.headers.get('retry-after')))
        return r

    def _post(self, messages: list[dict], schema: Optional[dict]) -> str:
        body: dict[str, Any] = {'model': self.model, 'messages': messages,
                                'temperature': self.cfg.get('temperature', 0.0),
                                'max_tokens': int(self.cfg.get('max_tokens', 4096))}
        if self.cfg.get('seed') is not None:
            body['seed'] = self.cfg['seed']
        if self.cfg.get('disable_thinking'):
            body['chat_template_kwargs'] = {'enable_thinking': False}
        if schema is not None and self.mode in ('schema', 'auto'):
            body['response_format'] = {'type': 'json_schema', 'json_schema': {'name': 'out', 'schema': schema}}
        elif self.mode == 'object':
            body['response_format'] = {'type': 'json_object'}
        body.update(self.cfg.get('extra_body') or {})          # v3.1.1
        r = self._send(body)
        if r.status_code in (400, 422) and self.mode == 'auto' and 'response_format' in body:
            self.mode = 'prompt'                       # endpoint không nhận json_schema -> dùng prompt từ nay
            body.pop('response_format', None)
            r = self._send(body)
        r.raise_for_status()
        self.calls += 1
        msg = r.json()['choices'][0]['message']
        return msg.get('content') or ''

    def ask(self, system: str, user: str, schema: Optional[dict], validate: Callable[[Any], Any]) -> Optional[Any]:
        """Trả kết quả đã qua validate(obj) (validate ném ValueError kèm lý do nếu sai), hoặc None."""
        messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        key = self._key(messages, schema)
        if self.cache and (self.cache / f'{key}.json').exists():
            try:
                return validate(json.loads((self.cache / f'{key}.json').read_text(encoding='utf-8')))
            except Exception:  # noqa: BLE001 - cache hỏng/không còn hợp lệ: gọi lại
                pass
        last = None
        for attempt in range(int(self.cfg.get('retries', 3))):
            try:
                text = self._post(messages, schema)
                obj = extract_json(text)
                out = validate(obj)
                if self.cache:
                    (self.cache / f'{key}.json').write_text(json.dumps(obj, ensure_ascii=False), encoding='utf-8')
                return out
            except (httpx.HTTPError, KeyError) as e:
                last = f'lỗi gọi API: {e!r}'[:300]
                time.sleep(min(2 ** attempt, 8))
            except (ValueError, TypeError) as e:
                last = str(e)[:300]
                messages = messages[:2] + [{'role': 'user', 'content':
                                            f'Câu trả lời trước không hợp lệ: {last}. Chỉ trả về JSON đúng yêu cầu, không giải thích.'}]
        self.failures += 1
        log.warning('LLM JSON thất bại sau %s lần: %s', self.cfg.get('retries', 3), last)
        return None
