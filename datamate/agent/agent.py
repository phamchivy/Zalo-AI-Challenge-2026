"""v2.0 — strategy 'agent': LLM giải câu hỏi bằng vòng lặp viết code → chạy → đọc kết quả (≤ max_steps lượt),
K lời giải song song theo các hướng khác nhau (pandas / SQL / bảng gốc / tự kiểm tra), mỗi lời giải là 1 phiếu.

Tuỳ chọn (configs, strategies[].options):
  styles: [canon_pandas, sql, raw_tables, canon_check]   hướng giải cho từng lời giải (lặp vòng nếu samples > len)
  temperatures: [0.2, 0.6, 0.6, 0.8]
  max_steps: 8, max_tokens_step: 1200, cell_timeout: 30
  shop_notes: true            v2.1 — ghi chú về shop (LLM, kiểm chứng, cache)
  rule_hints: true            v2.2 — đưa các dòng luật liên quan nhất vào phần câu hỏi
  unanswerable_factor: 0.6    trọng số phiếu 'không trả lời được' (cần nhiều phiếu đồng ý hơn)
  kernel: {per_shop: 4, max_workers: 24, mem_gb: 12}
"""
from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Optional

from datamate.agent.kernel import KernelPool
from datamate.agent.prompts import OBSERVATION, QUESTION_BLOCK, SHOP_BLOCK, STYLES, system_prompt
from datamate.agent.rule_index import RuleIndex
from datamate.agent.shop_notes import get_notes
from datamate.agent.verify import coerce, sanity_warnings
from datamate.core.log import get_logger
from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType
from datamate.execution.sandbox import write_pack
from datamate.reasoning.prompts import fill
from datamate.reasoning.strategies.base import Strategy

log = get_logger('agent')
_THINK = re.compile(r'<think>.*?</think>', re.S)
_PY = re.compile(r'```(?:python|py)?\s*\n(.*?)```', re.S)


def extract_cells(text: str) -> str:
    """Các khối ```python``` trong câu trả lời (bỏ phần <think>), nối theo thứ tự."""
    t = _THINK.sub('', text or '')
    blocks = [b.strip() for b in _PY.findall(t) if b.strip()]
    if not blocks and '```' in t:                    # khối chưa đóng ở cuối (bị cắt max_tokens)
        tail = t.split('```')[-1]
        tail = tail.split('\n', 1)[1] if tail.startswith(('python', 'py')) and '\n' in tail else tail
        if tail.strip():
            blocks = [tail.strip()]
    return '\n\n'.join(blocks)


def _compact(msgs: list[dict], budget: int):
    """Hội thoại quá dài: rút gọn kết quả chạy của các bước cũ (giữ 4 message đầu và 4 message cuối nguyên vẹn)."""
    total = sum(len(m['content']) for m in msgs)
    i = 4
    while total > budget and i < len(msgs) - 4:
        m = msgs[i]
        if len(m['content']) > 600:
            short = m['content'][:500] + '\n... (đã rút gọn)'
            total -= len(m['content']) - len(short)
            m['content'] = short
        i += 1


_CTX_ERR = re.compile(r'maximum context length|context length|too long|input_tokens', re.I)
_CTX_N = re.compile(r'(?:contains at least|contains|value=)\s*(\d{4,})')


def _fit(msgs: list[dict], max_chars: int) -> int:
    """Ép tổng độ dài hội thoại <= max_chars, theo thứ tự ít hại nhất:
    1) rút gọn mạnh các bước cũ (giữ 2 message cuối nguyên vẹn)  2) cắt phần thông tin shop  3) cắt rulebook."""
    def total():
        return sum(len(m['content']) for m in msgs)
    _compact(msgs, max_chars)
    for i in range(4, max(4, len(msgs) - 2)):
        if total() <= max_chars:
            break
        if len(msgs[i]['content']) > 250:
            msgs[i]['content'] = msgs[i]['content'][:200] + '\n... (đã rút gọn)'
    for idx, keep_min in ((1, 4000), (0, 6000)):
        over = total() - max_chars
        if over <= 0:
            break
        c = msgs[idx]['content']
        cut = min(over + 200, max(0, len(c) - keep_min))
        if cut > 0:
            msgs[idx]['content'] = c[:len(c) - cut] + '\n... (đã cắt bớt cho vừa ngữ cảnh)'
    return total()


def _health_text(h: dict) -> str:
    keep = {k: v for k, v in (h or {}).items() if v and k not in ('recon_orders',)}
    return json.dumps(keep, ensure_ascii=False, default=str) if keep else 'không có'


def _summary(cat) -> str:
    c = cat.canon
    parts = [f"{len(c.lines)} dòng hàng, {len(c.orders)} đơn"]
    tt = {}
    for n, m in cat.mapping.tables.items():
        tt.setdefault(m.ttype, []).append(n)
    parts.append('loại bảng: ' + '; '.join(f'{k}: {", ".join(v[:4])}' for k, v in tt.items()))
    if c.notes:
        parts.append('ghi chú chuẩn hoá: ' + '; '.join(c.notes[:6]))
    return ' | '.join(parts)


@register_strategy('agent')
class Agent(Strategy):
    uses_llm = True

    def __init__(self, scfg, env):
        super().__init__(scfg, env)
        o = scfg.options
        self.styles = o.get('styles', ['canon_pandas', 'sql', 'raw_tables', 'canon_check'])
        self.temps = o.get('temperatures', [0.2, 0.6, 0.6, 0.8])
        self.max_steps = int(o.get('max_steps', 8))
        self.max_tokens_step = int(o.get('max_tokens_step', 1200))
        self.cell_timeout = float(o.get('cell_timeout', env.cfg.sandbox_timeout_sec))
        self.use_notes = bool(o.get('shop_notes', True))
        self.use_rules = bool(o.get('rule_hints', True))
        self.unans_factor = float(o.get('unanswerable_factor', 0.6))
        # ngân sách ngữ cảnh (ký tự; tiếng Việt ~2.5 ký tự / token): giữ tổng < max-model-len 32k token
        self.cards_chars = int(o.get('cards_chars', 12000))
        self.readme_chars = int(o.get('readme_chars', 7000))
        self.history_chars = int(o.get('history_chars', 52000))
        # giới hạn thật theo token của server (--max-model-len). Ước lượng ký tự/token bắt đầu thận trọng và tự hiệu
        # chỉnh khi server báo vượt ngữ cảnh (đọc số token thật trong lỗi).
        self.ctx_tokens = int(o.get('context_tokens', 32768))
        self._cpt = float(o.get('chars_per_token', 2.0))
        self._shop_lock: dict[str, asyncio.Lock] = {}
        self._rule_index: Optional[RuleIndex] = None
        self._system: Optional[str] = None

    # -- tài nguyên dùng chung
    def kernel(self) -> KernelPool:
        if getattr(self.env, 'kernel', None) is None:
            k = self.scfg.options.get('kernel', {})
            self.env.kernel = KernelPool(timeout_sec=self.cell_timeout, per_shop=int(k.get('per_shop', 4)),
                                         max_workers=int(k.get('max_workers', 24)), mem_gb=float(k.get('mem_gb', 12)))
        return self.env.kernel

    def system(self) -> str:
        if self._system is None:
            self._system = system_prompt(self.env.kb.rulebook.text, self.cell_timeout)
        return self._system

    def rules(self) -> RuleIndex:
        if self._rule_index is None:
            self._rule_index = RuleIndex(self.env.kb.rulebook.text)
        return self._rule_index

    async def shop_block(self, cat) -> tuple[str, str]:
        """(pack path, khối thông tin shop) — tạo 1 lần cho mỗi shop."""
        bid = cat.business_id
        key = f'agent:{bid}'
        lock = self._shop_lock.setdefault(bid, asyncio.Lock())
        async with lock:
            if key not in self.env.shop_blocks:
                if bid not in self.env.packs:
                    loop = asyncio.get_running_loop()
                    self.env.packs[bid] = await loop.run_in_executor(None, write_pack, cat, self.env.cfg.cache_dir)
                cards = cat.cards(max_tables=30, max_cols=40, max_chars=self.cards_chars)
                notes = ''
                if self.use_notes:
                    try:
                        notes = await get_notes(self.env, cat, _summary(cat), cards)
                    except Exception as e:  # noqa: BLE001 - ghi chú lỗi không chặn agent
                        log.warning('%s: ghi chú shop lỗi %r', bid, e)
                c = cat.canon
                caps = ', '.join(sorted(k for k in ('lines', 'orders', 'products', 'units', 'customers', 'payments', 'purchases',
                                                     'stock', 'prices', 'lots', 'deliveries')
                                        if getattr(c, k, None) is not None and len(getattr(c, k)) > 0))
                self.env.shop_blocks[key] = fill(
                    SHOP_BLOCK, bid=bid, dmin=c.date_min, dmax=c.date_max, caps=caps or '(trống)',
                    health=_health_text(getattr(c, 'health', {})), readme=(cat.readme.text or '(không có README)')[:self.readme_chars],
                    cards=cards, notes=notes)
        return self.env.packs[bid], self.env.shop_blocks[key]

    def question_block(self, u, style: str) -> str:
        times = '; '.join(f'{r.expr or r.label}: {r.start} .. {r.end}' for r in u.ctx.times) or '(không đọc được, tự xác định)'
        ents = {k: v for k, v in (u.ctx.entities or {}).items() if k != 'unexplained'}
        exp = ', '.join(u.ctx.expected_types) if u.ctx.expected_types else '(tự xác định)'
        rules = self.rules().block(u.ctx.q.question) if self.use_rules else ''
        return fill(QUESTION_BLOCK, as_of=u.ctx.q.as_of_date.isoformat(), question=u.ctx.q.question, times=times,
                    entities=json.dumps(ents, ensure_ascii=False, default=str) if ents else '(không có)', expected=exp,
                    rules=('\n' + rules) if rules else '', style=STYLES.get(style, style), steps=self.max_steps)

    # -- một lời giải
    async def solve(self, i: int, u, pack: str, shop: str) -> 'object':
        env = self.env
        style = self.styles[i % len(self.styles)]
        temp = self.temps[i % len(self.temps)]
        sid = f'{u.ctx.q.id}:{i}:{uuid.uuid4().hex[:6]}'
        msgs = [{'role': 'system', 'content': self.system()},
                {'role': 'user', 'content': shop},
                {'role': 'assistant', 'content': 'Tôi đã đọc dữ liệu của shop. Mời đưa câu hỏi.'},
                {'role': 'user', 'content': self.question_block(u, style)}]
        meta_q = {'as_of_date': u.ctx.q.as_of_date.isoformat(), 'question': u.ctx.q.question}
        expected = set(u.ctx.expected_types) if u.ctx.expected_types else None
        fq = u.ctx.fq
        t0, tokens, cells, warned, nudges = time.time(), 0, [], False, 0
        status, err, final = 'empty', 'hết lượt chưa trả lời', None
        try:
            for step in range(1, self.max_steps + 1):
                if not env.llm_allowed():
                    err = 'hết thời gian'
                    break
                for _try in range(3):
                    limit = int((self.ctx_tokens - self.max_tokens_step - 400) * self._cpt)
                    n_chars = _fit(msgs, min(self.history_chars, limit))
                    res = await env.llm.chat(msgs, temperature=temp, seed=env.cfg.llm.seed + 1000 * i + step,
                                             max_tokens=self.max_tokens_step)
                    if not (res.error and _CTX_ERR.search(res.error)):
                        break
                    m = _CTX_N.search(res.error)
                    real = int(m.group(1)) if m else 0
                    new_cpt = n_chars / real * 0.9 if real else self._cpt * 0.75
                    log.info('vượt ngữ cảnh (%s token / %d ký tự): ký tự/token %.2f -> %.2f', real or '?', n_chars,
                             self._cpt, min(self._cpt, new_cpt))
                    self._cpt = max(0.8, min(self._cpt, new_cpt))
                tokens += res.tokens
                if res.error or not res.text:
                    status, err = 'error', res.error or 'LLM trả rỗng'
                    break
                msgs.append({'role': 'assistant', 'content': _THINK.sub('', res.text).strip()[:6000]})
                code = extract_cells(res.text)
                if not code:
                    nudges += 1
                    if nudges > 1:
                        status, err = 'error', 'không viết code'
                        break
                    msgs.append({'role': 'user', 'content': 'Hãy viết đúng một khối ```python``` để chạy (hoặc gọi answer(...)).'})
                    continue
                cells.append(code)
                ex = await self.kernel().execute(pack, sid, code, meta=meta_q, timeout=self.cell_timeout)
                if ex.answer and ex.status == 'ok':
                    t, v, hard = coerce(ex.answer.get('type'), ex.answer.get('value'))
                    if t == 'abstain':
                        status, err = 'empty', 'agent tự bỏ trống'
                        break
                    warns = list(hard)
                    if t not in (None, 'unanswerable'):
                        warns += sanity_warnings(t, v, fq)
                        if expected and t not in expected:
                            warns.append(f"câu hỏi cần kiểu {', '.join(sorted(expected))} nhưng đáp án là {t}")
                    if warns and not warned and step < self.max_steps:
                        warned = True
                        msgs.append({'role': 'user', 'content':
                                     f"Bạn vừa gửi answer({t!r}, {json.dumps(v, ensure_ascii=False, default=str)[:300]}). "
                                     f"Điểm đáng ngờ: {'; '.join(warns)}. Kiểm tra lại bằng code. Nếu vẫn đúng, gọi lại "
                                     f"answer với cùng giá trị; nếu không chắc, answer('abstain')."})
                        continue
                    if hard:
                        status, err = 'error', '; '.join(hard)
                        break
                    final = (t, v, ex.answer.get('reason'), warns)
                    status, err = 'ok', None
                    break
                body = (ex.stdout or '').strip() or '(không in gì)'
                if ex.error:
                    body += f'\nLỖI ({ex.status}):\n{ex.error}'
                left = self.max_steps - step
                tail = ('\nChỉ còn 1 lượt: lượt sau PHẢI gọi answer(...), hoặc answer("abstain") nếu không chắc.'
                        if left == 1 else '')
                msgs.append({'role': 'user', 'content': fill(OBSERVATION, step=step, steps=self.max_steps, body=body[:3500],
                                                             tail=tail)})
        finally:
            await self.kernel().close_session(sid)
        meta = {'style': style, 'temp': temp, 'steps': len(cells), 'rechecked': warned}
        program = '\n# ----- bước -----\n'.join(cells)[-12000:]
        if status != 'ok' or final is None:
            return self.cand(i, program=program, status=status if status in ('error', 'empty', 'timeout') else 'error',
                             error=err, elapsed=time.time() - t0, tokens=tokens, meta=meta)
        t, v, reason, warns = final
        meta['warnings'] = warns
        w = self.scfg.weight * (self.unans_factor if t == 'unanswerable' else 1.0)
        return self.cand(i, program=program, answer_type=AnswerType(t), answer=v, reason=reason, weight=w,
                         elapsed=time.time() - t0, tokens=tokens, meta=meta)

    async def run(self, u, cat, ix):
        if not self.env.llm_allowed():
            return [self.cand(status='skipped', error='llm tắt hoặc hết thời gian')]
        try:
            pack, shop = await self.shop_block(cat)
        except Exception as e:  # noqa: BLE001
            return [self.cand(status='error', error=f'không chuẩn bị được dữ liệu shop: {e!r}'[:300])]
        n = max(1, int(self.scfg.samples))
        outs = await asyncio.gather(*(self.solve(i, u, pack, shop) for i in range(n)), return_exceptions=True)
        cands = []
        for i, o in enumerate(outs):
            if isinstance(o, Exception):
                log.warning('%s agent %d crash: %r', u.ctx.q.id, i, o)
                cands.append(self.cand(i, status='error', error=repr(o)[:300]))
            else:
                cands.append(o)
        return cands
