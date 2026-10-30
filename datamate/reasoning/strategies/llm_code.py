"""Tầng B: LLM đọc README + rulebook + thẻ bảng, viết code pandas, chạy sandbox, tự sửa lỗi, nhiều mẫu để bỏ phiếu."""
from __future__ import annotations

import json
import time

from datamate.core.registry import register_strategy
from datamate.core.types import AnswerType
from datamate.execution.sandbox import write_pack
from datamate.reasoning.prompts import QUESTION_TEMPLATE, REPAIR_TEMPLATE, SHOP_TEMPLATE, code_messages, fill
from datamate.reasoning.strategies.base import Strategy
from datamate.runtime.llm_client import extract_code


def shop_block(env, cat) -> str:
    if cat.business_id not in env.shop_blocks:
        c = cat.canon
        env.shop_blocks[cat.business_id] = fill(
            SHOP_TEMPLATE, bid=cat.business_id, dmin=c.date_min, dmax=c.date_max, caps=', '.join(sorted(c.caps)),
            readme=(cat.readme.text or '(không có README)')[:9000], cards=cat.cards(max_tables=30, max_cols=45))
    return env.shop_blocks[cat.business_id]


def question_block(u) -> str:
    times = '; '.join(f'{r.expr or r.label}: {r.start}..{r.end}' for r in u.ctx.times) or '(không có)'
    ents = json.dumps(u.ctx.entities, ensure_ascii=False, default=str) if u.ctx.entities else '(không có)'
    return fill(QUESTION_TEMPLATE, as_of=u.ctx.q.as_of_date.isoformat(), question=u.ctx.q.question, times=times,
                entities=ents, intent=u.ctx.intent)


@register_strategy('llm_code')
class LLMCode(Strategy):
    uses_llm = True

    async def run(self, u, cat, ix):
        env = self.env
        if not env.llm_allowed():
            return [self.cand(status='skipped', error='llm disabled or out of time')]
        when = self.scfg.options.get('when', 'fallback')
        min_conf = float(self.scfg.options.get('skip_if_rule_conf', 0.8))
        if when == 'fallback' and u.parse.plan is not None and u.parse.confidence >= min_conf:
            return [self.cand(status='skipped', error='rule plan confident')]
        if cat.business_id not in env.packs:
            env.packs[cat.business_id] = write_pack(cat, env.cfg.cache_dir)
        pack = env.packs[cat.business_id]
        msgs0 = code_messages(env.kb.rulebook.text, shop_block(env, cat), question_block(u))
        meta = {'as_of_date': u.ctx.q.as_of_date.isoformat(), 'question': u.ctx.q.question}
        out = []
        temps = self.scfg.options.get('temperatures', [0.0, 0.4, 0.7])
        for i in range(self.scfg.samples):
            t0 = time.time()
            msgs = list(msgs0)
            temp = temps[i] if i < len(temps) else temps[-1]
            res = await env.llm.chat(msgs, temperature=temp, seed=env.cfg.llm.seed + i)
            if res.error or not res.text:
                out.append(self.cand(i, status='error', error=res.error or 'empty llm output', elapsed=time.time() - t0))
                continue
            code = extract_code(res.text)
            tokens = res.tokens
            ex = await env.sandbox.run(pack, code, meta)
            rounds = 0
            while ex.status != 'ok' and rounds < env.cfg.max_repair_rounds:
                rounds += 1
                msgs = msgs + [{'role': 'assistant', 'content': res.text},
                               {'role': 'user', 'content': fill(REPAIR_TEMPLATE, error=(ex.error or ex.status)[-1200:])}]
                res = await env.llm.chat(msgs, temperature=temp, seed=env.cfg.llm.seed + i)
                if res.error or not res.text:
                    break
                code = extract_code(res.text)
                tokens += res.tokens
                ex = await env.sandbox.run(pack, code, meta)
            if ex.status != 'ok' or not isinstance(ex.result, dict):
                out.append(self.cand(i, program=code, status='error' if ex.status != 'timeout' else 'timeout',
                                     error=(ex.error or ex.status or '')[-600:], elapsed=time.time() - t0, tokens=tokens))
                continue
            r = ex.result
            t = str(r.get('type') or '').lower()
            if t == 'abstain':
                out.append(self.cand(i, program=code, status='empty', error='model abstained', elapsed=time.time() - t0, tokens=tokens))
                continue
            try:
                at = AnswerType(t)
            except ValueError:
                at = None
            w = self.scfg.weight * (self.scfg.options.get('unanswerable_factor', 0.6) if at == AnswerType.UNANSWERABLE else 1.0)
            out.append(self.cand(i, program=code, answer_type=at, answer=r.get('answer'), reason=r.get('reason'), weight=w,
                                 elapsed=time.time() - t0, tokens=tokens))
        return out
