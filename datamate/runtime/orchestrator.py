"""Điều phối một lần chạy: nạp split -> build catalog song song -> hỏi từng câu (async) -> bỏ phiếu -> ghi kết quả.

Một tiến trình Python duy nhất: asyncio cho I/O (gọi vLLM, sandbox subprocess), process pool cho build catalog.
Ghi tăng dần + resume: chết giữa chừng chạy lại sẽ bỏ qua câu đã có. Quá time budget -> chỉ chạy tầng A, còn lại abstain.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

from datamate.core.config import Config
from datamate.core.log import get_logger
from datamate.core.registry import get_strategy
from datamate.core.types import AnswerType, Decision, Question, Trace
from datamate.data.catalog import build_all
from datamate.decision.voter import decide
from datamate.execution.sandbox import Sandbox
from datamate.knowledge.base import KnowledgeBase
from datamate.output.writer import SubmissionWriter
from datamate.reasoning.strategies.base import Env
from datamate.understand.entity_link import EntityIndex
from datamate.understand.expected_type import compatible
from datamate.understand.pipeline import understand

log = get_logger('orchestrator')


def _t0(t_start: float) -> float:
    """Mốc tính giờ: DATAMATE_T0 (epoch giây, predict.sh đặt lúc container khởi động, tức là tính cả thời gian nạp mô hình)
    nếu có, không thì lúc bắt đầu predict."""
    try:
        return min(t_start, float(os.environ['DATAMATE_T0']))
    except (KeyError, ValueError):
        return t_start


def load_questions(path: Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            out.append(Question(**json.loads(line)))
    return out


class Orchestrator:
    def __init__(self, cfg: Config, split_dir: str, out_dir: str, llm=None):
        self.cfg = cfg
        self.split = Path(split_dir)
        self.out = Path(out_dir)
        self._llm = llm

    async def _make_llm(self):
        if self._llm is not None:
            return self._llm
        if not self.cfg.llm.enabled:
            return None
        from datamate.runtime.llm_client import LLMClient
        llm = LLMClient(self.cfg.llm)
        t0 = time.time()
        while time.time() - t0 < self.cfg.llm.wait_ready_sec:
            if await llm.health():
                return llm
            await asyncio.sleep(5)
        log.error('LLM không sẵn sàng tại %s -> chạy chỉ tầng A', self.cfg.llm.base_urls)
        return None

    async def run(self, ids: Optional[list[str]] = None, businesses: Optional[list[str]] = None,
                  limit: Optional[int] = None, resume: bool = True) -> dict:
        t_start = time.time()
        kb = KnowledgeBase.load(self.split)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / 'rules.json').write_text(kb.rules.to_json(), encoding='utf-8')   # luật đã đọc (để soát / cho LLM sửa)
        qs = load_questions(self.split / 'questions.jsonl')
        all_ids = [q.id for q in qs]
        if ids:
            s = set(ids)
            qs = [q for q in qs if q.id in s]
        if businesses:
            s = set(businesses)
            qs = [q for q in qs if q.business_id in s]
        if limit:
            qs = qs[:limit]
        biz = sorted({q.business_id for q in qs})
        log.info('%d câu, %d doanh nghiệp', len(qs), len(biz))
        loop = asyncio.get_running_loop()
        cats = await loop.run_in_executor(None, build_all, self.split / 'data', biz, self.cfg.cache_dir, self.cfg.workers)
        ixs = {b: EntityIndex.from_catalog(c) for b, c in cats.items()}
        log.info('catalog xong sau %.1fs', time.time() - t_start)
        llm = await self._make_llm()
        env = Env(cfg=self.cfg, kb=kb, llm=llm, sandbox=Sandbox(self.cfg.sandbox_timeout_sec, max_parallel=self.cfg.workers),
                  deadline=_t0(t_start) + self.cfg.time_budget_sec)
        strategies = [get_strategy(s.name)(s, env) for s in self.cfg.strategies if s.enabled]
        writer = SubmissionWriter(str(self.out), resume=resume)
        sem = asyncio.Semaphore(self.cfg.question_concurrency)
        stats: Counter = Counter()
        llm_strats = [st for st in strategies if st.uses_llm]
        pending: dict[str, list] = defaultdict(list)        # business -> câu cần LLM (pha 2)

        def contract(u, cands):
            if not self.cfg.decision.type_contract:
                return
            exp = set(u.ctx.expected_types) if u.ctx.expected_types else None
            for c in cands:
                if c.status == 'ok' and c.answer_type is not None and not compatible(c.answer_type.value, exp):
                    c.status = 'error'
                    c.error = f'type_contract: {c.answer_type.value} not in {sorted(exp)}'

        def finish(q, intent, times, cands, d, t0):
            tr = Trace(qid=q.id, business_id=q.business_id, question=q.question, as_of_date=q.as_of_date, intent=intent,
                       times=times, candidates=cands, decision=d, elapsed_sec=round(time.time() - t0, 3), config=self.cfg.name)
            writer.write(d, tr)
            stats[d.type.value] += 1

        def confident(d: Decision) -> bool:
            return d.type != AnswerType.ABSTAIN and d.support + 1e-9 >= self.cfg.decision.llm_skip_support

        # ---- pha 1: tầng A (không LLM) cho mọi câu; câu chắc chắn ghi ngay, còn lại chờ pha 2
        async def phase1(q: Question):
            if q.id in writer.done:
                stats['resumed'] += 1
                return
            cat = cats.get(q.business_id)
            async with sem:
                t0 = time.time()
                cands, intent, times, u = [], '', [], None
                try:
                    if cat is None:
                        raise RuntimeError(f'không có dữ liệu cho {q.business_id}')
                    u = understand(q, cat, kb, ixs[q.business_id])
                    intent, times = u.ctx.intent, u.ctx.times
                    for st in strategies:
                        if not st.uses_llm:
                            cands += await st.run(u, cat, ixs[q.business_id])
                    contract(u, cands)
                    d = decide(q.id, cands, self.cfg.decision)
                except Exception as e:  # noqa: BLE001 - một câu lỗi không làm hỏng cả lần chạy
                    import traceback
                    log.warning('%s crash: %s', q.id, traceback.format_exc()[-500:])
                    d = Decision(qid=q.id, type=AnswerType.ABSTAIN, policy=f'crash: {e!r}'[:200])
                    u = None
                if u is not None and llm_strats and env.llm_allowed() and not confident(d):
                    pending[q.business_id].append((q, u, cands, intent, times, t0, d))
                else:
                    finish(q, intent, times, cands, d, t0)

        # ---- pha 2: strategy LLM cho câu chưa chắc, theo từng shop (tái dùng cache tiền tố của vLLM)
        async def phase2(item):
            q, u, cands, intent, times, t0, d0 = item
            async with sem:
                d = d0
                try:
                    if env.llm_allowed():
                        new = []
                        runs = [st.run(u, cats[q.business_id], ixs[q.business_id]) for st in llm_strats]
                        for r in await asyncio.gather(*runs, return_exceptions=True):
                            if isinstance(r, Exception):
                                log.warning('%s strategy crash: %r', q.id, r)
                            else:
                                new += r
                        contract(u, new)
                        cands = cands + new
                        d = decide(q.id, cands, self.cfg.decision)
                    else:
                        stats['llm_out_of_time'] += 1
                except Exception as e:  # noqa: BLE001
                    log.warning('%s phase2 crash: %r', q.id, e)
                finish(q, intent, times, cands, d, t0)

        # gom theo doanh nghiệp (tận dụng prefix cache của vLLM)
        by_biz = defaultdict(list)
        for q in qs:
            by_biz[q.business_id].append(q)
        ordered = [q for b in biz for q in by_biz[b]]
        await asyncio.gather(*(phase1(q) for q in ordered))
        n_pending = sum(len(v) for v in pending.values())
        log.info('pha 1 xong sau %.1fs: %d câu chắc chắn, %d câu chuyển cho LLM', time.time() - t_start,
                 len(qs) - n_pending - stats['resumed'], n_pending)
        await asyncio.gather(*(phase2(it) for b in biz for it in pending.get(b, [])))
        if getattr(env, 'kernel', None) is not None:
            await env.kernel.close()
        writer.finalize(all_ids if not (ids or businesses or limit) else [q.id for q in qs])
        if llm is not None:
            await llm.close()
        oot = stats.pop('llm_out_of_time', 0)
        summary = {'questions': len(qs), 'elapsed_sec': round(time.time() - t_start, 1), 'by_type': dict(stats),
                   'llm_questions': n_pending, 'llm_out_of_time': oot,
                   'llm_calls': getattr(llm, 'calls', 0), 'llm_tokens': getattr(llm, 'tokens', 0), 'config': self.cfg.name,
                   'catalogs': {b: c.summary() for b, c in cats.items()}}
        (self.out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        log.info('xong %d câu trong %.1fs: %s', len(qs), time.time() - t_start, dict(stats))
        return summary
