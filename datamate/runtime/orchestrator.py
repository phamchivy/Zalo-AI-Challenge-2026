"""Điều phối một lần chạy: nạp split -> build catalog song song -> hỏi từng câu (async) -> bỏ phiếu -> ghi kết quả.

Một tiến trình Python duy nhất: asyncio cho I/O (gọi vLLM, sandbox subprocess), process pool cho build catalog.
Ghi tăng dần + resume: chết giữa chừng chạy lại sẽ bỏ qua câu đã có. Quá time budget -> chỉ chạy tầng A, còn lại abstain.
"""
from __future__ import annotations

import asyncio
import json
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
from datamate.understand.pipeline import understand

log = get_logger('orchestrator')


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
                  deadline=t_start + self.cfg.time_budget_sec)
        strategies = [get_strategy(s.name)(s, env) for s in self.cfg.strategies if s.enabled]
        writer = SubmissionWriter(str(self.out), resume=resume)
        sem = asyncio.Semaphore(self.cfg.question_concurrency)
        stats: Counter = Counter()

        async def one(q: Question):
            if q.id in writer.done:
                stats['resumed'] += 1
                return
            cat = cats.get(q.business_id)
            async with sem:
                t0 = time.time()
                cands = []
                intent = ''
                times = []
                try:
                    if cat is None:
                        raise RuntimeError(f'không có dữ liệu cho {q.business_id}')
                    u = understand(q, cat, kb, ixs[q.business_id])
                    intent, times = u.ctx.intent, u.ctx.times
                    for st in strategies:
                        if not st.uses_llm:
                            cands += await st.run(u, cat, ixs[q.business_id])
                    llm_runs = [st.run(u, cat, ixs[q.business_id]) for st in strategies if st.uses_llm]
                    for r in await asyncio.gather(*llm_runs, return_exceptions=True):
                        if isinstance(r, Exception):
                            log.warning('%s strategy crash: %r', q.id, r)
                        else:
                            cands += r
                    d = decide(q.id, cands, self.cfg.decision)
                except Exception as e:  # noqa: BLE001 - một câu lỗi không làm hỏng cả lần chạy
                    import traceback
                    log.warning('%s crash: %s', q.id, traceback.format_exc()[-500:])
                    d = Decision(qid=q.id, type=AnswerType.ABSTAIN, policy=f'crash: {e!r}'[:200])
                tr = Trace(qid=q.id, business_id=q.business_id, question=q.question, as_of_date=q.as_of_date, intent=intent,
                           times=times, candidates=cands, decision=d, elapsed_sec=round(time.time() - t0, 3), config=self.cfg.name)
                writer.write(d, tr)
                stats[d.type.value] += 1

        # gom theo doanh nghiệp (tận dụng prefix cache của vLLM)
        by_biz = defaultdict(list)
        for q in qs:
            by_biz[q.business_id].append(q)
        ordered = [q for b in biz for q in by_biz[b]]
        await asyncio.gather(*(one(q) for q in ordered))
        writer.finalize(all_ids if not (ids or businesses or limit) else [q.id for q in qs])
        if llm is not None:
            await llm.close()
        summary = {'questions': len(qs), 'elapsed_sec': round(time.time() - t_start, 1), 'by_type': dict(stats),
                   'llm_calls': getattr(llm, 'calls', 0), 'llm_tokens': getattr(llm, 'tokens', 0), 'config': self.cfg.name,
                   'catalogs': {b: c.summary() for b, c in cats.items()}}
        (self.out / 'run_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
        log.info('xong %d câu trong %.1fs: %s', len(qs), time.time() - t_start, dict(stats))
        return summary
