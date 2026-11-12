"""v3.1.0 — Quyết định bằng đếm phiếu cho câu tầng A bỏ trống (pha LLM).

- Mỗi lời giải LLM (strategy dùng LLM) = 1 phiếu.
- Mỗi strategy không dùng LLM (tầng A: rule_plan, guard) có đáp án hợp lệ = 1 phiếu (tối đa 1 phiếu / strategy),
  KHÔNG có quyền phủ quyết: tầng A chỉ là một phiếu thường, vì câu hỏi / luật mới thì regex không quyết được.
- Trả lời khi đáp án nhiều phiếu nhất có >= min_votes phiếu VÀ nhiều phiếu hơn hẳn đáp án thứ hai (không hoà).
- 'unanswerable' cần >= unanswerable_min_votes phiếu (sai bị trừ, câu không trả lời được chỉ ~2%).
- Phải có ít nhất một phiếu LLM (pha này chỉ chạy khi tầng A đã bỏ trống: tầng A một mình không đủ).
"""
from __future__ import annotations

from collections import defaultdict

from datamate.core.config import DecisionConfig
from datamate.core.types import AnswerType, Candidate, Decision
from datamate.output.formatter import answer_key, normalize_answer


def consensus(qid: str, cands: list[Candidate], cfg: DecisionConfig, llm_strategies: set[str]) -> Decision:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    norm: dict[str, tuple[str, object]] = {}
    rule_voted: set[str] = set()
    n_votes = 0
    for c in cands:
        if c.status != 'ok' or c.answer_type is None:
            continue
        try:
            t, a = normalize_answer(c.answer_type, c.answer)
            if t is None or t == 'abstain':
                continue
            k = answer_key(t, a)
        except Exception as e:  # noqa: BLE001
            c.status, c.error = 'error', f'đáp án không chuẩn hoá được: {e!r}'[:200]
            continue
        if c.strategy not in llm_strategies:
            if c.strategy in rule_voted:
                continue                             # tầng A: tối đa 1 phiếu mỗi strategy
            rule_voted.add(c.strategy)
        groups[k].append(c)
        norm[k] = (t, a)
        n_votes += 1
    if not groups:
        return Decision(qid=qid, type=AnswerType.ABSTAIN, policy='no_valid_candidate')
    ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    best_k, best = ranked[0]
    second = len(ranked[1][1]) if len(ranked) > 1 else 0
    t, a = norm[best_k]
    n_llm = sum(1 for c in best if c.strategy in llm_strategies)
    need = cfg.unanswerable_min_votes if t == 'unanswerable' else cfg.min_votes
    tag = f'{len(best)}/{n_votes}'
    if len(best) < need or len(best) <= second or n_llm == 0:
        return Decision(qid=qid, type=AnswerType.ABSTAIN, confidence=len(best) / n_votes, support=len(best),
                        total=n_votes, policy=f'no_consensus({tag}) best={t}')
    reason = next((c.reason for c in best if c.reason), None)
    by = '+'.join(sorted({c.strategy for c in best}))
    return Decision(qid=qid, type=AnswerType(t), answer=a, reason=reason, confidence=len(best) / n_votes,
                    support=len(best), total=n_votes, policy=f'consensus({tag}:{by})')
