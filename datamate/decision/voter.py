"""Bỏ phiếu có trọng số giữa các candidate -> Decision (trả lời / unanswerable / abstain).

Chính sách (PA: đúng +1, sai -1, bỏ 0): chỉ trả lời khi tổng trọng số ủng hộ >= min_score VÀ tỉ lệ ủng hộ >= min_share.
'unanswerable' cần ngưỡng cao hơn (unanswerable_min_score) vì chỉ ~2% câu là unanswerable.
"""
from __future__ import annotations

from collections import defaultdict

from datamate.core.config import DecisionConfig
from datamate.core.types import AnswerType, Candidate, Decision
from datamate.output.formatter import answer_key, normalize_answer


def decide(qid: str, cands: list[Candidate], cfg: DecisionConfig) -> Decision:
    groups: dict[str, list[Candidate]] = defaultdict(list)
    norm: dict[str, tuple[str, object]] = {}
    total = 0.0
    for c in cands:
        if c.status != 'ok' or c.answer_type is None:
            continue
        try:                                    # một đáp án dạng lạ chỉ loại candidate đó, không làm hỏng cả câu
            t, a = normalize_answer(c.answer_type, c.answer)
            if t is None or t == 'abstain':
                continue
            k = answer_key(t, a)
        except Exception as e:  # noqa: BLE001
            c.status, c.error = 'error', f'đáp án không chuẩn hoá được: {e!r}'[:200]
            continue
        groups[k].append(c)
        norm[k] = (t, a)
        total += c.weight
    if not groups:
        return Decision(qid=qid, type=AnswerType.ABSTAIN, policy='no_valid_candidate')
    scored = sorted(((sum(c.weight for c in g), k) for k, g in groups.items()), reverse=True)
    best_score, best_k = scored[0]
    t, a = norm[best_k]
    share = best_score / total if total else 0.0
    support = groups[best_k]
    reason = next((c.reason for c in support if c.reason), None)
    need = cfg.unanswerable_min_score if t == 'unanswerable' else cfg.min_score
    if best_score + 1e-9 < need or share + 1e-9 < cfg.min_share:
        return Decision(qid=qid, type=AnswerType.ABSTAIN, confidence=share, support=best_score, total=total,
                        policy=f'low_support({best_score:.2f}/{total:.2f}) best={t}')
    return Decision(qid=qid, type=AnswerType(t), answer=a, reason=reason, confidence=share, support=best_score,
                    total=total, policy='+'.join(sorted({c.strategy for c in support})))
