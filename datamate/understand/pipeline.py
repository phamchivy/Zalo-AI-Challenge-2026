"""Hiểu câu hỏi: Question + catalog -> QContext (+ kết quả rule parser) dùng chung cho mọi strategy."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from datamate.core.textutil import fold
from datamate.core.types import QContext, Question
from datamate.understand import answerability
from datamate.understand.entity_link import EntityIndex
from datamate.understand.intent import ParseResult, parse_question
from datamate.understand.normalize import qnorm
from datamate.understand.time_resolver import TimeParse, resolve_time


@dataclass
class Understanding:
    ctx: QContext
    tp: TimeParse
    parse: ParseResult


def understand(q: Question, catalog, kb, ix: Optional[EntityIndex] = None) -> Understanding:
    qn = qnorm(q.question)
    qf = ' '.join(re.sub(r'(?<!\d),|,(?!\d)|[?!;"“”]', ' ', fold(q.question)).split())
    tp = resolve_time(qn, q.as_of_date, kb.calendar)
    canon = catalog.canon
    ix = ix or EntityIndex()
    parse = parse_question(qn, qf, q.as_of_date, tp, ix, canon.caps, canon.date_min, canon.date_max, kb.calendar)
    ctx = QContext(q=q, fq=qf, norm_question=qn, times=tp.ranges, intent=parse.intent, entities=parse.entities)
    reason = answerability.subjective(qn)
    if not reason and parse.plan is not None:
        reason = answerability.out_of_range(parse.plan.periods, canon.date_min, canon.date_max, q.as_of_date)
    elif not reason and tp.ranges:
        reason = answerability.out_of_range(tp.ranges, canon.date_min, canon.date_max, q.as_of_date)
    ctx.unanswerable_reason = reason
    if tp.flags:
        ctx.notes.append('time_flags=' + ','.join(sorted(tp.flags)))
    return Understanding(ctx=ctx, tp=tp, parse=parse)
