"""Hợp đồng dữ liệu giữa các module (v1). Mọi module chỉ trao đổi qua các object này."""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


class AnswerType(str, Enum):
    NUMBER = 'number'
    PERCENT = 'percent'
    STRING = 'string'
    DATE = 'date'
    BOOLEAN = 'boolean'
    LIST = 'list'
    TABLE = 'table'
    UNANSWERABLE = 'unanswerable'
    ABSTAIN = 'abstain'


AnswerValue = Union[int, float, str, bool, list, dict, None]


# ----------------------------------------------------------------------------- question level
class Question(BaseModel):
    id: str
    business_id: str
    question: str
    as_of_date: date


class TimeRange(BaseModel):
    start: date
    end: date                                   # inclusive
    label: str = 'main'                         # main | compare | series_i | snapshot
    expr: str = ''
    granularity: Literal['day', 'week', 'month', 'quarter', 'year', 'custom', 'tet', 'snapshot'] = 'custom'


class Plan(BaseModel):
    """DSL trung gian: LLM hoặc rule parser điền, executor deterministic tính.

    op:
      value            -> 1 số cho metric trong `periods[0]`
      values           -> list số, mỗi phần tử ứng 1 filter trong `split_by_values` (vd 2 kênh)
      series           -> list số theo từng period trong `periods`
      top              -> top-k theo `group_by`, xếp theo metric giảm dần (list hoặc string nếu k=1)
      argmax           -> phần tử có metric lớn nhất theo `group_by` ('day'|'month'|...) -> date/string
      share            -> % metric(filter) / metric(toàn shop) cùng kỳ
      pct_change       -> % (periods[0] - periods[1]) / periods[1]
      diff             -> periods[0] - periods[1]
      trend_up         -> bool tăng liên tục qua periods
      count_ge         -> đếm group (day/order/customer/product) có metric >= threshold
      pct_ge           -> % group có metric >= threshold trên tổng group
    """
    metric: str = 'revenue'                     # revenue | orders | cancelled_orders | cancel_rate | aov | qty | gross_profit | unit_price | lines
    op: str = 'value'
    periods: list[TimeRange] = Field(default_factory=list)
    filters: dict[str, Any] = Field(default_factory=dict)       # product, category, channel, payment, customer_group, shift, staff, customer, rx, ingredient, base_unit, size
    group_by: Optional[str] = None              # product | customer | day | month | staff | supplier
    split_by: Optional[str] = None              # cho op=values: tên filter (vd 'channel')
    split_values: list[Any] = Field(default_factory=list)
    k: Optional[int] = None
    threshold: Optional[float] = None
    threshold_metric: Optional[str] = None      # metric dùng cho count_ge (vd 'orders' của khách)
    answer_type: Optional[AnswerType] = None
    source: str = ''                            # rule | llm
    note: str = ''


class QContext(BaseModel):
    q: Question
    fq: str                                     # câu hỏi đã fold (không dấu, thường)
    norm_question: str                          # câu đã chuẩn hoá teencode (có thể vẫn không dấu)
    times: list[TimeRange] = Field(default_factory=list)
    intent: str = 'unknown'
    entities: dict[str, Any] = Field(default_factory=dict)      # product/customer/category... đã khớp vào dữ liệu
    unanswerable_reason: Optional[str] = None
    expected_types: Optional[list[str]] = None  # hợp đồng kiểu đáp án suy từ dạng câu hỏi (None = không ràng buộc)
    notes: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------- candidates & decision
class Candidate(BaseModel):
    cand_id: str
    strategy: str                               # rule_plan | llm_plan | llm_code | guard
    weight: float = 1.0
    program: str = ''                           # plan JSON / code
    status: Literal['ok', 'error', 'timeout', 'empty', 'skipped'] = 'ok'
    error: Optional[str] = None
    answer_type: Optional[AnswerType] = None
    answer: AnswerValue = None
    reason: Optional[str] = None
    elapsed: float = 0.0
    tokens: int = 0
    meta: dict[str, Any] = Field(default_factory=dict)      # đặc trưng độ tin cậy, luật thiếu... (để hiệu chỉnh)


class Decision(BaseModel):
    qid: str
    type: AnswerType
    answer: AnswerValue = None
    reason: Optional[str] = None
    confidence: float = 0.0
    support: float = 0.0
    total: float = 0.0
    policy: str = ''


class Trace(BaseModel):
    qid: str
    business_id: str
    question: str
    as_of_date: date
    intent: str = ''
    times: list[TimeRange] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    decision: Optional[Decision] = None
    elapsed_sec: float = 0.0
    config: str = ''
