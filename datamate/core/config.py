"""Config YAML -> object. Mọi tham số chạy (LLM, chiến lược, ngưỡng, thời gian) nằm ở configs/*.yaml."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    enabled: bool = True
    base_urls: list[str] = Field(default_factory=lambda: ['http://127.0.0.1:8000/v1'])   # >1 url = nhiều replica
    model: str = 'qwen'
    api_key: str = 'EMPTY'
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 2048
    timeout_sec: float = 180.0
    max_concurrency: int = 48
    guided_json: bool = True                    # vLLM structured output cho plan
    disable_thinking: bool = True               # Qwen3.x: tắt thinking cho nhanh/ổn định
    wait_ready_sec: int = 900                   # chờ vLLM load model trước khi chạy


class StrategyConfig(BaseModel):
    name: str
    enabled: bool = True
    weight: float = 1.0
    samples: int = 1
    options: dict[str, Any] = Field(default_factory=dict)


class DecisionConfig(BaseModel):
    min_score: float = 0.5                      # tổng trọng số ủng hộ tối thiểu để trả lời
    min_share: float = 0.6                      # tỉ lệ trọng số ủng hộ / tổng trọng số candidate hợp lệ
    unanswerable_min_score: float = 0.9         # unanswerable cần chắc hơn (sai = -1)
    # v1.1 — độ tin cậy tầng A theo đặc trưng (decision/calibration.py)
    calibration_path: Optional[str] = None      # file sinh bởi `main calibrate` (độ chính xác out-of-fold theo intent)
    unexplained_factor: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.0])   # 0 / 1 / >=2 từ lạ
    health_factor: dict[str, float] = Field(default_factory=lambda: {'recon_mismatch': 0.5, 'header_rows': 1.0,
                                                                     'duplicate_sources': 1.0})
    recon_mismatch_max: float = 0.2             # tỉ lệ đơn lệch (dòng hàng vs tiền hàng của đơn) tối đa coi là khoẻ
    type_contract: bool = True                  # loại candidate lệch kiểu so với dạng câu hỏi
    # v2 — chỉ gọi strategy LLM khi tầng A chưa đủ chắc: quyết định pha 1 đã trả lời với support >= ngưỡng thì bỏ qua LLM
    llm_skip_support: float = 0.9


class Config(BaseModel):
    name: str = 'baseline'
    cache_dir: str = '/tmp/datamate_cache'
    workers: int = 8                            # process pool build catalog
    question_concurrency: int = 32
    time_budget_sec: float = 5.4 * 3600         # dừng sinh candidate mới khi vượt, phần còn lại abstain
    sandbox_timeout_sec: float = 30.0
    max_repair_rounds: int = 1
    llm: LLMConfig = Field(default_factory=LLMConfig)
    strategies: list[StrategyConfig] = Field(default_factory=lambda: [
        StrategyConfig(name='guard', weight=1.0),
        StrategyConfig(name='rule_plan', weight=1.0),
        StrategyConfig(name='llm_plan', weight=0.6),
        StrategyConfig(name='llm_code', weight=0.5, samples=2),
    ])
    decision: DecisionConfig = Field(default_factory=DecisionConfig)

    def strategy(self, name: str) -> Optional[StrategyConfig]:
        for s in self.strategies:
            if s.name == name:
                return s
        return None


def load_config(path: Optional[str] = None, overrides: Optional[dict] = None) -> Config:
    data: dict = {}
    if path:
        data = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    # biến môi trường cho triển khai (predict.sh)
    if os.environ.get('DATAMATE_LLM_URLS'):
        data.setdefault('llm', {})['base_urls'] = os.environ['DATAMATE_LLM_URLS'].split(',')
    if os.environ.get('DATAMATE_LLM_MODEL'):
        data.setdefault('llm', {})['model'] = os.environ['DATAMATE_LLM_MODEL']
    if os.environ.get('DATAMATE_NO_LLM') == '1':
        data.setdefault('llm', {})['enabled'] = False
    return Config(**data)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
