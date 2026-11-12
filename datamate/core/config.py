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


class MappingLLMConfig(BaseModel):
    """v3.0.0 — LLM cho bước dựng dữ liệu (gán nhãn trạng thái/thanh toán, gán vai trò cột còn thiếu).
    Đổi model chỉ cần đổi base_url / model / api_key_env / json_mode (xem configs/llm_map_*.yaml).
    Biến môi trường ghi đè: DM_LLM_URL, DM_LLM_MODEL, DM_LLM_KEY (khoá API không bao giờ ghi trong file)."""
    enabled: bool = False
    base_url: str = 'http://127.0.0.1:8000/v1'
    model: str = 'qwen3-14b'
    api_key_env: str = 'DM_LLM_KEY'
    json_mode: str = 'auto'                     # schema | object | prompt | auto
    disable_thinking: bool = True               # Qwen3: tắt thinking; model khác bỏ qua tham số này
    temperature: float = 0.0
    seed: Optional[int] = 42
    max_tokens: int = 4096
    timeout_sec: float = 120.0
    retries: int = 3
    cache: bool = True
    enums: bool = True                          # gán nhãn trạng thái / thanh toán mà từ điển không hiểu
    roles: bool = True                          # gán vai trò cột còn thiếu (chọn bằng invariants)
    min_conf: float = 0.8                       # độ tin cậy tối thiểu để nhận một vai trò do LLM gán


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
    # v2.2 — vai trò của LLM sau khi đo trên public (agent tự tính số: đúng ~40% trên shop lạ, lỗ điểm):
    llm_keep_tier_a: bool = True                # tầng A đã trả lời (kể cả độ tin cậy < llm_skip_support) thì giữ nguyên
    llm_answer_types: Optional[list[str]] = None  # kiểu đáp án LLM được tự đưa ra khi tầng A bỏ trống; None = mọi kiểu
    llm_reject_zero: bool = True                # đáp án 0 / 0% / ±100% chỉ từ LLM -> bỏ trống
    # v2.5.0 — invariants báo đỏ -> bỏ trống câu phụ thuộc phần dữ liệu đó; đáp án vô lý -> bỏ trống
    data_guard: bool = True


class Config(BaseModel):
    name: str = 'baseline'
    cache_dir: str = '/tmp/datamate_cache'
    workers: int = 8                            # process pool build catalog
    question_concurrency: int = 32
    time_budget_sec: float = 5.4 * 3600         # dừng sinh candidate mới khi vượt, phần còn lại abstain
    sandbox_timeout_sec: float = 30.0
    max_repair_rounds: int = 1
    llm: LLMConfig = Field(default_factory=LLMConfig)
    mapping_llm: MappingLLMConfig = Field(default_factory=MappingLLMConfig)
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
    for env, k in (('DM_LLM_URL', 'base_url'), ('DM_LLM_MODEL', 'model')):
        if os.environ.get(env):
            data.setdefault('mapping_llm', {})[k] = os.environ[env]
    if os.environ.get('DATAMATE_NO_LLM') == '1':
        data.setdefault('llm', {})['enabled'] = False
    return Config(**data)


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out
