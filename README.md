# DataMate v1 — Zalo AI Challenge 2026

Agent trả lời câu hỏi tiếng Việt trên dữ liệu bán hàng của cửa hàng nhỏ (csv / xlsx / sqlite + README), chạy offline,
một container, LLM ≤ 15B qua vLLM. Chấm bằng Penalized Accuracy (+1 / −1 / 0) nên hệ thống **chỉ trả lời khi đủ chắc**.

## Kiến trúc 2 tầng

```
questions.jsonl ─┐
data/<biz>/ ─────┼─► data/ (raw_loader → profiler → schema_mapper → canonical → catalog [cache])
rulebook.md ─────┤
calendar.csv ────┘        │
                          ▼
                 understand/ (normalize → time_resolver → entity_link → intent → answerability)  → QContext + Plan?
                          │
        ┌─────────────────┼──────────────────────────┬───────────────────────────┐
        ▼                 ▼                          ▼                           ▼
   guard            rule_plan (tầng A)         llm_plan (tầng A')          llm_code (tầng B)
 (unanswerable)   Plan → Executor (pandas,    LLM điền Plan JSON →      LLM đọc README+rulebook+thẻ bảng
                  deterministic, rulebook)    Executor                   → code pandas → sandbox → tự sửa
        └─────────────────┴──────────────┬───────────┴───────────────────────────┘
                                         ▼
                        decision/voter (bỏ phiếu có trọng số, ngưỡng) → output/ (format + submission.jsonl + logs/)
```

* **Tầng A** (rule parser + thư viện metric): nhanh, chính xác tuyệt đối với dạng câu đã biết (doanh thu, số đơn, huỷ,
  AOV, SL bán, so sánh kỳ, tỉ trọng, xếp hạng, ngày/tháng cao nhất, ngưỡng, giá bán, công nợ, nhập hàng). LLM không bao giờ tự tính số.
* **Tầng B** (code agent): cho nghiệp vụ lạ / shop lạ (tồn kho, lô–hạn dùng, hiệu ứng tăng giá, hàng ế, trả góp, hoa hồng...).
  Code chạy trong tiến trình con có timeout, RAM giới hạn.
* Không có luật nào gắn với `business_id` hay `question_id` (cấm hardcode). Ánh xạ cột dựa trên **từ điển cột trong README**
  (bộ đề dùng từ vựng mô tả thống nhất) + đồng nghĩa tên cột + kiểm chứng bằng giá trị.

## Cấu trúc thư mục & người phụ trách

| Thư mục | Nội dung | Phụ trách |
|---|---|---|
| `datamate/core/` | types (hợp đồng dữ liệu), config, textutil, registry | Lead |
| `datamate/data/` | raw_loader, readme, profiler, schema_mapper, vocab, canonical, catalog | Data engineer |
| `datamate/knowledge/` | rulebook (đọc rulebook của split), calendar (Tết âm lịch) | Lead |
| `datamate/understand/` | normalize (teencode/Telex), time_resolver, entity_link, intent (rule parser), answerability | Backend 1 |
| `datamate/reasoning/` | executor (thư viện metric), prompts, strategies/{guard, rule_plan, llm_plan, llm_code} | Backend 1 (executor) + Lead (prompt/LLM) |
| `datamate/execution/` | sandbox + runner chạy code LLM | Backend 2 |
| `datamate/decision/` | voter | Lead |
| `datamate/output/` | formatter, writer (resume, logs) | Backend 2 |
| `datamate/runtime/` | llm_client (vLLM), orchestrator (asyncio), main (CLI) | Backend 2 |
| `datamate/offline/` | scorer (luật chấm BTC), eval (PA + breakdown + errors.jsonl) | Backend 2 |
| `configs/` | baseline (2 GPU), single_gpu, no_llm | Lead |
| `docker/` | Dockerfile, predict.sh | Backend 2 |
| `tests/` | unit + end-to-end (split tổng hợp) + regression trên train thật | mọi người |

## Chạy nhanh

```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # test không cần GPU

# Tầng A trên train (CPU, vài phút) + chấm
python -m datamate.runtime.main predict --split /path/train --out runs/a --config configs/no_llm.yaml --eval --fresh

# Đầy đủ (cần vLLM đang chạy, xem docs/DEPLOY.md)
python -m datamate.runtime.main predict --split /path/train --out runs/full --config configs/baseline.yaml --eval --fresh
python -m datamate.runtime.main predict --split /path/public_test --out runs/pub --config configs/baseline.yaml --fresh

# Debug
python -m datamate.runtime.main build --split /path/train --biz biz_123 -v      # xem ánh xạ cột / loại bảng
python -m datamate.runtime.main ask --split /path/train --id q_0c8f3e07768f --show-program
python -m datamate.runtime.main eval --split /path/train --pred runs/full/submission.jsonl
```

Kết quả: `submission.jsonl`, `logs/<qid>.json` (câu hỏi, intent, thời gian, mọi candidate kèm plan/code/lỗi, quyết định,
thời gian chạy — đúng yêu cầu log của BTC), `run_summary.json`, và khi `--eval`: `eval_report.json`, `errors.jsonl`.

## Quy trình cải tiến (mỗi vòng)

1. `make eval-train` → đọc `errors.jsonl` (câu sai, kèm plan/code của từng strategy) và breakdown theo intent/policy.
2. Sai ở tầng A → sửa `understand/intent.py` / `reasoning/executor.py` / `data/*`; sai ở tầng B → sửa prompt.
3. Thêm test regression cho lỗi vừa sửa; chạy `make test` + `make regress`.
4. Đổi ngưỡng bỏ phiếu trong `configs/*.yaml` để tối ưu PA (không cần sửa code).
5. Đánh giá tổng quát hoá: giữ hẳn 1 họ dữ liệu (vd sổ tay, SQLite) không nhìn khi phát triển, chỉ dùng để đo.

## Thêm strategy mới

```python
from datamate.core.registry import register_strategy
from datamate.reasoning.strategies.base import Strategy

@register_strategy('my_strategy')
class MyStrategy(Strategy):
    uses_llm = False
    async def run(self, u, cat, ix):          # u: Understanding, cat: BusinessCatalog, ix: EntityIndex
        return [self.cand(answer_type='number', answer=42)]
```
Import nó trong `reasoning/strategies/__init__.py` rồi thêm `- {name: my_strategy, weight: 0.5}` vào config.

## Trạng thái v1

* Regression nội bộ (3 shop có dữ liệu cục bộ, chỉ tầng A, không LLM): 277 câu, đúng 264, sai 0, bỏ 13 (PA 0.953).
  Số này **lạc quan** vì luật được phát triển trên chính các shop này — phải chạy `make eval-train` trên toàn bộ train.
* Chưa có ở tầng A (đang để tầng B): tồn kho / giá trị tồn, lô–hạn dùng, hiệu ứng tăng giá, hàng ế, nhập trừ bán, số đợt giao,
  lãi ròng, nghiệp vụ ngành mới (trả góp, hoa hồng, IMEI, voucher/điểm).
