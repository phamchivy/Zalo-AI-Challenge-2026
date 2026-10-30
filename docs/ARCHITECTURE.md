# Kiến trúc v1.1 — hợp đồng dữ liệu & điểm mở rộng

## 1. Luồng một lần chạy

```
predict --split S --out O
 ├─ KnowledgeBase.load(S)                 rulebook.md -> nguyên văn + RuleSet (rules.json: knobs, rankings, unknown)
 │                                         + override <S>/rules.json hoặc $DATAMATE_RULES (LLM/người điền), calendar.csv
 ├─ build_all(S/data, biz, cache)         process pool; cache pickle theo (hash code tầng data + kích thước/mtime file)
 │    raw_loader → profiler → schema_mapper(README + tên cột + giá trị) → canonical → BusinessCatalog
 ├─ EntityIndex.from_catalog              tên hàng/khách/nhóm/kênh/thanh toán/nhân viên có trong dữ liệu
 ├─ với mỗi câu (asyncio, gom theo shop):
 │    understand() → QContext + ParseResult(plan?, confidence, tier_b_hint)
 │    strategies không LLM (guard, rule_plan) → strategies LLM (llm_plan, llm_code) song song
 │    decide() → Decision → writer (submission.jsonl tăng dần + logs/<qid>.json)
 └─ finalize: đủ 1 dòng/id theo thứ tự đề, thiếu = abstain; run_summary.json
```
Train và test chạy **cùng một đường**; train chỉ khác là có `answers.jsonl` để `eval`. Không có bước "train model":
tập train dùng để (1) phát triển/kiểm luật, (2) sinh từ vựng chung (`offline/build_vocab.py`), (3) đo PA, chỉnh ngưỡng.

## 2. Đối tượng (datamate/core/types.py)

| Đối tượng | Sinh ở | Nội dung |
|---|---|---|
| `Question` | input | id, business_id, question, as_of_date |
| `BusinessCatalog` | data/catalog | raw tables, profiles, mapping (role từng cột, loại bảng), canon, readme |
| `TimeRange` | understand/time_resolver | start, end (gồm 2 đầu), label (main/compare/series_i/snapshot), granularity |
| `Plan` | intent / llm_plan | metric, op, periods, filters, group_by, split_by/values, k, threshold, answer_type |
| `QContext` | understand/pipeline | câu đã chuẩn hoá, thời gian, intent, thực thể, lý do unanswerable |
| `Candidate` | strategies | strategy, weight, program (plan/code), status, answer_type, answer, reason, elapsed |
| `Decision` | decision/voter | type, answer, confidence, support, policy |
| `Trace` | orchestrator | toàn bộ trên cho 1 câu → logs/<qid>.json |

## 3. Canonical (data/canonical.py)
`lines` (mỗi dòng hàng) · `orders` (mỗi đơn, gồm đơn không có dòng) · `products` · `units` · `customers` · `payments` · `purchases`
+ `caps` (năng lực: orders, status, customer_code, customer_group, category, time, channel, payment, staff, cost_line,
payments, purchases, debt, rx, ingredient, no_lines) + `date_min/date_max` (README "Dữ liệu có đến hết").

Luật đọc file đã áp: ô trống dòng sau = dòng đầu nhóm (ffill theo đơn) · dòng trùng trong cùng đơn tính 1 lần · ngày không năm
lấy năm của sheet/file · số lượng âm / bảng trả hàng / SL trả > 0 ⇒ cả đơn `returned` · voucher/dùng điểm/phí ship ⇒ `is_nonproduct`
· "Phí phục vụ" ⇒ `is_service` · khách: mã nếu có, không thì tên bỏ số điện thoại + bỏ dấu; khách lẻ = None.

## 4. Ánh xạ cột (data/schema_mapper.py)
`ROLES[role] = (kiểu, regex mô tả README, regex tên cột)`; điểm = max(mô tả 3, tên 2) − 2.5 nếu giá trị sai kiểu; mỗi cột 1 role,
mỗi role 1 cột / bảng; `classify_table` theo tên bảng + tập role; `RENAME` đổi role chung theo loại bảng
(vd `qty` ở bảng nhập → `p_qty`). Thêm từ đồng nghĩa = sửa regex, chạy `main build -v` để xem lại toàn bộ shop.

## 5. Luật dạng dữ liệu (knowledge/rules.py) — v1.1
`compile_rules(rulebook)` tách bảng luật thành câu (mục, tên luật, nội dung) rồi khớp với `KNOBS`
(mỗi knob = key, giá trị, regex tên luật, mọi regex nội dung phải khớp):
* knob khớp -> giá trị; không khớp (rulebook không nói / nói khác) -> `None` -> intent cần knob đó **bỏ trống**;
* câu trong mục xếp hạng có dạng "<đối tượng> <tính từ> nhất | ... nhiều X nhất" -> `RankingDef(phrase, entity, metric, min_orders)`;
* câu không khớp gì -> `unknown` (luật lạ) với `keys` = cụm từ nội dung của tên luật; câu hỏi chứa đủ cụm -> bỏ trống.
Executor khai báo knob + năng lực dữ liệu cho từng Plan (`Executor.requirements`), thiếu -> `RuleMissing`.
Thêm luật mới = thêm 1 dòng `K(...)` + đọc `self.r.get(key)` trong executor; LLM có thể điền cùng schema vào `rules.json`.

## 6. Bỏ phiếu & độ tin cậy (configs/*.yaml → decision)
Trọng số: guard 1.0 · rule_plan = base(intent) × f_từ_lạ × f_sức_khoẻ · llm_plan 0.6 · llm_code 0.5/mẫu.
`base(intent)` = cận dưới Wilson của độ chính xác out-of-fold trên train (`configs/calibration.yaml`, sinh bởi `main calibrate`),
chưa có file -> prior của parser. `f_từ_lạ` = [1, 0.5, 0] theo số từ lạ; `f_sức_khoẻ` theo `canon.health`
(lệch đối chiếu dòng hàng/đơn...). Trả lời khi `support ≥ min_score` (no_llm: 0.8 = độ chính xác tối thiểu) và
`support/total ≥ min_share`. Trước khi bỏ phiếu: **hợp đồng kiểu** (`understand/expected_type.py`) loại candidate lệch kiểu.

## 7. Việc tiếp theo theo người

**Data engineer — `data/`**
- Chạy `main build --split train -v` và `--split public_test -v`, soát role/loại bảng của từng shop; ghi lại shop sai vào test.
- Canonical cho tồn kho (thẻ kho + tồn cuối snapshot), lô/hạn dùng, lịch sử giá, giao hàng, chi phí → mở khoá tầng A cho
  inventory / expiry / price_effect / delivery / net_profit.
- Định dạng đặc biệt: SL kèm đơn vị khác dòng ("2 thùng"), quy đổi dạng "1 thùng = 4 bịch" trong danh mục, sổ ghi tắt.

**Backend 1 — `understand/` + `reasoning/executor.py`**
- Lấy `errors.jsonl` + các câu `rule_plan skipped` trên train, bổ sung intent: tồn theo đơn vị lớn [chẵn, lẻ], hàng ế, hiệu ứng
  tăng giá, thứ trong tuần, xếp hạng tăng dần, nhập trừ bán, đợt giao, lãi ròng.
- Mỗi intent mới: thêm test vào `tests/` với câu train thật (không dùng câu public/private).

**Backend 2 — `execution/`, `runtime/`, `docker/`, `offline/`**
- Dựng vLLM trên máy thuê, đo tok/s và thời gian/câu; pool sandbox giữ dữ liệu trong RAM (đỡ nạp pickle mỗi lần).
- Docker + predict.sh chạy `--network none`, đo tổng thời gian private giả lập; checksum, upload.
- Công cụ phân tích lỗi: gom lỗi theo intent/strategy, so sánh 2 lần chạy (ổn định |ΔPA|).

**Lead — prompt, voter, cấu hình**
- Prompt tầng B: thêm ví dụ code mẫu tổng quát (viết tay, không chép đáp án), hướng dẫn đọc README ngành mới.
- Tinh chỉnh trọng số/ngưỡng theo PA; hold-out 1 họ dữ liệu để đo khả năng tổng quát hoá.
