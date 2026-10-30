# Kiến trúc v1 — hợp đồng dữ liệu & điểm mở rộng

## 1. Luồng một lần chạy

```
predict --split S --out O
 ├─ KnowledgeBase.load(S)                 rulebook.md (nguyên văn + tham số ca/Tết), calendar.csv
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

## 5. Bỏ phiếu (configs/*.yaml → decision)
Trọng số mặc định: guard 1.0 · rule_plan 1.0 × confidence(0.45–0.9) · llm_plan 0.6 · llm_code 0.5/mẫu.
Trả lời khi `support ≥ min_score (0.5)` và `support/total ≥ min_share (0.6)`; unanswerable cần `≥ 0.9`.
`confidence` của rule giảm khi câu có từ lạ không khớp được (`unexplained`) → tự gọi tầng B kiểm chéo.
Tối ưu các ngưỡng này bằng PA trên train (không sửa code).

## 6. Việc tiếp theo (v1.1) theo người

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
