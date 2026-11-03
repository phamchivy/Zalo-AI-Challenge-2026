# DataMate v2.2: sửa tầng A theo log public + LLM không làm mất điểm

Nguồn phân tích: log `runs/v2_public` (1000 câu) và 3 lượt nộp ghép để đo riêng từng phần.
- Bản v1.3 (chỉ tầng A) đạt 0,380, trả lời 674 câu, tức khoảng 527 đúng và 147 sai.
- LLM ở v2: câu số / ngày / tên −8 điểm, câu trả lời 0 là −17, câu "không trả lời được" +6, làm mất đáp án tầng A khoảng −10.

Mọi chỗ sửa đều là **quy tắc đọc câu chung**: không theo mã câu hay mã shop, không viết riêng cho tập public. Đối chiếu
với log public bằng `scripts/parse_harness.py`: có **64 câu đổi cách đọc**, trong đó **49 câu trước đây tầng A đã trả
lời** (phần lớn là sai kỳ / sai loại câu).

## 1. Đọc thời gian

| Lỗi | Ví dụ | Trước | Sau |
|---|---|---|---|
| Tháng/năm viết tắt không có chữ "tháng" | "2/25 bị huỷ mấy đơn", "4/26 doanh thu" | Cả khoảng dữ liệu (2025-01..hôm nay) | Tháng 02/2025. Chỉ nhận khi năm 2 chữ số từ 20 đến 39, nên "1/5" (ngày/tháng) không đổi |
| Dải tháng viết tắt | "T1/2026 tới T3/2026", "từ th12/2025 đến th2/2026", "từ 02/2025 tới 04/2025" | Chỉ 2 tháng đầu và cuối, mất tháng giữa | Đủ mọi tháng (best_month / series / trend) |
| Nhiều tháng chung một năm | "các tháng 10, 11, 12 năm 2025", "tháng 3, tháng 4, tháng 5 năm 2025" | Năm gán sai (2026) hoặc thành cả năm | Đúng từng tháng của năm nêu |
| Mốc "tính luôn hôm nay" | "Tính luôn hnay thì 7 ngày gần nhất" | Chỉ hôm nay | 7 ngày, có hôm nay |
| "Đầu năm tới chừ" (không có "từ") | | Chỉ hôm nay | Từ 01/01 tới hôm nay |
| Tháng này so với tháng trước | "từ mùng 1 tới hết hôm nay so cùng kỳ tháng trước", "cùng số ngày đầu tháng rồi", "tháng này tới giờ so với tháng trước" | Chỉ hôm nay, hoặc lùi theo độ dài kỳ (03-15..03-31) | Cùng số ngày đầu tháng (rulebook §2 "Tháng này") |
| "Tháng M năm ngoái / năm rồi" | "tháng 5/2026 so với tháng 5 năm ngoái" | So với tháng liền trước | Tháng M của năm trước |
| "Tháng sau ... cao hơn tháng trước" | câu hỏi xu hướng | Thêm kỳ "tháng trước" của hôm nay vào chuỗi | Là quan hệ, không phải kỳ |
| "Tháng rồi" | | Không nhận | Tháng liền trước |

## 2. Loại câu hỏi

| Lỗi | Ví dụ | Trước | Sau |
|---|---|---|---|
| Giá trị trung bình một đơn đảo trật tự | "mỗi đơn trung bình đem về", "bình quân mỗi o đờ" | Doanh thu **tổng** | AOV |
| So sánh nhưng hỏi tháng cao nhất | "So sánh 3/25, 4/25, 5/25 xem tháng nào nhiều tiền nhất" | Chênh lệch 2 tháng | Tháng cao nhất |
| Hiệu ứng tăng giá không có chữ "giá" | "Từ khi Bạc xỉu (M) lên 35.000đ, 14 ngày đầu so 14 ngày trước" | Tỉ trọng doanh thu | price_effect |
| Câu công nợ "bán chịu", "ngày phải trả" | "mấy đơn bán chịu ghi ngày phải trả trong th2/2026" | Đếm **mọi** đơn của 1 ngày | Nghiệp vụ công nợ: parser công nợ, không có thì tầng B |
| Câu giao hàng | "mấy đơn đã giao từ 2 đợt", "đã chở đi bao nhiêu thùng" | Số đơn / số lượng bán | Tầng B. Chỉ "bao nhiêu đợt giao" mới tính số đợt |

## 3. Vai trò LLM (`decision`, đo trên public)

| | v2.1 | v2.2 (`configs/v2_agent.yaml`) |
|---|---|---|
| Câu tầng A đã trả lời nhưng độ tin cậy < 0,9 | Bỏ phiếu chung với agent, có thể bị agent đổi hoặc xoá | **Giữ nguyên đáp án tầng A** (`llm_keep_tier_a`) |
| Câu tầng A bỏ trống | Agent được trả lời mọi kiểu | Chỉ nhận **"không trả lời được"** (`llm_answer_types: [unanswerable]`) |
| Đáp án 0 / 0% / ±100% chỉ từ LLM | Nhận | Bỏ trống (`llm_reject_zero`) |

Mô phỏng trên log public: tầng A giữ đủ **674** câu như v1.3, cộng 12 câu "không trả lời được". Chặn 57 đáp án agent
(35 số, 8 tên, 6 boolean, 5 ngày, 3 %). Ba nút này cấu hình được; `--set decision.llm_answer_types=null` trả về cách cũ
để phân tích.

## 4. Khác
- `configs/dev_27b.yaml`: phát triển agent trên máy chủ công ty (Qwen3.6-27B, thinking, 6 câu song song). **Không dùng để nộp bài.**
- `scripts/parse_harness.py`: chạy phần đọc câu (thời gian + loại câu) trên mọi câu của một log, không cần dữ liệu shop.
  So hai lần chạy để biết một thay đổi ảnh hưởng tới những câu nào.

## Kiểm tra
- Test: 55 qua. Thêm `tests/test_v22.py`: tháng/năm viết tắt, dải tháng, mốc hôm nay, tháng này so tháng trước, tháng năm ngoái.
- 3 shop train (rulebook public dựng lại): **264 đúng, 0 sai, 13 bỏ**, giống hệt v2.1.2.
- Ước tính public: tầng A khoảng **+0,03 đến +0,08** (49 câu đã trả lời đổi cách đọc), LLM +0,006. Chưa đo, cần nộp để xác nhận.
