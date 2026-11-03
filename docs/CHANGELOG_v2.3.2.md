# DataMate v2.3.2 — Sửa lỗi chung tìm được khi kiểm toán tầng A bằng agent 27B

Nguồn: `runs/audit27b` (agent Qwen 27B giải độc lập 718 câu tầng A đã trả lời). 666 khớp, 27 nghi sai, 19 lệch yếu.
Mọi sửa là quy tắc chung (cấu trúc dữ liệu / ngôn ngữ / rulebook), không theo shop hay id câu.

## Trước → Sau

| # | Lỗi | v2.3.1 | v2.3.2 |
|---|---|---|---|
| 1 | Sổ ghi tay (khoá đơn theo nhóm STT), cột giảm chỉ ghi ở **dòng đầu đơn** ("Bớt") | coi là giảm từng dòng, **trừ vào doanh thu** (dev_011: 9/12 câu doanh thu lệch agent đúng bằng tiền Bớt) | nhận là **giảm cả đơn** → `order_discount`, không trừ doanh thu (rulebook §1). Kiểm sau khi đã có khoá đơn |
| 2 | Top-N lấy số trong ngày tháng / chữ số | `Th 9/25 khách…` → 25 người, `Tháng 1/26 sp nào…` → 26 món, `Top năm món` → 1 | số dính `/` `-` không phải N; `top năm`, `ba khách`… đọc thành số |
| 3 | `trong các khách hàng, ai … nhất` | trả list | trả string (cụm phạm vi không phải hỏi danh sách) |
| 4 | `dthu đơn tại quầy bao nhiêu, đơn qua điện thoại bao nhiêu` | trả **số đơn** | trả doanh thu |
| 5 | `bữa nào … nhiều tiền nhất` | tổng tiền | ngày cao nhất |
| 5 | `mấy đơng tiền hàng từ 760k trở lên` | doanh thu | đếm đơn theo ngưỡng |
| 5 | `… cộng lại được bao nhiêu tiền hàng` | lãi gộp ("lại được" bị nhận là "lãi được") | doanh thu |
| 5 | `bán được mấy đồng` | số lượng | doanh thu |
| 6 | Guard "ngoài khoảng dữ liệu" với kỳ là **hạn trả / hạn dùng** | unanswerable | không áp (ngày thuộc tính được nằm sau dữ liệu) |
| 6 | `máy mới lẫn máy cũ` khớp nhầm từ khoá liều dùng `mỗi lần mấy` | unanswerable | chỉ khi `mỗi lần mấy viên/gói/ml…` |
| 6 | `huỷ chủ yếu vì sao` | unanswerable (chủ quan) | không kết luận (lý do huỷ có thể nằm trong dữ liệu) |
| + | Đối soát tiền khách trả với tổng dòng hàng | — | `health.paid_eq_lines`, `paid_eq_lines_minus_od` trong run_summary (chẩn đoán) |
| + | Kernel worker in traceback BrokenPipe/ConnectionReset khi bị khởi động lại | traceback dài trong log | thoát êm |

## Đo offline
- Harness trên log v23: 5 câu đổi loại câu (4 đã trả lời — đều là lỗi ở bảng trên); thêm các câu đổi top-N / kiểu đáp án / guard.
- Train mini (no_llm): 264 đúng / 0 sai / 13 bỏ — không đổi. pytest 65 passed.
- **Chưa kiểm được trên train có sổ dạng "Bớt"** (không có trong train mini). Chạy kiểm train dev trước khi nộp (lệnh bên dưới).

## Chạy
```bash
git am datamate_v2.3.2.patch
# 1) kiểm train dev không tụt (so với v2.3.1)
python -m datamate.runtime.main predict --split <train> --split-file configs/split.yaml --part dev --out runs/v232_dev --config configs/no_llm.yaml --eval --fresh
# 2) public
python -m datamate.runtime.main predict --split <public_test> --out runs/v232_public --config configs/no_llm.yaml --fresh
python merge.py runs/v232_public/submission.jsonl runs/submission_v2.jsonl v232_best.jsonl --only-unans
```
