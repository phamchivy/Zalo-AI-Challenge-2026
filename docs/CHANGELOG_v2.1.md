# DataMate v2.1: sửa vì sao agent hầu như không ra đáp án (theo log so mô hình 200 câu dev)

Lần chạy Qwen2.5-Coder-14B trên 200 câu dev: 161 câu chuyển cho LLM, agent chỉ chốt được 11 câu (8 đúng, 3 sai), 140 câu bỏ
trống vì `low_support`. Đọc 5 log mẫu: đa số lời giải **"hết lượt chưa trả lời" sau 8 bước, mỗi bước chạy lại y nguyên
một đoạn code lỗi**.

## Nguyên nhân và cách sửa

| Hiện tượng trong log | Nguyên nhân | Sửa |
|---|---|---|
| `canon['lines']['date'] = pd.to_datetime(...)` ở bước sau vẫn như chưa đổi, agent lặp mãi | Mỗi lần đọc `canon['x']` kernel trả **một bản sao mới**, nên sửa cột chỉ tác dụng trên bản sao bị bỏ đi | Bảng theo **từng lời giải**: lần đầu sao chép, các lần sau trả đúng đối tượng đó, sửa cột hay gán lại bảng đều giữ ở bước sau. Lời giải khác không bị ảnh hưởng |
| `L[L['date'] >= '2026-07-01']` lỗi, agent đổi qua đổi lại | Cột ngày của canon là `datetime.date` (object). pandas không so được với chuỗi | Trong kernel, cột ngày được đổi sang **datetime64**: so được với chuỗi, `pd.Timestamp`, `datetime`, dùng được `.dt.month`. `date(...)` và `datetime.date(...)` trong code trả `pd.Timestamp`. Lỗi so với `datetime.date` kèm gợi ý cách viết đúng |
| `tables['Danh sách hàng hóa']` / `tables['Danh_sach_hang_hoa']` KeyError | LLM đoán tên bảng | Tìm tên lỏng (bỏ dấu, `_` thay khoảng trắng, hoa thường, đuôi file). Sai hẳn thì lỗi liệt kê đúng tên bảng |
| Chạy lại y nguyên code đã lỗi tới hết 8 lượt (mỗi lời giải 90–135k token) | Không phát hiện lặp | Code trùng một bước trước: **không chạy**, nhắc lỗi cũ. Lặp lần 2: dừng lời giải đó (`lặp lại code cũ`). Prompt: mỗi bước chỉ viết code mới, đọc dòng lỗi và sửa đúng chỗ |
| Tồn kho = 0 − tổng bán (âm), khách hàng = tên hàng / mã đơn, "ai được quý mến nhất" = khách mua nhiều nhất | LLM tự giả định khi shop không có dữ liệu | Phần thông tin shop ghi rõ **loại dữ liệu bộ chuẩn hoá không tìm thấy** (khách, tồn kho, nhân viên, công nợ, giờ bán ...), kèm hướng dẫn: tìm trong bảng gốc trước, không có thì `unanswerable`. Prompt cấm giả định thay thế |
| Đáp án `table` rỗng được chấp nhận; lỗi `OverflowError` làm hỏng lời giải / câu | Kiểm định chưa đủ | Bảng rỗng = lỗi. Số NaN / vô cực / > 1e15 = lỗi, không sập. `answer('date', Timestamp)` tự đổi sang 'YYYY-MM-DD'. Lý do cắt ở 300 ký tự |

## Tốc độ (GPU 100% vì KV cache không đủ cho 96 hội thoại dài)

| | v2.0.x | v2.1 |
|---|---|---|
| KV cache vLLM | bf16 | **fp8** (`KV_DTYPE`, chứa gấp đôi) |
| `question_concurrency` | 24 (≈ 96 yêu cầu cùng lúc) | **8** (≈ 32) |
| `max_num_seqs` | 128 | 64 |
| `max_steps` | 8 | **6** (thêm dừng sớm khi lặp) |
| `cards_chars` / `history_chars` | 12000 / 52000 | 8000 / 36000 |

Test: 51 qua (thêm: dừng khi lặp code; bảng theo lời giải + cột ngày datetime64 + tên bảng lỏng).

## Chưa đổi (chờ số liệu `sweep`)
Ngưỡng bỏ phiếu. Ví dụ biz_128 "còn mấy chai nước suối": 2 lời giải cùng ra 48, 2 lời giải hỏng, tổng 0,6 < 0,8 nên vẫn
bỏ trống. Sau khi sửa lặp, số lời giải hỏng sẽ giảm. Ngưỡng sẽ chỉnh bằng `main sweep` trên log dev, không đoán.

## v2.1.1
| Lỗi | Trước | Sau |
|---|---|---|
| `phase2 crash: float() argument must be ... not 'dict'` | Agent trả `table` với dòng dạng `{cột: giá trị}` (kiểu `df.to_dict('records')`), hoặc ô chứa giá trị lồng. Bước chuẩn hoá đáp án gọi `float()` trên dict nên cả câu mất phần LLM, chỉ còn đáp án tầng A | Dòng dạng dict được đổi theo `columns`; ô lạ đổi thành chuỗi. Bộ bỏ phiếu bọc từng đáp án: đáp án không chuẩn hoá được chỉ loại riêng đáp án đó |

## v2.1.2
| Lỗi | Trước | Sau |
|---|---|---|
| `sweep` lỗi `float(None)` | Ô `None` trong đáp án bảng (đã sửa ở v2.1.1). `sweep` chưa bọc lỗi từng đáp án | `sweep` bỏ qua đáp án không chuẩn hoá được |
| `sweep` đếm cả log của lần chạy cũ (148 log cho 100 câu) | `--fresh` không xoá `logs/` và ghi nối vào `submission.jsonl` cũ | `--fresh` xoá submission + logs cũ của thư mục `--out`. `sweep` chỉ đọc log của các câu có trong `submission.jsonl` |
