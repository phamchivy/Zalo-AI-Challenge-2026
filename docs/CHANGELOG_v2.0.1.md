# DataMate v2.0.1: sửa lỗi khi chạy trên máy GPU thật (Vast 2× 5090, CPU 96 nhân)

| Lỗi | Trước | Sau |
|---|---|---|
| Kernel chết ngay khi khởi động | Worker giới hạn bộ nhớ 12 GB. numpy, OpenBLAS và DuckDB mở 1 luồng cho mỗi nhân (96 nhân), nên vượt giới hạn bộ nhớ ngay lúc import. Log báo `cannot allocate memory for thread-local data: ABORT`, rồi `Broken pipe` và khởi động lại liên tục | Worker chỉ chạy 1 luồng tính toán (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), `MALLOC_ARENA_MAX=2`, DuckDB 2 luồng. Các biến được đặt trước khi tạo worker |
| Ghi chú shop bị bỏ hết | So tên bảng gần như tuyệt đối. LLM viết "Đơn hàng", "BaoCao::Đơn hàng" hay `tables['x']` đều không khớp (biz_106: 0 bảng, bỏ 18 khẳng định) | So lỏng: bỏ dấu, hoa thường, đuôi file và cú pháp `tables['..']`; khớp tên file hoặc tên sheet; chấp nhận khớp một phần nếu chỉ có một bảng phù hợp. Kiểm tra cột và phép nối vẫn giữ nguyên |
| `requirements.txt` ghi đè môi trường vLLM | Ghim cứng numpy 2.5.3 và pydantic 2.13.5, xung đột với numba / mistral-common của vLLM | Chuyển sang khoảng phiên bản. `setup.sh` cài kèm ràng buộc theo đúng bản numpy / pydantic / torch mà vLLM đang dùng, rồi chạy `pip check` |
| `setup.sh` không tải được mô hình | Dùng `huggingface-cli`, lệnh đã bị bỏ trong huggingface_hub mới | Dùng `hf download`, nếu không có thì dùng lại `huggingface-cli`. Bật `hf_transfer` để tải nhanh hơn |
| Đĩa nhỏ | Báo lỗi giữa chừng khi tải mô hình | Kiểm tra dung lượng đĩa trước (khoảng 35 GB mỗi mô hình + 15 GB), thiếu thì dừng ngay. Bỏ qua bước cài vLLM nếu template đã có sẵn |

Test: 49 passed, tầng A không đổi.

## v2.0.2
| Lỗi | Trước | Sau |
|---|---|---|
| Code LLM sửa bảng "không ăn" | LLM viết kiểu pandas cũ `df[col].fillna(0, inplace=True)`. Với pandas 3 (Copy-on-Write), lệnh này không sửa gì, chỉ in cảnh báo `ChainedAssignmentError` ra màn hình mà agent không thấy, nên kết quả sai âm thầm | Kernel biến cảnh báo này thành lỗi, kèm hướng dẫn cách viết lại để agent đọc được và sửa ở bước sau. Prompt thêm 1 dòng: không dùng `inplace=True` trên cột |

## v2.0.3
| Lỗi | Trước | Sau |
|---|---|---|
| Vượt ngữ cảnh 32k token (HTTP 400) | Ngân sách tính theo **ký tự** (giả định khoảng 2,5 ký tự/token) và chỉ rút gọn phần lịch sử. Với tiếng Việt và bảng số, mỗi token ít ký tự hơn, nên rulebook + thông tin shop + lịch sử vượt 32k token. Lời giải bị hỏng, server còn bị gửi lại y nguyên 3 lần | Ngân sách tính theo **token** (`context_tokens` 32768 trừ `max_tokens_step`, ước 2,0 ký tự/token). Trước mỗi lượt, hội thoại được ép cho vừa: rút gọn mạnh các bước cũ, rồi cắt bớt thông tin shop, cuối cùng mới cắt rulebook. Khi server vẫn báo vượt, agent đọc số token thật trong lỗi để **tự chỉnh tỉ lệ** ký tự/token, ép lại và gửi lại (tối đa 2 lần). Lỗi vượt ngữ cảnh không còn bị gửi lại y nguyên |
| Ghi chú shop sai tên bảng (biz_106 bỏ 18/18) | LLM tự viết tên bảng | Schema JSON **chỉ cho phép đúng tên bảng có thật** (enum, vLLM ép khi sinh). Prompt liệt kê tên hợp lệ. Log in các tên bị loại để dò lỗi |

## v2.0.4
| Lỗi | Trước | Sau |
|---|---|---|
| Cảnh báo từ code LLM (chia cho 0 ra NaN, ngày không rõ định dạng ...) | In ra console, làm rối log. Agent không thấy nên có thể trả lời NaN hoặc ngày đọc sai | Kernel thu các cảnh báo, gộp trùng, lấy tối đa 4 dòng và đưa vào kết quả chạy dưới mục "CẢNH BÁO:" để agent tự sửa. Console sạch |

## v2.0.5
| Lỗi | Trước | Sau |
|---|---|---|
| Vẫn vượt 32k token, cắt không giảm (83 550 ký tự cả 3 lần gửi lại) | Phần **câu hỏi** không bao giờ bị cắt, mà danh sách "thực thể đã khớp" có thể chứa hàng nghìn sản phẩm. vLLM chỉ báo "ít nhất 32 769 token" (cận dưới), nên tỉ lệ ký tự/token không giảm | Mỗi danh sách thực thể giữ tối đa 15 phần tử, toàn bộ tối đa 1 500 ký tự. Khi ép cho vừa, cắt thêm phần câu hỏi và 2 message cuối nếu vẫn còn quá dài. Lỗi chỉ cho cận dưới thì giảm tỉ lệ ký tự/token 25% mỗi lần. Log in kích thước từng message |
