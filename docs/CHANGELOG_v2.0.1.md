# DataMate v2.0.1: sửa lỗi khi chạy trên máy GPU thật (Vast 2× 5090, CPU 96 nhân)

| Lỗi | Trước | Sau |
|---|---|---|
| Kernel chết ngay khi khởi động | Worker giới hạn bộ nhớ 12 GB. numpy, OpenBLAS và DuckDB mở 1 luồng cho mỗi nhân (96 nhân), nên vượt giới hạn bộ nhớ ngay lúc import. Log báo `cannot allocate memory for thread-local data: ABORT`, rồi `Broken pipe` và khởi động lại liên tục | Worker chỉ chạy 1 luồng tính toán (`OMP/OPENBLAS/MKL_NUM_THREADS=1`), `MALLOC_ARENA_MAX=2`, DuckDB 2 luồng. Các biến được đặt trước khi tạo worker |
| Ghi chú shop bị bỏ hết | So tên bảng gần như tuyệt đối. LLM viết "Đơn hàng", "BaoCao::Đơn hàng" hay `tables['x']` đều không khớp (biz_106: 0 bảng, bỏ 18 khẳng định) | So lỏng: bỏ dấu, hoa thường, đuôi file và cú pháp `tables['..']`; khớp tên file hoặc tên sheet; chấp nhận khớp một phần nếu chỉ có một bảng phù hợp. Kiểm tra cột và phép nối vẫn giữ nguyên |
| `requirements.txt` ghi đè môi trường vLLM | Ghim cứng numpy 2.5.3 và pydantic 2.13.5, xung đột với numba / mistral-common của vLLM | Chuyển sang khoảng phiên bản. `setup.sh` cài kèm ràng buộc theo đúng bản numpy / pydantic / torch mà vLLM đang dùng, rồi chạy `pip check` |
| `setup.sh` không tải được mô hình | Dùng `huggingface-cli`, lệnh đã bị bỏ trong huggingface_hub mới | Dùng `hf download`, nếu không có thì dùng lại `huggingface-cli`. Bật `hf_transfer` để tải nhanh hơn |
| Đĩa nhỏ | Báo lỗi giữa chừng khi tải mô hình | Kiểm tra dung lượng đĩa trước (khoảng 35 GB mỗi mô hình + 15 GB), thiếu thì dừng ngay. Bỏ qua bước cài vLLM nếu template đã có sẵn |

Test: 49 passed, tầng A không đổi.
