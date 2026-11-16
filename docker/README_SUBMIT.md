# DataMate: đội Beyond Models (Zalo AI Challenge 2026)

Chạy: `bash /code/predict.sh` (không tham số).
- Đọc `/data` (chỉ đọc).
- Ghi `/result/submission.jsonl`, mỗi câu ghi ngay khi xong.
- Ghi `/result/logs/<id>.json`, mỗi câu một dòng JSON.
- Ghi `/result/system/` (log vLLM) và `/result/run_summary.json`.
- Offline hoàn toàn.

## Bố cục

| Đường dẫn | Nội dung |
|---|---|
| `/code/predict.sh` | Điểm vào: khởi động vLLM (Qwen3-14B-FP8, chia 2 GPU) rồi chạy `python -m datamate.runtime.main predict` |
| `/code/src/datamate/` | Mã suy luận |
| `/code/src/configs/qa_qwen14b.yaml` | Cấu hình cố định: model, seed, temperature, số luồng… BTC không cần sửa |
| `/code/src/configs/calibration.yaml` | Ngưỡng tin cậy tầng luật, hiệu chỉnh trên train (xem `training_code/`) |
| `/code/models/qwen3-14b-fp8/` | Trọng số và tokenizer Qwen3-14B-FP8 |
| `/code/training_code/` | Dữ liệu train, hiệu chỉnh, `data_manifest.json` |

## Cách làm

1. **Đọc dữ liệu.** Đọc csv / xlsx / sqlite và README của từng shop. Ánh xạ cột về bảng chuẩn (đơn hàng, dòng hàng, sản phẩm, khách…) bằng từ điển cột trong README và luật định dạng. LLM chỉ được dùng để gán vai trò cho cột lạ, và vai trò đó chỉ được giữ khi các kiểm tra bất biến của dữ liệu không xấu đi (thuật toán Hungarian chọn cách gán một–một). Đọc `rulebook.md` của chính tập đang chấm để lấy các luật nghiệp vụ.
2. **Tầng A (tất định).** Chuẩn hoá câu hỏi (teencode, viết tắt), đọc mốc thời gian theo `as_of_date` và `calendar.csv` (lịch âm, Tết), nhận dạng ý định, rồi lập kế hoạch (Plan JSON). Bộ thực thi pandas tính đáp án theo rulebook. Đáp án chỉ được giữ khi độ tin cậy, đã hiệu chỉnh trên train, vượt ngưỡng. Log ghi `code = src/datamate/reasoning/executor.py::Executor.run` và `inputs` = kế hoạch.
3. **Tầng LLM** (chỉ cho câu tầng A bỏ trống). Qwen3-14B viết code pandas và chạy trong kernel cô lập ngay trong container, có các hàm hỗ trợ `period`, `find`, `peek`, `lunar`, `solar`, `holidays`. Mỗi câu chạy 3 lời giải độc lập với 3 kiểu prompt, temperature 0.3 / 0.6 / 0.8, seed cố định, tối đa 10 bước. Chỉ trả lời khi **cả 3 lời giải cùng một đáp án**; tầng A tính là một phiếu thường. Không đủ phiếu thì bỏ trống (abstain).
4. **Chấm điểm phạt** (+1 / −1 / 0), nên hệ thống bỏ trống khi không chắc.
5. **Giới hạn thời gian.** Phần LLM tự dừng ở 19 800 s tính từ lúc container khởi động; các câu còn lại giữ đáp án tầng A hoặc bỏ trống.

Không có luật nào gắn với `business_id` hay `question_id`. Không có đáp án viết cứng.

## Mô hình

| Mô hình | Nguồn | Revision | Phát hành gốc | Giấy phép | Tham số |
|---|---|---|---|---|---|
| Qwen3-14B-FP8 (dùng để sinh và chạy code, gán vai trò cột) | https://huggingface.co/Qwen/Qwen3-14B-FP8 | ghi trong `training_code/data_manifest.json` (`inference_model.revision`) | 29/04/2025 | Apache-2.0 | 14.8B |

Tổng tham số mọi mô hình dùng khi suy luận: **14.8B** (≤ 15B). Không dùng mô hình embedding, rerank hay phân loại. vLLM 0.10.1.1, FP8, tensor-parallel 2, thinking tắt (`enable_thinking=False`). Vì vậy trường `thinking` trong log là `null`.

## Dữ liệu

Chỉ dùng **dữ liệu train BTC cung cấp** (30 shop, khoảng 3000 câu), và chỉ cho việc hiệu chỉnh ngưỡng. Không fine-tune, không sinh dữ liệu huấn luyện, không dùng dữ liệu public hay private test. Chi tiết xem `/code/training_code/README.md`.

## Tái lập

- Seed cố định (vLLM `--seed 42`, seed theo từng lời gọi).
- Tầng A tất định.
- Sai khác giữa hai lần chạy chỉ có thể đến từ việc vLLM gom batch trên GPU, ở các câu do LLM trả lời. Ngưỡng 3/3 phiếu giữ sai khác này nhỏ.
