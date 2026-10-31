# DataMate v2.0: agent nhiều bước (kèm v2.1 ghi chú shop, v2.2 luật liên quan, v2.3 chọn ngưỡng)

Tầng A (không LLM) giữ nguyên và vẫn chạy trước cho mọi câu. Câu tầng A **chưa đủ chắc** được chuyển cho agent: LLM
làm việc như một nhà phân tích, mỗi lượt viết code, chạy, đọc kết quả rồi tính tiếp. Mỗi câu có 4 lời giải theo 4 hướng
khác nhau, bỏ phiếu cùng tầng A. Tất cả chạy offline với một mô hình ≤15B qua vLLM.

## Trước / sau

| Hạng mục | v1.3 | v2.0 |
|---|---|---|
| Câu tầng A chưa chắc | Bỏ trống (hoặc 1 lần sinh code, không vòng lặp) | Agent: tối đa 8 lượt *viết code → chạy → xem kết quả*, có thể xem dữ liệu trước khi tính |
| Môi trường chạy code | Mỗi lần chạy nạp lại dữ liệu | **Kernel giữ trạng thái** theo shop: dữ liệu nạp 1 lần, biến của bước trước còn ở bước sau; có timeout, giới hạn RAM, tự khởi động lại khi treo; mỗi lời giải có bản sao bảng riêng (copy-on-write) |
| Công cụ cho LLM | pandas | pandas, `sql()` (DuckDB) trên bảng chuẩn hoá và bảng gốc, `answer(type, value, reason)` |
| Số lời giải / câu | 1–2, giống nhau | 4, mỗi lời giải một hướng: pandas trên canon / SQL / bảng gốc / tự kiểm tra bằng cách khác. Bỏ phiếu có trọng số cùng tầng A |
| Kiểm định đáp án | Hợp đồng kiểu | Thêm: chuẩn hoá kiểu; cảnh báo giá trị đáng ngờ (0, âm, % < 1 khi hỏi tỉ trọng, số đếm lẻ, sai số phần tử). Agent phải kiểm tra lại 1 lần, sai kiểu thì loại |
| Thứ tự chạy | Mọi strategy chạy cùng lúc cho mỗi câu | **2 pha**: pha 1 tầng A cho mọi câu, câu chắc (≥ `llm_skip_support`) ghi ngay; pha 2 chạy LLM theo từng shop (tái dùng cache tiền tố của vLLM) |
| Giờ chạy | Tính từ lúc bắt đầu predict | Tính từ lúc container khởi động (`DATAMATE_T0`, gồm cả nạp mô hình). Hết giờ thì câu còn lại lấy đáp án tầng A |
| **v2.1** Ghi chú về shop | — | LLM đọc README + mô tả bảng 1 lần mỗi shop: bảng nào chứa gì, nối bằng cột nào, quy ước riêng, điểm dễ sai, thông tin shop không ghi. **Kiểm chứng bằng dữ liệu** (bảng / cột phải tồn tại, phép nối phải khớp ≥ 50%), lưu cache |
| **v2.2** Luật liên quan | Cả rulebook trong prompt | Vẫn đưa cả rulebook (được cache), thêm 3–6 dòng luật liên quan nhất ngay cạnh câu hỏi (truy hồi theo IDF, không dùng LLM) |
| **v2.3** Chọn ngưỡng | `calibrate` (chỉ tầng A) | `main sweep`: chạy lại bộ bỏ phiếu trên logs dev với nhiều cặp ngưỡng, kèm bảng "số lời giải đồng ý → tỉ lệ đúng" |
| Giới hạn 15B tổng | 2 bản mô hình (mỗi GPU 1 bản) | Mặc định **1 bản chia 2 GPU** (tensor-parallel, FP8). Chế độ 2 bản để tuỳ chọn, chỉ dùng khi BTC cho phép |
| Thử nghiệm | Sửa config | `--set strategy.agent.options.shop_notes=false`, `--ids-file`, `make-subset` (tập dev để so mô hình) |
| Script GPU | — | `scripts/vast/setup.sh, serve.sh, stop.sh, compare_models.sh`, `scripts/bench_llm.py`, `docs/V2_RUNBOOK.md` |
| Test | 44 | 49 (agent với LLM giả: 2 bước, bỏ phiếu, kiểm tra lại, ghi chú shop bị loại khẳng định sai) |

## Đã kiểm tra trên máy không GPU
- **Tầng A không đổi:** 3 shop train, `no_llm` vẫn PA 1.000 (277/0/0).
- **Đường HTTP thật:** dùng server giả tương thích OpenAI. Agent chạy 2 bước, 4 lời giải, đáp án đi qua bỏ phiếu. Khi 4 lời giải không khớp tầng A thì bỏ trống, đúng thiết kế.
- **Kernel:** giữ biến giữa các bước; vòng lặp vô hạn bị ngắt sau timeout và session vẫn dùng được tiếp; lỗi trả về traceback gọn; `sql()` chạy được trên shop thật.

## Chưa đo được (cần GPU, theo `docs/V2_RUNBOOK.md`)
- **Thông lượng:** tp và dp trên 2× 5090.
- **So mô hình:** Qwen2.5-Coder-14B và Qwen3-14B trên 200 câu dev.
- **Bật / tắt từng phần:** v2.1, v2.2, số lời giải, chế độ suy nghĩ.
- **PA:** dev, holdout, public; thời gian 1000 câu public, dùng để ước tính private.

## Ngưỡng mặc định (`configs/v2_agent.yaml`, sẽ chỉnh theo `sweep`)
- **Trọng số phiếu:** tầng A theo hiệu chỉnh; mỗi lời giải agent 0.3.
- **Ngưỡng trả lời:** `min_score` 0.8, `min_share` 0.6.
  - Không có tầng A: cần ≥ 3/4 lời giải cùng đáp án.
  - Tầng A chưa chắc: chỉ cần ≥ 1–2 lời giải agent đồng ý.
- **"Không trả lời được" từ agent:** nhân 0.8, nên chỉ thắng khi cả 4 lời giải cùng kết luận.

## Chưa làm
- **LLM tự điền các tham số luật của tầng A.** Các tham số này chưa có mô tả nghĩa dạng chữ để LLM đối chiếu. Hiện agent đọc thẳng rulebook, và tầng A vẫn bỏ trống khi luật khác.
- **Docker theo image BTC.** Chờ BTC công bố image. `docker/predict.sh` đã hỗ trợ tp / dp, FP8 và `DATAMATE_T0`.
