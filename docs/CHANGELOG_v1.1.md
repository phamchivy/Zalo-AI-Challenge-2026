# DataMate v1.1 — tầng A tổng quát, precision-first

Mục tiêu: tầng A (không LLM) **chỉ trả lời khi chắc**. Luật được đọc từ rulebook của split đang chạy, không viết cứng theo
bộ train. Các giao diện (`rules.json`, `Plan`, hợp đồng kiểu, voter) giữ nguyên để sau này gắn LLM vào.

## Trước / sau

| Hạng mục | v1 | v1.1 |
|---|---|---|
| Luật nghiệp vụ | Viết cứng theo rulebook train; chỉ đọc từ rulebook giờ ca và ngày Tết | 74 nút luật trong `rules.json`, đọc từ rulebook của split. Nút không xác nhận được thì intent bỏ trống. Có thể ghi đè bằng `<split>/rules.json` hoặc `$DATAMATE_RULES` |
| Luật lạ / ngành mới | Không phát hiện, vẫn chạy logic cũ | Liệt kê luật lạ; câu hỏi chạm vào luật lạ thì bỏ trống. Rulebook public (dựng lại): 9 luật lạ — thu máy cũ, trả góp, bảo hành, lượt xe, voucher… |
| Luật bị bỏ ở public | Vẫn áp luật train | Nút về None, tự tắt intent: giá nhập bình quân, tồn theo đơn vị lớn, giá bán ngày D |
| Xếp hạng | Viết cứng (bán chạy, nhiều tiền) | Thêm định nghĩa đọc từ rulebook: khách trung thành / sộp / hay đổi ý, mặt hàng ưa chuộng, nhân viên chăm chỉ, ngày đắt khách (theo **số đơn**), giờ cao điểm, nhà cung cấp chính (theo **số phiếu**) |
| Câu không hiểu | Rơi về "doanh thu" | Bỏ fallback; thiếu từ khoá số đo hoặc tiêu chí xếp hạng thì bỏ trống |
| Kiểu đáp án | Không kiểm | Hợp đồng kiểu suy từ dạng câu hỏi, áp cho mọi strategy. Trên 633 câu train có đáp án: 0 vi phạm, chặn 14–16 câu v1 trả lời sai |
| `%` trong tên hàng ("Otrivin 0,1%") | Hiểu thành câu hỏi tỉ trọng | Ký tự `%` đứng ngay sau chữ số không coi là câu hỏi phần trăm |
| "tăng/giảm", "tawng/giamr" | Không nhận | Đổi thành "tăng hoặc giảm"; giữ nguyên tên "P/S", "8/2025" |
| Dòng "Tổng tháng…", "Cộng", "Total" | Tính như đơn (lỗi gấp 2–3 lần) | Bỏ; so khớp có dấu nên phân biệt được họ **Tống** với **Tổng** |
| Dòng không có hàng và số lượng nhưng có tiền | Tính vào doanh thu | Không coi là mặt hàng (vẫn giữ thông tin cấp đơn) |
| Nhiều bảng bán hàng trùng đơn | Cộng dồn | Bảng có ≥80% khoá nằm trong bảng khác thì bỏ |
| Thành tiền ghi trước giảm dòng | Dùng nguyên | Tự phát hiện rồi trừ giảm dòng (bản gốc giữ ở `amount_gross`) |
| Đối chiếu dòng hàng với tiền hàng của đơn | Không có | `health.recon_mismatch`; lệch quá ngưỡng thì giảm độ tin cậy |
| Ánh xạ cột ngang điểm | Chọn bừa | Đánh dấu role "không chắc"; intent cần role đó thì bỏ trống |
| Độ tin cậy tầng A | Hằng 0.9 × (0.6 / 0.45 nếu có từ lạ), nên luôn vượt ngưỡng 0.5 | Độ chính xác **out-of-fold** theo intent (cận dưới Wilson) × hệ số từ lạ [1, 0.5, 0] × hệ số sức khoẻ dữ liệu. no_llm: `min_score` 0.8 |
| Hiệu chỉnh | Không có | `main calibrate`: chia fold theo shop, báo cáo PA theo từng ngưỡng, ghi `configs/calibration.yaml` |
| Phép tính chung | — | Tồn theo thời điểm, giá trị tồn, lô gần hết hạn, vốn / đếm / liệt kê lô sắp hết hạn, hiệu ứng tăng giá, nhập trừ bán, hàng ế, đợt giao, trung bình theo thứ, đếm ngày đạt ngưỡng, xếp hạng theo ngày / giờ / số ngày hoạt động / số đơn huỷ |
| Mốc tuổi nợ | 0–7 / 8–15 / >15 viết cứng | Đọc mốc từ câu hỏi |
| Ca tối | 18:00–23:59 viết cứng | Bỏ (rulebook không định nghĩa thì bỏ trống) |
| Prompt LLM | Ghi cứng "đơn hợp lệ = completed/shipping" | Theo rulebook của split; mô tả thêm các bảng kho / giá / lô / giao |
| Test | 9 | 26 |

## Số đo được (máy dev, chỉ 3 shop train có dữ liệu ở đây)

| Tập | v1 | v1.1 |
|---|---|---|
| 3 shop train (biz_101/102/123, 277 câu), no_llm | PA **0.953** (264 đúng / 0 sai / 13 bỏ) | PA **1.000** (277 / 0 / 0) |
| 13 câu v1 bỏ trống (tồn kho, lô/hạn dùng, tăng giá, nhập trừ bán) | 0 / 13 | 13 / 13 |
| Rulebook rỗng (test) | Trả lời theo luật train | 0 sai, chỉ guard trả lời |
| Train đủ 30 shop | 0.650 | **cần chạy lại** (dữ liệu nằm trên máy team) |
| Public | −0.037 | **cần chạy lại** |

Cổng trên câu hỏi (hợp đồng kiểu, bỏ fallback, đếm ngày) chặn 21 trong 356 câu v1 sai trên train mà gần như không chặn câu
đúng. Phần lớn lỗi còn lại (dòng tổng, nguồn trùng, lệch cột) được xử lý ở tầng dữ liệu; chỉ đo được khi chạy lại trên dữ liệu thật.

## Quy trình chạy (Windows cmd, không cần make)

```bat
:: 1. kiểm tra luật đọc được từ rulebook public (không cần GPU)
python -m datamate.runtime.main rules --split D:\data\public_test
:: 2. kiểm tra sức khoẻ dữ liệu public (dòng tổng, nguồn trùng, lệch đối chiếu, cột không chắc)
python -m datamate.runtime.main build --split D:\data\public_test -v
:: 3. chạy "thô" trên train + hiệu chỉnh out-of-fold
python -m datamate.runtime.main predict --split D:\data\train --out runs\raw --config configs\no_llm_raw.yaml --eval --fresh
python -m datamate.runtime.main calibrate --split D:\data\train --run runs\raw --out configs\calibration.yaml
:: 4. train với ngưỡng (so với 0.650 của v1)
python -m datamate.runtime.main predict --split D:\data\train --out runs\v11_train --config configs\no_llm.yaml --eval --fresh
:: 5. public
python -m datamate.runtime.main predict --split D:\data\public_test --out runs\v11_public --config configs\no_llm.yaml --fresh
```

`calibrate` in PA out-of-fold cho từng `min_score` và đề xuất ngưỡng; sửa `decision.min_score` trong `configs/no_llm.yaml`
theo đề xuất. Con số out-of-fold là dự báo cho shop chưa thấy (public / private), **không chỉnh theo điểm public**.

## Đánh đổi có chủ ý
* Rulebook public không còn luật "Giá bán X ngày D", nên intent `unit_price` (100% ở train) bỏ trống trên public. Nếu team
  đọc rulebook và thống nhất cách hiểu, ghi `"unit_price.def": "line_base_before_disc_all"` vào `public_test/rules.json`.
  Cách này là quyết định của người hoặc LLM, ghi lại được, không nằm trong code.
* `min_score` 0.8 bỏ trống nhiều hơn v1: đổi độ phủ lấy độ chính xác (sai −1, bỏ 0).
