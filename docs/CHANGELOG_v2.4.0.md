# DataMate v2.4.0 — Invariants: kiểm canonical không cần đáp án (bước 1 kế hoạch v3)

Chỉ chẩn đoán: **không đổi dữ liệu, không đổi đáp án**. Submission giống hệt v2.3.2, nên không cần nộp.

## Trước → Sau
| | v2.3.2 | v2.4.0 |
|---|---|---|
| Biết shop nào dựng bảng sai | chỉ khi nộp / kiểm toán 27B | `invariants` chạy trên mọi split, kể cả private, không cần đáp án |
| Nhãn trạng thái lạ | âm thầm thành `completed` | lộ ra ở I15 kèm nhãn và số dòng |
| Giảm cả đơn bị trừ vào dòng (kiểu "Bớt") | không phát hiện | I1/I2 đỏ, ghi chú "Σ dòng + giảm đơn = tiền hàng" |
| Tiền sai bậc ("45" vs "45k") | không phát hiện | MONEY_scale |

## 12 invariant (theo tên cột canon hiện tại)
| Mã | Kiểm | Phát hiện |
|---|---|---|
| I1_line_arith | thành tiền = SL × đơn giá − giảm dòng (có xét hệ số quy đổi) | map nhầm cột tiền; giảm cấp đơn bị trừ vào dòng; đơn giá sau giảm |
| I2_goods_total | Σ dòng = tổng tiền hàng của đơn | thiếu/trùng dòng, ffill sai, giảm đơn trừ vào dòng |
| I3_payable | khách phải trả = tiền hàng − giảm đơn + phí giao (+VAT) | map nhầm cột tổng/giảm/phí |
| I6_one_status | mỗi đơn một trạng thái | khoá đơn sai, trạng thái sai nguồn |
| I15_status_mapped | mọi nhãn trạng thái gốc gán được 4 trạng thái | nhãn lạ (Ghi chú, cách gọi mới) |
| I15_payment_mapped | hình thức thanh toán thuộc tập chuẩn | nhãn thanh toán lạ |
| I9_customer_join | mã khách có trong bảng khách; vãng lai không có nhóm | tách tên/SĐT sai, khách lẻ vs nhóm "lẻ" |
| I12_purchase_total | tổng phiếu nhập = Σ dòng nhập | map nhầm bảng nhập |
| MONEY_scale | đơn giá cùng mặt hàng không lệch ≥ ×300 | parse tiền sai bậc |
| DATE_parsed / DATE_range | ngày đọc được; không vượt ngày cuối README | năm lấy sai (năm theo sheet), định dạng lạ |
| STOCK_card | tồn trên thẻ kho không âm | nền cho tồn tính ngược (v2.4.x) |

Mức: xanh / vàng / đỏ theo tỉ lệ vi phạm (ngưỡng trong `LEVELS`). Rủi ro shop = 3 × số đỏ + số vàng. Bảng xếp theo rủi ro × số câu.

## Kiểm trên train mini
- biz_123 (sqlite, đủ bảng): xanh toàn bộ.
- biz_101/102: I1 vàng (~1,3%). Dòng có thành tiền thấp hơn SL × giá mà không có cột giảm dòng. Đây là dữ liệu thật; đáp án train dùng thành tiền, và hệ thống đang dùng đúng cột này.
- pytest 69 passed (thêm `tests/test_v240.py`: dữ liệu sạch xanh; giảm cả đơn trừ vào dòng → đỏ; nhãn lạ → đỏ; tiền sai bậc → đỏ).

## Chạy
```bash
git am datamate_v2.4.0.patch
git tag v2.3.2 HEAD~1      # mốc baseline (nếu chưa có)
git tag v2.4.0
python -m datamate.runtime.main invariants --split <train> --out runs/invariants_train
python -m datamate.runtime.main invariants --split /home/vht/phamchivy/datamate/public_test/public_test --out runs/invariants_public
```
Gửi lại `runs/invariants_train.md` và `runs/invariants_public.md`, kèm 2 file `.json` để xem ví dụ. Đây là đầu vào cho v2.4.1 (sửa canonical theo các ô đỏ).
