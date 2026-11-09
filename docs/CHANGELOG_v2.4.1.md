# DataMate v2.4.1 — Sửa canonical theo báo cáo invariants (bản nộp đầu tuần)

Nguồn: `invariants_train.md` / `invariants_public.md` (v2.4.0). Mọi sửa là quy tắc theo dữ liệu, không theo tên shop.

## Trước → Sau
| # | Lỗi (invariant phát hiện) | Shop bị | v2.4.0 | v2.4.1 |
|---|---|---|---|---|
| 1 | Nhãn `Thu hộ (COD)` không gán được (khớp nguyên chuỗi, vướng dấu ngoặc) → đơn COD **bị tính vào công nợ** (sai §4), câu "doanh thu COD" sai | biz_106, biz_109, biz_124, dev_003 (4–5% số đơn) | payment = `thu ho (cod)` | `cod`. Thanh toán gán theo **từ khoá** sau khi bỏ dấu câu (cod/thu hộ, ghi nợ/trả góp, ví/momo, quẹt thẻ, chuyển khoản/ngân hàng/QR, tiền mặt) |
| 2 | Tiền **cấp đơn ghi theo nghìn đồng**, tiền dòng theo đồng (I2 đỏ 100%) → nợ, khách đã trả, phí giao sai ×1000 | dev_007 | tiền hàng 129 vs Σ dòng 129 000 | Cột tổng của đơn ≈ Σ dòng / 1000 trên ≥ 80% đơn → nhân 1000 các ô tiền cấp đơn cùng quy ước |
| 3 | **Topping ghi chung ô với món** (`Trà đào (M) + Trân châu đen (7.000đ)`), luật cà phê §6: hai mặt hàng | dev_002 | một dòng, tên lạ, tiền gộp → top món / SL / doanh thu theo món sai | tách dòng topping (SL = SL món, tiền = SL × giá topping, cùng mặt hàng với topping ghi dòng riêng). Chỉ áp khi tiền xác nhận ≥ 80% dòng; tổng doanh thu không đổi |

## Chỉnh bộ đo
| Invariant | v2.4.0 | v2.4.1 |
|---|---|---|
| I1 | vàng ở mọi shop phẳng (giảm ~5% không ghi cột, dữ liệu thật) | ngưỡng 3%/8%; **chiều "thành tiền > SL×giá" > 1% → đỏ** (gộp topping/phụ phí, đơn giá sai cột) |
| I3 | đỏ ~10% ở 7 shop train (phí giao 15k không có cột) | lệch là 1–2 hằng số → **vàng** + chú thích "phí giao / voucher chưa gán" |
| I15 | shop lấy trạng thái từ Ghi chú: không kiểm được | đếm ghi chú có từ khoá trạng thái mà không gán được |

## Kiểm
- Train mini (no_llm): 264 đúng / 0 sai / 13 bỏ (không đổi). pytest 73 passed (+`tests/test_v241.py`).
- Chưa đo trên train đầy đủ/public (dữ liệu ở máy công ty). Kỳ vọng: biz_106/109/124 tăng ở nhóm công nợ; dev_007, dev_002 hết đỏ.

## Chạy (máy công ty)
```bash
git am datamate_v2.4.1.patch && git tag v2.4.1
# 1) invariants: dev_007, dev_002 phải hết đỏ; không shop nào từ xanh sang đỏ
python -m datamate.runtime.main invariants --split <train> --out runs/invariants_train_241
python -m datamate.runtime.main invariants --split <public_test> --out runs/invariants_public_241
# 2) train dev: so với v2.3.2 (PA 0.8855, sai 4) — không được tụt
python -m datamate.runtime.main predict --split <train> --split-file configs/split.yaml --part dev --out runs/v241_dev --config configs/no_llm.yaml --eval --fresh
# 3) public + merge như v2.3.2
python -m datamate.runtime.main predict --split <public_test> --out runs/v241_public --config configs/no_llm.yaml --fresh
python merge.py runs/v241_public/submission.jsonl runs/submission_v2.jsonl v241_best.jsonl --only-unans
```
Nộp `v241_best.jsonl` khi bước 1–2 đạt. So với 0.521.
