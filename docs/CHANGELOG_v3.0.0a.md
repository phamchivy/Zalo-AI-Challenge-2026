# DataMate v3.0.0-a — Suy cấu trúc dữ liệu từ giá trị + kiểm tra độ phủ (phần không cần LLM của bước 4)

Nguồn: bộ shop làm bẩn trên shop dev (v2.5.1). Đổi tên cột làm bảng chi tiết hoá đơn của shop POS tách bảng
(biz_106, biz_124) mất hết role -> hệ thống lấy tiền cấp đơn làm doanh thu -> 59 câu sai im lặng.
Invariants không chặn được vì bảng biến mất thì "không kiểm được" bị coi như an toàn.

## Thay đổi (không dùng tên cột, không regex, không từ khoá)
### B. `datamate/data/structure.py`: suy cấu trúc từ quan hệ giá trị giữa các bảng
| Bằng chứng | Suy ra |
|---|---|
| Cột gần như duy nhất trong bảng danh mục, được nhiều bảng khác tham chiếu (giá trị con ⊂ giá trị cha ≥ 95%) | khoá chính `product_code` / `customer_code` + cột tên (chữ, gần duy nhất, dài nhất) |
| Cột ở bảng khác có giá trị ⊂ khoá chính đó | khoá ngoại `product_code` / `customer_code` |
| Cột ⊂ mã đơn của bảng đơn | bảng con của đơn, `order_id` |
| Ba cột số với thành tiền ≈ SL × đơn giá (− giảm dòng) trên ≥ 80% dòng | `qty`, `unit_price`, `amount`, `line_discount`; bảng thành `sales_lines` |
| Cột số ở bảng đơn = Σ thành tiền các dòng của đơn (≥ 90% đơn) | `goods_total` |

Chỉ **bổ sung role còn thiếu**, không ghi đè role regex đã gán (evidence = `structure`). Shop sạch không đổi.

### A. Invariant `COVERAGE_child`
Bảng con của đơn (khoá ⊂ mã đơn, nhiều dòng/mã) mà vẫn không hiểu được là dòng hàng / thanh toán / trả hàng / lô / giao
hàng -> **đỏ** -> chặn mọi câu dựa trên đơn hàng của shop đó (bỏ trống thay vì trả sai).

## Kết quả
Shop POS 17 bảng dựng từ biz_123 (sqlite → CSV, có đáp án), bộ làm bẩn:

| Phép | v2.5.1 đúng / sai / bỏ | v3.0.0-a đúng / sai / bỏ |
|---|---|---|
| clean | 111 / 0 / 7 | 111 / 0 / 7 |
| **ren** (đổi tên cột) | 11 / **24** / 83 (PA −0.110) | **88 / 0 / 30** (PA 0.746) |
| status | 17 / 0 / 101 | 17 / 0 / 101 (phần của LLM, v3.0.0) |

- Train mini sạch: 264/0/13, không đổi; bộ làm bẩn mini không đổi.
- pytest 80 passed (+`tests/test_v300a.py`: tên cột vô nghĩa vẫn suy đúng bảng dòng + tiền hàng; bảng con không có
  quan hệ SL × giá → COVERAGE đỏ).
- Thời gian build catalog không đổi đáng kể.

## Còn lại cho v3.0.0 (cần LLM)
- Nhãn trạng thái/thanh toán/kênh lạ (phép `status`).
- Role không có quan hệ giá trị để suy (tổng phải trả, đã trả, đơn vị tính…): LLM chấm ứng viên + Hungarian + chọn bằng
  invariants.

## Chạy (máy công ty)
```bash
git am datamate_v3.0.0a.patch && git tag v3.0.0a
T=/home/vht/phamchivy/datamate/train; P=/home/vht/phamchivy/datamate/public_test/public_test
# 1) train không tụt (dev 2105/1/260, holdout 591/0/43)
python -m datamate.runtime.main predict --split $T --split-file configs/split.yaml --part dev --out runs/v300a_dev --config configs/no_llm.yaml --eval --fresh
python -m datamate.runtime.main predict --split $T --split-file configs/split.yaml --part holdout --out runs/v300a_holdout --config configs/no_llm.yaml --eval --fresh
# 2) bộ làm bẩn dev, phép ren (đã có runs/dirty_dev_ren): mục tiêu sai 59 -> 0, bỏ < 389
python -m datamate.runtime.main predict --split runs/dirty_dev_ren --out runs/dirty_dev_ren_pred_300a --config configs/no_llm.yaml --fresh --set workers=2
python scripts/make_dirty.py report --split runs/dirty_dev_ren --pred runs/dirty_dev_ren_pred_300a --errors --shops
# 3) invariants public (COVERAGE_child phải xanh ở mọi shop) + public
python -m datamate.runtime.main invariants --split $P --out runs/invariants_public_300a
python -m datamate.runtime.main predict --split $P --out runs/v300a_public --config configs/no_llm.yaml --fresh
python merge.py runs/v300a_public/submission.jsonl runs/submission_v2.jsonl v300a_best.jsonl --only-unans
```
Nộp `v300a_best.jsonl` khi 1–3 đạt.
