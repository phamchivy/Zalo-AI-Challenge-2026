# DataMate v2.5.0 — Tự phòng thủ khi dữ liệu lạ + bộ đo tổng quát không dùng public

Hướng: hệ thống không thể biết trước mọi kiểu lỗi của private. Thay vì sửa tay từng lỗi, **invariants chạy lúc dự đoán**:
phần dữ liệu nào đáng ngờ thì các câu dựa vào phần đó **bỏ trống (0)** thay vì trả lời sai (−1). Kèm bộ đo "shop train làm bẩn"
để đo tổng quát hoá mà không cần public.

## 1. Chặn theo rủi ro dữ liệu (`datamate/decision/data_guard.py`)
- Khi build catalog: chạy invariants, lưu mức từng invariant vào `canon.health['invariants']` (+ `inv_risk`).
- Quyết định cuối: invariant **đỏ** → bỏ trống các câu có intent phụ thuộc. Ví dụ:
  - trạng thái/ngày đỏ → câu dựa trên đơn;
  - số học tiền đỏ → câu tiền;
  - I2/I3/thanh toán đỏ → công nợ;
  - quy đổi đơn vị đỏ → câu số lượng.
  Không chặn cả shop. Không chặn câu `unanswerable`.
- **Kiểm hợp lý đáp án:** đại lượng không thể âm mà ra âm, tỉ trọng/tỉ lệ huỷ ngoài 0–100%, NaN → bỏ trống.
- Tắt được bằng `--set decision.data_guard=false`.

## 2. Invariant mới
- **I5_unit_factor:** dòng bán theo đơn vị khác đơn vị cơ bản mà hệ số = 1, tức bảng quy đổi không gán được. Trên bản sạch 0 dòng; khi đổi tên cột trong bảng quy đổi thì 100% dòng → đỏ.

## 3. Sửa chung (tìm bởi invariants public, không theo shop)
| Lỗi | v2.4.1 | v2.5.0 |
|---|---|---|
| Số nhóm nghìn bằng khoảng trắng `1 851 000`, `113 889` | đọc thành **1**, **113** (dev_007: tiền đơn ≥ 1 triệu thành 1000). Bản sửa "nghìn đồng" ở v2.4.1 chỉ đúng tình cờ cho số < 1 triệu | đọc đúng (không gộp `1 gói`, `2 100ml`) |
| Ghi chú `cancel` / `refund` / `pending` (dev_013, 471 dòng) | không thành trạng thái → đơn huỷ bị tính doanh thu | dùng chung từ điển trạng thái |

## 4. Bộ đo shop làm bẩn (`scripts/make_dirty.py`)
Biến đổi giữ nguyên ngữ nghĩa (đáp án train vẫn đúng):
- `ren`: đổi tên cột sang từ đồng nghĩa, README đổi theo.
- `money`: tiền kiểu `1 234 000` / `1tr234` / `234k` / `1.234.000đ`.
- `status`: nhãn trạng thái chưa có trong từ điển (`khách bùng`, `đã chốt đơn`…).
- `thousand`: tiền cấp đơn ghi theo nghìn.
- `clean`: bản đối chứng.

Kết quả trên train mini (3 shop × 5 phép, 1385 câu):

| Phép | Không chặn (v2.4.1) đúng / sai / bỏ | **v2.5.0** đúng / sai / bỏ |
|---|---|---|
| clean | 264 / 0 / 13 | 264 / 0 / 13 |
| ren | 259 / **5** / 13 | 256 / **0** / 21 |
| money | 264 / 0 / 13 | 264 / 0 / 13 |
| status | 148 / **116** / 13 (PA 0.116) | 111 / **0** / 166 (PA 0.401) |
| thousand | 264 / 0 / 13 | 264 / 0 / 13 (shop mini không có tiền cấp đơn, phép này chưa được thử thật) |

Nhận xét:
- Nhãn lạ hoặc tên cột lạ trước đây gây **sai im lặng**. Giờ chuyển thành bỏ trống.
- Phần bỏ trống ở `status` (153 câu) là chỗ v3.0 (LLM gán nhãn theo giá trị thật) phải lấy lại.

## Kiểm
- pytest 78 passed (+`tests/test_v250.py`).
- Train mini sạch không đổi: 264/0/13.

## Chưa làm
Bước 2 của kế hoạch (2 chính sách: lãi gộp khi thiếu giá vốn dòng, giá bán trước/sau giảm) để sau.

## Chạy (máy công ty)
```bash
git am datamate_v2.5.0.patch && git tag v2.5.0
T=/home/vht/phamchivy/datamate/train; P=/home/vht/phamchivy/datamate/public_test/public_test
# 1) không tụt trên train
python -m datamate.runtime.main predict --split $T --split-file configs/split.yaml --part dev --out runs/v250_dev --config configs/no_llm.yaml --eval --fresh
python -m datamate.runtime.main predict --split $T --split-file configs/split.yaml --part holdout --out runs/v250_holdout --config configs/no_llm.yaml --eval --fresh
# 2) bộ đo làm bẩn trên shop holdout (8 shop): sai không được tăng so với clean
python scripts/make_dirty.py make --split $T --split-file configs/split.yaml --part holdout --out runs/dirty_holdout
python -m datamate.runtime.main predict --split runs/dirty_holdout --out runs/dirty_holdout_pred --config configs/no_llm.yaml --fresh
python scripts/make_dirty.py report --split runs/dirty_holdout --pred runs/dirty_holdout_pred
# 3) invariants public + public
python -m datamate.runtime.main invariants --split $P --out runs/invariants_public_250
python -m datamate.runtime.main predict --split $P --out runs/v250_public --config configs/no_llm.yaml --fresh
python merge.py runs/v250_public/submission.jsonl runs/submission_v2.jsonl v250_best.jsonl --only-unans
```
