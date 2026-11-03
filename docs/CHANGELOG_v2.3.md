# DataMate v2.3 — Tầng A: đọc câu khẩu ngữ / viết tắt (CPU, không LLM)

Nguồn lỗi: log `runs/v22_public` (0.444). Mọi sửa là **quy tắc ngôn ngữ chung** (không theo business_id/question_id, không đụng data train). Năm của tháng không ghi năm lấy theo luật rulebook (tháng chưa tới → năm trước).

## Trước → Sau

| Nhóm | v2.2 | v2.3 |
|---|---|---|
| `Th2, th3, th4/2026 tháng nào…` | chỉ đọc 1 tháng (th4) → so sai | đọc đủ 3 tháng |
| `Th3 …`, `Th5 …`, `Th4 …` (không năm) | bỏ qua → kỳ = toàn bộ dữ liệu (sai) | tháng theo luật năm của rulebook |
| `Tháng 5/2025 … so với tháng trước` | thiếu kỳ so sánh | thêm cờ `prev_period` |
| Mốc thời gian có trong câu nhưng parser không dùng | vẫn trả lời (dễ sai kỳ) | tính vào token chưa giải thích → hạ confidence / bỏ |
| `lên hay xuống`, `hơn kém`, `so dzới`, `nhiều tiềng`, `nhta`, `bih quân`, `chốt được`, `góp mặt`, `chong tháng`, `o đờ` | không nhận | chuẩn hoá về dạng chuẩn |
| `bán được nhiều X nhất`, `được gọi/đặt`, `ít nhất 1 lần` (không phải xếp hạng), `lên giá … N ngày` | nhận sai / không nhận | nhận đúng loại câu |

## Đo offline (parse_harness trên log v22_public, không cần dữ liệu shop)
- 44 câu đổi cách đọc: **4 câu đã trả lời** (đều là sửa kỳ sai ở trên), **~39 câu đang abstain** nay parse được (compare_period 8, day_count_thr 8, best_day 4, series 4, price_effect 3, …).
- Câu địa chỉ (`khách ở Trường Chinh … hay khách ở Lạc Long Quân`) vẫn còn 4 token chưa giải thích → không trả lời.
- Train mini (no_llm): **264 đúng / 0 sai / 13 bỏ** — không đổi so với v2.2.
- pytest: 60 passed (thêm `tests/test_v23.py`).

## Chạy
```bash
git am datamate_v2.3.patch
python -m datamate.runtime.main predict --split <thư_mục_public_test> --out runs/v23_public --config configs/no_llm.yaml --fresh
python merge.py runs/v23_public/submission.jsonl runs/submission_v2.jsonl v23_best.jsonl --only-unans
```
Gợi ý nộp: `runs/v23_public/submission.jsonl` trước (đo riêng tầng A), sau đó `v23_best.jsonl`.
