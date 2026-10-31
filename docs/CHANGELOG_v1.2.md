# DataMate v1.2: đo mức tổng quát bằng holdout, sửa lỗi dữ liệu theo cấu trúc

Mục tiêu: sửa các nhóm lỗi tìm thấy trên train/public **theo cấu trúc dữ liệu, không theo tên shop**, và có một con số đo
mức tổng quát mà không bị "học thuộc" (holdout). Vẫn giữ precision-first: không chắc thì bỏ trống, để phần còn lại cho LLM.

## Trước / sau

| Hạng mục | v1.1 | v1.2 |
|---|---|---|
| Tiền viết tắt `1tr440`, `1tr5`, `1ty2`, `3tr450k` | `to_num` đọc ra **1.0** (biz_117: 5009 dòng, biz_127: 1067 dòng hỏng) | 1 440 000 / 1 500 000 / 1 200 000 000 / 3 450 000. Bộ profiler cũng nhận đây là cột số |
| Tên cột có đơn vị / ngoặc (`Thành tiền (đ)`, `TongTienVND`, `Khách hàng (tên)`) | Không ánh xạ (dev_003: 0 dòng hàng) | Thử thêm biến thể bỏ hậu tố tiền / phần ngoặc (điểm 1.9 < tên đầy đủ 2.0) |
| Từ đồng nghĩa cột | Thiếu `S.L`, `SL (ĐV cơ bản)`, `Giá 1 đv`, `Mã H.hóa`, `Item`, `Giá trị`, `Mã phiếu`, `Ngày TT`, `Tiền thu`, `Công nợ`, `Bớt` | Đã có (từ vựng chung, không gắn shop) |
| `Hàng` / `Hạng` | `fold()` trùng nhau nên "Hàng" bị nhận là **hạng khách** (dev_011/014) | Phân biệt bằng chữ có dấu: Hàng là tên hàng, Hạng là nhóm khách |
| Gợi ý loại bảng | Tên file trước, nên sheet `BanHang_LichSuGia` bị coi là bảng bán | Tên **sheet** trước, rồi mới tới tên file |
| Sổ quỹ / phiếu thu (`SoQuy`: Mã phiếu, Giá trị) | Thành bảng đơn hàng; công nợ dùng cột "đã trả" cuối kỳ nên ra **0** cho ngày quá khứ (biz_106) | Nhận đúng là bảng thanh toán, công nợ tính theo ngày trả |
| Sổ thu có tên nhưng không đọc được cột | Âm thầm dùng cột "đã trả" của đơn | `health.payments_unmapped` nên công nợ tại ngày X **bỏ trống** |
| Sổ ghi tay không có mã đơn, STT chỉ ở dòng đầu | Mỗi dòng là một đơn (số đơn, AOV, khách sai) | Khoá đơn theo nhóm STT, **chỉ khi được xác nhận bằng dữ liệu**: tiền khách trả khớp tổng nhóm hơn tiền của dòng đầu, hoặc cột cấp đơn chỉ có ở dòng đầu |
| "Giảm" lặp y nguyên trên mọi dòng của đơn (`Tiền giảm` của dev_003) | Bị coi là giảm từng dòng, có thể trừ nhiều lần | Phát hiện là giảm **cấp đơn**, chuyển sang `order_discount`; nếu bảng còn cột giảm dòng thật (`Giảm (đ)`) thì dùng cột đó. Bộ phát hiện "thành tiền trước giảm" không kích hoạt trong trường hợp này |
| Khoá bỏ bảng trùng nguồn | Chỉ mã đơn, nên mã đánh lại mỗi năm có thể làm mất cả bảng năm sau | (mã đơn, tháng của ngày đã đọc) |
| Điều kiện thanh toán không nối được ("trả bằng tiền tươi", COD mà dữ liệu không có) | Bỏ điều kiện và trả **tổng không lọc** (sai) | Bỏ trống, gợi ý tầng B |
| Thuộc tính khách không có trong dữ liệu ("khách nam", giới tính, độ tuổi) | Trả tổng không lọc | Bỏ trống |
| "cộng lại", "bán lại" | Bị hiểu thành **lãi gộp** (bắt chữ "lai" đứng riêng) | Chỉ nhận lãi gộp / lợi nhuận / lãi bao nhiêu / lãi nhất... |
| "doanh thu của Máy bơm…" | Thành câu hỏi số lượng (đơn vị "bơm" lấy từ tên hàng) | Bỏ tên hàng đã nhận ra trước khi tìm đơn vị; có "doanh thu / doanh số" thì là doanh thu |
| "có **nhiêu** ngày đạt…" | Không nhận nên trả doanh thu cả tháng | Đếm ngày đạt ngưỡng |
| "lần lượt", hai lần "bao nhiêu" | Có thể trả một số | Hợp đồng kiểu chỉ nhận list / table. Trên 3000 câu train: 40 câu bị siết, đáp án gốc đều là list hoặc unanswerable (0 vi phạm) |
| Đo tổng quát | OOF theo shop (lạc quan vì đã đọc lỗi mọi shop) | `configs/split.yaml`: **dev** (shop đã xem + phần còn lại) và **holdout** (khoảng 8 shop chưa xem, rải đều theo ngành và họ dữ liệu, chọn theo md5 nên chạy lại vẫn ra cùng kết quả) |
| Báo cáo holdout | — | `--part holdout`: chỉ in PA tổng + PA theo kiểu đáp án, ghi `eval_report_holdout.json`, **không** ghi `errors.jsonl` |
| Hiệu chỉnh | Trên mọi shop train | `calibrate --split-file`: chỉ dev; từ chối chạy trên holdout |
| Test | 26 | 39 |

## Số đo được (máy dev, chỉ có 3 shop train)

| Tập | v1.1 | v1.2 |
|---|---|---|
| 3 shop train (277 câu), `no_llm_raw` | PA 1.000 (277 / 0 / 0) | PA 1.000 (277 / 0 / 0): không lùi |
| Như trên, rulebook public dựng lại | 264 / 0 sai / 13 bỏ | 264 / 0 sai / 13 bỏ |
| Train đủ 30 shop, public | 0.8297 (thô), public 0.132 | **cần chạy lại trên máy team** (dữ liệu biz_106/117/127/dev_003/dev_011 không có ở đây) |

Các lỗi train trong `errors.jsonl` mà v1.2 nhắm tới: biz_117 và biz_127 (tiền viết tắt: phần lớn số lỗi), biz_106
(sổ quỹ, COD), biz_109 và biz_130 (hai giá trị / lần lượt), biz_129 (tiền tươi), biz_112 (nhiêu ngày), biz_116 (máy bơm),
biz_122 (khách nam), biz_124 (cộng lại). Vẫn **chưa** sửa: tuổi nợ 0–7 ngày của biz_124 (chưa rõ nguyên nhân).

## Quy trình mới (Windows cmd)

```bat
:: 1. chia dev / holdout MỘT LẦN (shop đã xem lỗi -> --seen, luôn vào dev)
python -m datamate.runtime.main split-info --split D:\data\train
python -m datamate.runtime.main make-split --split D:\data\train --seen biz_101 biz_102 biz_104 biz_105 biz_106 biz_109 biz_110 biz_112 biz_116 biz_117 biz_122 biz_123 biz_124 biz_127 biz_129 biz_130 --out configs\split.yaml
:: 2. chạy thô trên dev + hiệu chỉnh chỉ trên dev (được xem errors.jsonl của dev)
python -m datamate.runtime.main predict --split D:\data\train --split-file configs\split.yaml --part dev --out runs\dev_raw --config configs\no_llm_raw.yaml --eval --fresh
python -m datamate.runtime.main calibrate --split D:\data\train --run runs\dev_raw --split-file configs\split.yaml --out configs\calibration.yaml
:: 3. MỘT con số trên holdout (không mở logs\ của lần chạy này)
python -m datamate.runtime.main predict --split D:\data\train --split-file configs\split.yaml --part holdout --out runs\holdout --config configs\no_llm.yaml --eval --fresh
:: 4. public
python -m datamate.runtime.main predict --split D:\data\public_test --out runs\v12_public --config configs\no_llm.yaml --fresh
```

Đọc kết quả: PA trên holdout gần PA của dev nghĩa là tổng quát tốt. Nếu holdout thấp hơn nhiều thì code đang khớp theo dev.
`min_score` chọn theo PA out-of-fold của dev, **không** chỉnh theo holdout hay public.

## Đánh đổi có chủ ý
* Danh sách `--seen` gồm cả các shop có lỗi xuất hiện trong `errors.jsonl` đã đọc ở vòng này (112, 116, 122, 129, 130), vì
  code đã sửa theo câu hỏi của chúng. Train có 30 shop, nên còn 14 shop chưa xem để chọn holdout.
* Cổng thanh toán và thuộc tính làm tăng số câu bỏ trống: một số câu trước đây đúng do may mắn (shop chỉ có một hình thức
  thanh toán) giờ sẽ bỏ trống.
* Sổ thu có tên nhưng không đọc được cột thì công nợ bỏ trống, kể cả khi cột "đã trả" của đơn tình cờ đúng.
* Khoá đơn theo STT chỉ bật khi dữ liệu xác nhận. Không xác nhận được thì giữ cách cũ (mỗi dòng một đơn) và ghi chú `no order id`.
