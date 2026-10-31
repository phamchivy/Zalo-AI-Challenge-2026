# DataMate v1.3: bộ phạt "từ lạ" không còn bám từ vựng train

Phân tích 1000 log public (v1.2, `no_llm`): **204 câu đã có plan nhưng bị bỏ trống** vì bộ phạt "từ lạ". Bộ phạt này dùng
danh sách từ lấy từ câu hỏi **train**, nên cách viết của public (*tiệm, shop, mik, đh, wan, cko, jùm, trog*, phương ngữ
*mô / ni / rứa / chừ / bữa*) đều bị coi là "từ lạ". Câu có 1 từ lạ bị nhân hệ số 0.5, từ 2 từ trở lên bị nhân 0.

Tuy vậy, không thể chỉ bỏ bộ phạt: đọc lại các câu này thấy nhiều câu bị **hiểu sai** đúng vì những từ đó ("ngày **mô**",
"mấy **bữa**", "11 **củ**", "**cất hàng**", "**đh**"). Bỏ phạt mà không sửa phần hiểu câu thì sẽ trả lời sai.

## Trước / sau

| Hạng mục | v1.2 | v1.3 |
|---|---|---|
| Từ lạ | Mọi từ không có trong từ vựng câu hỏi train | Chỉ tính 4 dấu hiệu "nhắc tới thực thể mà chưa nối được": (1) từ **viết hoa giữa câu** (tên riêng); (2) token **chữ + số** kiểu tên hàng (`100g`, `25W`, `D10`); (3) **cặp từ có trong tên hàng / nhóm hàng của chính shop** mà chưa nối; (4) sau "khách" là **cặp từ có trong tên khách của shop**. Cộng thêm từ chưa hiểu sau khi đã loại từ đệm |
| Từ đệm, xưng hô, tự xưng | Bị phạt | Danh sách viết tay, tiếng Việt chung (tiệm, shop, mik, ad, ơi, nha, z, rứa, nghen…): không phạt |
| Chat / phương ngữ | Không hiểu | Chuẩn hoá: đh→đơn hàng, wan→quán, cko→cho, kua→của, trog→trong, jùm/dúp→giùm/giúp, mik→mình, ngàynào→ngày nào, cất hàng→nhập hàng, lên hay xuống→tăng hay giảm, tới giờ / tới chừ / tới bấy nay→đến hết hôm nay. Theo ngữ cảnh: "ngày/khách **mô**"→nào, "hôm/tháng **ni**"→nay, "mấy **bữa**"→mấy ngày, "11 **củ**"→11 triệu, "**mí** %"→mấy % |
| Ngưỡng tiền viết tắt | "từ 3tr7", "11 củ" không đọc được | 3 700 000, 11 000 000 |
| Liệt kê nhiều kỳ không có từ so sánh ("doanh thu th7, th8, th9") | Chỉ trả kỳ đầu | Trả danh sách theo thứ tự (chỉ khi các kỳ rời nhau và cùng loại) |
| "tháng sau **luôn hơn** tháng trước" | Doanh thu một kỳ | Câu hỏi đúng/sai về xu hướng |
| Test | 39 | 44 |

Bảo vệ: chuẩn hoá mới **không làm đổi câu nào** trong 3000 câu train (đã đo). Các ánh xạ dễ phá tên hàng ('họng', 'chén',
'mì') đã bị loại hoặc chỉ áp dụng theo ngữ cảnh.

## Số đo

| Tập | v1.2 | v1.3 |
|---|---|---|
| 3 shop train, `no_llm` | 1.000 (277/0/0) | 1.000 (277/0/0) |
| Như trên, rulebook public dựng lại (`no_llm_raw`) | 264 / 0 sai / 13 bỏ | 264 / 0 sai / 13 bỏ |
| Public: câu có plan bị phạt (mô phỏng lại trên log v1.2, chưa có dữ liệu shop) | 204 / 734 | khoảng 118 / 734, tức khoảng 139 câu được gỡ phạt |
| Public: PA | 0.283 | **cần chạy lại**. Ước tính +0.04 đến +0.08 nếu số câu gỡ phạt đúng 65–75% |

## Cách kiểm chứng (bắt buộc trước khi nộp)

```bat
python -m datamate.runtime.main predict --split D:\data\train --split-file configs\split.yaml --part dev --out runs\v13_dev --config configs\no_llm.yaml --eval --fresh
python -m datamate.runtime.main predict --split D:\data\train --split-file configs\split.yaml --part holdout --out runs\v13_holdout --config configs\no_llm.yaml --eval --fresh
python -m datamate.runtime.main predict --split D:\data\public_test --out runs\v13_public --config configs\no_llm.yaml --fresh
```

Điều kiện giữ v1.3: số câu sai của **holdout không tăng** (v1.2: 0 sai, PA 0.927). Nếu holdout có câu sai mới thì báo lại
PA và số câu sai theo kiểu đáp án; không mở log holdout.
