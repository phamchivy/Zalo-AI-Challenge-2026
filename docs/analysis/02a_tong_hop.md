# Báo cáo Phân tích Họ câu hỏi và Dữ liệu (Việc 2a)

## 1. Bảng Ma trận Họ câu hỏi × Split (Số lượng & Tỉ lệ %)

### Số lượng câu hỏi theo họ:
| Họ câu hỏi | Public Test | Train Data | Tổng cộng |
| --- | --- | --- | --- |
| Q01 | 577 | 765 | 1342 |
| Q02 | 44 | 61 | 105 |
| Q03 | 58 | 58 | 116 |
| Q04 | 12 | 9 | 21 |
| Q05 | 17 | 45 | 62 |
| Q06 | 44 | 39 | 83 |
| Q07 | 2 | 0 | 2 |
| Q08 | 87 | 68 | 155 |
| Q09 | 20 | 19 | 39 |
| Q10 | 21 | 48 | 69 |
| Q11 | 1 | 0 | 1 |
| Q12 | 14 | 12 | 26 |
| Q13 | 18 | 31 | 49 |
| Q14 | 3 | 2 | 5 |
| Q15 | 26 | 29 | 55 |
| Q17 | 6 | 1 | 7 |
| Q18 | 4 | 6 | 10 |
| Q19 | 42 | 7 | 49 |
| Q20 | 4 | 0 | 4 |
| Tổng | 1000 | 1200 | 2200 |

### Tỉ lệ % theo họ trên từng tập dữ liệu:
| Họ câu hỏi | Public Test (%) | Train Data (%) |
| --- | --- | --- |
| Q01 | 57.70% | 63.75% |
| Q02 | 4.40% | 5.08% |
| Q03 | 5.80% | 4.83% |
| Q04 | 1.20% | 0.75% |
| Q05 | 1.70% | 3.75% |
| Q06 | 4.40% | 3.25% |
| Q07 | 0.20% | 0.00% |
| Q08 | 8.70% | 5.67% |
| Q09 | 2.00% | 1.58% |
| Q10 | 2.10% | 4.00% |
| Q11 | 0.10% | 0.00% |
| Q12 | 1.40% | 1.00% |
| Q13 | 1.80% | 2.58% |
| Q14 | 0.30% | 0.17% |
| Q15 | 2.60% | 2.42% |
| Q17 | 0.60% | 0.08% |
| Q18 | 0.40% | 0.50% |
| Q19 | 4.20% | 0.58% |
| Q20 | 0.40% | 0.00% |

---

## 2. Thống kê Thuộc tính Ngôn ngữ trên Dữ liệu

Tổng số câu hỏi phân tích: **2200 câu** (1,000 Public Test + 1,200 Train Data mẫu).
- **`khau_ngu`**: Tổng **797 câu** (36.2%) — (Public: 361 câu | Train: 436 câu)
- **`go_sai`**: Tổng **26 câu** (1.2%) — (Public: 20 câu | Train: 6 câu)
- **`viet_tat`**: Tổng **620 câu** (28.2%) — (Public: 358 câu | Train: 262 câu)
- **`nhieu_ve`**: Tổng **541 câu** (24.6%) — (Public: 284 câu | Train: 257 câu)
- **`loc`**: Tổng **142 câu** (6.5%) — (Public: 80 câu | Train: 62 câu)

---

## 3. Họ câu hỏi có chênh lệch lớn nhất giữa Train và Public

- **Họ xuất hiện nhiều nhất ở cả 2 tập:** `Q01` (Chiếm ~57.7% Public và 63.75% Train).
- **Các họ câu hỏi có sự lệch tỷ trọng đáng chú ý:**
  - **`Q19` (Đặc thù ngành):** Ở Public chiếm **4.2%** (42 câu) nhưng ở Train mẫu chỉ chiếm **0.58%** (7 câu).
  - **`Q08` (Đếm theo ngưỡng):** Ở Public chiếm **8.7%** (87 câu) cao hơn hẳn ở Train (**5.67%**).
  - **`Q05` (Kỳ/Ngày cao/thấp nhất):** Tập Train có tỷ lệ **3.75%** cao hơn Public (**1.7%**).

---

## 4. 3 Ví dụ tiêu biểu cho mỗi Họ câu hỏi

### Họ Q01
1. "Tháng 11 năm 2025, nhóm trà có mặt hàng nào bán được nhiều ly nhất?"
2. "Tháng 5/2026 bán ra bao nhiêu cây Ống PVC Bình Minh D50 cây 4m?"
3. "Rieng ngay hom nay doanh thu quan la bao nhieu?"

### Họ Q02
1. "Tuần rồi trung bình mổi đơn quán bán được nhiêu tiền vậy?"
2. "Th2/2026 không tính xe quay lại bảo hành, bình quân 1 phiếu sửa hết bnhiu tiền?"
3. "Dthu bình quân mỗi đơn tuần trước là bnhiu z?"

### Họ Q03
1. "Dịp Tết âm lịch 2026 có doanh thu hơn hay kém 17 ngày trước Tết bao nhiêu %?"
2. "Tháng này tới hôm nay, doanh thu so với cùng kỳ tháng trước thay đổi bao nhiêu phần trăm?"
3. "Thang 4 nam 2026 so thang truoc tien ban len hay xuong bao nhieu % z?"

### Họ Q04
1. "Kể giúp dthu 3 tháng liền từ T4/2025 tới T6/2025, theo thứ tự thời gian."
2. "Th3/2026, dthu các đơn trả xi ô đi và các đơn trả ví điện tử lần lượt là bnhiu?"
3. "Dthu th5/2026, th6/2026, th7/2026 lần lượt bnhiu?"

### Họ Q05
1. "2/2026 hàng mô lãi gộp nhiều nhất rứa?"
2. "Tại ngày 11/05/2026 ai đang nợ tiệm nhiều nhất, cho top 3 kèm số nợ từng người nha."
3. "Trong T5/2026, ngày mô dthu lên cao nhất?"

### Họ Q06
1. "Tuần rồi 5 món bán chạy nhất là những món nàoo?"
2. "Ai là khách có số đơn mua nhiều nhất trong tháng 4 năm 2025?"
3. "T1/2026 có bao nhiêu mặt hàng có mặt trong ít nhất 187 đơn vậy?"

### Họ Q07
1. "Trong tháng 3/2026, giờ cao điểm trong ngày của cửa hàng là mấy giờ?"
2. "Th7/2026 nhóm khách thân thiết mang về dthu bnhiu?"

### Họ Q08
1. "Từ đầu tháng này tới hết hnay, dthu cao hay thấp hơn cùng kỳ tháng trước mấy %?"
2. "Th6/2025 khách có từ 4 đơn trở lên chiếm mấy % tổng số khách mua?"
3. "Từ đầu năm tới hnay tiệm có bnhiu đơn hàng rồi?"

### Họ Q09
1. "Th5/2025 thuốc kê đơn chiếm bao nhiêu % tổng dthu?"
2. "T8/2025 tỉ trọng nhóm khai vị trong dthu là bnhiu %?"
3. "7/25 nhóm nước ép chiếm mấy % dthu cả quán?"

### Họ Q10
1. "6-2026 bị huỷ mấy đơn nhỉ?"
2. "Th3/2025 có mấy đơn bị huỷ vì giao trễ z?"
3. "Tỉ nệ huỷ đơn th7/2026 bnhiu %?"

### Họ Q11
1. "chị ơi, bữa 14/11/2025 mik tăng giá bo luc lac khoai tay tu 119k len 125k á, mún bít 7 ngay tu bua do so vs 7 ngay trc do bán nhìu hay ít hơn mí %"

### Họ Q12
1. "Thang nay nhap cua ncc nao nhieu tien nhat z?"
2. "Xem giúp mình tháng 12/2025, nhà cung cấp chính của cửa hàng là ai?"
3. "Tiền nhập hàng th này, tổng bnhiu?"

### Họ Q13
1. "9/2025 cộng các khoản tiền thu máy cũ trừ vào đơn, hết nhiêu tiền z?"
2. "Kể giúp các sp còn lô tồn hết hạn trong 90 ngày tới."
3. "Tháng 2 năm 2026, nhóm vitamin khoáng chất thì mặt hàng nào thu về nhiều tiền nhất?"

### Họ Q14
1. "Lô Oresol 245 DHG hộp 20 gói gần hết hạn nhất, hsd là ngày mấy z?"
2. "Tính tại ngày 08/08/2026, lô Augmentin 625mg hộp 2 vỉ x 7 viên cận date nhứt hết hạn ngày nào?"
3. "nè mik hỏi xíu, hiện kòn bn lô chua ban het mà hạn dùng rơi zô 120 ngay toi"

### Họ Q15
1. "Tổng tiền khách trả nợ 12-2025?"
2. "Đến hết 17/12/2025 chia tiền nợ còn lại theo tuổi nợ 0–7 ngày, 8–15 ngày và trên 15 ngày, từng nhóm có bao nhiêu tiền?"
3. "Đến hết ngày 15/05/2026, cửa hàng còn bao nhiêu khách hàng chưa trả hết nợ?"

### Họ Q17
1. "Lãi ròng của quán trong 10/2025 là bao nhiêu?"
2. "trog tháng 6 wán mik lãi gộp đc bnhiu dzạ 🤔"
3. "th7/2025 sp nao lãi gộp cao nhut z ta"

### Họ Q18
1. "Tới hôm nay, xe đã giao tổng cộng mấy cuộn Dây điện Cadivi VCmd 2x2.5 cuộn 100m cho khách đặt tháng 4/2025?"
2. "Thang nay tiem giao hang tong cong may dot?"
3. "T7/2025 tiem giao duoc bao nhieu dot hang?"

### Họ Q19
1. "Tháng 8/2025 thợ Đoàn Bảo Phúc lãnh bao nhiêu tiền huê hồng?"
2. "Thuốc kê đơn th này bán được bnhiu tiền gồi?"
3. "Đến hết hôm nay, các hợp đồng trả góp còn lại bao nhiêu tiền chưa góp, tính cả kỳ chưa tới hạn?"

### Họ Q20
1. "Quán mình bạn nhân viên nào phục vụ khách thân thiện nhất nhỉ?"
2. "Khách thân thiện nhất tháng 12/2025 là ai?"
3. "Tháng 10/2025, trong các khách hàng, ai khó tính nhất?"

