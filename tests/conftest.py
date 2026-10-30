"""Fixture: split tổng hợp nhỏ (không dùng dữ liệu cuộc thi) để test end-to-end không cần GPU."""
import csv
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

README = """# Dữ liệu biz_t01

Một file phẳng, mỗi dòng một dòng hàng.

## File
- `ban_hang.csv` (csv)

## Cột trong `ban_hang.csv`
- **Mã đơn**: mã đơn
- **Ngày**: ngày bán
- **Trạng thái**: trạng thái đơn
- **Khách**: tên khách (Khách lẻ nếu không lưu)
- **Tên hàng**: tên hàng
- **ĐVT**: đơn vị tính
- **SL**: số lượng
- **Đơn giá**: đơn giá
- **Thành tiền**: thành tiền sau giảm

## Quy ước
- Dữ liệu có đến hết 31/07/2025.
- **Quy đổi đơn vị**: số lượng ghi theo đơn vị của dòng:
  - Coca lon: 1 thùng = 24 lon
"""

ROWS = [
    # Mã đơn, Ngày, Trạng thái, Khách, Tên hàng, ĐVT, SL, Đơn giá, Thành tiền
    ('O1', '01/06/2025', 'Hoàn thành', 'Khách lẻ', 'Coca lon', 'lon', '2', '10,000', '20,000'),
    ('O1', '01/06/2025', 'Hoàn thành', 'Khách lẻ', 'Bánh mì', 'cái', '1', '15,000', '15,000'),
    ('O2', '01/06/2025', 'Đã hủy', 'Anh Tú', 'Coca lon', 'thùng', '1', '220,000', '220,000'),
    ('O3', '02/06/2025', 'Hoàn thành', 'Chị Lan', 'Coca lon', 'thùng', '1', '220,000', '220,000'),
    ('O3', '02/06/2025', 'Hoàn thành', 'Chị Lan', 'Bánh mì', 'cái', '2', '15,000', '30,000'),
    ('O4', '02/06/2025', 'Trả hàng', 'Chị Lan', 'Bánh mì', 'cái', '1', '15,000', '15,000'),
    ('O5', '03/06/2025', 'Đang giao', 'Anh Tú', 'Coca lon', 'lon', '3', '10,000', '30,000'),
    ('O5', '03/06/2025', 'Đang giao', 'Anh Tú', 'Coca lon', 'lon', '3', '10,000', '30,000'),   # dòng trùng -> tính 1 lần
    ('O6', '03/06/2025', 'Hoàn thành', 'Chị Lan', 'Bánh mì', 'cái', '1', '15,000', '15,000'),
    ('O6', '03/06/2025', 'Hoàn thành', '', 'Coca lon', 'lon', '1', '10,000', '10,000'),           # ô trống = dòng đầu
    ('O7', '01/07/2025', 'Hoàn thành', 'Khách lẻ', 'Coca lon', 'lon', '1', '10,000', '10,000'),
]

# Rulebook tổng hợp tối thiểu cho fixture (đúng dạng bảng luật của đề) — tầng A chỉ trả lời khi luật được xác nhận.
RULEBOOK = """# Rulebook test

## 1. Đơn hàng và doanh thu
| Luật | Mặc định |
|---|---|
| Đơn tính vào doanh thu | đơn **không huỷ** và **không trả hàng**. Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn**. Đơn đang giao vẫn tính |
| Đơn thuộc kỳ nào | theo **ngày đặt**, không theo ngày giao hay ngày thu tiền |
| Đơn bị huỷ | đơn có trạng thái huỷ. Đơn trả hàng **không** phải đơn huỷ |
| Số đơn | chỉ tính đơn không huỷ, không trả; cả khi đếm số đơn có một mặt hàng |
| Doanh thu một đơn | tổng **thành tiền các dòng hàng** (đã trừ giảm giá trên từng dòng). **Không** trừ giảm giá trên cả đơn, **không** cộng thuế VAT, **không** gồm phí giao hàng |
| Tỉ lệ đơn huỷ trong kỳ | phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**) |

## 2. Thời gian
| Cách nói | Nghĩa |
|---|---|
| Ca sáng, ca chiều | theo giờ tạo đơn: ca sáng 06:00–13:59, ca chiều 14:00–21:59 |

## 3. Xếp hạng
| Luật | Mặc định |
|---|---|
| Mặt hàng | Dòng **Phí phục vụ** không phải mặt hàng: không xếp hạng, không đếm |
| Bán chạy | theo **số lượng quy về đơn vị cơ bản**, không theo số dòng hay số tiền |
| Bán được bao nhiêu <đơn vị> | cộng số lượng quy về đơn vị cơ bản của các mặt hàng **có đơn vị cơ bản là <đơn vị>** |
| Xác định khách | Shop có mã khách: mỗi **mã khách** là một khách; đơn không ghi mã (**khách lẻ**) không phải khách. Khách mua **nhiều tiền** nhất xếp theo doanh thu (§1). |
| Giá trị trung bình một đơn | làm tròn về đồng, nửa đồng làm tròn lên |
| Khách trung thành nhất | khách có **nhiều ngày có mua nhất** trong kỳ |
"""

QUESTIONS = [
    ('t1', 'doanh thu tháng 6/2025 là bao nhiêu', '2025-07-31', 'number', 340000),
    ('t2', 'tháng 6/2025 có bao nhiêu đơn', '2025-07-31', 'number', 4),
    ('t3', 'tỉ lệ huỷ đơn tháng 6/2025 là bao nhiêu %', '2025-07-31', 'percent', 16.67),
    ('t4', 'tháng 6/2025 bán được bao nhiêu lon Coca lon', '2025-07-31', 'number', 30),
    ('t5', 'mặt hàng bán chạy nhất tháng 6/2025', '2025-07-31', 'string', 'Coca lon'),
    ('t6', 'khách nào mua nhiều tiền nhất tháng 6/2025', '2025-07-31', 'string', 'Chị Lan'),
    ('t7', 'giá trị trung bình một đơn tháng 6/2025', '2025-07-31', 'number', 85000),
    ('t8', 'doanh thu tháng 3/2024 bao nhiêu', '2025-07-31', 'unanswerable', None),
    ('t9', 'số đơn bị huỷ trong tháng 6/2025', '2025-07-31', 'number', 1),
    ('t10', 'dthu hqua', '2025-06-04', 'number', 55000),
]


@pytest.fixture(scope='session')
def mini_split(tmp_path_factory):
    root = tmp_path_factory.mktemp('split')
    d = root / 'data' / 'biz_t01'
    d.mkdir(parents=True)
    (d / 'README.md').write_text(README, encoding='utf-8')
    with open(d / 'ban_hang.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Mã đơn', 'Ngày', 'Trạng thái', 'Khách', 'Tên hàng', 'ĐVT', 'SL', 'Đơn giá', 'Thành tiền'])
        w.writerows(ROWS)
    (root / 'rulebook.md').write_text(RULEBOOK, encoding='utf-8')
    (root / 'calendar.csv').write_text('date,weekday,lunar_day,lunar_month,lunar_year,lunar_leap,holiday\n', encoding='utf-8')
    with open(root / 'questions.jsonl', 'w', encoding='utf-8') as fq, open(root / 'answers.jsonl', 'w', encoding='utf-8') as fa:
        for qid, q, asof, t, a in QUESTIONS:
            fq.write(json.dumps({'id': qid, 'business_id': 'biz_t01', 'question': q, 'as_of_date': asof}, ensure_ascii=False) + '\n')
            fa.write(json.dumps({'id': qid, 'type': t, 'answer': a, 'alias': []}, ensure_ascii=False) + '\n')
    return root
