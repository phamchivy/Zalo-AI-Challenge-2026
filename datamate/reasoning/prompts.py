"""Prompt cho tầng B. Thứ tự message cố định để vLLM prefix-cache: [system+rulebook] -> [shop context] -> [câu hỏi].
Câu hỏi được xử lý theo nhóm doanh nghiệp nên prefix (rulebook + shop) trùng nhau giữa các câu -> rẻ & nhanh."""
from __future__ import annotations

import json

from datamate.core.types import Plan

CANON_DOC = """canon['lines']  — mỗi dòng hàng bán đã chuẩn hoá (dùng cho doanh thu/số đơn/SL/xếp hạng):
  order_id, date (datetime.date, ngày đặt), time ('HH:MM:SS' hoặc None), status ∈ {completed, cancelled, returned, shipping}
  (đơn có trả hàng một phần đã gán returned cho CẢ đơn), product_key, product_code, product_name, category, unit, qty,
  qty_base (SL quy về đơn vị cơ bản), base_unit, factor, unit_price, line_discount, amount (thành tiền dòng sau giảm dòng),
  cost_unit/cogs (giá vốn trên dòng bán, nếu có), customer_key (None = khách lẻ), customer_code, customer_display,
  customer_group, is_walkin, channel/channel_f, payment/payment_n ∈ {cash, transfer, ewallet, cod, debt, card, ...}, staff,
  is_service (Phí phục vụ: tính doanh thu, không phải mặt hàng), is_nonproduct (voucher/dùng điểm/phí ship: bỏ),
  order_total/paid/debt/ship_fee/order_discount (cấp đơn, nếu có).
canon['orders'] — mỗi đơn: order_id, date, time, status, customer_key, customer_display, payment_n, channel_f, revenue,
  n_lines, order_total, paid, debt, ship_fee ... (gồm cả đơn không có dòng hàng)
canon['products'] — product_key, product_code, product_name, category, base_unit, cost, price, rx, ingredient
canon['units'] — product_key, unit_f, factor (số đơn vị cơ bản trong 1 đơn vị), base_unit
canon['payments'] — order_id, pay_date, amount, method (mỗi lần khách trả tiền)
canon['purchases'] — dòng nhập: receipt_no, p_date, supplier, product_key, product_name, p_qty, p_unit, p_unit_cost,
  p_amount, p_total (tổng phiếu), p_status, p_cancelled
canon['stock'] — thẻ kho: product_key, ts (thời điểm), balance (tồn sau phát sinh, đơn vị cơ bản), seq
canon['prices'] — lịch sử giá: product_key, eff_date, price
canon['lots'] — lô: lot, product_key, expiry, received_base, unit_cost_base, recv_date; canon['lot_sales'] — order_id, lot, qty_base, return_qty
canon['deliveries'] — đợt giao: order_id, deliv_date, deliv_no
Bảng có thể rỗng nếu shop không ghi. Đơn nào tính doanh thu / số đơn, làm tròn, tiêu chí xếp hạng... lấy theo RULEBOOK của
split này (không giả định theo bộ khác); status đã chuẩn hoá về 4 giá trị ở trên. Dòng is_nonproduct không phải hàng.
Nếu canon thiếu thông tin hoặc có vẻ sai, dùng tables (bảng gốc, mọi ô là chuỗi: dùng to_num/parse_date)."""

CODE_SYSTEM = """Bạn là chuyên gia phân tích dữ liệu bán hàng của cửa hàng nhỏ Việt Nam. Viết MỘT đoạn code Python (pandas) \
tính đáp án chính xác cho câu hỏi dựa trên dữ liệu của shop.

QUY TẮC BẮT BUỘC
1. Áp dụng đúng RULEBOOK bên dưới (luật mặc định khi câu hỏi không nói khác) và README của shop (đặc tả dữ liệu, quy ước riêng).
2. Chỉ dùng dữ liệu có sẵn, không bịa. Dữ liệu không có loại thông tin câu hỏi cần, hoặc kỳ hỏi nằm hoàn toàn ngoài
   khoảng dữ liệu -> result = {"type": "unanswerable", "reason": "..."}. Không chắc chắn -> result = {"type": "abstain"}.
3. Code PHẢI gán biến: result = {"type": T, "answer": V}
   T ∈ number | percent | string | date | boolean | list | table | unanswerable | abstain.
   number: số (tiền làm tròn về đồng, nửa đồng làm tròn lên: round_money(x)); percent: số phần trăm làm tròn 2 chữ số
   (12.4 nghĩa là 12,4%); date: 'YYYY-MM-DD' (hỏi theo tháng: 'YYYY-MM'); string: tên nguyên văn trong dữ liệu;
   list: theo thứ tự câu hỏi nêu / giảm dần theo tiêu chí, hoà thì theo mã tăng dần; table: {"columns": [...], "rows": [[...]]}.
4. Biến có sẵn (không cần import): pd, np, re, math, date, datetime, timedelta, json, tables (dict tên->DataFrame gốc),
   canon (dict DataFrame đã chuẩn hoá), meta (as_of_date, data_min, data_max, caps), fold (bỏ dấu, chữ thường),
   to_num, parse_date, month_end(y, m), round_money.
5. Không đọc/ghi file, không mạng, không in nhiều. Code chạy tối đa 30 giây.

MÔ TẢ canon
{canon_doc}

RULEBOOK
{rulebook}
"""

SHOP_TEMPLATE = """# DOANH NGHIỆP {bid}
Khoảng dữ liệu: {dmin} .. {dmax}. Năng lực canon: {caps}

## README
{readme}

## BẢNG GỐC (tables) — [loại bảng] tên, cột [kiểu role=vai trò ánh xạ], ví dụ giá trị
{cards}
"""

QUESTION_TEMPLATE = """# CÂU HỎI
Hôm nay (as_of_date) = {as_of}
Câu hỏi: {question}

Gợi ý từ bộ phân tích câu (có thể sai, tự kiểm tra lại):
- Khoảng thời gian đọc được: {times}
- Thực thể khớp trong dữ liệu: {entities}
- Ý định dự đoán: {intent}

Viết code trong một khối ```python```. Chỉ đưa code."""

REPAIR_TEMPLATE = """Code trước bị lỗi khi chạy:
```
{error}
```
Sửa lại và đưa toàn bộ code mới trong một khối ```python```."""


PLAN_SYSTEM = """Bạn chuyển câu hỏi kinh doanh tiếng Việt thành Plan JSON cho thư viện metric có sẵn (không tự tính số).
metric: revenue | orders | qty | aov | customers | gross_profit | cancelled_orders | cancel_rate
op: value (1 số) | series (list theo từng kỳ) | values (list theo từng giá trị split_values của split_by) |
    pct_change (% kỳ 1 so với kỳ 2) | diff (kỳ 1 - kỳ 2) | share (% metric có filter / toàn shop cùng kỳ) |
    trend_up (tăng liên tục qua các kỳ) | top (xếp hạng group_by, k phần tử) | argmax (ngày/tháng cao nhất) |
    count_ge (đếm nhóm có chỉ số >= threshold) | pct_ge (% nhóm như trên)
group_by: product | customer | staff | category | day | month | order
filters (chỉ dùng giá trị có trong danh sách thực thể được cung cấp): product (product_key), category, channel, payment,
  customer_group, shift (morning|afternoon), staff, customer (customer_key), rx (true/false), size (M|L), base_unit
periods: list {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} theo rulebook (§2 thời gian). Kỳ chính đứng trước, kỳ so sánh sau.
Nếu câu hỏi không thuộc các metric trên -> {"supported": false}.
RULEBOOK (trích):
{rulebook}"""

PLAN_SCHEMA = {
    'type': 'object',
    'properties': {
        'supported': {'type': 'boolean'},
        'metric': {'type': 'string', 'enum': ['revenue', 'orders', 'qty', 'aov', 'customers', 'gross_profit',
                                              'cancelled_orders', 'cancel_rate']},
        'op': {'type': 'string', 'enum': ['value', 'series', 'values', 'pct_change', 'diff', 'share', 'trend_up', 'top',
                                          'argmax', 'count_ge', 'pct_ge']},
        'periods': {'type': 'array', 'items': {'type': 'object', 'properties': {'start': {'type': 'string'}, 'end': {'type': 'string'}},
                                               'required': ['start', 'end']}},
        'filters': {'type': 'object'},
        'group_by': {'type': 'string'},
        'split_by': {'type': 'string'},
        'split_values': {'type': 'array', 'items': {'type': 'string'}},
        'k': {'type': 'integer'},
        'threshold': {'type': 'number'},
        'threshold_metric': {'type': 'string'},
        'answer_type': {'type': 'string', 'enum': ['number', 'percent', 'string', 'date', 'boolean', 'list']},
    },
    'required': ['supported'],
}


def fill(template: str, **kw) -> str:
    """Thay {key} bằng giá trị (không dùng str.format vì prompt chứa ngoặc nhọn JSON)."""
    for k, v in kw.items():
        template = template.replace('{' + k + '}', str(v))
    return template


def code_messages(rulebook: str, shop_block: str, q_block: str) -> list[dict]:
    return [
        {'role': 'system', 'content': fill(CODE_SYSTEM, canon_doc=CANON_DOC, rulebook=rulebook)},
        {'role': 'user', 'content': shop_block},
        {'role': 'assistant', 'content': 'Đã nắm dữ liệu của shop. Mời đưa câu hỏi.'},
        {'role': 'user', 'content': q_block},
    ]


def plan_messages(rulebook_excerpt: str, entity_block: str, q_block: str) -> list[dict]:
    return [
        {'role': 'system', 'content': fill(PLAN_SYSTEM, rulebook=rulebook_excerpt)},
        {'role': 'user', 'content': entity_block + '\n\n' + q_block + '\nTrả về đúng một JSON.'},
    ]


def describe_plan(p: Plan) -> str:
    return json.dumps(p.model_dump(mode='json', exclude_none=True), ensure_ascii=False)
