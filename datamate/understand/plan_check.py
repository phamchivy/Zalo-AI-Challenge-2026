"""v3.3.3 — LLM kiểm tra CÁCH TẦNG A ĐÃ HIỂU câu hỏi (không tự giải, không thêm regex).

Vì sao: trên bộ dev viết lại, 10/10 câu tầng A sai thật là ĐỌC NHẦM (khớp một phần): "Q2/2026" -> tháng 2,
"từ đầu năm tới hết hôm nay" -> hôm nay, "sl đh" -> số lượng hàng, "đếm ... từ 180.000đ" -> tổng doanh thu...
Chuẩn hoá câu hỏi (v3.3.0) chỉ sửa ĐẦU VÀO; regex vẫn là bên đọc nên đọc nhầm câu chuẩn y như câu gốc.
Ở đây LLM nhìn ĐẦU RA của việc đọc: code in kế hoạch có cấu trúc của tầng A ra câu chữ (đại lượng, phép tính,
khoảng ngày + thứ, cụm chữ đã khớp, bộ lọc, các luật đã áp kèm dòng rulebook làm căn cứ), LLM so với câu hỏi + RULEBOOK.

Code ở đây KHÔNG đọc câu hỏi: chỉ dịch Plan (đã có) sang tiếng Việt. Phần hiểu nghĩa là của LLM.
Bác chỉ khi LLM chỉ ra được chỗ lệch cụ thể, và đủ `min_reject` lần hỏi cùng bác. Câu bị bác -> bỏ trống -> chuyển cho
LLM giải như câu tầng A bỏ trống; tầng A vẫn là 1 phiếu khi đếm phiếu (keep_vote) nên bác nhầm chỉ mất điểm khi LLM
không ra cùng đáp án.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from datamate.core.types import Plan

log = logging.getLogger('plan_check')

WD = ['thứ Hai', 'thứ Ba', 'thứ Tư', 'thứ Năm', 'thứ Sáu', 'thứ Bảy', 'Chủ nhật']

METRIC_VN = {
    'revenue': 'doanh thu (tiền bán hàng)', 'orders': 'số đơn hàng', 'cancelled_orders': 'số đơn bị huỷ',
    'cancel_rate': 'tỉ lệ đơn huỷ (%)', 'all_orders': 'tổng số đơn (mọi trạng thái)',
    'aov': 'giá trị trung bình mỗi đơn', 'qty': 'số lượng hàng bán ra (đơn vị sản phẩm)',
    'gross_profit': 'lợi nhuận gộp', 'unit_price': 'đơn giá', 'lines': 'số dòng hàng',
    'customers': 'số khách hàng (không tính khách lẻ)', 'active_days': 'số ngày có bán',
    'debt': 'công nợ khách', 'purchase_total': 'tổng tiền nhập hàng', 'purchase_avg_price': 'giá nhập trung bình',
    'purchase_supplier': 'nhà cung cấp (nhập hàng)', 'purchase_minus_sales': 'lượng nhập trừ lượng bán',
    'price_effect': 'ảnh hưởng của việc đổi giá', 'dead_stock': 'hàng tồn không bán được',
    'delivery_count': 'số lần giao hàng', 'inventory_value': 'giá trị tồn kho', 'stock_qty': 'số lượng tồn kho',
    'lot_nearest': 'lô hàng gần hết hạn nhất', 'expiry': 'hàng sắp hết hạn', 'min_orders': 'số đơn tối thiểu',
}
GROUP_VN = {'product': 'mặt hàng', 'customer': 'khách hàng', 'day': 'ngày', 'month': 'tháng', 'staff': 'nhân viên',
            'supplier': 'nhà cung cấp', 'order': 'đơn hàng', 'hour': 'giờ trong ngày', 'category': 'nhóm hàng',
            'week': 'tuần', 'weekday': 'thứ trong tuần'}
FILTER_VN = {'product': 'mặt hàng', 'category': 'nhóm hàng', 'channel': 'kênh bán', 'payment': 'hình thức thanh toán',
             'customer_group': 'nhóm khách', 'shift': 'ca', 'staff': 'nhân viên', 'customer': 'khách',
             'rx': 'thuốc kê đơn', 'ingredient': 'hoạt chất', 'base_unit': 'đơn vị', 'size': 'size', 'walkin': 'khách lẻ'}
LABEL_VN = {'main': 'kỳ chính', 'compare': 'kỳ so sánh', 'snapshot': 'thời điểm'}
GRAN_VN = {'day': 'ngày', 'week': 'tuần', 'month': 'tháng', 'quarter': 'quý', 'year': 'năm', 'custom': 'khoảng tuỳ ý',
           'tet': 'dịp Tết', 'snapshot': 'thời điểm'}
TIME_KNOBS = {'week': ['time.week_start'], 'month': ['time.month']}

SYS = ('Bạn là người KIỂM TRA. Một chương trình tự động đã đọc câu hỏi của chủ cửa hàng và lập KẾ HOẠCH TÍNH. '
       'Việc của bạn: xác định kế hoạch đó có tính ĐÚNG thứ câu hỏi hỏi, theo đúng RULEBOOK của cửa hàng, hay không. '
       'Bạn KHÔNG tính toán và KHÔNG trả lời câu hỏi.\n'
       'So từng phần:\n'
       '1. thoi_gian: ngày bắt đầu – kết thúc (kèm thứ) có đúng khoảng câu hỏi nói không, tính theo NGÀY HỎI đã cho '
       '(tháng / quý / tuần / năm, "hôm nay", "từ đầu năm đến nay", "cùng kỳ năm ngoái", "tháng trước"...). '
       'Chú ý cụm chữ máy đã khớp: nếu nó chỉ là MỘT PHẦN của cách nói thời gian trong câu thì thường là đọc nhầm.\n'
       '2. dai_luong: doanh thu / số đơn / số lượng hàng / số khách / lợi nhuận... có đúng thứ được hỏi không.\n'
       '3. phep_tinh: một con số, đếm theo ngưỡng, top N, so sánh %, chênh lệch, tỉ trọng, danh sách...; '
       'đếm cái gì, ngưỡng bao nhiêu, N bằng mấy, kỳ nào so với kỳ nào.\n'
       '4. loc: mặt hàng / nhóm / kênh / thanh toán / khách / ca... có đúng và đủ điều kiện của câu hỏi không.\n'
       '5. luat: các luật kế hoạch đã áp có khớp RULEBOOK bên dưới không (vd kỳ theo ngày đặt hay ngày giao, '
       'tuần bắt đầu thứ mấy, "quá N" có gồm N không, trạng thái đơn nào được tính).\n'
       'Nguyên tắc:\n'
       '- Chỉ kết luận SAI khi chỉ ra được một chỗ CỤ THỂ kế hoạch khác câu hỏi hoặc trái RULEBOOK.\n'
       '- Diễn đạt khác mà cùng nghĩa là ĐÚNG. Chi tiết câu hỏi không nói và RULEBOOK có quy định mặc định thì theo mặc '
       'định là ĐÚNG. Không đánh giá con số kết quả to hay nhỏ.\n'
       '- Câu hỏi dùng viết tắt / không dấu / khẩu ngữ (vd dthu = doanh thu, đh = đơn hàng, sl = số lượng, '
       'Q2 = quý 2): hiểu theo nghĩa thông dụng rồi mới so.\n'
       'Chỉ trả về JSON: {"dung": true|false, "loi": [{"phan": "thoi_gian|dai_luong|phep_tinh|loc|luat", '
       '"cau_hoi_noi": "<câu hỏi nói gì>", "ke_hoach_lam": "<kế hoạch làm gì>"}]}  ("loi" rỗng khi đúng).')
SCHEMA = {'type': 'object', 'properties': {
    'dung': {'type': 'boolean'},
    'loi': {'type': 'array', 'items': {'type': 'object', 'properties': {
        'phan': {'type': 'string'}, 'cau_hoi_noi': {'type': 'string'}, 'ke_hoach_lam': {'type': 'string'}}}}},
    'required': ['dung', 'loi']}
PARTS = {'thoi_gian', 'dai_luong', 'phep_tinh', 'loc', 'luat'}


# ----------------------------------------------------------------------------- Plan -> câu chữ
def _d(x) -> str:
    if isinstance(x, str):
        x = date.fromisoformat(x[:10])
    return f'{x.day:02d}/{x.month:02d}/{x.year} ({WD[x.weekday()]})'


def _m(name: Optional[str]) -> str:
    return METRIC_VN.get(name or '', name or '?')


def _g(name: Optional[str]) -> str:
    return GROUP_VN.get(name or '', name or '?')


def _short(v: Any, n: int = 300) -> str:
    s = json.dumps(v, ensure_ascii=False, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[:n] + '…'


def _op_text(p: Plan) -> str:
    op, met, gb = p.op, _m(p.metric), _g(p.group_by)
    if op == 'value':
        return f'một con số: {met} trong kỳ chính'
    if op == 'values':
        return f'một danh sách số: {met} cho từng {FILTER_VN.get(p.split_by or "", p.split_by)} = {_short(p.split_values)} (đúng thứ tự này)'
    if op == 'series':
        return f'một danh sách số: {met} của từng kỳ liệt kê bên dưới (đúng thứ tự)'
    if op in ('top', 'bottom'):
        s = f'{"top" if op == "top" else "bét"} {p.k or 1} {gb} theo {met} {"giảm" if op == "top" else "tăng"} dần'
        if p.threshold_metric == 'min_orders' and p.threshold:
            s += f', chỉ xét {gb} có từ {p.threshold:g} đơn trở lên'
        return s
    if op == 'argmax':
        return f'{gb} có {met} {"nhỏ" if p.note == "min" else "lớn"} nhất'
    if op == 'share':
        base = [k for k in (p.note.split(',') if p.note else []) if k]
        den = 'chỉ giữ lọc ' + ', '.join(FILTER_VN.get(k, k) for k in base) if base else 'toàn cửa hàng, không lọc'
        return f'tỉ lệ % = {met} (áp mọi bộ lọc) / {met} ({den}), cùng kỳ chính'
    if op == 'pct_change':
        return f'% thay đổi {met} = (kỳ chính − kỳ so sánh) / kỳ so sánh × 100'
    if op == 'pct_change_filters':
        a, b = (p.split_values + [None, None])[:2]
        return f'% chênh {met} giữa {_short(a)} và {_short(b)} = (thứ nhất − thứ hai) / thứ hai × 100, cùng kỳ chính'
    if op == 'diff':
        return f'chênh lệch {met} = kỳ chính − kỳ so sánh'
    if op == 'trend_up':
        return f'đúng/sai: {met} tăng liên tục qua các kỳ liệt kê'
    if op in ('count_ge', 'pct_ge'):
        cmp = '>' if p.note == 'gt' else '≥'
        what = f'{_m(p.threshold_metric or p.metric)} của mỗi {gb}'
        head = 'đếm số' if op == 'count_ge' else 'tỉ lệ % số'
        return f'{head} {gb} có {what} {cmp} {p.threshold:g}' if p.threshold is not None else f'{head} {gb} theo ngưỡng'
    if op == 'weekday_avg':
        wd = WD[int(p.threshold) - 1] if p.threshold else '?'
        return f'trung bình {met} mỗi {wd} trong kỳ chính (tổng chia số ngày {wd})'
    if op == 'unit_price':
        return ('đơn giá bán của mặt hàng: đơn giá ghi trên dòng bán (trước chiết khấu dòng, quy về đơn vị cơ bản), '
                'lấy cả dòng của đơn huỷ / trả, chọn giá xuất hiện nhiều nhất trong kỳ')
    return f'phép "{op}" trên {met}' + (f' theo {gb}' if p.group_by else '')


def _special(p: Plan) -> Optional[list[str]]:
    """Plan có nghĩa kỳ khác thường (executor tự dựng khoảng ngày): in đúng khoảng executor thật sự tính."""
    from datetime import timedelta
    if p.metric == 'price_effect':
        n = int(p.threshold or 0)
        if p.periods and n > 0:
            D = date.fromisoformat(str(p.periods[0].start)[:10])
            a = f'{_d(D)} → {_d(D + timedelta(days=n - 1))}'
            b = f'{_d(D - timedelta(days=n))} → {_d(D - timedelta(days=1))}'
            when = f'ngày đổi giá D = {_d(D)}'
        else:
            a, b, when = f'{n} ngày kể từ D (gồm D)', f'{n} ngày ngay trước D', 'D = ngày tăng giá tìm trong lịch sử giá'
        return ['- Đại lượng: số lượng bán của mặt hàng (quy về đơn vị cơ bản)',
                f'- Phép tính: % thay đổi = (kỳ sau − kỳ trước) / kỳ trước × 100, N = {n} ngày, {when}',
                f'- Thời gian (kỳ sau): {a}', f'- Thời gian (kỳ trước): {b}']
    if p.metric == 'dead_stock' and p.periods:
        t = p.periods[0]
        return ['- Đại lượng: mặt hàng không bán được (hàng ế)',
                f'- Phép tính: {"đếm số" if p.op == "count" else "danh sách"} mặt hàng TỪNG bán trước {_d(t.start)} nhưng '
                f'KHÔNG bán được ngày nào trong {_d(t.start)} → {_d(t.end)}'
                + (f'; máy đọc từ cụm chữ "{t.expr}"' if t.expr else '')]
    return None


def _knobs_shown(p: Plan, knobs: list[str]) -> list[str]:
    """Bỏ những nút luật mà dòng 'Phép tính' đã ghi cách áp thật (tránh LLM hiểu nhầm cách executor tính)."""
    if p.op in ('count_ge', 'pct_ge'):
        knobs = [k for k in knobs if k != 'threshold.strict']          # so sánh >/≥ thật đã ghi ở dòng phép tính
    if p.op == 'unit_price':
        knobs = [k for k in knobs if not k.startswith(('revenue.', 'orders.'))]   # đơn giá lấy cả dòng đơn huỷ / trả
    return knobs


def render_plan(plan: Plan, rules=None, answer_type: Optional[str] = None, answer: Any = None) -> str:
    """Kế hoạch tầng A -> câu chữ cho LLM kiểm tra. Không đọc câu hỏi."""
    p = plan
    sp = _special(p)
    out = sp or [f'- Đại lượng: {_m(p.metric)}', f'- Phép tính: {_op_text(p)}']
    if sp:
        pass
    elif p.periods:
        for i, t in enumerate(p.periods):
            lab = LABEL_VN.get(t.label) or (f'kỳ {i + 1}' if t.label.startswith('series') else t.label)
            n = (date.fromisoformat(str(t.end)[:10]) - date.fromisoformat(str(t.start)[:10])).days + 1
            if t.granularity == 'snapshot' or t.label == 'snapshot':
                s = f'- Thời gian ({lab}): tính đến hết ngày {_d(t.end)}'
            else:
                s = f'- Thời gian ({lab}): {_d(t.start)} → {_d(t.end)}, {n} ngày, kiểu {GRAN_VN.get(t.granularity, t.granularity)}'
            if t.expr:
                s += f'; máy đọc từ cụm chữ "{t.expr}"'
            out.append(s)
    else:
        out.append('- Thời gian: không giới hạn (toàn bộ dữ liệu)')
    F = {k: v for k, v in (p.filters or {}).items() if v not in (None, '', [])}
    out.append('- Bộ lọc: ' + ('; '.join(f'{FILTER_VN.get(k, k)} = {_short(v, 120)}' for k, v in F.items()) if F else 'không có'))
    if rules is not None:
        try:
            from datamate.reasoning.executor import Executor
            knobs, _ = Executor.requirements(SimpleNamespace(r=rules), p)
        except Exception:  # noqa: BLE001
            knobs = []
        for t in p.periods or []:
            knobs += TIME_KNOBS.get(t.granularity, [])
        knobs = _knobs_shown(p, list(dict.fromkeys(knobs)))
        if knobs:
            out.append('- Luật đã áp:')
            ev = getattr(rules, 'evidence', {}) or {}
            for k in knobs:
                v = rules.get(k)
                s = f'  • {k} = {_short(v, 120)}'
                if ev.get(k):
                    s += f'  (căn cứ rulebook: "{_short(ev[k].strip(), 180)}")'
                out.append(s)
    if answer_type:
        out.append(f'- Kết quả trả ra: kiểu {answer_type}' + (f', giá trị {_short(answer, 200)}' if answer is not None else ''))
    return '\n'.join(out)


# ----------------------------------------------------------------------------- gọi LLM
def _parse(text: str) -> Optional[dict]:
    from datamate.runtime.llm_client import extract_json
    o = extract_json(text or '')
    if not isinstance(o, dict) or 'dung' not in o:
        return None
    errs = []
    for e in o.get('loi') or []:
        if isinstance(e, dict) and str(e.get('phan', '')).strip() in PARTS and str(e.get('cau_hoi_noi', '')).strip() \
                and str(e.get('ke_hoach_lam', '')).strip():
            errs.append({'phan': str(e['phan']).strip(), 'cau_hoi_noi': str(e['cau_hoi_noi'])[:200],
                         'ke_hoach_lam': str(e['ke_hoach_lam'])[:200]})
    dung = o.get('dung')
    if isinstance(dung, str):
        dung = dung.strip().lower() in ('true', 'đúng', 'dung', 'yes')
    # SAI chỉ được tính khi chỉ ra được chỗ lệch cụ thể
    return {'reject': (dung is False) and bool(errs), 'errors': errs}


async def check_one(llm, question: str, as_of, plan_text: str, rulebook_text: str, cfg, cache_dir: Optional[str]) -> dict:
    """Hỏi `cfg.samples` lần; bác khi >= cfg.min_reject lần bác (có chỗ lệch cụ thể). Lỗi gọi / JSON hỏng = không bác."""
    asd = as_of if isinstance(as_of, date) else date.fromisoformat(str(as_of)[:10])
    sys = SYS + ('\n\n=== RULEBOOK của cửa hàng ===\n' + rulebook_text[:cfg.rulebook_chars] if rulebook_text and cfg.rulebook_chars else '')
    user = f'Ngày hỏi (hôm nay): {_d(asd)}\nCâu hỏi: {question}\n\nKẾ HOẠCH TÍNH của chương trình:\n{plan_text}'
    model = getattr(getattr(llm, 'cfg', None), 'model', '')
    cdir = Path(cache_dir) / 'plan_check' if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)

    async def sample(i: int) -> Optional[dict]:
        key = hashlib.sha1(f'{model}\n{i}\n{sys}\n{user}'.encode('utf-8')).hexdigest()
        cf = cdir / f'{key}.json' if cdir else None
        if cf is not None and cf.exists():
            try:
                return json.loads(cf.read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001
                pass
        for _ in range(2):
            r = await llm.chat([{'role': 'system', 'content': sys}, {'role': 'user', 'content': user}], json_schema=SCHEMA,
                               max_tokens=cfg.max_tokens, temperature=cfg.temperature, seed=1000 + i)
            v = _parse(r.text) if not r.error else None
            if v is not None:
                if cf is not None:
                    cf.write_text(json.dumps(v, ensure_ascii=False), encoding='utf-8')
                return v
        return None

    res = await asyncio.gather(*(sample(i) for i in range(max(1, cfg.samples))))
    votes = [r for r in res if r is not None]
    n_rej = sum(1 for r in votes if r['reject'])
    errs = [e for r in votes if r['reject'] for e in r['errors']]
    return {'reject': n_rej >= cfg.min_reject, 'n_reject': n_rej, 'n_valid': len(votes), 'errors': errs[:4],
            'plan_text': plan_text}


def reason_text(v: dict) -> str:
    return '; '.join(f'[{e["phan"]}] câu hỏi: {e["cau_hoi_noi"]} / máy đọc: {e["ke_hoach_lam"]}' for e in v.get('errors', [])[:2])
