"""v3.0.0 — LLM ở bước dựng dữ liệu (bước 4 kế hoạch v3), theo khung Magneto: ứng viên rẻ -> LLM chấm -> gán một-một
-> dữ liệu (invariants) quyết định cuối. LLM KHÔNG có quyết định cuối.

1. Gán nhãn (enums): giá trị trạng thái / hình thức thanh toán mà từ điển không hiểu -> LLM xếp vào tập đóng
   (+ 'unknown'). Chỉ nhãn từ điển không hiểu mới được gửi; nhãn từ điển đã hiểu giữ nguyên. LLM nói 'unknown'
   thì giữ unknown -> invariant I15 đỏ -> câu liên quan bị chặn.
2. Gán vai trò cột còn thiếu (roles): với bảng bán hàng thiếu vai trò cốt lõi, gửi PROFILE NÉN của các cột chưa gán
   (kiểu, % trống, số giá trị khác nhau, ~8 giá trị mẫu, mô tả README) + danh sách vai trò còn thiếu. LLM chấm 0–1
   từng cặp; Hungarian gán một-một; chỉ nhận cặp >= min_conf và hợp kiểu giá trị. Catalog dựng canonical với và
   không với các vai trò này, giữ bản có invariants tốt hơn; ngang nhau giữ bản không LLM.

Prompt chỉ chứa profile (không gửi bảng thô). Không có tên shop hay luật riêng.
"""
from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np

from datamate.core.textutil import fold
from datamate.data.schema_mapper import ALLOWED, ROLES, BusinessMapping, _value_ok
from datamate.data.vocab import STATUS, norm_payment, norm_status

log = logging.getLogger(__name__)

STATUS_SET = ['completed', 'cancelled', 'returned', 'shipping', 'unknown']
PAYMENT_SET = ['cash', 'transfer', 'ewallet', 'card', 'cod', 'debt', 'other', 'unknown']
PAYMENT_KNOWN = {'cash', 'transfer', 'ewallet', 'card', 'cod', 'debt'}
MAX_LABELS = 60

# vai trò cốt lõi LLM được phép bổ sung (mô tả ngắn cho prompt)
CORE_ROLES = {
    'order_id': 'mã đơn / hoá đơn / phiếu bán',
    'date': 'ngày bán / ngày đặt',
    'datetime': 'thời điểm đặt (ngày + giờ)',
    'status': 'trạng thái đơn (hoàn tất / huỷ / trả hàng / đang giao)',
    'customer_code': 'mã khách hàng',
    'customer_name': 'tên khách hàng',
    'product_code': 'mã mặt hàng',
    'product_name': 'tên mặt hàng',
    'unit': 'đơn vị tính của dòng bán',
    'qty': 'số lượng bán của dòng',
    'unit_price': 'đơn giá một đơn vị (trước giảm)',
    'line_discount': 'tiền giảm của riêng dòng',
    'amount': 'thành tiền của dòng',
    'goods_total': 'tổng tiền hàng của đơn (trước giảm cả đơn)',
    'order_discount': 'tiền giảm của cả đơn',
    'ship_fee': 'phí giao hàng',
    'order_total': 'số khách phải trả cho cả đơn',
    'paid': 'số tiền khách đã trả',
    'debt': 'số tiền khách còn nợ',
    'payment': 'hình thức thanh toán',
    'channel': 'kênh bán',
    'staff': 'nhân viên bán',
}
SALES_TYPES = {'sales_orders', 'sales_lines', 'sales_flat', 'sales', 'other'}
# v3.0.1: vai trò hỏi LLM theo loại bảng (bảng đơn không hỏi vai trò dòng hàng và ngược lại) -> ít nhiễu hơn
ORDER_ROLES = {'order_id', 'date', 'datetime', 'status', 'customer_code', 'customer_name', 'goods_total',
               'order_discount', 'ship_fee', 'order_total', 'paid', 'debt', 'payment', 'channel', 'staff'}
LINE_ROLES = {'order_id', 'product_code', 'product_name', 'unit', 'qty', 'unit_price', 'line_discount', 'amount'}
ROLES_BY_TYPE = {'sales_orders': ORDER_ROLES, 'sales_lines': LINE_ROLES}
STATUS_HOLDERS = {'sales_orders', 'sales_flat', 'sales'}     # loại bảng mang đơn hàng (có thể có trạng thái đơn)
UNRESOLVED_CONF = 0.5         # LLM nói có cột trạng thái (>= mức này) mà cuối cùng không gán được -> báo đỏ

_SYS = ('Bạn là bộ phân loại dữ liệu bán hàng của cửa hàng nhỏ ở Việt Nam. Chỉ dựa vào giá trị và mô tả được cung cấp. '
        'Không chắc thì chọn "unknown" (nhãn) hoặc "skip" (cột). Chỉ trả về một đối tượng JSON, không giải thích.')


# ----------------------------------------------------------------------------- 1. nhãn
def _label_values(raw, bm: BusinessMapping, role: str) -> dict[str, int]:
    cnt: dict[str, int] = {}
    for m in bm.tables.values():
        c = m.roles.get(role)
        if not c or m.table not in raw or c not in raw[m.table].df.columns:
            continue
        vc = raw[m.table].df[c].dropna().astype(str).str.strip().value_counts()
        for v, n in vc.items():
            if v:
                cnt[v] = cnt.get(v, 0) + int(n)
    return cnt


def _enum_prompt(kind: str, labels: list[tuple[str, int]], desc: str) -> tuple[str, dict]:
    if kind == 'status':
        cats = ('completed = đơn đã hoàn tất / thành công / đã giao / đã nhận; cancelled = đơn bị huỷ, khách không lấy, '
                'không giao được; returned = khách trả lại hàng (kể cả trả một phần), hoàn hàng; shipping = chưa xong: '
                'đang giao, đang xử lý, chờ giao; unknown = không đủ căn cứ')
        ex = {v: k for k, vs in STATUS.items() for v in list(vs)[:2]}
        allowed = STATUS_SET
    else:
        cats = ('cash = tiền mặt; transfer = chuyển khoản / ngân hàng / QR; ewallet = ví điện tử (Momo, ZaloPay…); '
                'card = quẹt thẻ; cod = thu hộ khi giao hàng; debt = ghi nợ / mua chịu / trả góp; other = hình thức khác; '
                'unknown = không đủ căn cứ')
        ex = {'tiền mặt': 'cash', 'chuyển khoản': 'transfer', 'thu hộ (COD)': 'cod', 'ghi nợ': 'debt'}
        allowed = PAYMENT_SET
    lines = '\n'.join(f'- {v!r} ({n} dòng)' for v, n in labels)
    user = (f'Cột: {desc}\nXếp mỗi nhãn dưới đây vào đúng một loại: {cats}.\n'
            f'Ví dụ cách xếp (từ cửa hàng khác): {ex}\n\nNhãn cần xếp:\n{lines}\n\n'
            'Trả về JSON dạng {"map": {"<nhãn đúng như trên>": "<loại>", ...}} với đủ mọi nhãn.')
    schema = {'type': 'object', 'properties': {'map': {'type': 'object', 'additionalProperties': {'type': 'string', 'enum': allowed}}},
              'required': ['map']}
    return user, schema


def label_enums(client, raw, bm: BusinessMapping) -> dict[str, dict[str, str]]:
    """{'status': {fold(nhãn): loại}, 'payment': {...}} cho các nhãn từ điển không hiểu."""
    out: dict[str, dict[str, str]] = {'status': {}, 'payment': {}}
    for kind in ('status', 'payment'):
        cnt = _label_values(raw, bm, kind)
        if kind == 'status':
            unk = {v: n for v, n in cnt.items() if norm_status(v) in (None, 'unknown')}
        else:
            unk = {v: n for v, n in cnt.items() if norm_payment(v) not in PAYMENT_KNOWN}
        if not unk:
            continue
        labels = sorted(unk.items(), key=lambda x: -x[1])[:MAX_LABELS]
        desc = 'trạng thái đơn hàng' if kind == 'status' else 'hình thức thanh toán của đơn'
        user, schema = _enum_prompt(kind, labels, desc)
        allowed = set(STATUS_SET if kind == 'status' else PAYMENT_SET)
        want = {v for v, _ in labels}

        def validate(obj, allowed=allowed, want=want):
            mp = obj.get('map') if isinstance(obj, dict) else None
            if not isinstance(mp, dict):
                raise ValueError('thiếu khoá "map"')
            bad = {k: v for k, v in mp.items() if v not in allowed}
            if bad:
                raise ValueError(f'loại không hợp lệ: {list(bad.items())[:3]}; chỉ được dùng {sorted(allowed)}')
            miss = want - set(mp)
            if len(miss) > 0.2 * len(want):
                raise ValueError(f'thiếu {len(miss)} nhãn, ví dụ {list(miss)[:3]}')
            return {k: v for k, v in mp.items() if k in want}

        res = client.ask(_SYS, user, schema, validate)
        if not res:
            continue
        for v, cat in res.items():
            if cat in ('unknown', 'other') and kind == 'status':
                continue
            if cat == 'unknown':
                continue
            out[kind][fold(v)] = cat
    return out


# ----------------------------------------------------------------------------- 2. vai trò cột
def _profile_col(c: str, p, desc: Optional[str]) -> dict:
    top = list((p.top or {}).keys())[:8] or list(p.samples or [])[:8]
    kind = 'ngày' if p.date_rate >= 0.6 else 'số' if p.num_rate >= 0.9 else 'chữ'
    d = {'cột': c, 'kiểu': kind, 'trống_%': round(100 * p.null_rate), 'số_giá_trị_khác_nhau': int(p.n_distinct),
         'mẫu': [str(x)[:40] for x in top]}
    if desc:
        d['mô_tả_README'] = desc[:160]
    return d


def _role_candidates(tm, prof) -> tuple[list[str], list[str]]:
    allowed = ALLOWED.get('sales')
    by_type = ROLES_BY_TYPE.get(tm.ttype)
    roles = [r for r in CORE_ROLES if r not in tm.roles and (allowed is None or r in allowed)
             and (by_type is None or r in by_type)]
    # cột hằng (1 giá trị: mã cửa hàng, mã chi nhánh…) không mang thông tin vai trò -> không gửi
    cols = [c for c, p in prof.columns.items()
            if not c.startswith('__') and c not in tm.col_roles and p.null_rate < 0.95 and p.n_distinct >= 2]
    return roles, cols


def _assign(M):
    """Gán một-một cột <-> vai trò, tổng độ tin cậy lớn nhất. scipy (Hungarian) nếu có; không có thì tham lam
    (v3.3.2: thiếu scipy từng làm hỏng im lặng toàn bộ bước gán vai trò bằng LLM)."""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(-M)
    except ImportError:
        cells = sorted(((M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1]) if M[i, j] > 0), reverse=True)
        ri, ci, ui, uj = [], [], set(), set()
        for _, i, j in cells:
            if i not in ui and j not in uj:
                ri.append(i); ci.append(j); ui.add(i); uj.add(j)
        return ri, ci


def suggest_roles(client, profiles, bm: BusinessMapping, min_conf: float = 0.8,
                  raw_out: Optional[dict] = None) -> dict[str, dict[str, tuple[str, float]]]:
    """{bảng: {role: (cột, conf)}} — chỉ cho bảng bán hàng còn thiếu vai trò cốt lõi.
    raw_out (nếu truyền): {bảng: [(cột, role, conf), ...]} — toàn bộ đề xuất thô của LLM (để chẩn đoán / báo đỏ)."""
    out: dict[str, dict[str, tuple[str, float]]] = {}
    for t, tm in bm.tables.items():
        if tm.ttype not in SALES_TYPES or t not in profiles:
            continue
        prof = profiles[t]
        roles, cols = _role_candidates(tm, prof)
        if not roles or not cols:
            continue
        known = {r: c for r, c in tm.roles.items()}
        cprof = [_profile_col(c, prof.columns[c], tm.descs.get(c)) for c in cols[:40]]
        role_desc = {r: CORE_ROLES[r] for r in roles}
        user = (f'Bảng "{t}" ({prof.n_rows} dòng), loại đoán: {tm.ttype}. Các cột ĐÃ biết vai trò: {known}.\n'
                f'Vai trò còn thiếu (chỉ chọn trong danh sách này): {role_desc}\n'
                f'Các cột CHƯA biết vai trò (profile nén):\n{cprof}\n\n'
                'Với mỗi cột chưa biết, chọn một vai trò còn thiếu hoặc "skip" nếu không thuộc vai trò nào, kèm độ tin '
                'cậy 0–1. Mỗi vai trò dùng cho tối đa một cột. Trả về JSON '
                '{"assign": [{"column": "<tên cột>", "role": "<vai trò|skip>", "conf": <0..1>}, ...]}.')
        schema = {'type': 'object', 'properties': {'assign': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'column': {'type': 'string', 'enum': cols[:40]},
                                             'role': {'type': 'string', 'enum': roles + ['skip']},
                                             'conf': {'type': 'number'}},
            'required': ['column', 'role', 'conf']}}}, 'required': ['assign']}
        colset, roleset = set(cols[:40]), set(roles) | {'skip'}

        def validate(obj, colset=colset, roleset=roleset):
            a = obj.get('assign') if isinstance(obj, dict) else None
            if not isinstance(a, list):
                raise ValueError('thiếu mảng "assign"')
            res = []
            for x in a:
                if not isinstance(x, dict) or x.get('column') not in colset or x.get('role') not in roleset:
                    raise ValueError(f'phần tử không hợp lệ: {x!r}'[:200])
                res.append((x['column'], x['role'], float(x.get('conf', 0))))
            return res

        res = client.ask(_SYS, user, schema, validate)
        if not res:
            continue
        if raw_out is not None:
            raw_out[t] = [(c, r, round(conf, 3)) for c, r, conf in res if r != 'skip']
        pairs = [(c, r, conf) for c, r, conf in res if r != 'skip' and conf >= min_conf
                 and _value_ok(ROLES[r][0], prof.columns[c])]
        if not pairs:
            continue
        cs = sorted({c for c, _, _ in pairs})
        rs = sorted({r for _, r, _ in pairs})
        M = np.zeros((len(cs), len(rs)))
        for c, r, conf in pairs:
            M[cs.index(c), rs.index(r)] = max(M[cs.index(c), rs.index(r)], conf)
        ri, ci = _assign(M)                                # Hungarian: gán một-một, tổng độ tin cậy lớn nhất
        got = {rs[j]: (cs[i], float(M[i, j])) for i, j in zip(ri, ci) if M[i, j] >= min_conf}
        if got:
            out[t] = got
    return out


def apply_roles(bm: BusinessMapping, sugg: dict) -> BusinessMapping:
    bm2 = copy.deepcopy(bm)
    for t, rs in sugg.items():
        tm = bm2.tables[t]
        for r, (c, conf) in rs.items():
            if r in tm.roles or c in tm.col_roles:
                continue
            tm.roles[r] = c
            tm.col_roles[c] = r
            tm.evidence[r] = f'llm:{conf:.2f}'
        if tm.ttype in ('sales', 'other'):
            has_prod = bool({'product_code', 'product_name'} & set(tm.roles))
            has_line = bool({'qty', 'amount'} & set(tm.roles))
            if has_prod and has_line:
                tm.ttype = 'sales_lines' if 'order_id' in tm.roles else 'sales_flat'
    return bm2


def without_roles(sugg: dict, drop: set[str]) -> dict:
    out = {t: {r: v for r, v in rs.items() if r not in drop} for t, rs in sugg.items()}
    return {t: rs for t, rs in out.items() if rs}


def status_unresolved(raw_out: dict, bm: BusinessMapping, min_conf: float = UNRESOLVED_CONF) -> list[str]:
    """Bảng (loại có thể mang trạng thái đơn) mà LLM nói có cột trạng thái nhưng mapping cuối vẫn không có 'status'.

    Khi đó mặc định 'mọi đơn hoàn tất' không còn an toàn (đơn huỷ/trả bị tính vào doanh thu, số đơn) -> báo đỏ.
    Căn cứ là đề xuất của LLM trên giá trị cột, không phải tên cột."""
    if any('status' in tm.roles for tm in bm.tables.values()):
        return []                      # shop đã có cột trạng thái ở đâu đó: canonical đã dùng nó
    bad = []
    for t, res in (raw_out or {}).items():
        tm = bm.tables.get(t)
        if tm is None or tm.ttype not in STATUS_HOLDERS:
            continue
        if any(r == 'status' and conf >= min_conf for _, r, conf in res):
            bad.append(t)
    return bad


def inv_score(levels: dict[str, str]) -> tuple:
    """Thấp hơn = tốt hơn: (số đỏ, số vàng, −số invariant kiểm được và xanh)."""
    v = list(levels.values())
    return v.count('red'), v.count('yellow'), -v.count('green')
