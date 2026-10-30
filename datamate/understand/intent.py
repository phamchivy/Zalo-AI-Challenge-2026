"""Bộ phân tích luật (rule parser) câu hỏi -> Plan cho tầng A.

Chỉ nhận những dạng câu chắc chắn hiểu đúng (doanh thu, số đơn, huỷ, AOV, SL bán, so sánh kỳ, tỉ trọng, xếp hạng,
ngày/tháng cao nhất, ngưỡng, giá bán...). Không chắc -> trả None để tầng B (LLM) xử lý. Câu thuộc nghiệp vụ tầng A chưa
hỗ trợ (công nợ, tồn kho, lô/hạn dùng, nhập hàng, hiệu ứng tăng giá, hàng ế, giao hàng, lãi ròng...) -> None.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

from datamate.core.textutil import add_months, month_end, to_num
from datamate.core.types import AnswerType, Plan, TimeRange
from datamate.understand import entity_link as EL
from datamate.understand.time_resolver import TimeParse

TIER_B_ONLY = [
    ('debt', r'\b(con no|cong no|no nhieu|tra no|thu no|ghi no|mua chiu|no them|tuoi no|khach no|dang no|so no|tien no|thu hoi|chua tra( het)?|tra het no|tra bot|con thieu|no cua hang)\b'),
    ('inventory', r'\b(ton kho|con ton|trong kho|het hang|con (lai )?(bao nhieu|may)|ton cuoi|hang ton|chan .* le|con lai)\b'),
    ('expiry', r'\b(het han|han dung|han su dung|hsd|lo\b.*het han|lo .* gan het)\b'),
    ('price_effect', r'\b(tang gia|doi gia|len gia|giam gia xuong|ha gia)\b'),
    ('purchase', r'\b(nhap|phieu nhap|gia nhap|nha cung cap|tien nhap)\b'),
    ('dead_stock', r'\b(bi e|hang e|mat hang e|san pham e|mon e|e hang|ban cham|khong ban duoc|chua ban duoc|e nhat|ton nhieu)\b'),
    ('delivery', r'\b(dot giao|dot hang|chuyen giao|cho giao|lan giao|xe giao|giao bao nhieu dot)\b'),
    ('net_profit', r'\b(lai rong|chi phi|tru het chi phi|loi nhuan rong)\b'),
    ('special', r'\b(hoa hong|tra gop|bao hanh|imei|thu may cu|may cu|sua xe|tho|voucher|diem tich luy|dung diem|khuyen mai)\b'),
    ('lots', r'\b(lo nao|so lo|theo lo)\b'),
]

MONEY = r'(\d[\d.,]*)\s*(k|nghin|ngan|tr|trieu|d|dong|vnd)?\b'

_VOCAB_FILE = Path(__file__).with_name('vocab_generic.txt')
GENERIC = set(_VOCAB_FILE.read_text(encoding='utf-8').split()) if _VOCAB_FILE.exists() else set()
GENERIC |= {'doanh', 'thu', 'tien', 'bao', 'nhieu', 'hom', 'nay', 'qua', 'thang', 'nam', 'quy', 'tuan', 'ngay', 'cua', 'hang',
            'ban', 'duoc', 'don', 'khach', 'mat', 'san', 'pham', 'nhat', 'nhieu', 'la', 'cho', 'toi', 'biet', 'hoi', 'xem', 'giup',
            'gium', 'check', 'coi', 'trong', 'tong', 'cong', 'so', 'may', 'cac', 'nhung', 'nao', 'gi', 'ai', 'co', 'khong', 'va',
            'voi', 'tu', 'den', 'het', 'luc', 'tai', 'theo', 'tinh', 'size', 'top', 'ly', 'mon', 'nhom', 'ca', 'sang', 'chieu',
            'gia', 'tri', 'trung', 'binh', 'quan', 'moi', 'mot', 've', 'ra', 'lan', 'luot', 'lien', 'tiep', 'ket', 'thuc', 'o',
            'truoc', 'sau', 'cung', 'ky', 'tet', 'am', 'lich', 'dip', 'mua', 'huy', 'ti', 'le', 'phan', 'tram', 'tang', 'giam',
            'hon', 'kem', 'thay', 'doi', 'chenh', 'lech', 'cao', 'thap', 'it', 'dat', 'tro', 'len', 'tren', 'duoi', 'it', 'nhat',
            'chiem', 'dong', 'gop', 'trong', 'bang', 'dien', 'thoai', 'chuyen', 'khoan', 'mat', 'vi', 'cod', 'quay', 'online',
            'nhan', 'vien', 'ma', 'ke', 'don', 'hoat', 'chat', 'chay', 'hot', 'dong', 'nguoi', 'dau', 'cuoi', 'gan', 'day',
            'vay', 'nhi', 'a', 'ha', 'z', 'v', 'e', 'oi', 'nha', 'do', 'the', 'ne', 'total', 'revenue', 'kg', 'lon', 'chai', 'goi',
            'hop', 'vien', 'cai', 'thung', 'loc', 'tui', 'cay', 'bao', 'met', 'cuon', 'dia', 'phan', 'to', 'chen', 'suat', 'chiec',
            'phai', 'loi', 'nhuan', 'lai', 'gop', 'mang', 'lai', 'xuat', 'hien', 'khac', 'nhau', 'thuoc', 'si', 'than', 'thiet',
            'vip', 'thuong', 'quen', 'chi', 'tieu', 'kiem', 'bill', 'order', 'hoa', 'dot', 'giao', 'tuc', 'luc', 'nay', 'roi'}


def unexplained(qn: str, linked_names: list[str], tp: TimeParse) -> list[str]:
    """Từ trong câu không thuộc từ vựng chung, không thuộc thời gian/thực thể đã khớp -> dấu hiệu thiếu thực thể."""
    from datamate.understand.normalize import telex_fix
    cover = set()
    for n in linked_names:
        for t in EL.fold(n).replace('(', ' ').replace(')', ' ').split():
            cover.add(t)
            cover.add(telex_fix(t))
    for r in tp.ranges:
        cover.update((r.expr or '').split())
    out = []
    for t in qn.split():
        if re.fullmatch(r'[a-z]+', t) and len(t) >= 2 and t not in GENERIC and t not in cover:
            out.append(t)
    return out


@dataclass
class ParseResult:
    intent: str
    plan: Optional[Plan] = None
    confidence: float = 0.0
    reason: str = ''
    tier_b_hint: str = ''
    entities: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- period helpers
def prev_period(tr: TimeRange) -> TimeRange:
    g = tr.granularity
    if g == 'month' and tr.start.day == 1 and tr.end == month_end(tr.start.year, tr.start.month):
        p = add_months(tr.start, -1)
        return TimeRange(start=p, end=month_end(p.year, p.month), label='compare', granularity='month')
    if g == 'quarter':
        p = add_months(tr.start, -3)
        return TimeRange(start=p, end=month_end(add_months(p, 2).year, add_months(p, 2).month), label='compare', granularity='quarter')
    if g == 'year' and tr.start.month == 1 and tr.start.day == 1:
        n = (tr.end - tr.start).days
        s = date(tr.start.year - 1, 1, 1)
        return TimeRange(start=s, end=min(date(s.year, 12, 31), s + timedelta(days=n)), label='compare', granularity='year')
    n = (tr.end - tr.start).days + 1
    return TimeRange(start=tr.start - timedelta(days=n), end=tr.start - timedelta(days=1), label='compare', granularity=g)


def mtd_compare(tr: TimeRange) -> TimeRange:
    """Tháng này đến hôm nay vs cùng số ngày tháng trước (rulebook §2)."""
    p = add_months(tr.start, -1)
    end = min(tr.end.day, month_end(p.year, p.month).day)
    return TimeRange(start=p, end=date(p.year, p.month, end), label='compare', granularity='month')


def yoy(tr: TimeRange) -> TimeRange:
    def sh(d: date) -> date:
        try:
            return d.replace(year=d.year - 1)
        except ValueError:
            return d.replace(year=d.year - 1, day=28)
    return TimeRange(start=sh(tr.start), end=sh(tr.end), label='compare', granularity=tr.granularity)


# ----------------------------------------------------------------------------- parser
def _k(qn: str) -> Optional[int]:
    m = re.search(r'\btop\s*(\d+)\b', qn) or re.search(r'\b(\d+)\s+(?:mat hang|san pham|mon|hang|khach|khach hang|nguoi|nhan vien|nhom|loai|thuoc)\b', qn)
    if m:
        k = int(m[1])
        return k if 1 <= k <= 50 else None
    return None


def _money_threshold(qn: str) -> Optional[tuple[float, bool]]:
    m = re.search(r'(?:tu|tren|hon|dat|it nhat|toi thieu|qua|>=|>|lon hon|tro len tu)\s+' + MONEY, qn)
    if not m:
        return None
    v = to_num(m[1] + (m[2] or ''))
    if v is None:
        return None
    if m[2] in ('k', 'nghin', 'ngan') and v < 1000:
        v *= 1000
    strict = bool(re.search(r'\b(tren|hon|qua|lon hon|>)\s+' + re.escape(m[1]), qn)) and 'tro len' not in qn
    return v, strict


def _int_threshold(qn: str, unit: str) -> Optional[tuple[int, bool]]:
    m = re.search(r'(?:tu|tren|hon|it nhat|toi thieu|qua|>=|>|lon hon|>=)\s*(\d+)\s*' + unit, qn) or \
        re.search(r'(\d+)\s*' + unit + r'\s*(?:khac nhau\s*)?tro len', qn)
    if not m:
        return None
    strict = bool(re.search(r'\b(tren|hon|qua|lon hon)\s*' + m[1], qn)) and 'tro len' not in qn
    return int(m[1]), strict


def parse_question(qn: str, qf: str, as_of: date, tp: TimeParse, ix: EL.EntityIndex, caps: set[str],
                   data_min: Optional[date], data_max: Optional[date], cal=None) -> ParseResult:
    # ---- nghiệp vụ chuyên biệt (công nợ, nhập hàng) hoặc để tầng B (vẫn gửi thực thể đã khớp làm gợi ý cho LLM)
    hint_ents = {}
    _p = EL.link_products(ix, qf)
    if _p:
        hint_ents['products'] = [{'product_key': k, 'name': d} for k, d, _ in _p]
    _c = EL.link_customer(ix, qf) if re.search(r'\bkhach\b', qn) else None
    if _c:
        hint_ents['customer'] = {'customer_key': _c[0], 'name': _c[1]}
    for name, rx in TIER_B_ONLY:
        if re.search(rx, qn):
            if name == 'debt':
                r = parse_debt(qn, qf, as_of, tp, ix, caps)
                if r is not None:
                    return r
            if name == 'purchase':
                r = parse_purchase(qn, qf, as_of, tp, ix, caps, data_min)
                if r is not None:
                    return r
            return ParseResult(intent=name, tier_b_hint=name, reason='tier B intent', entities=hint_ents)

    P = list(tp.ranges)
    filters: dict[str, Any] = {}
    ents: dict[str, Any] = {}
    prods = EL.link_products(ix, qf)
    if prods:
        ents['products'] = [(k, d) for k, d, _ in prods]
    cat = EL.link_category(ix, qf) if re.search(r'\b(nhom|danh muc|loai|nganh)\b', qn) else None
    cg = EL.link_cgroup(ix, qf) if re.search(r'\bkhach\b|\bnhom\b', qn) else None
    if cat and cg and EL.fold(cat) == EL.fold(cg):
        cat = None if re.search(r'khach', qn) else cat
    if cg:
        ents['customer_group'] = cg
    if cat and not cg:
        ents['category'] = cat
    chans = EL.link_channels(ix, qn) if re.search(r'\b(qua|tai|kenh|nguon|ban tren|ban o|online|offline|san)\b', qn) else []
    pays = EL.link_payments(ix, qn) if re.search(r'\b(tra bang|thanh toan|hinh thuc|bang|tra qua|cod|tien mat|chuyen khoan|vi dien tu|quet the|ghi no)\b', qn) else []
    staff = EL.link_staff(ix, qf) if re.search(r'\bnhan vien\b|\bnv\d', qn) else None
    shift = EL.shift_filter(qn)
    size = EL.size_filter(qn)
    unit = EL.asked_unit(qn)
    rx = None
    if re.search(r'\bke don\b', qn):
        rx = not bool(re.search(r'\b(khong|ko) ke don\b', qn))
    ingr = EL.link_ingredient(ix, qf) if 'hoat chat' in qn else None
    cust = None
    if re.search(r'\bkhach\b', qn) and not re.search(r'\b(khach nao|nhung khach|top|khach hang nao|ai)\b', qn):
        cust = EL.link_customer(ix, qf)

    if shift:
        filters['shift'] = shift
    if size:
        filters['size'] = size
    if rx is not None:
        filters['rx'] = rx
    if ingr:
        filters['ingredient'] = ingr
    if staff:
        filters['staff'] = staff
    if cust:
        filters['customer'] = cust[0]
        ents['customer'] = cust[1]
    if cg:
        filters['customer_group'] = cg
    elif cat:
        filters['category'] = cat
    if len(chans) == 1:
        filters['channel'] = chans[0]
    if len(pays) == 1:
        filters['payment'] = pays[0]
    if prods and len(prods) == 1:
        filters['product'] = prods[0][0]
    if 'hoat chat' in qn and not ingr:
        return ParseResult(intent='ingredient', tier_b_hint='ingredient', reason='ingredient not found')
    if re.search(r'\bnhan vien\b', qn) and not staff and not re.search(r'nhan vien (nao|ban gioi|gioi nhat|xuat sac)', qn):
        return ParseResult(intent='staff', tier_b_hint='staff', reason='staff not linked')
    if re.search(r'\bnhom\b', qn) and not (cat or cg):
        return ParseResult(intent='group', tier_b_hint='group', reason='group not linked')

    whole = TimeRange(start=data_min or date(2000, 1, 1), end=as_of, label='all', granularity='custom')
    p0 = P[0] if P else None

    def mk(metric, op='value', periods=None, **kw) -> Plan:
        return Plan(metric=metric, op=op, periods=periods if periods is not None else ([p0] if p0 else [whole]),
                    filters={**filters, **kw.pop('extra_filters', {})}, source='rule', **kw)

    linked = [d for _, d, _ in prods] + [x for x in (cat, cg, ingr) if x] + ([cust[1]] if cust else []) + ([staff] if staff else [])
    for k_, v_ in ix.staff.items():
        if staff and v_ == staff:
            linked.append(k_)
    unk = unexplained(qn, linked, tp)
    if unk:
        ents['unexplained'] = unk
    conf_factor = 1.0 if not unk else (0.6 if len(unk) == 1 else 0.45)

    def res(intent, plan, conf=0.9):
        return ParseResult(intent=intent, plan=plan, confidence=conf * conf_factor, entities=ents)

    # ---- chọn metric cơ bản
    def base_metric() -> str:
        if re.search(r'\b(trung binh|binh quan)\b.*\b(don|bill|hoa don)\b|\bgia tri trung binh\b', qn):
            return 'aov'
        if re.search(r'\b(lai gop|loi gop|lai|loi)\b', qn) and not re.search(r'\bdoanh thu\b', qn):
            return 'gross_profit'
        if re.search(r'\b(so|may|bao nhieu|dem|tong so) (don|hoa don|luot mua)\b|\bmay don\b|\bso don\b', qn):
            return 'orders'
        if re.search(r'\b(bao nhieu|may|so) khach\b', qn) and not re.search(r'\bkhach (nao|hang nao)\b', qn):
            return 'customers'
        if unit or re.search(r'\b(so luong|ban (ra |duoc )?(bao nhieu|may)(?! tien))\b', qn) and not re.search(r'\btien\b|doanh thu', qn):
            return 'qty'
        return 'revenue'

    # ---- 1. xu hướng
    if re.search(r'\b(tang lien tuc|luon cao hon|lien tuc tang|thang sau .* cao hon)\b', qn):
        if len(P) >= 2:
            return res('trend_bool', mk(base_metric(), 'trend_up', periods=P, answer_type=AnswerType.BOOLEAN))
        return ParseResult(intent='trend_bool', reason='need >=2 periods')

    # ---- 2. so sánh kỳ (%), chênh lệch
    is_cmp = re.search(r'\b(so voi|so sanh|hon hay kem|tang hay giam|thay doi|tang truong|chenh lech|nhieu hay it hon|cao hay thap hon|giam bao nhieu|tang bao nhieu|ca chieu .* ca sang)\b', qn)
    if is_cmp:
        met = base_metric()
        pct = '%' in qn or re.search(r'\b(ti le|phan tram|tang truong)\b', qn)
        if shift is None and re.search(r'ca chieu', qn) and re.search(r'ca sang', qn):
            filters.pop('shift', None)
            plan = mk(met, 'pct_change_filters', split_values=[{'shift': 'afternoon'}, {'shift': 'morning'}])
            return res('compare_shift', plan)
        if 'ca chieu' in qn and 'ca sang' in qn:
            filters.pop('shift', None)
            plan = mk(met, 'pct_change_filters', split_values=[{'shift': 'afternoon'}, {'shift': 'morning'}])
            return res('compare_shift', plan)
        if not P:
            return ParseResult(intent='compare', reason='no period')
        rel = [p for p in P if re.fullmatch(r'(thang|tuan|quy|nam) truoc', p.expr or '')]
        main = [p for p in P if p not in rel]
        if rel and main and 'mtd_compare' not in tp.flags:
            # "tháng 4/2026 so với tháng trước" = tháng liền trước của kỳ được hỏi, không phải của hôm nay
            P = main
            p0 = main[0]
            tp.flags.add('prev_period')
        if 'mtd_compare' in tp.flags:
            pp = [p0, mtd_compare(p0)]
        elif 'before_tet' in tp.flags and tp.n_before:
            pp = [p0, TimeRange(start=p0.start - timedelta(days=tp.n_before), end=p0.start - timedelta(days=1), label='compare')]
        elif 'yoy' in tp.flags:
            pp = [p0, yoy(p0)]
        elif 'prev_period' in tp.flags or (len(P) == 1 and re.search(r'\b(ky truoc|thang truoc|tuan truoc|quy truoc|nam truoc|hom truoc|lien truoc)\b', qn)):
            pp = [p0, prev_period(p0)]
        elif len(P) >= 2:
            # kỳ đứng sau 'so voi' là kỳ gốc
            mcmp = re.search(r'\bso (voi|sanh voi)\b', qn)
            if mcmp and P[0].expr and qn.find(P[0].expr) > mcmp.start():
                pp = [P[1], P[0]]
            else:
                pp = [P[0], P[1]]
        else:
            return ParseResult(intent='compare', reason='no compare period')
        if pct:
            return res('compare_period', mk(met, 'pct_change', periods=pp))
        return res('diff', mk(met, 'diff', periods=pp), 0.75)

    # ---- 3. tỉ lệ huỷ, số đơn huỷ
    if re.search(r'\bhuy\b', qn):
        if '%' in qn or re.search(r'\bti le\b', qn):
            return res('cancel_rate', mk('cancel_rate', answer_type=AnswerType.PERCENT))
        if re.search(r'\b(don|hoa don)\b', qn):
            return res('cancel_count', mk('cancelled_orders'))
        return ParseResult(intent='cancel', reason='ambiguous cancel')

    # ---- 4. ngưỡng (count / % thỏa điều kiện)
    if re.search(r'\b(bao nhieu|may|so|dem)\s+(don|hoa don)\b', qn) and re.search(r'\b(tien hang|doanh thu|gia tri|tri gia|tong tien)\b', qn):
        thr = _money_threshold(qn)
        if thr:
            return res('order_count_thr', mk('revenue', 'count_ge', group_by='order', threshold=thr[0], threshold_metric='revenue',
                                             note='gt' if thr[1] else ''))
    if re.search(r'\bkhach\b', qn) and re.search(r'\bdon\b', qn) and re.search(r'tro len|it nhat|tu \d|tren \d|hon \d', qn):
        thr = _int_threshold(qn, r'don')
        if thr:
            op = 'pct_ge' if ('%' in qn or 'ti le' in qn) else 'count_ge'
            return res('customer_thr', mk('orders', op, group_by='customer', threshold=thr[0], threshold_metric='orders', note='gt' if thr[1] else ''))
    if re.search(r'\b(mat hang|san pham|mon)\b', qn) and re.search(r'\b(xuat hien|co mat|duoc mua|nam trong|co trong)\b', qn):
        thr = _int_threshold(qn, r'don')
        if thr:
            return res('product_count', mk('orders', 'count_ge', group_by='product', threshold=thr[0], threshold_metric='orders', note='gt' if thr[1] else ''))

    # ---- 5. tỉ trọng
    if re.search(r'\b(chiem|ti trong|phan cua|dong gop|tren tong|so voi tong|bang bao nhieu %|chiem bao nhieu|ti le doanh thu|% doanh thu|% tong doanh thu|phan tram doanh thu)\b', qn) or \
            (('%' in qn or 'ti le' in qn) and (filters.keys() & {'category', 'customer_group', 'rx', 'channel', 'payment', 'product', 'ingredient'})):
        if filters.keys() & {'category', 'customer_group', 'rx', 'channel', 'payment', 'product', 'ingredient', 'size', 'shift', 'staff', 'customer'}:
            return res('share', mk('revenue' if base_metric() not in ('orders', 'qty') else base_metric(), 'share', answer_type=AnswerType.PERCENT))
        return ParseResult(intent='share', reason='share without filter')

    # ---- 6. ngày / tháng cao nhất
    SUP = r'\b(cao nhat|nhieu nhat|dat nhat|lon nhat|dong nhat|nhieu tien nhat|thap nhat|it nhat|it tien nhat|ky luc)\b'
    if re.search(r'\b(ngay nao|hom nao)\b', qn) and re.search(SUP, qn):
        met = base_metric()
        return res('best_day', mk(met if met != 'customers' else 'revenue', 'argmax', group_by='day', answer_type=AnswerType.DATE,
                                  note='min' if re.search(r'thap nhat|it nhat', qn) else ''))
    if re.search(r'\bthang nao\b', qn) and re.search(SUP, qn):
        met = base_metric()
        return res('best_month', mk(met, 'argmax', periods=P if len(P) > 1 else ([p0] if p0 else [whole]), group_by='month',
                                    answer_type=AnswerType.DATE, note='min' if re.search(r'thap nhat|it nhat', qn) else ''))

    # ---- 7. xếp hạng
    k = _k(qn)
    rank_words = r'\b(ban chay|chay nhat|ban nhieu nhat|ban duoc nhieu nhat|hot nhat|nhieu nhat|nhieu tien nhat|cao nhat|lon nhat|best seller|ban dat nhat|loi nhat|lai nhat|gioi nhat|xuat sac nhat|top|chi nhieu|chi tieu nhieu)\b'
    if re.search(rank_words, qn):
        many = k is not None and k > 1
        if re.search(r'\b(khach|ai mua|nguoi mua)\b', qn):
            if 'customers' not in caps:
                return ParseResult(intent='top_customer', reason='no customer identity', tier_b_hint='top_customer')
            met = 'orders' if re.search(r'\b(nhieu don|so don|nhieu lan|nhieu hoa don)\b', qn) else 'revenue'
            filters.pop('customer', None)
            return res('top_customer', mk(met, 'top', group_by='customer', k=k or 1, answer_type=AnswerType.LIST if many or re.search(r'\b(nhung|cac|liet ke)\b', qn) else None))
        if re.search(r'\b(nhan vien|nguoi ban|thu ngan|ai ban)\b', qn):
            filters.pop('staff', None)
            return res('top_staff', mk('revenue', 'top', group_by='staff', k=k or 1), 0.7)
        if re.search(r'\bnhom (hang )?nao\b|\bdanh muc nao\b', qn):
            return res('top_category', mk('revenue' if 'doanh thu' in qn or 'tien' in qn else 'qty', 'top', group_by='category', k=k or 1), 0.7)
        if re.search(r'\b(mat hang|san pham|hang|mon|thuoc|sp|loai|mau|item)\b', qn) or re.search(r'ban chay|hot nhat|best seller', qn):
            if re.search(r'\b(lai|loi|loi nhuan|lai gop|loi gop)\b', qn):
                met = 'gross_profit'
            elif re.search(r'\b(doanh thu|nhieu tien|tien)\b', qn) and not re.search(r'ban chay', qn):
                met = 'revenue'
            else:
                met = 'qty'
            filters.pop('product', None)
            lst = many or re.search(r'\b(nhung|cac|liet ke)\b', qn)
            return res('top_product', mk(met, 'top', group_by='product', k=k or 1, answer_type=AnswerType.LIST if lst else None))

    # ---- 8. giá bán
    if re.search(r'\b(gia ban|don gia|gia bao nhieu|voi gia|ban .* gia|gia (mot|1))\b', qn) and prods and p0 and p0.granularity == 'day':
        return res('unit_price', mk('revenue', 'unit_price'))

    # ---- 9. nhiều thực thể -> list theo thứ tự câu hỏi
    met = base_metric()
    if len(chans) >= 2:
        return res('values_channel', mk(met, 'values', split_by='channel', split_values=chans))
    if len(pays) >= 2:
        return res('values_payment', mk(met, 'values', split_by='payment', split_values=pays))
    if prods and len(prods) >= 2 and re.search(r'\b(va|lan luot|moi)\b', qn):
        filters.pop('product', None)
        return res('values_product', mk(met, 'values', split_by='product', split_values=[p[0] for p in prods]), 0.75)
    if len(P) >= 2 and ('series' in tp.flags or re.search(r'\b(moi thang|tung thang|lan luot|moi ky|tung quy|moi quy)\b', qn)):
        return res('series', mk(met, 'series', periods=P))

    # ---- 10. giá trị đơn lẻ
    if met == 'qty':
        if unit and not prods and not size:
            if unit not in ix.base_units:
                return ParseResult(intent='qty_unit', reason=f'no product with base unit {unit}', tier_b_hint='qty_unit')
            filters['base_unit'] = unit
        if prods and unit:
            ents['asked_unit'] = unit
        return res('qty_sold', mk('qty'), 0.85 if (prods or unit or size) else 0.6)
    if met in ('revenue', 'orders', 'aov', 'customers', 'gross_profit'):
        if met == 'revenue' and not re.search(r'\b(doanh thu|tien|thu ve|ban duoc|doanh so|tien hang|tien ban)\b', qn):
            return ParseResult(intent='unknown', reason='no metric keyword')
        conf = 0.9 if P else 0.6
        return res(met if met != 'orders' else 'order_count', mk(met), conf)
    return ParseResult(intent='unknown', reason='no rule')


# ----------------------------------------------------------------------------- công nợ
def _snapshot(tp: TimeParse, as_of: date) -> TimeRange:
    snaps = [r for r in tp.ranges if r.granularity == 'snapshot']
    if snaps:
        return snaps[0]
    days = [r for r in tp.ranges if r.granularity == 'day']
    d = days[0].end if days else as_of
    return TimeRange(start=date(2000, 1, 1), end=d, label='snapshot', granularity='snapshot')


def parse_debt(qn: str, qf: str, as_of: date, tp: TimeParse, ix: EL.EntityIndex, caps: set[str]) -> Optional[ParseResult]:
    if 'debt' not in caps:
        return None
    k = _k(qn)
    flows = [r for r in tp.ranges if r.granularity != 'snapshot']
    # nợ thêm & trả được của một khách trong kỳ
    if re.search(r'\b(no them|mua chiu them|ghi no them)\b', qn) and re.search(r'\btra\b', qn):
        cust = EL.link_customer(ix, qf)
        if not cust or not flows:
            return None
        return ParseResult(intent='debt_flow', confidence=0.85, entities={'customer': cust[1]},
                           plan=Plan(metric='debt', op='flow', periods=[flows[0]], filters={'customer': cust[0]}, source='rule',
                                     answer_type=AnswerType.LIST))
    # thu nợ trong kỳ
    if re.search(r'\b(thu (duoc|hoi|ve)?.*(tra no|no)|tra no|tra bot no|thu hoi .* no|thu no)\b', qn) and flows and \
            not re.search(r'\b(con no|dang no)\b', qn):
        return ParseResult(intent='debt_collected', confidence=0.85,
                           plan=Plan(metric='debt', op='collected', periods=[flows[0]], source='rule'))
    snap = _snapshot(tp, as_of)
    if re.search(r'\btuoi no\b|0\s*[-–]\s*7 ngay', qn):
        return ParseResult(intent='debt_aging', confidence=0.85,
                           plan=Plan(metric='debt', op='aging', periods=[snap], source='rule', answer_type=AnswerType.LIST))
    m = re.search(r'(?:da dat|dat|da mua|mua|no) (?:da )?(qua|hon|tren|tu) (\d+) ngay', qn)
    if m:
        return ParseResult(intent='debt_over_days', confidence=0.8,
                           plan=Plan(metric='debt', op='over_days', periods=[snap], threshold=int(m[2]),
                                     note='ge' if m[1] == 'tu' else '', source='rule'))
    if re.search(r'\b(nhieu nhat|lon nhat|top)\b', qn) and re.search(r'\bkhach\b', qn):
        return ParseResult(intent='debt_top', confidence=0.85,
                           plan=Plan(metric='debt', op='top', periods=[snap], k=k or 3, source='rule', answer_type=AnswerType.TABLE))
    if re.search(r'\b(bao nhieu|may|so) khach\b', qn):
        return ParseResult(intent='debt_customers', confidence=0.85,
                           plan=Plan(metric='debt', op='customers', periods=[snap], source='rule'))
    if re.search(r'\b(con no|cong no|chua thu|chua tra|no)\b', qn):
        return ParseResult(intent='debt_total', confidence=0.8,
                           plan=Plan(metric='debt', op='total', periods=[snap], source='rule'))
    return None


# ----------------------------------------------------------------------------- nhập hàng
def parse_purchase(qn: str, qf: str, as_of: date, tp: TimeParse, ix: EL.EntityIndex, caps: set[str],
                   data_min: Optional[date]) -> Optional[ParseResult]:
    if 'purchases' not in caps:
        return None
    flows = [r for r in tp.ranges if r.granularity != 'snapshot']
    if not flows:
        return None
    p0 = flows[0]
    if re.search(r'\b(tru|tru di|chenh|so voi so ban)\b', qn):
        return None                              # nhập trừ bán: để tầng B
    if re.search(r'\bnha cung cap\b', qn) and re.search(r'\b(nhieu nhat|cao nhat|lon nhat|nao)\b', qn):
        return ParseResult(intent='purchase_supplier', confidence=0.85,
                           plan=Plan(metric='purchase_supplier', op='top', periods=[p0], source='rule'))
    if re.search(r'\b(binh quan|trung binh)\b', qn) and re.search(r'\b(gia|tien)\b', qn):
        prods = EL.link_products(ix, qf)
        if len(prods) != 1:
            return None
        return ParseResult(intent='purchase_price', confidence=0.8, entities={'products': [(prods[0][0], prods[0][1])]},
                           plan=Plan(metric='purchase_avg_price', op='value', periods=[p0], filters={'product': prods[0][0]}, source='rule'))
    if re.search(r'\b(tong|bao nhieu tien|het bao nhieu|gia tri|tien nhap|tien hang nhap)\b', qn) and not EL.link_products(ix, qf):
        return ParseResult(intent='purchase_total', confidence=0.85,
                           plan=Plan(metric='purchase_total', op='value', periods=[p0], source='rule'))
    return None
