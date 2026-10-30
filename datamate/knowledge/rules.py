"""Rulebook -> RuleSet (rules.json): luật nghiệp vụ dạng DỮ LIỆU, tách khỏi code tính toán.

Ba thành phần:
  knobs        : nút cấu hình (key -> value) mà executor đọc. Nút không xác nhận được từ rulebook = None
                 -> intent cần nút đó BỎ TRỐNG (không lấy mặc định của bộ train).
  rankings     : định nghĩa xếp hạng ("Khách trung thành nhất = nhiều ngày có mua nhất") -> (đối tượng, số đo, ràng buộc).
  unknown      : câu luật trong rulebook không khớp nút / định nghĩa nào ("luật lạ"). Câu hỏi chạm vào luật lạ -> bỏ trống.

Nguồn điền (cùng một schema, ưu tiên giảm dần):
  1. file override `rules.json` (LLM hoặc người điền) — `KnowledgeBase.load` đọc nếu có,
  2. bộ đọc regex dưới đây (không cần LLM).
Mỗi detector chỉ nhận một câu luật khi khớp TOÀN BỘ các mẫu của nó -> luật đổi nghĩa thì nút về None, không đoán.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from datamate.core.textutil import fold

UNKNOWN = None


# ----------------------------------------------------------------------------- statements
@dataclass
class Statement:
    section: str            # '1. don hang va doanh thu' / '6. theo nganh > ca phe'
    name: str               # cột trái của bảng luật (đã fold)
    text: str               # cột phải (đã fold)
    raw: str = ''           # dòng gốc (để báo cáo)


def _clean(s: str) -> str:
    s = s.replace('**', '').replace('`', '').replace('−', '-').replace('–', '-').replace('—', '-')
    return fold(s)


def parse_statements(text: str) -> list[Statement]:
    out: list[Statement] = []
    sec, sub = '', ''
    for line in text.splitlines():
        s = line.strip()
        if s.startswith('### '):
            sub = _clean(s[4:])
            continue
        if s.startswith('## '):
            sec, sub = _clean(s[3:]), ''
            continue
        where = f'{sec} > {sub}' if sub else sec
        if s.startswith('|'):
            cells = [c.strip() for c in s.strip('|').split('|')]
            if len(cells) < 2 or re.fullmatch(r'[-: ]*', cells[0]) or not cells[0]:
                continue
            name, body = _clean(cells[0]), _clean(' | '.join(cells[1:]))
            if (name, body) in (('luat', 'mac dinh'), ('cach noi', 'nghia'), ('cach ghi', 'hieu la'), ('tu', 'nghia'),
                                ('type', 'format | dung | sai')):
                continue
            out.append(Statement(where, name, body, s))
        elif s.startswith('- ') and sec:
            out.append(Statement(where, '', _clean(s[2:]), s))
    return out


# ----------------------------------------------------------------------------- knob detectors
@dataclass
class Knob:
    key: str
    value: Any
    name: str = ''                       # regex trên tên luật (cột trái); '' = không xét
    text: tuple = ()                     # mọi regex phải khớp trong cột phải
    section: str = ''                    # regex trên tên mục
    doc: str = ''


def K(key, value, name='', text=(), section='', doc=''):
    return Knob(key, value, name, tuple([text] if isinstance(text, str) else text), section, doc)


# Mỗi knob mô tả một quy ước nghiệp vụ. Thêm luật mới = thêm 1 dòng ở đây + đọc knob ở executor.
KNOBS: list[Knob] = [
    # §1 đơn hàng và doanh thu
    K('revenue.valid_status', ['completed', 'shipping'], r'^don tinh vao doanh thu',
      (r'khong huy', r'khong tra hang', r'dang giao van tinh')),
    K('revenue.partial_return_whole_order', True, r'^don tinh vao doanh thu', (r'ke ca chi tra mot phan', r'bo ca don')),
    K('revenue.money_words_mean_revenue', True, r'^ban duoc bao nhieu tien', r'la doanh thu'),
    K('revenue.period_by', 'order_date', r'^don thuoc ky nao', r'theo ngay dat'),
    K('cancel.def', 'status_cancelled', r'^don bi huy', (r'trang thai huy', r'tra hang khong phai don huy')),
    K('orders.valid_only', True, r'^so don', r'chi tinh don khong huy, khong tra'),
    K('revenue.order_def', 'lines_net', r'^doanh thu mot don',
      (r'thanh tien cac dong hang', r'da tru giam gia tren tung dong', r'khong tru giam gia tren ca don',
       r'khong cong thue', r'khong gom phi giao')),
    K('payment.by_order_method', True, r'^doanh thu theo hinh thuc thanh toan', r'ghi tren don'),
    K('payment.cod_separate', True, r'^doanh thu theo hinh thuc thanh toan', r'cod la hinh thuc rieng'),
    K('customer_group.walkin_none', True, r'^doanh thu theo nhom khach', r'khong thuoc nhom nao'),
    K('share.group_denominator', 'shop_total', r'^doanh thu theo nhom khach', r'chia cho doanh thu ca shop'),
    K('cash_shift_report', 'cash_by_pay_time', r'^bao cao tien mat theo ca', (r'chi ghi tien mat', r'gio thu tien')),
    K('cancel_rate.denominator', 'all_orders', r'^ti le don huy', (r'moi don dat trong ky', r'moi trang thai')),
    K('pct_change.base', 'previous', r'^phan tram thay doi', (r'so voi ky truoc', r'giam thi ra so am')),
    K('diff.order', 'asked_minus_base', r'^chenh lech', r'ky hoi - ky so'),
    # §2 thời gian (time_resolver áp dụng; nút ở đây để xác nhận độ phủ)
    K('time.day_flow', 'within_day', r'^ngay x, hom nay', r'chi trong ngay do'),
    K('time.snapshot', 'from_start', r'^den het ngay x', r'tu dau du lieu'),
    K('time.range_inclusive', True, r'^tu x den y', r'gom ca x va y'),
    K('time.n_days_incl_today', True, r'^n ngay gan nhat', r'ke ca hom nay'),
    K('threshold.strict', True, r'^qua n', (r'khong gom n', r'tu n tro len thi gom n')),
    K('time.week_start', 1, r'^tuan$', r'bat dau thu hai'),
    K('time.month', 'full', r'^thang m/y', r'tu ngay 1 den het ngay cuoi thang'),
    K('time.mtd_compare', 'same_days', r'^thang nay', r'cung so ngay'),
    K('time.ytd', True, r'^tu dau nam', r'01/01'),
    K('time.no_year', 'this_or_last', r'^thang m, quy q khong ghi nam', r'nam nay neu khong sau hom nay'),
    K('time.tet', 'calendar', r'^tet nam y', r'23 thang chap'),
    K('time.shift', 'order_time', r'^ca sang, ca chieu', r'theo gio tao don'),
    K('weekday_avg.denominator', 'calendar_weekdays', r'^trung binh moi ngay thu x', r'ke ca ngay khong co don'),
    # §3 xếp hạng
    K('product.service_fee_excluded', True, r'^mat hang', r'phi phuc vu khong phai mat hang'),
    K('category.exact', True, r'^nhom hang', r'khong gop nhom khac ten'),
    K('best_seller.by', 'qty_base', r'^ban chay', r'so luong quy ve don vi co ban'),
    K('qty_by_unit', 'base_unit_match', r'^ban duoc bao nhieu', r'co don vi co ban la'),
    K('top_profit.by', 'gross_profit', r'^hang loi nhat', r'tong lai gop'),
    K('customer.identity', 'code_else_name', r'^xac dinh khach', (r'moi ma khach la mot khach', r'khach le')),
    K('customer.top_money_by', 'revenue', r'^xac dinh khach', r'nhieu tien nhat xep theo doanh thu'),
    K('aov.rounding', 'half_up', r'^gia tri trung binh mot don', r'nua dong lam tron len'),
    # §4 công nợ
    K('debt.include_returned', True, r'^don duoc tinh vao no', (r'khong huy', r'don tra hang van tinh')),
    K('debt.order_owed', 'order_total_incl_ship', r'^no mot don tai ngay x', (r'tong don', r'gom phi giao', r'con no > 0')),
    K('debt.by_customer', 'walkin_in_total_only', r'^no theo khach', r'khach le van tinh vao tong no'),
    K('debt.exclude_cod_marketplace', True, r'^don cod', r'khong phai khach no'),
    K('debt.age_from', 'order_date', r'^tuoi no', (r'tinh tu ngay dat', r'0 ngay')),
    K('debt.flow', 'after_purchase_day', r'^no them, tra no', (r'chua tra ngay trong ngay mua', r'sau ngay mua')),
    # §5 kho, nhập hàng, giá vốn
    K('purchase.total', 'receipt_total_excl_cancelled', r'^tien nhap hang trong ky', (r'ngay phieu', r'bo phieu huy')),
    K('purchase.top_supplier_by', 'amount', r'^tien nhap hang trong ky', r'nha cung cap nhap nhieu nhat = .*tien nhap cao nhat'),
    K('purchase.minus_sales', 'valid_sales_same_unit', r'^nhap tru ban', (r'don tinh vao doanh thu', r'cung don vi')),
    K('purchase.avg_price', 'qty_weighted', r'^gia nhap binh quan', r'theo so luong nhap'),
    K('stock.big_unit_format', 'pair', r'^ton theo don vi lon', r'list 2 so'),
    K('inventory.value', 'stock_x_catalog_cost_base', r'^gia tri hang ton kho', (r'ton het hom nay', r'gia von mot don vi co ban')),
    K('stockout.def', 'end_of_day_zero', r'^mat hang het hang cuoi ngay', r'het ngay'),
    K('dead_stock.def', 'sold_before_not_in', r'^mat hang khong ban duoc trong ky', (r'da tung ban truoc ky', r'trong ky khong co don')),
    K('unit_price.def', 'line_base_before_disc_all', r'^gia ban x ngay d', (r'mot don vi co ban', r'truoc chiet khau', r'don huy')),
    K('price_effect.def', 'n_from_d_vs_n_before', r'^hieu ung tang gia', (r'n ngay ke tu d', r'n ngay ngay truoc d', r'bo don huy')),
    K('price_effect.first_raise', True, r'^hieu ung tang gia', r'lay lan dau'),
    K('cogs.method', 'line_cost', r'^lai gop', r'gia von ghi tren dong ban'),
    K('cogs.avg', 'weighted', r'^gia von$', r'binh quan gia quyen'),
    K('net_profit', 'gross_minus_all_expenses', r'^lai rong mot thang', r'moi khoan'),
    # §6 theo ngành
    K('size.suffix', True, r'^size m, size l', r'\(m\) hoac \(l\)'),
    K('topping.separate', True, r'^topping ghi chung o', r'hai mat hang'),
    K('rx.def', 'catalog_flag', r'^thuoc ke don', r'ghi ke don trong danh muc'),
    K('rx.share_by', 'revenue', r'^thuoc ke don', r'ti le ke don tinh theo doanh thu'),
    K('expiry.window', 'd_to_d_plus_n', r'^hang het han trong n ngay toi', r'tu d den het ngay d \+ n'),
    K('expiry.capital', 'lot_cost_base', r'^hang het han trong n ngay toi', r'gia nhap cua chinh lo'),
    K('ingredient.exact', True, r'^doanh thu theo hoat chat', r'mot loai rieng'),
    K('lot.nearest', 'earliest_unexpired_in_stock', r'^lo gan het han nhat', r'han som nhat'),
    K('delivery.count', 'per_order_batch', r'^giao nhieu dot', r'moi dot cua moi don tinh mot dot'),
    # §7, §8 cách ghi trong file
    K('unanswerable.no_data_not_zero', True, '', r'khong co du lieu khac voi bang 0'),
    K('file.same_name', True, r'^ten goi, nhan', r'cung chi mot thu'),
    K('file.group_ffill', True, r'^o de trong o cac dong sau', r'giong o do o dong dau nhom'),
    K('status.no_column', 'note', r'^so khong co cot trang thai', (r'cot ghi chu', r'don hoan thanh')),
    K('status.values', 'four', r'^trang thai don', r'hoan thanh, huy, tra hang, dang giao'),
    K('file.dedupe_same_order', True, r'^hai dong cua cung mot don', r'tinh mot lan'),
    K('file.rows_not_by_order', 'each_row', r'^bang khong ghi theo don', r'moi dong la mot lan ghi'),
    K('file.date_no_year', 'sheet_year', r'^ngay khong ghi nam', r'nam cua sheet'),
    K('unit.base', 'catalog_or_readme', r'^don vi co ban, don vi lon', r'he so quy doi'),
]
# câu luật mô tả format chấm điểm / bao quát -> không phải luật nghiệp vụ, coi là đã phủ
_FORMAT_SECTIONS = r'^9\. dap an'
_GENERIC_NAMES = {'number', 'percent', 'date', 'boolean', 'string', 'list', 'table', 'unanswerable'}


# ----------------------------------------------------------------------------- ranking definitions
ENTITY_HEADS = [
    ('nha cung cap', 'supplier'), ('mat hang', 'product'), ('san pham', 'product'), ('mon', 'product'), ('hang', 'product'),
    ('khach', 'customer'), ('nhan vien', 'staff'), ('tho', 'staff'), ('ngay', 'day'), ('gio', 'hour'),
]
METRIC_PHRASES = [
    (r'nhieu ngay co mua nhat', 'active_days'),
    (r'nhieu ngay lam viec nhat', 'active_days'),
    (r'gia tri trung binh mot don cao nhat', 'aov'),
    (r'nhieu don huy nhat', 'cancelled_orders'),
    (r'co mat trong nhieu don nhat', 'orders'),
    (r'nhieu phieu nhap nhat', 'receipts'),
    (r'tien nhap cao nhat', 'purchase_amount'),
    (r'nhieu don nhat', 'orders'),
    (r'theo tong lai gop', 'gross_profit'),
    (r'theo doanh thu', 'revenue'),
    (r'theo so luong quy ve don vi co ban', 'qty'),
]
_OPTIONAL = {'duoc', 'hay', 'nhat', 'la', 'rat', 'ban'}


@dataclass
class RankingDef:
    phrase: str                         # cụm khớp trong câu hỏi (đã fold), vd 'trung thanh'
    heads: list[str]                    # đầu danh từ chỉ đối tượng, vd ['khach']
    entity: str                         # customer | product | staff | day | hour | supplier
    metric: str
    min_orders: Optional[int] = None
    answer: str = 'name'                # name | date | hour
    source: str = ''


def _ranking_from(st: Statement) -> Optional[RankingDef]:
    name = st.name
    if not name or not (name.endswith('nhat') or 'cao diem' in name or name.endswith('chinh')):
        return None
    metric = next((m for rx, m in METRIC_PHRASES if re.search(rx, st.text)), None)
    if metric is None:
        return None
    heads, rest = [], name
    # 'nhan vien, tho cham chi nhat' -> heads [nhan vien, tho], tính từ 'cham chi'
    parts = [p.strip() for p in re.split(r',| hoac ', name)]
    for p in parts:
        for h, ent in ENTITY_HEADS:
            if p == h or p.startswith(h + ' '):
                heads.append(h)
                rest = p[len(h):].strip()
                break
    ents = {e for h, e in ENTITY_HEADS if h in heads}
    if len(ents) != 1:
        return None
    entity = ents.pop()
    core = ' '.join(t for t in rest.split() if t not in _OPTIONAL)
    if not core:
        return None                      # 'mat hang nhat'? không có tính từ phân biệt
    mo = re.search(r'tu (\d+) don tro len', st.text)
    ans = 'date' if entity == 'day' else ('hour' if entity == 'hour' else 'name')
    return RankingDef(phrase=core, heads=heads, entity=entity, metric=metric,
                      min_orders=int(mo[1]) if mo else None, answer=ans, source=st.raw[:200])


# ----------------------------------------------------------------------------- rule set
_STOP = {'la', 'cua', 'va', 'cac', 'theo', 'mot', 'nhung', 'trong', 'cho', 'thi', 'khi', 'co', 'khong', 'duoc', 'x', 'y',
         'n', 'm', 'q', 'd', 'nao', 'nhat', 'ca', 'cung', 'hay', 've', 'tu', 'den', 'het'}


@dataclass
class RuleSet:
    knobs: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, str] = field(default_factory=dict)
    rankings: list[RankingDef] = field(default_factory=list)
    unknown: list[dict] = field(default_factory=list)
    source: str = 'regex'

    # -- truy vấn
    def get(self, key: str, default: Any = UNKNOWN) -> Any:
        return self.knobs.get(key, default)

    def has(self, key: str) -> bool:
        return self.knobs.get(key) is not None

    def missing(self, keys) -> list[str]:
        return [k for k in keys if not self.has(k)]

    def touches_unknown(self, qn: str) -> Optional[dict]:
        """Câu hỏi chạm vào một luật lạ (mọi từ nội dung của tên luật đều có trong câu) -> trả luật đó."""
        toks = set(qn.split())
        for u in self.unknown:
            for alt in u.get('keys', []):
                ws = alt.split()
                if len(ws) >= 2 and all(w in toks for w in ws) and alt in qn:
                    return u
        return None

    def ranking_for(self, qn: str) -> Optional[RankingDef]:
        for r in sorted(self.rankings, key=lambda r: -len(r.phrase)):
            if re.search(r'\b' + re.escape(r.phrase) + r'\b', qn) and any(re.search(r'\b' + h + r'\b', qn) for h in r.heads):
                return r
        return None

    # -- IO
    def to_json(self) -> str:
        return json.dumps({'source': self.source, 'knobs': self.knobs, 'evidence': self.evidence,
                           'rankings': [asdict(r) for r in self.rankings], 'unknown': self.unknown},
                          ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, s: str) -> 'RuleSet':
        d = json.loads(s)
        return cls(knobs=d.get('knobs', {}), evidence=d.get('evidence', {}),
                   rankings=[RankingDef(**r) for r in d.get('rankings', [])], unknown=d.get('unknown', []),
                   source=d.get('source', 'file'))

    def merged(self, override: 'RuleSet') -> 'RuleSet':
        """override (LLM / người) thắng; knob override = null nghĩa là cố ý tắt."""
        k = dict(self.knobs)
        k.update(override.knobs)
        ev = dict(self.evidence)
        ev.update(override.evidence)
        rk = override.rankings or self.rankings
        unk = override.unknown if override.unknown is not None and override.source != 'regex' and override.unknown != [] else self.unknown
        return RuleSet(knobs=k, evidence=ev, rankings=rk, unknown=unk, source=f'{self.source}+{override.source}')


def _keys_of(name: str) -> list[str]:
    out = []
    for alt in re.split(r',|/| hoac ', name):
        ws = [w for w in re.sub(r'[^a-z0-9 ]', ' ', alt).split() if w not in _STOP]
        if len(ws) >= 2:
            out.append(' '.join(ws))
    return out


def compile_rules(text: str) -> RuleSet:
    rs = RuleSet()
    sts = parse_statements(text)
    for k in KNOBS:
        rs.knobs.setdefault(k.key, UNKNOWN)
    for st in sts:
        covered = False
        for k in KNOBS:
            if k.section and not re.search(k.section, st.section):
                continue
            if k.name and not re.search(k.name, st.name):
                continue
            if not k.name and not k.text:
                continue
            if all(re.search(t, st.text) for t in k.text):
                if rs.knobs.get(k.key) is None:
                    rs.knobs[k.key] = k.value
                    rs.evidence[k.key] = st.raw[:240]
                covered = True
            elif k.name and re.search(k.name, st.name):
                covered = covered or False         # đúng tên luật nhưng nội dung đổi -> giữ None, ghi unknown bên dưới
        rd = _ranking_from(st) if re.search(r'^3\.|^5\.|^6\.', st.section) else None
        if rd:
            rs.rankings.append(rd)
            covered = True
        if not covered and (re.search(_FORMAT_SECTIONS, st.section) or st.name in _GENERIC_NAMES
                            or re.search(r'^7\.|^10\.', st.section)):
            covered = True
        if not covered and st.name:
            rs.unknown.append({'section': st.section, 'name': st.name, 'text': st.text[:300], 'keys': _keys_of(st.name)})
    return rs


def load_rules(rulebook_text: str, override_path: Optional[Path] = None) -> RuleSet:
    rs = compile_rules(rulebook_text)
    if override_path and Path(override_path).exists():
        rs = rs.merged(RuleSet.from_json(Path(override_path).read_text(encoding='utf-8')))
    return rs
