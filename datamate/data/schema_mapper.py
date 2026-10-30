"""Ánh xạ cột gốc -> vai trò canonical (role) + phân loại bảng.

Nguồn bằng chứng (ưu tiên giảm dần):
  1. Từ điển cột trong README ("- **Cột**: mô tả") — bộ đề dùng từ vựng mô tả thống nhất giữa các shop.
  2. Câu quy ước trong README dạng "`bang.Cot` là ..." / "**Cột** là ...".
  3. Tên cột (fold + tách camelCase) khớp từ đồng nghĩa.
  4. Kiểm chứng bằng giá trị (cột ngày phải đọc ra ngày, cột tiền phải là số...).
Không có luật nào gắn với business_id cụ thể.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from datamate.core.textutil import fold, fold_name
from datamate.data.profiler import ColumnProfile, TableProfile
from datamate.data.readme import ReadmeInfo

# ----------------------------------------------------------------------------- role vocabulary
# role: (kind, desc_regex, name_regex). kind: id|date|datetime|time|num|text|bool
ROLES: dict[str, tuple[str, str, str]] = {
    # --- đơn / dòng bán
    'order_id': ('id', r'^(ma don|so hoa don|ma hoa don|so don)\b',
                 r'^(ma (hoa don|don|don hang|chung tu|bill|hd|phieu ban|giao dich)|so (ct|chung tu|hd|hoa don|don|bill|phieu)|order id|orderid|don so|bill|hd|ma bill|invoice( id| no)?|order( no)?)$'),
    'line_no': ('num', r'^so thu tu dong trong don', r'^(stt|so thu tu|no|tt dong|dong|line)$'),
    'date': ('date', r'^(ngay ban|ngay dat|ngay sua)\b', r'^(ngay|ngay ban|ngay thang|ngay dat( hang)?|ngay giao dich|ngay lap|ngay chung tu|ngay hach toan|ngay hd|date|ngayban|ngay tao don|thoi gian( ban)?)$'),
    'datetime': ('datetime', r'^thoi diem dat don', r'^(thoi gian (lap|tao|tao don|dat)|ngay gio( lap| ban)?|created at|tao luc)$'),
    'time': ('time', r'^gio dat don', r'^(gio|gio ban|gio tao don|gio dat|time)$'),
    'status': ('text', r'^trang thai don', r'^((trang thai|tinh trang)( (don|don hang|giao|xu ly|hoa don|ban))?|ket qua|status|xu ly|tien do)$'),
    'customer_code': ('id', r'^ma khach', r'^(ma (kh|khach|khach hang|khach mua)|so kh|customer id|ma kh mua|id khach)$'),
    'customer_name': ('text', r'^ten khach', r'^(khach|khach hang|ten (khach|khach hang|kh)|doi tac|ben mua|ho ten khach|nguoi mua|customer)$'),
    'customer_group': ('text', r'^nhom khach', r'^(nhom( khach| khach hang| kh)?|loai khach( hang)?|hang)$'),
    'product_code': ('id', r'^ma hang\b', r'^(ma (hang|sp|hang hoa|san pham|hh|mon|vach|hang ban)|sku|id sp|ma sku|barcode)$'),
    'product_name': ('text', r'^(ten hang|ten)\b(?!.*khach)', r'^(ten (hang|san pham|hang hoa|mon|mat hang|thuoc)|noi dung|hang|mat hang|hang hoa|hang ban|chi tiet hang|product|san pham|mon|ten goi|ten sp)$'),
    'category': ('text', r'^nhom hang', r'^(nhom hang( hoa)?|danh muc|phan loai|loai hang|nhom|category|nganh hang)( \d cap)?$'),
    'unit': ('text', r'^don vi tinh', r'^(dvt|don vi( tinh)?|unit|d vi|dv)$'),
    'base_unit': ('text', r'^don vi co ban', r'^(don vi co (ban|so)|dvt co ban|don vi goc)$'),
    'qty': ('num', r'^so luong( \(|$)', r'^(so luong|sl|slg|sluong|so luong san pham|qty|quantity)$'),
    'qty_base': ('num', r'^so luong da doi ra don vi co ban', r'^(sl quy doi|so luong quy doi|quy ra dv le|sl co ban|so luong le|sl quy ra le|slquydoi|so luong co ban)$'),
    'unit_price': ('num', r'^don gia', r'^(don gia( \(?d\)?)?|gia ban|gia|gia le|gia san pham|gia don vi|gia niem yet|dg)$'),
    'line_discount': ('num', r'^tien giam cua ca dong', r'^(chiet khau|khuyen mai|tien giam|giam gia dong)$'),
    'amount': ('num', r'^thanh tien', r'^(thanh tien|tri gia|t tien|tt|thanh tien dong)$'),
    'cost_unit': ('num', r'^gia von (luc ban|mot don vi ban)', r'^(gia von luc ban|gia von hang ban)$'),
    'channel': ('text', r'^kenh ban', r'^(kenh( ban( hang)?)?|nguon( don)?|ban qua|channel)$'),
    'payment': ('text', r'^hinh thuc thanh toan', r'^(phuong thuc( tt| thanh toan)?|hinh thuc( tt| thanh toan)?|pttt|httt|payment)$'),
    'staff': ('text', r'^ma nhan vien ban', r'^(ma (nv|nhan vien)( ban)?|nguoi ban|nhan vien( ban)?|thu ngan|nv ban)$'),
    'goods_total': ('num', r'^(tong )?tien hang cua (ca )?don', r'^(tong tien hang|tien hang( truoc giam)?)$'),
    'order_discount': ('num', r'^giam gia cua ca don', r'^(giam gia hd|giam gia don|ck don|chiet khau don)$'),
    'order_total': ('num', r'^(tong tien don|so khach phai tra|khach can tra)', r'^(tong (tien|bill|thanh toan|sau giam gia|cong|tien don)|tong tien|can tra|khach can tra)$'),
    'paid': ('num', r'^(da thanh toan|khach da tra|so tien khach tra)', r'^(da tra|khach da tra|da thu|khach tra|da thanh toan)$'),
    'debt': ('num', r'^(con no|so khach con no|khach con no)', r'^(con no|chua tra|con lai|con phai thu|no)$'),
    'ship_fee': ('num', r'^phi giao', r'^(phi (ship|giao( hang)?|van chuyen)|ship)$'),
    'vat': ('num', r'^thue', r'^(thue( gtgt| vat)?|vat)$'),
    'note': ('text', r'^ghi chu', r'^(ghi chu|note|dien giai|mo ta)$'),
    # --- danh mục
    'price': ('num', r'^gia ban hien tai', r'^(gia ban( le)?|gia)$'),
    'cost': ('num', r'^gia von$', r'^(gia von( hang| hien tai)?)$'),
    'rx': ('bool', r'^thuoc ke don', r'^(ke don|thuoc ke don|ban theo don)$'),
    'ingredient': ('text', r'^hoat chat', r'^(hoat chat|thanh phan( chinh)?)$'),
    'big_unit': ('text', r'^don vi lon', r'^(don vi quy doi|dv nhap|don vi lon)$'),
    'factor': ('num', r'^he so', r'^(ti le quy doi|ty le quy doi|he so( quy doi)?|bang)$'),
    'phone': ('text', r'^so dien thoai', r'^(dien thoai|sdt|so dien thoai|phone)$'),
    'address': ('text', r'^dia chi', r'^(dia chi( lien he)?|addr|address)$'),
    # --- thanh toán / trả hàng
    'receipt_id': ('id', r'^ma phieu thu', r'^(ma phieu thu|phieu thu|so phieu thu)$'),
    'pay_datetime': ('datetime', r'^thoi diem thu', r'^(thoi gian thu|ngay gio thu|ngay thu)$'),
    'pay_amount': ('num', r'^so tien$', r'^(so tien( thu)?|tien)$'),
    'return_id': ('id', r'^ma phieu tra', r'^(ma phieu tra)$'),
    'return_qty': ('num', r'^so luong khach tra lai', r'^(sl tra( quy doi| dvt co ban)?|so luong tra( dvt co ban)?|sltraquydoi)$'),
    'refund': ('num', r'^tien hoan', r'^(tien hoan tra|hoan tien)$'),
    # --- generic (đổi tên theo loại bảng ở RENAME)
    'name': ('text', r'^(ten|ho ten)( |$)', r'^(ten|ten goi|ho ten|name)$'),
    'discount': ('num', r'^giam gia$', r'^(giam gia|ck|chiet khau|khuyen mai)$'),
    'table_no': ('text', r'^ban$', r'^(ban|phong ban|so ban)$'),
    'area': ('text', r'^khu vuc', r'^(khu vuc)$'),
    'n_guests': ('num', r'^so khach', r'^(so khach)$'),
    # --- nhập hàng
    'receipt_no': ('id', r'^ma phieu nhap', r'^(ma (phieu )?nhap( hang| kho)?|so phieu( nhap)?|ma pn)$'),
    'p_date': ('date', r'^ngay nhap', r'^(ngay nhap( kho| hang)?)$'),
    'supplier': ('text', r'^nha cung cap', r'^(nha cung cap|ten ncc|ncc)$'),
    'supplier_code': ('id', r'^ma nha cung cap', r'^(ma (ncc|nha cung cap))$'),
    'p_qty': ('num', r'^so luong nhap', r'^(sl nhap|so luong nhap|sl nhap quy doi|slnhapquydoi)( dvt co ban)?$'),
    'p_unit': ('text', r'^don vi nhap', r'^(dv nhap|don vi nhap|dvt nhap)$'),
    'p_unit_cost': ('num', r'^gia nhap', r'^(gia nhap|don gia nhap|dongianhap)$'),
    'p_amount': ('num', r'^tien nhap cua ca dong', r'^(thanh tien nhap)$'),
    'p_total': ('num', r'^tong tien phieu nhap', r'^(can tra ncc|tong (gia tri|tien) (nhap|phieu)|tong tien phieu|tonggiatrinhap)$'),
    'p_status': ('text', r'^trang thai phieu nhap', r'^(trang thai phieu)$'),
    'lot': ('id', r'^so lo', r'^(lo|ma lo|so lo)$'),
    'expiry': ('date', r'^han su dung', r'^(hsd|han su dung|han dung)$'),
    # --- tồn kho / giá / chi phí / giao hàng
    'stock': ('num', r'^ton kho', r'^(ton cuoi( ky)?|ton kho|ton|toncuoi)$'),
    'eff_date': ('date', r'^ngay bat dau ap dung gia', r'^(ngay ap dung)$'),
    'new_price': ('num', r'^gia ban ap dung', r'^(gia moi)$'),
    'old_price': ('num', r'^gia truoc lan doi', r'^(gia cu)$'),
    'exp_date': ('date', r'^ngay chi', r'^(ngay chi)$'),
    'exp_item': ('text', r'^(muc|khoan) chi', r'^(muc|khoan chi|noi dung chi)$'),
    'deliv_date': ('date', r'^ngay giao', r'^(ngay giao( hang)?)$'),
    'deliv_no': ('num', r'^dot giao', r'^(lan giao|dot giao|dot)$'),
}

# role generic -> role theo loại bảng
RENAME = {
    'sales_lines': {'discount': 'line_discount', 'name': 'product_name'},
    'sales_flat': {'discount': 'line_discount'},
    'sales_orders': {'discount': 'order_discount', 'line_discount': 'order_discount', 'name': 'customer_name'},
    'products': {'name': 'product_name', 'unit_price': 'price', 'unit': 'base_unit'},
    'customers': {'name': 'customer_name'},
    'staff': {'name': 'staff_name', 'customer_name': 'staff_name', 'product_name': 'staff_name'},
    'purchases': {'order_id': 'receipt_no', 'qty': 'p_qty', 'unit': 'p_unit', 'unit_price': 'p_unit_cost',
                  'amount': 'p_amount', 'date': 'p_date', 'datetime': 'p_date', 'status': 'p_status', 'order_total': 'p_total',
                  'name': 'product_name', 'pay_amount': 'p_amount'},
    'lots': {'qty_base': 'qty_base', 'p_qty': 'p_qty_base', 'receipt_no': 'receipt_no'},
    'expenses': {'pay_amount': 'exp_amount', 'amount': 'exp_amount', 'date': 'exp_date', 'note': 'exp_item'},
    'price_history': {'price': 'new_price', 'unit_price': 'new_price', 'date': 'eff_date', 'name': 'product_name'},
    'inventory': {'name': 'product_name', 'unit': 'base_unit', 'datetime': 'inv_datetime', 'date': 'inv_datetime'},
    'deliveries': {'date': 'deliv_date'},
    'payments': {'amount': 'pay_amount', 'datetime': 'pay_datetime', 'date': 'pay_datetime'},
    'returns': {},
    'units': {'name': 'product_name', 'big_unit': 'unit'},
    'drugs': {'name': 'product_name'},
    'suppliers': {'name': 'supplier', 'product_name': 'supplier', 'customer_name': 'supplier'},
}

# loại bảng: (tên bảng fold match, role bắt buộc/gợi ý)
TABLE_NAME_HINTS = [
    ('suppliers', r'nha cung cap|ncc\b'),
    ('branches', r'chi nhanh'),
    ('payments', r'thanh toan|phieu thu|so quy|thu tien'),
    ('returns', r'tra hang|phieu tra'),
    ('purchases', r'nhap|phieu nhap'),
    ('units', r'quy doi|don vi sp|don vi'),
    ('lots', r'\blo\b|lo hang|ban theo lo|don lo'),
    ('price_history', r'lich su gia|bang gia'),
    ('inventory', r'ton kho|the kho|bao cao ton'),
    ('drugs', r'thuoc'),
    ('customers', r'khach hang|khachhang'),
    ('staff', r'nhan vien|tho\b'),
    ('expenses', r'chi phi'),
    ('deliveries', r'giao hang|phieu giao'),
    ('products', r'san pham|hang hoa|danh muc|hanghoa|sanpham|mat hang'),
    ('sales', r'hoa don|don hang|ban hang|chi tiet|bill|order|so ban|banhang|ban'),
]


@dataclass
class TableMapping:
    table: str
    ttype: str                                   # sales_lines | sales_orders | sales_flat | products | units | customers | payments | returns | purchases | ...
    roles: dict[str, str] = field(default_factory=dict)      # role -> column
    col_roles: dict[str, str] = field(default_factory=dict)  # column -> role
    evidence: dict[str, str] = field(default_factory=dict)   # role -> 'readme' | 'name' | 'value'
    descs: dict[str, str] = field(default_factory=dict)      # column -> mô tả README (nếu có)


@dataclass
class BusinessMapping:
    tables: dict[str, TableMapping]

    def of_type(self, *ttypes: str) -> list[TableMapping]:
        return [m for m in self.tables.values() if m.ttype in ttypes]


# ----------------------------------------------------------------------------- README extra descriptions
_CONV_COL = re.compile(r'`(?:[\w ]+\.)?([^`]+?)`\s+là\s+([^;.\n]+)')
_CONV_BOLD = re.compile(r'\*\*([^*]+?)\*\*\s+là\s+([^;.\n]+)')


def readme_extra_descs(info: ReadmeInfo) -> dict[str, str]:
    """'`chi_tiet_don.TriGia` là thành tiền của dòng' -> {'TriGia': 'thành tiền của dòng'}."""
    out: dict[str, str] = {}
    for m in list(_CONV_COL.finditer(info.text)) + list(_CONV_BOLD.finditer(info.text)):
        col, desc = m.group(1).strip(), m.group(2).strip()
        if len(col) <= 40:
            out.setdefault(fold(col), desc)
    return out


# ----------------------------------------------------------------------------- core
def _value_ok(kind: str, p: ColumnProfile) -> bool:
    if p.null_rate >= 0.999:
        return False
    if kind in ('date', 'datetime'):
        return p.date_rate + p.serial_rate + p.dm_rate >= 0.6
    if kind == 'num':
        return p.num_rate >= 0.8
    if kind == 'time':
        return True
    if kind == 'text':
        return p.num_rate < 0.95 or p.n_distinct <= 3
    return True


def _desc_score(desc: Optional[str], role: str) -> float:
    if not desc:
        return 0.0
    rx = ROLES[role][1]
    return 3.0 if re.search(rx, fold(desc)) else 0.0


def _name_score(col: str, role: str) -> float:
    return 2.0 if re.search(ROLES[role][2], fold_name(col)) else 0.0


def classify_table(name: str, roles: set[str], n_rows: int) -> str:
    fn = fold_name(name.split('::')[0]) + ' ' + fold(name)
    hint = next((t for t, rx in TABLE_NAME_HINTS if re.search(rx, fn)), None)
    has_prod = bool(roles & {'product_name', 'product_code'})
    has_line = bool(roles & {'qty', 'amount', 'qty_base'})
    has_order = 'order_id' in roles
    if hint in ('payments',) and ({'pay_amount', 'pay_datetime', 'receipt_id'} & roles):
        return 'payments'
    if hint in ('returns',) and 'order_id' in roles:
        return 'returns'
    if hint in ('suppliers', 'branches') and not has_line:
        return hint
    if hint in ('purchases', 'lots', 'price_history', 'inventory', 'drugs', 'expenses', 'deliveries', 'staff', 'units'):
        if hint == 'units' and 'factor' not in roles and has_order and has_line:
            pass   # bảng tên có "don vi" nhưng là dòng bán
        else:
            return hint
    if 'factor' in roles and has_prod and not has_order:
        return 'units'
    if hint == 'customers' or (roles & {'customer_code', 'customer_name'} and not has_prod and not has_order and n_rows < 20000
                               and roles & {'phone', 'address', 'customer_group'}):
        return 'customers'
    if has_prod and has_line and (has_order or roles & {'date'}):
        return 'sales_flat' if roles & {'date', 'datetime', 'status', 'customer_name', 'customer_code'} else 'sales_lines'
    if has_order and roles & {'date', 'datetime', 'status', 'order_total', 'goods_total'} and not has_prod:
        return 'sales_orders'
    if has_prod and not has_order and n_rows <= 5000:
        return 'products'
    return hint or 'other'


# vai trò hợp lệ theo loại bảng (role ngoài danh sách bị bỏ để tránh nhầm, vd 'gia' ở bảng nhập)
ALLOWED = {
    'sales': {'order_id', 'line_no', 'date', 'datetime', 'time', 'status', 'customer_code', 'customer_name', 'customer_group',
              'product_code', 'product_name', 'category', 'unit', 'base_unit', 'qty', 'qty_base', 'unit_price', 'line_discount',
              'amount', 'cost_unit', 'channel', 'payment', 'staff', 'goods_total', 'order_discount', 'order_total', 'paid',
              'debt', 'ship_fee', 'vat', 'note', 'discount', 'name', 'table_no', 'area', 'n_guests', 'return_qty'},
    'products': {'product_code', 'product_name', 'category', 'base_unit', 'unit', 'price', 'cost', 'rx', 'ingredient', 'name', 'unit_price', 'stock'},
    'customers': {'customer_code', 'customer_name', 'customer_group', 'phone', 'address', 'name'},
    'payments': {'receipt_id', 'order_id', 'pay_datetime', 'pay_amount', 'payment', 'date', 'datetime'},
    'returns': {'return_id', 'order_id', 'datetime', 'date', 'refund'},
    'drugs': {'product_code', 'product_name', 'rx', 'ingredient', 'name'},
    'staff': {'staff', 'name', 'customer_name', 'product_name'},
    'suppliers': {'supplier', 'supplier_code', 'name', 'phone', 'address', 'product_name', 'customer_name'},
    'units': {'product_code', 'product_name', 'unit', 'big_unit', 'factor', 'base_unit', 'unit_price', 'price', 'name'},
    'purchases': {'receipt_no', 'p_date', 'supplier', 'supplier_code', 'product_code', 'product_name', 'p_qty', 'p_unit',
                  'p_unit_cost', 'p_amount', 'p_total', 'p_status', 'note', 'lot', 'expiry', 'line_no', 'order_id', 'qty',
                  'unit', 'unit_price', 'amount', 'date', 'datetime', 'status', 'order_total', 'name', 'base_unit', 'qty_base', 'pay_amount'},
}


def map_table(tp: TableProfile, info: ReadmeInfo, extra: dict[str, str]) -> TableMapping:
    fname = tp.files[0] if tp.files else tp.name
    sheet = tp.sheets[0] if tp.sheets else None
    descs = {}
    for c in tp.columns:
        d = info.desc_for(fname, sheet, c) or extra.get(fold(c))
        if d:
            descs[c] = d
    # điểm (col, role)
    cand: list[tuple[float, str, str, str]] = []
    for c, p in tp.columns.items():
        if c.startswith('__'):
            continue
        dl = fold(descs.get(c, ''))
        if 'cot trong' in dl or 'khong co du lieu' in dl:
            continue
        for role, (kind, _, _) in ROLES.items():
            sd, sn = _desc_score(descs.get(c), role), _name_score(c, role)
            s = max(sd, sn)
            if s <= 0:
                continue
            if not _value_ok(kind, p):
                s -= 2.5
            if s > 0:
                cand.append((s + 0.1 * (sd > 0) + 0.05 * (sn > 0), c, role, 'readme' if sd else 'name'))
    rough = {r for _, _, r, _ in cand}
    ttype = classify_table(tp.name, rough, tp.n_rows)
    allowed = ALLOWED.get('sales' if ttype.startswith('sales') else ttype)
    tm = TableMapping(table=tp.name, ttype=ttype, descs=descs)
    ren = RENAME.get(ttype, {})
    for s, c, role, ev in sorted(cand, key=lambda x: -x[0]):
        if allowed is not None and role not in allowed:
            continue
        role = ren.get(role, role)
        if role in tm.roles or c in tm.col_roles:
            continue
        tm.roles[role] = c
        tm.col_roles[c] = role
        tm.evidence[role] = ev
    _fix_dates(tm, tp)
    return tm


def _fix_dates(tm: TableMapping, tp: TableProfile) -> None:
    """Có cả date & datetime: giữ cả hai; chỉ có datetime -> dùng làm date. Cột date mà chứa giờ -> datetime."""
    if 'date' not in tm.roles and 'datetime' in tm.roles:
        tm.roles['date'] = tm.roles['datetime']
    # nhiều cột dạng ngày mà chưa gán: thử cột có tên 'ngay'/'thoi gian' cho bảng sales
    if tm.ttype.startswith('sales') and 'date' not in tm.roles:
        for c, p in tp.columns.items():
            if c not in tm.col_roles and p.date_rate >= 0.8 and re.search(r'ngay|thoi gian|date', fold_name(c)):
                tm.roles['date'] = c
                tm.col_roles[c] = 'date'
                tm.evidence['date'] = 'value'
                break


def map_business(profiles: dict[str, TableProfile], info: ReadmeInfo) -> BusinessMapping:
    extra = readme_extra_descs(info)
    return BusinessMapping(tables={n: map_table(tp, info, extra) for n, tp in profiles.items()})
