"""v2.5.0 — Bộ đo tổng quát hoá không dùng public: "làm bẩn có kiểm soát" các shop train.

Mỗi phép biến đổi giữ nguyên NGỮ NGHĨA (đáp án train vẫn đúng) nhưng đổi HÌNH THỨC dữ liệu theo các kiểu đã gặp ở public /
có thể gặp ở private. Hệ thống tổng quát tốt phải: (a) tự xử lý được kiểu đã biết; (b) gặp kiểu lạ thì BỎ TRỐNG thay vì
trả lời sai. Chỉ số chính: số câu SAI không được tăng so với bản sạch.

Phép biến đổi (chỉ áp lên bảng CSV/XLSX; SQLite giữ nguyên):
  ren       đổi tên cột đã map sang tên đồng nghĩa khác (README đổi theo cho nhất quán)
  money     viết tiền kiểu sổ tay: '1 234 000', '1tr234', '234k', '1.234.000đ' (lẫn trong cùng cột)
  status    nhãn trạng thái đổi sang cách gọi chưa có trong từ điển ('khách bùng', 'đã chốt đơn'…)
  thousand  tiền cấp đơn (tổng đơn, đã trả, phí giao…) ghi theo nghìn đồng
  clean     bản sao không đổi (đối chứng)

Dùng:
  python scripts/make_dirty.py make   --split <train> --split-file configs/split.yaml --part holdout --out runs/dirty_split
  python -m datamate.runtime.main predict --split runs/dirty_split --out runs/dirty_pred --config configs/no_llm.yaml --fresh
  python scripts/make_dirty.py report --split runs/dirty_split --pred runs/dirty_pred
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from datamate.core.textutil import fold, to_num  # noqa: E402
from datamate.data.vocab import norm_status  # noqa: E402

PERTS = ['clean', 'ren', 'money', 'status', 'thousand']
MONEY_ROLES = {'amount', 'unit_price', 'goods_total', 'order_total', 'paid', 'debt', 'ship_fee', 'order_discount',
               'line_discount'}
ORDER_MONEY = {'goods_total', 'order_total', 'paid', 'debt', 'ship_fee', 'order_discount'}
# tên cột đồng nghĩa CHỈ dùng để sinh bộ đo (không nằm trong code dự đoán)
SYN = {
    'amount': ['Tiền bán dòng', 'Cộng dòng'], 'qty': ['Sl xuất', 'Số lg bán'], 'unit_price': ['Giá 1 đơn vị bán', 'Giá lẻ bán'],
    'date': ['Thời điểm lập', 'Ngày chứng từ'], 'status': ['Tình trạng đơn', 'TT xử lý'],
    'order_id': ['Số chứng từ', 'Mã phiếu bán'], 'customer_name': ['Người mua', 'Tên người mua'],
    'product_name': ['Mặt hàng bán', 'Tên món hàng'], 'paid': ['Khách đưa', 'Đã thu khách'],
    'goods_total': ['Cộng tiền hàng', 'Tiền hàng đơn'], 'order_total': ['Phải thu', 'Tổng phải thu'],
    'unit': ['ĐVT bán', 'Đơn vị bán ra'], 'customer_code': ['Mã người mua'], 'product_code': ['Mã mặt hàng'],
}
STATUS_NEW = {'completed': 'đã chốt đơn', 'cancelled': 'khách bùng', 'returned': 'khách gửi trả lại', 'shipping': 'đang đi giao'}


def _pick(opts: list[str], key: str) -> str:
    return opts[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(opts)]


def _money_str(v: float, i: int) -> str:
    v = int(round(v))
    neg = v < 0
    v = abs(v)
    k = i % 3
    if k == 0:
        s = f'{v:,}'.replace(',', ' ')
    elif k == 1:
        if v >= 1_000_000 and v % 1000 == 0:
            tr, rest = divmod(v // 1000, 1000)
            s = f'{tr}tr{rest:03d}' if rest else f'{tr}tr'
        elif v % 1000 == 0 and v > 0:
            s = f'{v // 1000}k'
        else:
            s = f'{v:,}'.replace(',', '.') + 'đ'
    else:
        s = f'{v:,}'.replace(',', '.') + 'đ'
    return '-' + s if neg else s


def _plans(cat) -> dict[str, dict]:
    """bảng thô -> {cột: role} cho các role cần làm bẩn."""
    out: dict[str, dict] = defaultdict(dict)
    for m in cat.mapping.tables.values():
        for role, col in m.roles.items():
            out[m.table][col] = role
    return out


def _transform_frame(df: pd.DataFrame, colroles: dict, pert: str, tag: str) -> tuple[pd.DataFrame, dict]:
    ren = {}
    df = df.copy()
    for col, role in colroles.items():
        if col not in df.columns:
            continue
        if pert == 'ren' and role in SYN:
            new = _pick(SYN[role], f'{tag}|{col}')
            if new not in df.columns:
                ren[col] = new
        elif pert == 'money' and role in MONEY_ROLES:
            vals = df[col].map(to_num)
            df[col] = [(_money_str(v, i) if v is not None and v == v else o) for i, (v, o) in enumerate(zip(vals, df[col]))]
        elif pert == 'thousand' and role in ORDER_MONEY:
            vals = df[col].map(to_num)
            df[col] = [(f'{v / 1000:g}' if v is not None and v == v and v != 0 else o) for v, o in zip(vals, df[col])]
        elif pert == 'status' and role == 'status':
            df[col] = df[col].map(lambda x: STATUS_NEW.get(norm_status(x), x) if isinstance(x, str) and x.strip() else x)
    if ren:
        df = df.rename(columns=ren)
    return df, ren


def _write_csv(src: Path, dst: Path, colroles: dict, pert: str, tag: str) -> dict:
    raw = src.read_bytes()
    enc = 'utf-8-sig' if raw.startswith(b'\xef\xbb\xbf') else 'utf-8'
    df = pd.read_csv(src, dtype=str, keep_default_na=False, encoding=enc)
    df, ren = _transform_frame(df, colroles, pert, tag)
    df.to_csv(dst, index=False, encoding=enc)
    return ren


def _write_xlsx(src: Path, dst: Path, colroles: dict, pert: str, tag: str) -> dict:
    import openpyxl
    wb = openpyxl.load_workbook(src)
    ren_all = {}
    want = set(colroles)
    for ws in wb.worksheets:
        hdr_row, hdr, best = None, {}, 0
        for r in range(1, min(ws.max_row, 10) + 1):
            vals = {str(c.value).strip(): c.column for c in ws[r] if c.value is not None}
            k = len(want & set(vals))
            if k > best:
                hdr_row, hdr, best = r, vals, k
        if hdr_row is None or best < min(2, len(want)):
            continue
        for col, ci in hdr.items():
            role = colroles.get(col)
            if role is None:
                continue
            if pert == 'ren' and role in SYN:
                new = _pick(SYN[role], f'{tag}|{col}')
                ws.cell(hdr_row, ci).value = new
                ren_all[col] = new
                continue
            for i, r in enumerate(range(hdr_row + 1, ws.max_row + 1)):
                cell = ws.cell(r, ci)
                v = cell.value
                if v is None or v == '':
                    continue
                if pert == 'money' and role in MONEY_ROLES:
                    x = to_num(v)
                    if x is not None:
                        cell.value = _money_str(x, i)
                elif pert == 'thousand' and role in ORDER_MONEY:
                    x = to_num(v)
                    if x:
                        cell.value = f'{x / 1000:g}'
                elif pert == 'status' and role == 'status' and isinstance(v, str):
                    cell.value = STATUS_NEW.get(norm_status(v), v)
    wb.save(dst)
    return ren_all


def make(args):
    from datamate.data.catalog import build_catalog
    split, out = Path(args.split), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    (out / 'data').mkdir(parents=True)
    for f in ('rulebook.md', 'calendar.csv'):
        if (split / f).exists():
            shutil.copy(split / f, out / f)
    shops = args.biz
    if not shops and args.split_file:
        import yaml
        sp = yaml.safe_load(Path(args.split_file).read_text(encoding='utf-8'))
        shops = sp.get(args.part or 'holdout') or []
    if not shops:
        shops = sorted(p.name for p in (split / 'data').iterdir() if p.is_dir())
    perts = args.perts or PERTS
    qs = [json.loads(l) for l in (split / 'questions.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    ans = {a['id']: a for a in (json.loads(l) for l in (split / 'answers.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip())}
    qo, ao = [], []
    for b in shops:
        src = split / 'data' / b
        cat = build_catalog(src, args.cache)
        plans = _plans(cat)
        for pert in perts:
            nb = f'{b}__{pert}'
            dst = out / 'data' / nb
            shutil.copytree(src, dst)
            renames = {}
            if pert != 'clean':
                for tname, t in cat.raw.items():
                    if t.kind not in ('csv', 'xlsx') or tname not in plans:
                        continue
                    for fp in t.files:
                        p = Path(fp)
                        rel = p.relative_to(src) if p.is_absolute() and src in p.parents else Path(p.name)
                        d = dst / rel
                        if not d.exists():
                            continue
                        try:
                            if t.kind == 'csv' and t.header_row == 0:
                                renames.update(_write_csv(src / rel, d, plans[tname], pert, f'{b}|{tname}'))
                            elif t.kind == 'xlsx':
                                renames.update(_write_xlsx(src / rel, d, plans[tname], pert, f'{b}|{tname}'))
                        except Exception as e:  # noqa: BLE001
                            print(f'  ! {nb} {rel}: {e!r}'[:200])
                if renames:
                    rd = next((p for p in dst.iterdir() if p.name.lower() == 'readme.md'), None)
                    if rd:
                        txt = rd.read_text(encoding='utf-8')
                        for old, new in renames.items():
                            txt = re.sub(r'(\*\*|`)' + re.escape(old) + r'(\*\*|`)', lambda m: m[1] + new + m[2], txt)
                        rd.write_text(txt, encoding='utf-8')
            for q in qs:
                if q['business_id'] == b and q['id'] in ans:
                    qo.append({**q, 'id': f"{q['id']}__{pert}", 'business_id': nb})
                    ao.append({**ans[q['id']], 'id': f"{q['id']}__{pert}"})
            print(f'{nb}: đổi tên {len(renames)} cột' if pert == 'ren' else nb)
    (out / 'questions.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in qo) + '\n', encoding='utf-8')
    (out / 'answers.jsonl').write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in ao) + '\n', encoding='utf-8')
    print(f'{len(shops)} shop × {len(perts)} phép = {len(qo)} câu -> {out}')


def report(args):
    from datamate.offline.scorer import score_one
    split, pred = Path(args.split), Path(args.pred)
    gold = {a['id']: a for a in (json.loads(l) for l in (split / 'answers.jsonl').read_text(encoding='utf-8').splitlines() if l.strip())}
    sub = {r['id']: r for r in (json.loads(l) for l in (pred / 'submission.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip())}
    agg = defaultdict(lambda: [0, 0, 0])
    per_shop = defaultdict(lambda: [0, 0, 0])
    for qid, g in gold.items():
        pert = qid.rsplit('__', 1)[-1]
        r = sub.get(qid, {'type': 'abstain'})
        s = score_one({'type': r.get('type'), 'answer': r.get('answer')}, g) if r.get('type') != 'abstain' else 0
        k = 0 if s == 1 else (1 if s == -1 else 2)
        agg[pert][k] += 1
        logs = pred / 'logs' / f'{qid}.json'
        biz = json.loads(logs.read_text(encoding='utf-8'))['business_id'] if logs.exists() else '?'
        per_shop[biz][k] += 1
    print(f'{"phép":<10} {"đúng":>6} {"sai":>5} {"bỏ":>5} {"PA":>7}')
    base = agg.get('clean')
    for p in PERTS:
        if p not in agg:
            continue
        ok, bad, ab = agg[p]
        n = ok + bad + ab
        extra = ''
        if base and p != 'clean':
            extra = f'   (so với sạch: đúng {ok - base[0]:+d}, sai {bad - base[1]:+d}, bỏ {ab - base[2]:+d})'
        print(f'{p:<10} {ok:>6} {bad:>5} {ab:>5} {(ok - bad) / n:>7.3f}{extra}')
    if args.shops:
        for b in sorted(per_shop):
            ok, bad, ab = per_shop[b]
            print(f'  {b:<24} đúng={ok} sai={bad} bỏ={ab}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make')
    p.add_argument('--split', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--split-file', default=None)
    p.add_argument('--part', default=None)
    p.add_argument('--biz', nargs='*')
    p.add_argument('--perts', nargs='*', choices=PERTS)
    p.add_argument('--cache', default='/tmp/datamate_cache')
    p = sub.add_parser('report')
    p.add_argument('--split', required=True)
    p.add_argument('--pred', required=True)
    p.add_argument('--shops', action='store_true')
    a = ap.parse_args()
    {'make': make, 'report': report}[a.cmd](a)


if __name__ == '__main__':
    main()
