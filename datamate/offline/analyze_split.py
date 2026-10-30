"""Phân tích toàn bộ một split (train/public) -> report.json để thiết kế schema mapper.
Chạy:
    pip install pandas openpyxl
    python analyze_split.py /path/to/train  -o report_train.json
    python analyze_split.py /path/to/public -o report_public.json
Chỉ đọc dữ liệu, không ghi gì vào thư mục split.
"""
import argparse, json, re, sqlite3, sys, time, unicodedata, hashlib
from collections import Counter, defaultdict
from pathlib import Path
import pandas as pd

SAMPLE_ROWS = 50_000      # số dòng tối đa dùng để profile mỗi bảng (đếm tổng dòng vẫn đầy đủ)
MAX_VALUESET = 40         # cột có <= ngần này giá trị khác nhau -> ghi đủ value set

def fold(s):
    s = unicodedata.normalize('NFD', str(s).lower())
    return ' '.join(''.join(c for c in s if unicodedata.category(c) != 'Mn').replace('đ', 'd').split())

def shape(v):
    """Mẫu định dạng: chữ số->9, chữ->a (giữ ký tự đặc biệt)."""
    s = str(v)
    s = re.sub(r'\d', '9', s); s = re.sub(r'9+', '9', s)
    s = re.sub(r'[^\W\d_]+', 'a', s)
    return s[:30]

def profile_df(df, n_total):
    cols = []
    sdf = df.head(SAMPLE_ROWS).copy()
    seen = Counter(); new = []
    for c in sdf.columns:
        c = str(c); seen[c] += 1; new.append(c if seen[c] == 1 else f'{c}__dup{seen[c]-1}')
    sdf.columns = new
    for c in sdf.columns:
        s = sdf[c]
        nn = s.dropna()
        nn = nn[nn.astype(str).str.strip() != '']
        vc = nn.astype(str).value_counts()
        col = {
            'name': str(c), 'pandas_dtype': str(s.dtype),
            'null_rate': round(1 - len(nn) / max(len(s), 1), 4),
            'n_distinct_sample': int(vc.size),
            'samples': [str(x)[:60] for x in nn.sample(min(6, len(nn)), random_state=0).tolist()] if len(nn) else [],
            'shapes': dict(Counter(shape(x) for x in nn.astype(str).head(5000)).most_common(6)),
        }
        if 0 < vc.size <= MAX_VALUESET:
            col['value_set'] = {str(k)[:60]: int(v) for k, v in vc.items()}
        else:
            col['top_values'] = {str(k)[:60]: int(v) for k, v in vc.head(8).items()}
        cols.append(col)
    return {'n_rows': int(n_total), 'n_cols': len(df.columns), 'columns': cols}

def read_csv_any(p):
    for enc in ('utf-8-sig', 'utf-8', 'cp1258', 'latin-1'):
        for sep in (None,):
            try:
                return pd.read_csv(p, dtype=str, encoding=enc, sep=sep, engine='python'), enc
            except Exception:
                continue
    raise RuntimeError(f'cannot read {p}')

def analyze_file(p):
    ext = p.suffix.lower(); out = {'file': p.name, 'ext': ext, 'size_kb': round(p.stat().st_size / 1024, 1), 'tables': []}
    try:
        if ext in ('.csv', '.tsv', '.txt'):
            df, enc = read_csv_any(p); out['encoding'] = enc
            out['tables'].append({'table': p.stem, **profile_df(df, len(df))})
        elif ext in ('.xlsx', '.xlsm', '.xls'):
            sheets = pd.read_excel(p, sheet_name=None, dtype=str, header=None)
            out['sheet_names'] = list(sheets)
            # nếu các sheet cùng header -> gộp để profile 1 lần
            groups = defaultdict(list)
            for name, raw in sheets.items():
                raw = raw.dropna(how='all').dropna(axis=1, how='all')
                if raw.empty: groups['__empty__'].append(name); continue
                head = raw.head(20)
                scores = head.notna().sum(axis=1) + head.apply(lambda r: sum(isinstance(x, str) and not re.fullmatch(r'[\d.,/ -]+', x) for x in r), axis=1)
                hdr_row = scores.idxmax()
                header = tuple(str(x) for x in raw.loc[hdr_row].tolist())
                groups[header].append((name, raw, hdr_row))
            for header, items in groups.items():
                if header == '__empty__': out['empty_sheets'] = items; continue
                dfs = []
                for name, raw, hdr_row in items:
                    d = raw.loc[raw.index > hdr_row].copy(); d.columns = list(header); d['__sheet__'] = name; dfs.append(d)
                df = pd.concat(dfs, ignore_index=True)
                first = items[0][1]
                out['tables'].append({'table': f'{p.stem}[{items[0][0]}..{items[-1][0]}]', 'sheets': [i[0] for i in items],
                                      'header_row_index': int(items[0][2]),
                                      'rows_above_header': [[str(x)[:60] for x in r] for r in first.loc[first.index < items[0][2]].head(5).values.tolist()],
                                      'raw_top_rows': [[str(x)[:40] for x in r] for r in first.head(6).values.tolist()],
                                      **profile_df(df, len(df))})
        elif ext in ('.sqlite', '.db', '.sqlite3'):
            con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
            for (t,) in con.execute("select name from sqlite_master where type in ('table','view')"):
                n = con.execute(f'select count(*) from "{t}"').fetchone()[0]
                df = pd.read_sql(f'select * from "{t}" limit {SAMPLE_ROWS}', con)
                fks = [list(r[2:5]) for r in con.execute(f'pragma foreign_key_list("{t}")')]
                out['tables'].append({'table': t, 'foreign_keys': fks, **profile_df(df, n)})
            con.close()
        else:
            out['note'] = 'unhandled extension'
    except Exception as e:
        out['error'] = repr(e)[:300]
    return out

# ---------------- questions
INTENTS = [
    ('debt_aging', r'tuoi no|0.?7 ngay|tren 15 ngay'), ('debt_top', r'(no|cong no).*(nhieu nhat|lon nhat)'),
    ('debt_flow', r'no them|tra (bot )?no duoc|mua chiu them'), ('debt_collected', r'tra no|thu (hoi )?.*no|tra bot no'),
    ('debt_total', r'con no|cong no|chua tra het no|con thieu'), ('expiry', r'het han|han dung|han su dung'),
    ('inventory_value', r'(gia von|tien hang).*(kho|ton)'), ('inventory_units', r'ton kho|trong kho|chan.*le|cuoi ngay.*con'),
    ('price_effect', r'tang gia|doi gia|len gia'), ('purchase_price', r'gia nhap|nhap .*binh quan|nhap vao gia'),
    ('purchase_supplier', r'nha cung cap'), ('purchase_total', r'nhap hang|phieu nhap|tien hang nhap'),
    ('net_profit', r'lai rong|tru (het )?chi phi'), ('gross_profit', r'lai gop|loi gop|loi nhat'),
    ('unit_price', r'gia ban|don gia|gia bao nhieu'), ('dead_stock', r'\be\b|ban cham|khong ban|ko ban'),
    ('cancel_rate', r'(ti le|%|phan tram).*huy|huy.*(%|phan tram)'), ('cancel_count', r'huy'),
    ('deliveries', r'dot (giao|hang)|cho giao'), ('trend_bool', r'tang lien tuc|luon cao hon|co phai'),
    ('compare_period', r'so (voi|vs)|hon hay kem|tang hay giam|tang truong|thay doi|chenh'),
    ('top_customer', r'khach .*(nhieu tien|chi tieu|mua nhieu|tong tien)|ai mua|khach nao'),
    ('staff', r'nhan vien'), ('top_product', r'ban chay|nhieu nhat|hot nhat|best seller|chay nhat'),
    ('best_period', r'ngay nao|thang nao'), ('customer_count', r'khach'),
    ('product_count', r'(may|bao nhieu|so) (mat hang|san pham)'), ('share', r'chiem|ti trong|phan cua|dong gop'),
    ('avg_order', r'trung binh|binh quan'), ('order_count', r'\b(don|bill|order)\b'),
    ('qty_sold', r'ban (ra |duoc |dc )?(bao nhieu|may|bn|bnhiu|tong cong)|so (ly|dia|chai|kg|vien|hop|goi)'),
    ('revenue', r'doanh thu|dthu|revenue|tien|thu ve'),
]
def intent(q):
    f = fold(q)
    for k, pat in INTENTS:
        if re.search(pat, f): return k
    return 'other'

def time_expr(q):
    f = fold(q); tags = []
    for k, pat in [('hom_nay', r'hom nay|hnay|bua nay'), ('hom_qua', r'hom qua|hqua|bua qua'), ('tuan_nay', r'tuan nay'),
                   ('tuan_truoc', r'tuan (truoc|roi|trc)'), ('thang_nay', r'thang nay'), ('thang_truoc', r'thang (truoc|roi|trc)'),
                   ('thang_M_Y', r'(thang|th|t)? ?\d{1,2}(/| nam |-)(\d{4}|\d{2})\b'), ('quy', r'quy \d'), ('dau_nam', r'dau nam|1/1'),
                   ('n_ngay', r'\d+ ngay (gan nhat|ke tu|tiep|toi)'), ('ngay_cu_the', r'\d{1,2}/\d{1,2}/\d{4}'),
                   ('tet', r'\btet\b'), ('cung_ky', r'cung ky'), ('ca', r'ca (sang|chieu)')]:
        if re.search(pat, f): tags.append(k)
    return tags or ['none']

def analyze_questions(split):
    qp = split / 'questions.jsonl'
    if not qp.exists(): return None
    Q = [json.loads(l) for l in open(qp, encoding='utf-8') if l.strip()]
    ap = split / 'answers.jsonl'
    A = {j['id']: j for j in (json.loads(l) for l in open(ap, encoding='utf-8') if l.strip())} if ap.exists() else {}
    rep = {'n_questions': len(Q), 'has_answers': bool(A),
           'per_business': dict(Counter(q['business_id'] for q in Q)),
           'as_of_range': [min(q['as_of_date'] for q in Q), max(q['as_of_date'] for q in Q)],
           'no_diacritics_rate': round(sum(fold(q['question']) == q['question'].lower() for q in Q) / len(Q), 4),
           'len_chars': {'mean': round(sum(len(q['question']) for q in Q) / len(Q), 1), 'max': max(len(q['question']) for q in Q)},
           'time_expr': dict(Counter(t for q in Q for t in time_expr(q['question'])).most_common()),
           }
    ic = defaultdict(Counter)
    for q in Q:
        ic[intent(q['question'])][A[q['id']]['type'] if A else '?'] += 1
    rep['intent'] = {k: {'n': sum(v.values()), 'types': dict(v)} for k, v in sorted(ic.items(), key=lambda kv: -sum(kv[1].values()))}
    rep['examples_per_intent'] = {k: [q['question'] for q in Q if intent(q['question']) == k][:6] for k in ic}
    if A:
        rep['answer_types'] = dict(Counter(a['type'] for a in A.values()).most_common())
        un = [q for q in Q if A.get(q['id'], {}).get('type') == 'unanswerable']
        rep['unanswerable'] = {'n': len(un), 'rate': round(len(un) / len(Q), 4),
                               'per_business': dict(Counter(q['business_id'] for q in un)),
                               'all': [{'biz': q['business_id'], 'as_of': q['as_of_date'], 'q': q['question']} for q in un]}
        rep['list_lengths'] = dict(Counter(len(a['answer']) for a in A.values() if a['type'] == 'list'))
        rep['table_shapes'] = dict(Counter(f"{len(a['answer']['columns'])}c:{','.join(a['answer']['columns'])}" for a in A.values() if a['type'] == 'table'))
        rep['alias_usage'] = sum(1 for a in A.values() if a.get('alias'))
        rep['negative_numbers'] = sum(1 for a in A.values() if a['type'] in ('number', 'percent') and isinstance(a['answer'], (int, float)) and a['answer'] < 0)
        rep['non_integer_numbers'] = sum(1 for a in A.values() if a['type'] == 'number' and isinstance(a['answer'], float) and a['answer'] != int(a['answer']))
    return rep

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('split'); ap.add_argument('-o', '--out', default='report.json')
    args = ap.parse_args(); split = Path(args.split); t0 = time.time()
    data_dir = split / 'data' if (split / 'data').is_dir() else split
    rep = {'split': str(split), 'top_level': sorted(p.name for p in split.iterdir())}
    rb = split / 'rulebook.md'
    if rb.exists():
        txt = rb.read_text(encoding='utf-8'); rep['rulebook'] = {'sha1': hashlib.sha1(txt.encode()).hexdigest(), 'chars': len(txt),
            'headings': [l.strip() for l in txt.splitlines() if l.startswith('#')]}
    cal = split / 'calendar.csv'
    if cal.exists(): c = pd.read_csv(cal); rep['calendar'] = {'rows': len(c), 'range': [c.iloc[0, 0], c.iloc[-1, 0]], 'columns': list(c.columns)}
    rep['businesses'] = {}
    bizs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    for i, b in enumerate(bizs, 1):
        print(f'[{i}/{len(bizs)}] {b.name}', file=sys.stderr, flush=True)
        readme = b / 'README.md'
        rep['businesses'][b.name] = {
            'readme': readme.read_text(encoding='utf-8') if readme.exists() else None,
            'files': [analyze_file(f) for f in sorted(b.iterdir()) if f.is_file() and f.name != 'README.md' and not f.name.startswith('~$')],
            'lock_files': [f.name for f in b.iterdir() if f.name.startswith('~$')],
        }
    rep['format_summary'] = dict(Counter(f['ext'] for v in rep['businesses'].values() for f in v['files']))
    rep['questions'] = analyze_questions(split)
    rep['elapsed_sec'] = round(time.time() - t0, 1)
    Path(args.out).write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    print(f'done -> {args.out} ({Path(args.out).stat().st_size/1024:.0f} KB, {rep["elapsed_sec"]}s)', file=sys.stderr)

if __name__ == '__main__':
    main()
