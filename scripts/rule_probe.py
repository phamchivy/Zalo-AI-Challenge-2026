"""v3.3.1 — Kiểm tra tầng A có ĐỌC rulebook hay đang dùng nghĩa cũ của train (nguồn "sai tự tin" khi rulebook đổi).

Ý tưởng: sửa một điều khoản trong BẢN SAO rulebook train (đổi định nghĩa, giữ chủ đề) rồi xem tầng A:
  - nhận ra (nút luật đổi giá trị / thành None / điều khoản thành 'luật lạ')  -> tốt: tính theo luật mới hoặc bỏ trống
  - KHÔNG nhận ra (nút luật y nguyên)                                         -> tầng A sẽ tự tin tính theo nghĩa cũ
Không cần đáp án, không dùng public. Chỉ dùng rulebook train.

  python scripts/rule_probe.py knobs --split $TRAIN                 # vài giây: so nút luật đọc được
  python scripts/rule_probe.py diff  --split $TRAIN --other <public_test>/rulebook.md   # điều khoản đổi / mới
  python scripts/rule_probe.py run   --split $TRAIN --split-file configs/split.yaml --part dev --out runs/rule_probe
        # vài phút / đột biến: chạy tầng A (no_llm) trên các câu dev liên quan, so quyết định với rulebook gốc
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# (tên, đoạn gốc trong rulebook train, đoạn thay, regex chọn câu hỏi liên quan trên câu đã bỏ dấu)
MUTATIONS = [
    ('ban_chay_theo_tien', '| Bán chạy | theo **số lượng quy về đơn vị cơ bản**, không theo số dòng hay số tiền;',
     '| Bán chạy | theo **doanh thu** của mặt hàng, không theo số lượng;', r'ban chay|ban (duoc )?nhieu nhat|hot nhat'),
    ('tuan_tu_chu_nhat', '| Tuần | bắt đầu **thứ Hai**. **Tuần này**: từ thứ Hai của tuần chứa hôm nay đến hết hôm nay. '
     '**Tuần trước, tuần rồi**: **cả** tuần liền trước, thứ Hai đến Chủ nhật |',
     '| Tuần | bắt đầu **Chủ nhật**. **Tuần này**: từ Chủ nhật của tuần chứa hôm nay đến hết hôm nay. '
     '**Tuần trước, tuần rồi**: **cả** tuần liền trước, Chủ nhật đến thứ Bảy |', r'\btuan\b'),
    ('ti_le_huy_mau_khac', 'phần trăm đơn huỷ trong **mọi đơn đặt trong kỳ** (theo ngày đặt, **mọi trạng thái**: cả đơn huỷ, '
     'đơn trả, đơn đang giao)', 'phần trăm đơn huỷ trong **các đơn hoàn thành và đơn huỷ** đặt trong kỳ (theo ngày đặt; '
     '**không** tính đơn trả, đơn đang giao)', r'(ti le|phan tram|%).*huy|huy.*(ti le|phan tram|%)'),
    ('tra_mot_phan_van_tinh', 'Đơn có trả hàng, **kể cả chỉ trả một phần**, là đơn trả hàng: bỏ **cả đơn**.',
     'Đơn trả hàng **một phần** vẫn tính, chỉ bỏ **phần hàng bị trả**.', r'doanh thu|bao nhieu tien|thu ve|ban duoc'),
    ('tru_giam_gia_don', '**Không** trừ giảm giá trên cả đơn (khuyến mãi theo tổng tiền đơn, voucher shop)',
     '**Có** trừ giảm giá trên cả đơn (khuyến mãi theo tổng tiền đơn, voucher shop)', r'doanh thu|bao nhieu tien|thu ve'),
    ('ky_theo_ngay_giao', 'theo **ngày đặt**, không theo ngày giao hay ngày thu tiền',
     'theo **ngày giao**, không theo ngày đặt hay ngày thu tiền', r'doanh thu|so don|bao nhieu don'),
    ('so_don_tinh_don_tra', '| Số đơn | chỉ tính đơn không huỷ, không trả;',
     '| Số đơn | tính **mọi đơn trừ đơn huỷ** (đơn trả vẫn tính);', r'so don|bao nhieu don|may don'),
    ('nhan_vien_theo_so_don', '| Nhân viên bán giỏi nhất | theo **doanh thu** (§1) của nhân viên trong kỳ, không theo số đơn |',
     '| Nhân viên bán giỏi nhất | theo **số đơn** của nhân viên trong kỳ, không theo doanh thu |', r'nhan vien'),
    ('n_ngay_khong_gom_hom_nay', 'N ngày liền nhau **kể cả hôm nay**, ngày cuối là hôm nay',
     'N ngày liền nhau **không gồm hôm nay**, ngày cuối là hôm qua', r'ngay (gan nhat|qua|vua qua)'),
    ('qua_n_gom_n', '| Quá N, hơn N, trên N, dưới N | **không** gồm N. **Từ N trở lên** thì gồm N |',
     '| Quá N, hơn N, trên N, dưới N | **gồm cả** N |', r'\b(qua|hon|tren|duoi) \d'),
    ('gio_ca_khac', 'ca sáng 06:00–13:59, ca chiều 14:00–21:59', 'ca sáng 07:00–12:59, ca chiều 13:00–20:59',
     r'\bca (sang|chieu)\b'),
    ('tet_khac', 'từ ngày dương của **23 tháng Chạp** (năm âm Y − 1) đến **hết** ngày dương của **mùng 10 tháng Giêng**',
     'từ ngày dương của **mùng 1 tháng Giêng** (năm âm Y) đến **hết** ngày dương của **mùng 5 tháng Giêng**', r'\btet\b'),
    ('tuoi_no_tu_han', 'số ngày tính từ **ngày đặt**, không từ hạn thanh toán.',
     'số ngày tính từ **hạn thanh toán**, không từ ngày đặt.', r'tuoi no|no (qua|hon|tren)'),
    ('gia_von_nhap_cuoi', 'Giá vốn của mỗi dòng bán bằng số lượng nhân với **giá vốn ghi trên dòng bán**',
     'Giá vốn của mỗi dòng bán bằng số lượng nhân với **giá nhập lần cuối của mặt hàng**', r'lai gop|loi nhuan|\blai\b'),
]


def _variant(text: str, old: str, new: str):
    return text.replace(old, new, 1) if old in text else None


def cmd_knobs(a):
    from datamate.knowledge.rules import compile_rules
    text = (Path(a.split) / 'rulebook.md').read_text(encoding='utf-8')
    base = compile_rules(text)
    bad = 0
    print(f"{'đột biến':26s} | nút luật đổi | luật lạ mới | kết luận")
    for name, old, new, _ in MUTATIONS:
        t2 = _variant(text, old, new)
        if t2 is None:
            print(f'{name:26s} | (không tìm thấy đoạn gốc trong rulebook — bỏ qua)')
            continue
        rs = compile_rules(t2)
        diff = {k: (base.knobs.get(k), rs.knobs.get(k)) for k in set(base.knobs) | set(rs.knobs)
                if base.knobs.get(k) != rs.knobs.get(k)}
        unk = [u['name'] for u in rs.unknown if u['name'] not in {x['name'] for x in base.unknown}]
        rk = {r.phrase for r in rs.rankings} ^ {r.phrase for r in base.rankings}
        ok = bool(diff or unk or rk)
        bad += not ok
        verdict = 'nhận ra' if ok else '**KHÔNG nhận ra -> tính theo nghĩa cũ**'
        print(f"{name:26s} | {', '.join(f'{k}:{v[0]}->{v[1]}' for k, v in diff.items())[:60] or '-':12s} | "
              f"{', '.join(unk)[:30] or '-':11s} | {verdict}")
    print(f'\n{bad}/{len(MUTATIONS)} đột biến tầng A không nhận ra.')


def cmd_diff(a):
    """So rulebook train với rulebook khác (vd public — đó là đầu vào được phát, không phải đáp án): điều khoản nào ĐỔI,
    và tầng A có nút luật kiểm được điều khoản đó không. Điều khoản đổi mà không nút nào kiểm = tầng A dùng nghĩa cũ."""
    from datamate.knowledge.rules import KNOBS, compile_rules, parse_statements
    t1 = (Path(a.split) / 'rulebook.md').read_text(encoding='utf-8')
    t2 = Path(a.other).read_text(encoding='utf-8')
    s1 = {(s.section.split(' > ')[0][:2], s.name): s for s in parse_statements(t1) if s.name}
    s2 = {(s.section.split(' > ')[0][:2], s.name): s for s in parse_statements(t2) if s.name}
    r2 = compile_rules(t2)
    unk = {u['name'] for u in r2.unknown}

    def knob_hit(st):
        return [k.key for k in KNOBS if (not k.section or re.search(k.section, st.section))
                and (not k.name or re.search(k.name, st.name)) and (k.name or k.text)
                and all(re.search(t, st.text) for t in k.text)]
    changed = [(k, s2[k]) for k in s2 if k in s1 and s1[k].text != s2[k].text]
    new = [(k, s2[k]) for k in s2 if k not in s1]
    print(f'{len(changed)} điều khoản ĐỔI nội dung, {len(new)} điều khoản MỚI (so với rulebook train)\n')
    for title, rows in (('ĐỔI', changed), ('MỚI', new)):
        for k, st in rows:
            kh = knob_hit(st)
            state = ('luật lạ -> tầng A bỏ trống' if st.name in unk else
                     f'nút {kh} kiểm' if kh else '**không nút nào kiểm -> tầng A dùng nghĩa cũ / bỏ qua**')
            print(f'[{title}] §{k[0].strip(". ")} {st.name[:45]:45s} | {state}')
            if title == 'ĐỔI' and a.verbose:
                print(f'      train : {s1[k].text[:160]}\n      khác  : {st.text[:160]}')


def cmd_run(a):
    from datamate.core.textutil import fold
    split = Path(a.split)
    text = (split / 'rulebook.md').read_text(encoding='utf-8')
    qs = [json.loads(l) for l in (split / 'questions.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    if a.split_file:
        from datamate.offline.split import load_part
        shops = load_part(a.split_file, a.part or 'dev')
        qs = [q for q in qs if q['business_id'] in shops]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def run(sp: Path, ids: list[str], o: Path):
        idf = o.with_suffix('.ids')
        idf.write_text('\n'.join(ids), encoding='utf-8')
        subprocess.run([sys.executable, '-m', 'datamate.runtime.main', 'predict', '--split', str(sp), '--ids-file', str(idf),
                        '--out', str(o), '--config', a.config, '--fresh'], check=True, cwd=str(ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        res = {}
        for f in (o / 'logs').glob('*.json'):
            t = json.loads(f.read_text(encoding='utf-8'))
            res[t['qid']] = (t['decision']['type'], json.dumps(t['decision'].get('answer'), ensure_ascii=False))
        return res

    rel = {name: [q['id'] for q in qs if re.search(kw, fold(q['question']))][:a.max_q] for name, _, _, kw in MUTATIONS}
    all_ids = sorted({i for v in rel.values() for i in v})
    print(f'chạy rulebook gốc trên {len(all_ids)} câu dev liên quan...')
    base = run(split, all_ids, out / 'base')
    print(f"\n{'đột biến':26s} {'n':>4s} {'đáp án y nguyên':>16s} {'đổi đáp án':>11s} {'thành bỏ trống':>15s}")
    for name, old, new, _ in MUTATIONS:
        t2 = _variant(text, old, new)
        ids = [i for i in rel[name] if base.get(i, ('abstain',))[0] != 'abstain']
        if t2 is None or not ids:
            print(f'{name:26s} (bỏ qua: không có đoạn gốc hoặc không có câu tầng A trả lời)')
            continue
        vs = out / f'split_{name}'
        vs.mkdir(exist_ok=True)
        for n in ('data', 'questions.jsonl', 'answers.jsonl', 'calendar.csv', 'README.md'):
            if (split / n).exists() and not (vs / n).exists():
                os.symlink((split / n).resolve(), vs / n)
        (vs / 'rulebook.md').write_text(t2, encoding='utf-8')
        r = run(vs, ids, out / f'run_{name}')
        c = Counter('same' if r.get(i) == base[i] else 'abstain' if r.get(i, ('abstain',))[0] == 'abstain' else 'changed'
                    for i in ids)
        print(f"{name:26s} {len(ids):4d} {c['same']:16d} {c['changed']:11d} {c['abstain']:15d}")
    print('\n"đáp án y nguyên" nhiều = tầng A không phản ứng với luật đã đổi (trừ khi dữ liệu shop không bị luật đó ảnh hưởng).')


def main(argv=None):
    ap = argparse.ArgumentParser('rule_probe')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('knobs')
    p.add_argument('--split', required=True)
    p = sub.add_parser('diff')
    p.add_argument('--split', required=True, help='split train (lấy rulebook.md)')
    p.add_argument('--other', required=True, help='rulebook.md khác để so (vd public_test/rulebook.md)')
    p.add_argument('-v', '--verbose', action='store_true')
    p = sub.add_parser('run')
    p.add_argument('--split', required=True)
    p.add_argument('--split-file', default=None)
    p.add_argument('--part', default='dev')
    p.add_argument('--config', default='configs/no_llm.yaml')
    p.add_argument('--max-q', type=int, default=40)
    p.add_argument('--out', required=True)
    a = ap.parse_args(argv)
    {'knobs': cmd_knobs, 'run': cmd_run, 'diff': cmd_diff}[a.cmd](a)


if __name__ == '__main__':
    main()
