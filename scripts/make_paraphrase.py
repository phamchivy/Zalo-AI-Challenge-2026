"""v3.2.0a — Bộ "câu hỏi viết lại" từ train: đo và hiệu chỉnh độ tin cậy của tầng A trên CÁCH HỎI CHƯA GẶP.

Vì sao: regex của tầng A được viết khi nhìn chính câu train, nên hiệu chỉnh trên train cho độ tin cậy ~1 với mọi intent
(calibrate luôn chọn ngưỡng 0). Câu viết lại giữ nguyên nghĩa + đáp án nhưng đổi cách hỏi -> độ chính xác đo trên đó
gần với public / private hơn.

Dùng ĐỂ ĐO VÀ HIỆU CHỈNH, KHÔNG ĐỂ VÁ: không thêm regex theo câu viết lại cụ thể.

  make  : chọn N câu dev (ngẫu nhiên, cố định seed), LLM viết lại mỗi câu theo 1 kiểu (xoay vòng các kiểu), lọc câu
          mất số (ngày, số lượng, ngưỡng) -> ghi một split tổng hợp: <out>/{questions,answers}.jsonl + data/rulebook/calendar
          trỏ về train. Chạy tầng A + calibrate trên split này như split thường.
  report: so tầng A trên câu gốc và câu viết lại (đúng / sai / bỏ theo kiểu viết lại và theo intent).

Ví dụ (máy công ty, GLM):
  python scripts/make_paraphrase.py make --split $TRAIN --split-file configs/split.yaml --part dev \\
      --config configs/qa_glm.yaml --n 400 --out runs/para_dev
  python -m datamate.runtime.main predict --split runs/para_dev --out runs/para_dev_A --config configs/no_llm.yaml --fresh --eval
  python -m datamate.runtime.main calibrate --split runs/para_dev --run runs/para_dev_A --out configs/calibration_para.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STYLES = {
    'khau_ngu': 'văn nói, khẩu ngữ / phương ngữ của chủ shop nhỏ (vd "nhiêu", "rứa", "dzậy", "hông", "tiệm mình")',
    'viet_tat': 'viết tắt kiểu chat (vd "dthu", "đh", "sl", "th3", "T4/26", "kh", "sp") nhưng vẫn đọc hiểu được',
    'go_sai': 'gõ vội: thiếu dấu hoặc sai dấu ở vài từ, không viết hoa, có thể dính chữ nhẹ',
    'dien_dat': 'diễn đạt khác hẳn: đổi trật tự, đổi cấu trúc câu, dùng từ đồng nghĩa (không đổi số liệu)',
    'dai_dong': 'câu dài hơn, thêm lời dẫn / ngữ cảnh không làm đổi nghĩa (vd "cho mình hỏi chút", "để chốt sổ")',
}

SYS = ('Bạn viết lại câu hỏi của chủ cửa hàng về dữ liệu bán hàng. GIỮ NGUYÊN 100% NGHĨA: cùng đại lượng được hỏi, cùng '
       'khoảng thời gian, cùng điều kiện lọc, cùng mặt hàng / khách / nhóm, cùng số lượng phần tử (top N), cùng kiểu đáp '
       'án. Giữ mọi con số, ngày tháng, tên riêng (có thể viết tắt ngày tháng nhưng không bỏ năm nếu câu gốc có năm). '
       'Không thêm điều kiện mới, không bỏ điều kiện. Chỉ trả về JSON {"q": "<câu viết lại>"}.')


def _jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def _nums(s: str) -> set[str]:
    return {x.lstrip('0') or '0' for x in re.findall(r'\d+', s)}


def cmd_make(a):
    from datamate.core.config import load_config
    from datamate.llm.json_chat import JSONChat
    from datamate.offline.split import load_part
    split = Path(a.split)
    qs = _jsonl(split / 'questions.jsonl')
    gold = {g['id']: g for g in _jsonl(split / 'answers.jsonl')}
    if a.split_file:
        shops = load_part(a.split_file, a.part or 'dev')
        if (a.part or 'dev') == 'holdout' and not a.allow_holdout:
            raise SystemExit('holdout chỉ dùng để KIỂM (thêm --allow-holdout), không hiệu chỉnh trên đó')
        qs = [q for q in qs if q['business_id'] in shops]
    qs = [q for q in qs if q['id'] in gold]
    random.Random(a.seed).shuffle(qs)
    qs = qs[:a.n]
    cfg = load_config(a.config)
    mcfg = cfg.mapping_llm.model_dump()
    mcfg['cache'] = True
    client = JSONChat(mcfg, cfg.cache_dir)
    styles = list(STYLES)

    def one(i_q):
        i, q = i_q
        st = styles[i % len(styles)]
        user = f'Kiểu viết lại: {STYLES[st]}\nCâu gốc: {q["question"]}'

        def ok(o):
            if not isinstance(o, dict) or not isinstance(o.get('q'), str) or len(o['q'].strip()) < 5:
                raise ValueError('thiếu trường "q"')
            if o['q'].strip() == q['question'].strip():
                raise ValueError('câu viết lại trùng câu gốc')
            miss = _nums(q['question']) - _nums(o['q'])
            if miss:
                raise ValueError(f'mất số {sorted(miss)}: phải giữ mọi con số / ngày tháng của câu gốc')
            return o['q'].strip()
        try:
            return q, st, client.ask(SYS, user, None, ok)
        except Exception as e:  # noqa: BLE001
            return q, st, None

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(one, enumerate(qs)))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ('data', 'rulebook.md', 'calendar.csv', 'README.md'):
        src, dst = split / name, out / name
        if src.exists() and not dst.exists():
            os.symlink(src.resolve(), dst)
    nq, ng, meta, bad = [], [], [], Counter()
    for q, st, para in res:
        if not para:
            bad[st] += 1
            continue
        pid = f"{q['id']}__{st}"
        nq.append({**q, 'id': pid, 'question': para})
        ng.append({**gold[q['id']], 'id': pid})
        meta.append({'id': pid, 'orig_id': q['id'], 'style': st, 'orig': q['question'], 'para': para})
    for fn, rows in (('questions.jsonl', nq), ('answers.jsonl', ng), ('paraphrase_meta.jsonl', meta)):
        (out / fn).write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    print(f'{len(nq)} câu viết lại ({len(qs) - len(nq)} bị loại: {dict(bad)}), theo kiểu: '
          f'{dict(Counter(m["style"] for m in meta))}; LLM gọi {client.calls}, lỗi {client.failures}')
    print(f'đã ghi {out}/questions.jsonl, answers.jsonl, paraphrase_meta.jsonl')
    for m in meta[:5]:
        print(f"  [{m['style']}] {m['orig']}\n      -> {m['para']}")


def cmd_report(a):
    """Tầng A trên câu gốc (run_orig) và câu viết lại (run_para): bao nhiêu câu đúng ở gốc nhưng sai / bỏ khi viết lại."""
    from datamate.offline.scorer import score_one
    from datamate.output.formatter import normalize_answer
    para = Path(a.para_split)
    meta = {m['id']: m for m in _jsonl(para / 'paraphrase_meta.jsonl')}
    gold = {g['id']: g for g in _jsonl(para / 'answers.jsonl')}

    def outcome(run, qid, g):
        lf = Path(run) / 'logs' / f'{qid}.json'
        if not lf.exists():
            return None, None
        tr = json.loads(lf.read_text(encoding='utf-8'))
        d = tr['decision']
        if d['type'] in ('abstain',):
            return 0, tr.get('intent')
        t, ans = normalize_answer(d['type'], d.get('answer'))
        return score_one({'type': t, 'answer': ans}, g), tr.get('intent')
    by_style, by_int, flips = defaultdict(Counter), defaultdict(Counter), Counter()
    for pid, m in meta.items():
        g = gold[pid]
        so, io = outcome(a.run_orig, m['orig_id'], {**g, 'id': m['orig_id']})
        sp, ip = outcome(a.run_para, pid, g)
        if so is None or sp is None:
            continue
        lab = {1: 'đúng', -1: 'sai', 0: 'bỏ'}
        by_style[m['style']][lab[sp]] += 1
        by_int[io][lab[sp]] += 1
        flips[f'{lab[so]} -> {lab[sp]}'] += 1
        if io != ip:
            flips['intent đổi'] += 1
    print('Gốc -> viết lại:', dict(flips))
    print('\nTheo kiểu viết lại (câu viết lại):')
    for s, c in sorted(by_style.items()):
        n = sum(c.values())
        print(f"  {s:10s} n={n:4d} đúng={c['đúng']:4d} sai={c['sai']:4d} bỏ={c['bỏ']:4d}  PA={(c['đúng'] - c['sai']) / n:+.3f}")
    print('\nTheo intent của câu gốc (sai nhiều nhất trước):')
    for i, c in sorted(by_int.items(), key=lambda kv: -kv[1]['sai'])[:25]:
        n = sum(c.values())
        print(f"  {str(i):18s} n={n:4d} đúng={c['đúng']:4d} sai={c['sai']:4d} bỏ={c['bỏ']:4d}")


def main(argv=None):
    ap = argparse.ArgumentParser('make_paraphrase')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make')
    p.add_argument('--split', required=True)
    p.add_argument('--split-file', default=None)
    p.add_argument('--part', choices=['dev', 'holdout'], default=None)
    p.add_argument('--allow-holdout', action='store_true')
    p.add_argument('--config', default='configs/qa_glm.yaml', help='lấy khối mapping_llm (địa chỉ / model / khoá)')
    p.add_argument('--n', type=int, default=400)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--workers', type=int, default=32)
    p.add_argument('--out', required=True)
    p = sub.add_parser('report')
    p.add_argument('--para-split', required=True)
    p.add_argument('--run-orig', required=True, help='lần chạy tầng A trên split gốc (có logs của câu gốc)')
    p.add_argument('--run-para', required=True, help='lần chạy tầng A trên split viết lại')
    a = ap.parse_args(argv)
    {'make': cmd_make, 'report': cmd_report}[a.cmd](a)


if __name__ == '__main__':
    main()
