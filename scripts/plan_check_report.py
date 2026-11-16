"""v3.3.3 — báo cáo bước LLM kiểm tra kế hoạch tầng A.

  python scripts/plan_check_report.py RUN_DIR [--split SPLIT_DIR] [--show 40]

Không nhãn (public): số câu được kiểm tra / bị bác, bác vì phần nào, danh sách câu bị bác kèm lý do.
Có nhãn (train / dev / dev viết lại: SPLIT_DIR có answers.jsonl):
  bảng tầng A (đúng / sai) x (giữ / bác) -> lời lỗ nếu câu bị bác chỉ bỏ trống,
  và lời lỗ thật sau khi LLM giải lại các câu bị bác (đáp án cuối so với đáp án tầng A).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datamate.offline.scorer import score_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--split', default=None)
    ap.add_argument('--show', type=int, default=40)
    a = ap.parse_args()
    run = Path(a.run)
    gold = {}
    if a.split and (Path(a.split) / 'answers.jsonl').exists():
        for l in (Path(a.split) / 'answers.jsonl').read_text(encoding='utf-8').splitlines():
            if l.strip():
                g = json.loads(l)
                gold[g['id']] = g
    traces = []
    for p in sorted((run / 'logs').glob('*.json')):
        t = json.loads(p.read_text(encoding='utf-8'))
        if t.get('plan_check'):
            traces.append(t)
    rej = [t for t in traces if t['plan_check'].get('reject')]
    parts = Counter(e['phan'] for t in rej for e in t['plan_check'].get('errors', [])[:1])
    print(f'Kiểm tra {len(traces)} câu tầng A đã trả lời: bác {len(rej)} ({100 * len(rej) / max(len(traces), 1):.1f}%)')
    print('  bác vì: ' + ', '.join(f'{k}={v}' for k, v in parts.most_common()))
    near = sum(1 for t in traces if not t['plan_check']['reject'] and t['plan_check'].get('n_reject', 0) > 0)
    print(f'  chỉ 1 lần bác (chưa đủ phiếu, vẫn giữ): {near}')
    if gold:
        tab = Counter()
        fin = Counter()
        for t in traces:
            g = gold.get(t['qid'])
            if g is None:
                continue
            sa = score_one(t['plan_check']['tier_a'], g)
            r = t['plan_check']['reject']
            tab[(sa, r)] += 1
            if r:
                d = t.get('decision') or {}
                fin[(sa, score_one({'type': d.get('type'), 'answer': d.get('answer')}, g))] += 1
        n = sum(tab.values())
        print(f'\nCó nhãn ({n} câu):          giữ   bác')
        print(f'  tầng A ĐÚNG          {tab[(1, False)]:>5} {tab[(1, True)]:>5}')
        print(f'  tầng A SAI           {tab[(-1, False)]:>5} {tab[(-1, True)]:>5}')
        if tab[(0, False)] + tab[(0, True)]:
            print(f'  (tầng A bỏ trống)    {tab[(0, False)]:>5} {tab[(0, True)]:>5}')
        gain = tab[(-1, True)] - tab[(1, True)]
        print(f'  Nếu câu bị bác chỉ bỏ trống: {gain:+d} điểm ({gain / max(len(gold), 1):+.4f} PA trên {len(gold)} câu)')
        if fin:
            real = sum((s1 - s0) * c for (s0, s1), c in fin.items())
            print(f'  Sau khi LLM giải lại câu bị bác: {real:+d} điểm ({real / max(len(gold), 1):+.4f} PA)')
            for (s0, s1), c in sorted(fin.items()):
                print(f'    tầng A {s0:+d} -> cuối {s1:+d}: {c}')
    print(f'\nCâu bị bác (tối đa {a.show}):')
    for t in rej[:a.show]:
        tag = ''
        if gold.get(t['qid']):
            tag = {1: '[A ĐÚNG] ', -1: '[A SAI] ', 0: '[A bỏ] '}[score_one(t['plan_check']['tier_a'], gold[t['qid']])]
        e = (t['plan_check'].get('errors') or [{}])[0]
        print(f"- {tag}{t['qid']}: {t['question']}\n    [{e.get('phan')}] câu hỏi: {e.get('cau_hoi_noi')} | máy: {e.get('ke_hoach_lam')}"
              f"\n    tầng A = {json.dumps(t['plan_check']['tier_a'].get('answer'), ensure_ascii=False)[:80]}"
              f" -> cuối = {(t.get('decision') or {}).get('type')} {json.dumps((t.get('decision') or {}).get('answer'), ensure_ascii=False)[:60]}")


if __name__ == '__main__':
    main()
