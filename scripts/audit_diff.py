"""v2.3.1 — Kiểm toán tầng A bằng agent LLM độc lập (không cần đáp án vàng, không tốn lượt nộp để dò).

Chạy agent trên các câu tầng A ĐÃ trả lời (llm_skip_support > 1 ép mọi câu sang pha 2; llm_keep_tier_a giữ đáp án
tầng A trong submission). Agent không thấy đáp án tầng A. Script so đáp án tầng A với các lời giải của agent:

  agree    : >= 1 lời giải agent khớp tầng A
  suspect  : >= min_votes lời giải agent cùng một đáp án KHÁC tầng A và không lời giải nào khớp tầng A
  weak     : agent có đáp án khác nhưng không đủ phiếu
  no_agent : agent không ra đáp án hợp lệ

Dùng:
  python scripts/audit_diff.py <run_audit>/logs <submission_tầng_A.jsonl> <out_dir> [--min-votes 2]
Ra: <out_dir>/audit.tsv (từng câu), audit_summary.txt (theo intent/shop), suspect_ids.txt,
    probe_drop_suspect.jsonl (submission tầng A với câu suspect -> abstain, để nộp đo nhóm).
Điểm probe - điểm gốc = (số suspect sai - số suspect đúng) / tổng câu.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path

from datamate.offline.scorer import score_one
from datamate.output.formatter import answer_key, normalize_answer


def _key(t, a):
    try:
        t2, a2 = normalize_answer(t, a)
        return None if t2 in (None, 'abstain') else (t2, a2, answer_key(t2, a2))
    except Exception:  # noqa: BLE001
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs')
    ap.add_argument('submission')
    ap.add_argument('out')
    ap.add_argument('--min-votes', type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sub = {}
    for line in open(a.submission, encoding='utf-8-sig'):
        if line.strip():
            r = json.loads(line)
            sub[r['id']] = r
    rows, by = [], defaultdict(Counter)
    for f in sorted(glob.glob(f'{a.logs}/*.json')):
        d = json.load(open(f, encoding='utf-8'))
        s = sub.get(d['qid'])
        if s is None or s.get('type') == 'abstain':
            continue
        gold = {'type': s['type'], 'answer': s.get('answer')}            # coi đáp án tầng A là "vàng" để so khớp có dung sai
        votes = Counter()
        rep, agree, n_ok = {}, 0, 0
        for c in d['candidates']:
            if c['strategy'] in ('rule_plan', 'guard') or c['status'] != 'ok' or not c.get('answer_type'):
                continue
            k = _key(c['answer_type'], c['answer'])
            if k is None:
                continue
            n_ok += 1
            if score_one({'type': k[0], 'answer': k[1]}, gold) == 1:
                agree += 1
            else:
                votes[k[2]] += 1
                rep[k[2]] = (k[0], k[1])
        top_k, top_n = (votes.most_common(1)[0] if votes else (None, 0))
        if n_ok == 0:
            st = 'no_agent'
        elif agree:
            st = 'agree'
        elif top_n >= a.min_votes:
            st = 'suspect'
        else:
            st = 'weak'
        alt = rep.get(top_k)
        rows.append({'id': d['qid'], 'biz': d['business_id'], 'intent': d.get('intent') or '', 'status': st,
                     'agree': agree, 'n_agent': n_ok, 'alt_votes': top_n, 'tierA': json.dumps(s.get('answer'), ensure_ascii=False)[:120],
                     'alt': json.dumps(alt, ensure_ascii=False)[:120] if alt else '', 'q': d['question'].replace('\t', ' ')})
        by[d.get('intent') or ''][st] += 1
        by['__all__'][st] += 1
    cols = ['status', 'id', 'biz', 'intent', 'agree', 'n_agent', 'alt_votes', 'tierA', 'alt', 'q']
    order = {'suspect': 0, 'weak': 1, 'no_agent': 2, 'agree': 3}
    rows.sort(key=lambda r: (order[r['status']], r['intent'], r['biz']))
    with open(out / 'audit.tsv', 'w', encoding='utf-8') as fo:
        fo.write('\t'.join(cols) + '\n')
        for r in rows:
            fo.write('\t'.join(str(r[c]) for c in cols) + '\n')
    sus = [r['id'] for r in rows if r['status'] == 'suspect']
    (out / 'suspect_ids.txt').write_text('\n'.join(sus) + '\n', encoding='utf-8')
    with open(out / 'probe_drop_suspect.jsonl', 'w', encoding='utf-8') as fo:
        for qid, r in sub.items():
            fo.write(json.dumps({'id': qid, 'type': 'abstain'} if qid in set(sus) else r, ensure_ascii=False) + '\n')
    lines = [f'{"intent":<24} {"n":>4} {"agree":>6} {"suspect":>8} {"weak":>5} {"no_agent":>8}']
    for k, c in sorted(by.items(), key=lambda x: -sum(x[1].values())):
        lines.append(f'{k:<24} {sum(c.values()):>4} {c["agree"]:>6} {c["suspect"]:>8} {c["weak"]:>5} {c["no_agent"]:>8}')
    (out / 'audit_summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:1] + [l for l in lines if l.startswith('__all__')]))
    print(f'suspect: {len(sus)} -> {out}/suspect_ids.txt, {out}/probe_drop_suspect.jsonl')


if __name__ == '__main__':
    main()
