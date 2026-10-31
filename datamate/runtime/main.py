"""CLI.

  python -m datamate.runtime.main predict --split /data --out /result --config configs/baseline.yaml
  python -m datamate.runtime.main eval    --split /path/train --pred /result/submission.jsonl
  python -m datamate.runtime.main build   --split /path/train            # chỉ build catalog + in tóm tắt mapping
  python -m datamate.runtime.main ask     --split /path/train --id q_0c8f3e07768f   # debug 1 câu (in candidates)
  python -m datamate.runtime.main rules   --split /path/public_test            # in luật đọc được từ rulebook (rules.json)
  python -m datamate.runtime.main calibrate --split /path/train --run runs/raw --out configs/calibration.yaml
  python -m datamate.runtime.main make-split --split /path/train --seen biz_101 biz_102 --out configs/split.yaml
  python -m datamate.runtime.main predict --split /path/train --split-file configs/split.yaml --part holdout --eval ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import Counter
from pathlib import Path

from datamate.core.config import load_config


def _cfg(args):
    over = {}
    if getattr(args, 'no_llm', False):
        over['llm'] = {'enabled': False}
    return load_config(args.config, over)


def _part(args) -> tuple[set | None, bool]:
    """(tập shop của --part, có phải holdout không)."""
    from datamate.offline.split import load_part
    sf = getattr(args, 'split_file', None)
    if not sf:
        return None, False
    part = getattr(args, 'part', None) or 'dev'
    return load_part(sf, part), part == 'holdout'


def cmd_predict(args):
    from datamate.runtime.orchestrator import Orchestrator
    cfg = _cfg(args)
    shops, sealed = _part(args)
    biz = args.biz
    if shops is not None:
        biz = sorted(shops & set(biz)) if biz else sorted(shops)
    orch = Orchestrator(cfg, args.split, args.out)
    s = asyncio.run(orch.run(ids=args.id, businesses=biz, limit=args.limit, resume=not args.fresh))
    print(json.dumps({k: v for k, v in s.items() if k != 'catalogs'}, ensure_ascii=False, indent=1))
    if args.eval and (Path(args.split) / 'answers.jsonl').exists():
        from datamate.offline.eval import evaluate, format_report
        rep = evaluate(args.split, str(Path(args.out) / 'submission.jsonl'), out_dir=args.out,
                       only_answered_ids=bool(args.id or args.biz or args.limit), businesses=shops, sealed=sealed)
        print(format_report(rep))


def cmd_eval(args):
    from datamate.offline.eval import evaluate, format_report
    shops, sealed = _part(args)
    rep = evaluate(args.split, args.pred, out_dir=args.report or str(Path(args.pred).parent), only_answered_ids=args.subset,
                   businesses=shops, sealed=sealed)
    print(format_report(rep, top=args.top))


def cmd_split_info(args):
    from datamate.offline.split import split_info
    info = split_info(args.split)
    fam = Counter((v['industry'], v['family']) for v in info.values())
    for b, v in info.items():
        print(f"{b}: câu={v['questions']:4d} ngành={v['industry']:10s} họ={v['family']:18s} files={v['files'][:5]}")
    print('\nnhóm (ngành, họ dữ liệu):', dict(fam))


def cmd_make_split(args):
    from datamate.offline.split import make_split
    seen = [s for x in (args.seen or []) for s in x.replace(',', ' ').split()]
    doc = make_split(args.split, seen, n_holdout=args.n_holdout, out=args.out)
    print(f"holdout ({len(doc['holdout'])}): {doc['holdout']}")
    print(f"dev ({len(doc['dev'])}), trong đó đã xem: {len(doc['seen'])}")
    print(f"đã ghi {args.out}")


def cmd_build(args):
    from datamate.data.catalog import build_all
    cfg = _cfg(args)
    data = Path(args.split) / 'data'
    biz = args.biz or sorted(p.name for p in data.iterdir() if p.is_dir())
    t0 = time.time()
    cats = build_all(data, biz, cfg.cache_dir, cfg.workers)
    for b, c in cats.items():
        s = c.summary()
        print(f"{b}: lines={s['lines']} orders={s['orders']} caps={s['caps']} range={s['date_min']}..{s['date_max']} notes={s['notes'][:4]}")
        h = {k: v for k, v in (s.get('health') or {}).items() if v}
        if h:
            print(f'   health: {h}')
        if args.verbose:
            for n, m in c.mapping.tables.items():
                unc = f"  [không chắc: {', '.join(m.uncertain)}]" if getattr(m, 'uncertain', None) else ''
                print(f'   - {n} [{m.ttype}] ' + ', '.join(f'{r}={col}' for r, col in m.roles.items()) + unc)
    print(f'built {len(cats)} in {time.time() - t0:.1f}s')


def cmd_ask(args):
    from datamate.runtime.orchestrator import Orchestrator
    cfg = _cfg(args)
    out = args.out or '/tmp/datamate_ask'
    orch = Orchestrator(cfg, args.split, out)
    asyncio.run(orch.run(ids=args.id, resume=False))
    for i in args.id:
        p = Path(out) / 'logs' / f'{i}.json'
        if p.exists():
            d = json.loads(p.read_text(encoding='utf-8'))
            print(json.dumps({k: d[k] for k in ('question', 'as_of_date', 'intent', 'decision')}, ensure_ascii=False, indent=1))
            for c in d['candidates']:
                print(f"  - {c['strategy']:10s} {c['status']:7s} w={c['weight']:.2f} {c.get('answer_type')} {str(c.get('answer'))[:120]} {str(c.get('error') or '')[:160]}")
                if args.show_program and c.get('program'):
                    print('    ' + c['program'].replace('\n', '\n    ')[:3000])


def cmd_rules(args):
    from datamate.knowledge.base import KnowledgeBase
    kb = KnowledgeBase.load(args.split)
    rs = kb.rules
    if args.out:
        Path(args.out).write_text(rs.to_json(), encoding='utf-8')
    miss = [k for k, v in rs.knobs.items() if v is None]
    print(f'knobs: {sum(v is not None for v in rs.knobs.values())}/{len(rs.knobs)} xác nhận; chưa xác nhận: {miss}')
    print(f'xếp hạng theo định nghĩa: {[(r.phrase, r.entity, r.metric) for r in rs.rankings]}')
    print(f'luật lạ (câu hỏi chạm vào -> tầng A bỏ trống): {len(rs.unknown)}')
    for u in rs.unknown:
        print(f"   - [{u['section']}] {u['name']}  keys={u['keys']}")


def cmd_calibrate(args):
    from datamate.offline.calibrate import calibrate
    shops, sealed = _part(args)
    if sealed:
        raise SystemExit('calibrate chỉ chạy trên dev, không hiệu chỉnh trên holdout')
    doc = calibrate(args.split, args.run, args.out, folds=args.folds, min_n=args.min_n, businesses=shops)
    print(f"{doc['questions']} câu; PA out-of-fold theo ngưỡng min_score:")
    for t, r in doc['oof_by_threshold'].items():
        print(f"  min_score={t:>5}: PA={r['PA']:+.4f} đúng={r['correct']} sai={r['wrong']} precision={r['precision']}")
    print(f"=> đề xuất decision.min_score = {doc['recommended_min_score']}  (ghi vào configs/no_llm.yaml)")
    print(f'đã ghi {args.out}')


def main(argv=None):
    ap = argparse.ArgumentParser('datamate')
    sub = ap.add_subparsers(dest='cmd', required=True)
    for name in ('predict', 'build', 'ask'):
        p = sub.add_parser(name)
        p.add_argument('--split', required=True)
        p.add_argument('--config', default=None)
        p.add_argument('--no-llm', action='store_true')
        p.add_argument('--biz', nargs='*')
        if name in ('predict', 'ask'):
            p.add_argument('--out', default=None if name == 'ask' else '/result')
            p.add_argument('--id', nargs='*')
        if name == 'predict':
            _split_args(p)
            p.add_argument('--limit', type=int)
            p.add_argument('--fresh', action='store_true', help='không resume, chạy lại từ đầu')
            p.add_argument('--eval', action='store_true', help='chấm ngay nếu split có answers.jsonl')
        if name == 'build':
            p.add_argument('-v', '--verbose', action='store_true')
        if name == 'ask':
            p.add_argument('--show-program', action='store_true')
    p = sub.add_parser('eval')
    p.add_argument('--split', required=True)
    p.add_argument('--pred', required=True)
    p.add_argument('--report', default=None)
    p.add_argument('--subset', action='store_true', help='chỉ chấm các id có trong pred')
    p.add_argument('--top', type=int, default=20)
    _split_args(p)
    p = sub.add_parser('split-info')
    p.add_argument('--split', required=True)
    p = sub.add_parser('make-split')
    p.add_argument('--split', required=True, help='split train')
    p.add_argument('--seen', nargs='*', help='shop đã xem lỗi (luôn vào dev), vd biz_101 biz_102 hoặc "biz_101,biz_102"')
    p.add_argument('--n-holdout', type=int, default=8)
    p.add_argument('--out', default='configs/split.yaml')
    p = sub.add_parser('rules')
    p.add_argument('--split', required=True)
    p.add_argument('--out', default=None)
    p = sub.add_parser('calibrate')
    p.add_argument('--split', required=True, help='split có answers.jsonl (train)')
    p.add_argument('--run', required=True, help='thư mục kết quả của lần chạy configs/no_llm_raw.yaml')
    p.add_argument('--out', default='configs/calibration.yaml')
    p.add_argument('--folds', type=int, default=5)
    p.add_argument('--min-n', type=int, default=8)
    _split_args(p)
    args = ap.parse_args(argv)
    {'predict': cmd_predict, 'eval': cmd_eval, 'build': cmd_build, 'ask': cmd_ask, 'rules': cmd_rules,
     'calibrate': cmd_calibrate, 'split-info': cmd_split_info, 'make-split': cmd_make_split}[args.cmd](args)


def _split_args(p):
    p.add_argument('--split-file', default=None, help='configs/split.yaml (sinh bởi make-split)')
    p.add_argument('--part', choices=['dev', 'holdout'], default=None, help='mặc định dev khi có --split-file')


if __name__ == '__main__':
    sys.exit(main())
