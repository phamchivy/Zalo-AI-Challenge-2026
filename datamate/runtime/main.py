"""CLI.

  python -m datamate.runtime.main predict --split /data --out /result --config configs/baseline.yaml
  python -m datamate.runtime.main eval    --split /path/train --pred /result/submission.jsonl
  python -m datamate.runtime.main build   --split /path/train            # chỉ build catalog + in tóm tắt mapping
  python -m datamate.runtime.main ask     --split /path/train --id q_0c8f3e07768f   # debug 1 câu (in candidates)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from datamate.core.config import load_config


def _cfg(args):
    over = {}
    if getattr(args, 'no_llm', False):
        over['llm'] = {'enabled': False}
    return load_config(args.config, over)


def cmd_predict(args):
    from datamate.runtime.orchestrator import Orchestrator
    cfg = _cfg(args)
    orch = Orchestrator(cfg, args.split, args.out)
    s = asyncio.run(orch.run(ids=args.id, businesses=args.biz, limit=args.limit, resume=not args.fresh))
    print(json.dumps({k: v for k, v in s.items() if k != 'catalogs'}, ensure_ascii=False, indent=1))
    if args.eval and (Path(args.split) / 'answers.jsonl').exists():
        from datamate.offline.eval import evaluate, format_report
        rep = evaluate(args.split, str(Path(args.out) / 'submission.jsonl'), out_dir=args.out,
                       only_answered_ids=bool(args.id or args.biz or args.limit))
        print(format_report(rep))


def cmd_eval(args):
    from datamate.offline.eval import evaluate, format_report
    rep = evaluate(args.split, args.pred, out_dir=args.report or str(Path(args.pred).parent), only_answered_ids=args.subset)
    print(format_report(rep, top=args.top))


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
        if args.verbose:
            for n, m in c.mapping.tables.items():
                print(f'   - {n} [{m.ttype}] ' + ', '.join(f'{r}={col}' for r, col in m.roles.items()))
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
    args = ap.parse_args(argv)
    {'predict': cmd_predict, 'eval': cmd_eval, 'build': cmd_build, 'ask': cmd_ask}[args.cmd](args)


if __name__ == '__main__':
    sys.exit(main())
