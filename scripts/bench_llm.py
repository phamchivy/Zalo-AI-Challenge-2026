"""Đo thông lượng vLLM với tải giống agent: tiền tố dài dùng chung (system + rulebook + shop) + câu hỏi ngắn khác nhau.

  python scripts/bench_llm.py --split /workspace/data/public_test --concurrency 16 64 128 --n 192
In: token sinh / giây, thời gian trả lời trung bình, và ước tính số câu agent chạy được trong 5 giờ.
"""
import argparse
import asyncio
import os
import random
import time
from pathlib import Path

import httpx


async def one(client, url, model, prefix, i, max_tokens):
    msgs = [{'role': 'system', 'content': prefix},
            {'role': 'user', 'content': f'Câu {i}: viết một đoạn code pandas khoảng 15 dòng tính doanh thu tháng {i % 12 + 1} '
                                        f'theo từng nhóm hàng, giải thích ngắn trước khi viết code.'}]
    t0 = time.time()
    r = await client.post(url, json={'model': model, 'messages': msgs, 'max_tokens': max_tokens, 'temperature': 0.6,
                                     'chat_template_kwargs': {'enable_thinking': False}})
    r.raise_for_status()
    u = r.json().get('usage', {})
    return time.time() - t0, u.get('prompt_tokens', 0), u.get('completion_tokens', 0)


async def run(args):
    urls = (os.environ.get('DATAMATE_LLM_URLS') or 'http://127.0.0.1:8000/v1').split(',')
    rb = Path(args.split) / 'rulebook.md'
    prefix = (rb.read_text(encoding='utf-8') if rb.exists() else 'Luật bán hàng. ' * 1500)
    prefix = (prefix + '\n') * max(1, args.prefix_repeat)
    async with httpx.AsyncClient(timeout=600) as client:
        for c in args.concurrency:
            sem = asyncio.Semaphore(c)

            async def guarded(i):
                async with sem:
                    return await one(client, urls[i % len(urls)].rstrip('/') + '/chat/completions', args.model, prefix, i,
                                     args.max_tokens)
            t0 = time.time()
            res = await asyncio.gather(*(guarded(i) for i in range(args.n)))
            wall = time.time() - t0
            out_tok = sum(r[2] for r in res)
            in_tok = sum(r[1] for r in res)
            lat = sum(r[0] for r in res) / len(res)
            tps = out_tok / wall
            # agent: ~ steps * samples lời gọi mỗi câu, mỗi lời gọi ~ max_tokens/2 token sinh
            calls_per_q = args.steps * args.samples
            q_per_5h = 5 * 3600 * (args.n / wall) / calls_per_q
            print(f'concurrency={c:4d}  {args.n} lời gọi trong {wall:6.1f}s  sinh {tps:7.0f} tok/s  '
                  f'đọc {in_tok / wall:8.0f} tok/s  trễ TB {lat:5.1f}s  => ~{q_per_5h:5.0f} câu agent / 5h '
                  f'({args.samples} lời giải x {args.steps} bước)')
            await asyncio.sleep(1)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='.')
    ap.add_argument('--model', default='qwen')
    ap.add_argument('--n', type=int, default=128)
    ap.add_argument('--concurrency', type=int, nargs='+', default=[16, 64, 128])
    ap.add_argument('--max-tokens', type=int, default=500)
    ap.add_argument('--prefix-repeat', type=int, default=1)
    ap.add_argument('--steps', type=int, default=5, help='số bước trung bình mỗi lời giải (ước tính)')
    ap.add_argument('--samples', type=int, default=4)
    random.seed(0)
    asyncio.run(run(ap.parse_args()))
