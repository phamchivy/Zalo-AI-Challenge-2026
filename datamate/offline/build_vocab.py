"""Sinh từ vựng chung (từ xuất hiện trong câu hỏi của >= N doanh nghiệp khác nhau) từ questions.jsonl của train.
Dùng để phát hiện 'từ lạ' (vd tên hàng chưa khớp được) -> hạ độ tin cậy của rule parser.
  python -m datamate.offline.build_vocab /path/train/questions.jsonl datamate/understand/vocab_generic.txt
"""
import json
import re
import sys
from collections import defaultdict

from datamate.understand.normalize import qnorm


def main(src: str, dst: str, min_biz: int = 6):
    seen = defaultdict(set)
    for line in open(src, encoding='utf-8'):
        if not line.strip():
            continue
        q = json.loads(line)
        for tok in set(qnorm(q['question']).split()):
            if re.fullmatch(r'[a-z%]+', tok):
                seen[tok].add(q['business_id'])
    words = sorted(w for w, b in seen.items() if len(b) >= min_biz)
    open(dst, 'w', encoding='utf-8').write('\n'.join(words) + '\n')
    print(f'{len(words)} words -> {dst}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6)
