"""Chia shop train thành dev (được xem lỗi, được sửa code theo) và holdout (không bao giờ mở lỗi, chỉ lấy 1 con số PA).

Vì sao: `calibrate` chia fold theo shop nhưng code đã được sửa sau khi đọc lỗi của mọi shop train, nên PA out-of-fold vẫn
lạc quan. Holdout là các shop chưa ai xem -> PA trên holdout mới đo đúng mức tổng quát cho public / private.

  split-info : mỗi shop -> số câu, định dạng file, "họ" dữ liệu (theo đuôi file + số bảng), ngành đoán từ tên hàng.
  make-split : shop đã xem (--seen) luôn vào dev; holdout chọn từ shop chưa xem, rải đều theo họ dữ liệu / ngành,
               thứ tự cố định theo md5(business_id) -> chạy lại ra cùng kết quả. Ghi configs/split.yaml.
Không đọc câu hỏi hay đáp án của shop để chọn (chỉ đọc tên file, README và vài trăm dòng dữ liệu).
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

import yaml

from datamate.core.textutil import fold

DATA_EXT = {'.csv', '.tsv', '.txt', '.xlsx', '.xlsm', '.xls', '.sqlite', '.db', '.sqlite3', '.json'}

# ngành đoán từ tên hàng + README (từ vựng chung theo ngành, không gắn với shop nào). Đếm số lần khớp, lấy ngành nhiều nhất.
INDUSTRY = {
    'nha_thuoc': r'\b(vien|vi x|siro|\d+ ?mg|hoat chat|ke don|thuoc (tay|ke don|bo))\b',
    'dien_thoai': r'\b(iphone|samsung|galaxy|xiaomi|oppo|imei|op lung|sac \d+w|tai nghe|tra gop)\b',
    'sua_xe': r'\b(thay nhot|nhot may|bien so|lop xe|ruot xe|vo xe|bugi|ma phanh)\b',
    'ca_phe': r'\b(ca phe|tra sua|tra dao|bac xiu|sinh to|da xay|latte|espresso)\b',
    'nha_hang': r'\b(lau|nuong|xao|chien|goi|bia tiger|bia sai gon|heineken|com chien)\b',
    'vat_lieu': r'\b(xi mang|thep|gach|ong pvc|ngoi|cat xay|da 1x2|son dulux|luoi b40)\b',
    'me_be': r'\b(bim|ta dan|ta quan|sua bot|binh sua|nuoc giat em be)\b',
    'tap_hoa': r'\b(mi hao hao|snack|nuoc suoi|sua tuoi|nuoc ngot|dau an|gao st25|trung ga)\b',
}


def guess_industry(text: str) -> str:
    f = fold(text)
    score = {k: len(re.findall(rx, f)) for k, rx in INDUSTRY.items()}
    best = max(score, key=score.get)
    return best if score[best] >= 3 else 'khac'


def _sample_text(biz_dir: Path, rows: int = 300) -> str:
    """README + một ít ô chữ trong dữ liệu (đủ để đoán ngành, không đọc câu hỏi / đáp án)."""
    from datamate.data.raw_loader import load_business
    parts = []
    rd = next((f for f in biz_dir.iterdir() if f.name.lower() == 'readme.md'), None)
    if rd:
        parts.append(rd.read_text(encoding='utf-8', errors='ignore'))
    try:
        for t in load_business(biz_dir).values():
            df = t.df.head(rows)
            txt = [str(x) for x in df.select_dtypes(include='object').values.ravel().tolist() if isinstance(x, str)]
            parts.append(' '.join(x for x in txt if x and not x.replace('.', '').isdigit()))
    except Exception:  # noqa: BLE001
        pass
    return '\n'.join(parts)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]


def _md5(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def family(files: list[Path]) -> str:
    """Họ dữ liệu theo cấu trúc file: đuôi file + số file (1 / 2-3 / 4+)."""
    ext = sorted({f.suffix.lower().lstrip('.') for f in files})
    n = len(files)
    size = '1' if n == 1 else ('2-3' if n <= 3 else '4+')
    return f"{'+'.join(ext) or 'none'}:{size}"


def split_info(split_dir: str) -> dict[str, dict]:
    split = Path(split_dir)
    nq = Counter(q['business_id'] for q in _jsonl(split / 'questions.jsonl')) if (split / 'questions.jsonl').exists() else Counter()
    out = {}
    for b in sorted(p for p in (split / 'data').iterdir() if p.is_dir()):
        files = [f for f in b.rglob('*') if f.is_file() and f.suffix.lower() in DATA_EXT and f.name.lower() != 'readme.md']
        out[b.name] = {'questions': nq.get(b.name, 0), 'family': family(files), 'industry': guess_industry(_sample_text(b)),
                       'files': sorted(f.name for f in files)[:12]}
    return out


def make_split(split_dir: str, seen: list[str], n_holdout: int = 8, out: Optional[str] = None) -> dict:
    info = split_info(split_dir)
    seen_s = {s for s in seen if s in info}
    unseen = [b for b in info if b not in seen_s]
    # nhóm theo (ngành, họ dữ liệu); lấy lần lượt mỗi nhóm một shop (thứ tự md5) để holdout đa dạng nhất
    groups: dict[tuple, list[str]] = defaultdict(list)
    for b in sorted(unseen, key=_md5):
        groups[(info[b]['industry'], info[b]['family'])].append(b)
    order = sorted(groups, key=lambda g: _md5('|'.join(g)))
    holdout: list[str] = []
    while len(holdout) < min(n_holdout, len(unseen)):
        for g in order:
            if groups[g] and len(holdout) < n_holdout:
                holdout.append(groups[g].pop(0))
    dev = sorted(b for b in info if b not in holdout)
    doc = {
        'created': date.today().isoformat(),
        'method': f'holdout = {len(holdout)} shop chưa xem, rải đều theo (ngành, họ dữ liệu), thứ tự md5(business_id)',
        'holdout': sorted(holdout),
        'dev': dev,
        'seen': sorted(seen_s),
        'notes': ['KHÔNG mở errors/logs của shop holdout, KHÔNG sửa code theo câu hỏi của holdout.',
                  'Chỉ báo cáo PA tổng + PA theo kiểu đáp án của holdout; nếu đã lỡ xem lỗi holdout -> chuyển shop đó sang dev '
                  'và chạy lại make-split với --seen bổ sung.'],
        'holdout_profile': {b: {k: info[b][k] for k in ('industry', 'family', 'questions')} for b in sorted(holdout)},
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return doc


def load_part(split_file: Optional[str], part: Optional[str]) -> Optional[set[str]]:
    """Tập business_id của phần dev / holdout; None nếu không dùng split-file."""
    if not split_file:
        return None
    doc = yaml.safe_load(Path(split_file).read_text(encoding='utf-8'))
    part = part or 'dev'
    if part not in ('dev', 'holdout'):
        raise ValueError(f'--part phải là dev hoặc holdout, không phải {part!r}')
    return set(doc.get(part) or [])
