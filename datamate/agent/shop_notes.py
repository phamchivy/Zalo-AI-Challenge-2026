"""v2.1 — "ghi chú về shop": LLM đọc README + mô tả bảng MỘT lần cho mỗi shop, ghi lại bảng nào chứa gì, nối bảng
bằng cột nào, quy ước riêng, điểm dễ sai. Mọi khẳng định có thể kiểm tra bằng dữ liệu đều được kiểm tra:
  - bảng / cột không tồn tại -> bỏ
  - phép nối giữa hai cột: tỉ lệ giá trị khớp < 50% -> bỏ
Kết quả lưu cache theo (shop, nội dung README + cấu trúc bảng) và đưa vào phần "thông tin shop" của prompt agent.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from datamate.core.log import get_logger
from datamate.reasoning.prompts import fill
from datamate.runtime.llm_client import extract_json

log = get_logger('shop_notes')

NOTES_SCHEMA = {
    'type': 'object',
    'properties': {
        'tables': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'name': {'type': 'string'}, 'meaning': {'type': 'string'}, 'grain': {'type': 'string'},
            'key_columns': {'type': 'array', 'items': {'type': 'string'}}}, 'required': ['name', 'meaning']}},
        'joins': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'left_table': {'type': 'string'}, 'left_column': {'type': 'string'},
            'right_table': {'type': 'string'}, 'right_column': {'type': 'string'}},
            'required': ['left_table', 'left_column', 'right_table', 'right_column']}},
        'conventions': {'type': 'array', 'items': {'type': 'string'}},
        'pitfalls': {'type': 'array', 'items': {'type': 'string'}},
        'not_recorded': {'type': 'array', 'items': {'type': 'string'}},
    },
    'required': ['tables', 'conventions'],
}

NOTES_SYSTEM = """Bạn là kỹ sư dữ liệu. Đọc README và mô tả các bảng của một cửa hàng, viết GHI CHÚ ngắn gọn để một \
người khác viết code trả lời câu hỏi kinh doanh trên dữ liệu này mà không phải đọc lại toàn bộ.
Chỉ ghi điều có căn cứ trong README hoặc mô tả bảng; không đoán. Trả về đúng một JSON theo schema:
- tables: mỗi bảng: name (đúng tên bảng như trong mô tả), meaning (bảng chứa gì), grain (mỗi dòng là gì: một đơn / một dòng
  hàng / một lần thu tiền ...), key_columns (cột khoá / cột quan trọng, đúng tên cột).
- joins: các cặp cột nối bảng (đúng tên bảng và tên cột).
- conventions: quy ước riêng của shop (vd: trạng thái ghi trong cột ghi chú; số thứ tự chỉ ghi ở dòng đầu đơn; tiền ghi
  tắt '1tr5'; cột X là tiền trước giảm; thu máy cũ không trừ doanh thu ...).
- pitfalls: điểm dễ tính sai (dòng tổng cộng lẫn trong bảng, bảng trùng nhau, đơn vị tính khác nhau ...).
- not_recorded: loại thông tin shop KHÔNG ghi (vd không có giờ bán, không có nhóm khách) — câu hỏi cần chúng thì không trả lời được."""

NOTES_USER = """# SHOP {bid}
Khoảng ngày có dữ liệu: {dmin} .. {dmax}
Bộ chuẩn hoá tự động đã nhận: {summary}

## README
{readme}

## Mô tả bảng
{cards}"""


def _key(cat) -> str:
    h = hashlib.md5()
    h.update((cat.readme.text or '').encode('utf-8'))
    for n in sorted(cat.raw):
        h.update(n.encode('utf-8'))
        h.update('|'.join(map(str, cat.raw[n].df.columns)).encode('utf-8'))
    return h.hexdigest()[:12]


def _norm(s: str) -> str:
    return ' '.join(str(s).strip().lower().split())


def _find_table(cat, name: str) -> Optional[str]:
    if name in cat.raw:
        return name
    n = _norm(name)
    for t in cat.raw:
        if _norm(t) == n or _norm(t.split('::')[-1]) == n or _norm(t.split('::')[0]) == n:
            return t
    return None


def _find_col(df, col: str) -> Optional[str]:
    if col in df.columns:
        return col
    n = _norm(col)
    return next((c for c in df.columns if _norm(c) == n), None)


def validate(cat, notes: dict) -> dict:
    """Bỏ khẳng định sai: bảng / cột không tồn tại, phép nối có tỉ lệ khớp thấp."""
    out = {'tables': [], 'joins': [], 'conventions': [], 'pitfalls': [], 'not_recorded': [], 'dropped': 0}
    for t in notes.get('tables') or []:
        tn = _find_table(cat, str(t.get('name', '')))
        if not tn:
            out['dropped'] += 1
            continue
        df = cat.raw[tn].df
        keys = [c for c in (_find_col(df, k) for k in (t.get('key_columns') or [])) if c]
        out['tables'].append({'name': tn, 'meaning': str(t.get('meaning', ''))[:300], 'grain': str(t.get('grain', ''))[:150],
                              'key_columns': keys[:12]})
    for j in notes.get('joins') or []:
        lt, rt = _find_table(cat, str(j.get('left_table', ''))), _find_table(cat, str(j.get('right_table', '')))
        if not lt or not rt:
            out['dropped'] += 1
            continue
        lc, rc = _find_col(cat.raw[lt].df, str(j.get('left_column', ''))), _find_col(cat.raw[rt].df, str(j.get('right_column', '')))
        if not lc or not rc:
            out['dropped'] += 1
            continue
        a = set(cat.raw[lt].df[lc].dropna().astype(str).str.strip().head(20000))
        b = set(cat.raw[rt].df[rc].dropna().astype(str).str.strip())
        cover = len(a & b) / max(len(a), 1)
        if cover < 0.5:
            out['dropped'] += 1
            continue
        out['joins'].append({'left': f'{lt}.{lc}', 'right': f'{rt}.{rc}', 'match': round(cover, 2)})
    for k in ('conventions', 'pitfalls', 'not_recorded'):
        out[k] = [str(x)[:300] for x in (notes.get(k) or []) if str(x).strip()][:12]
    return out


def render(v: dict) -> str:
    if not v:
        return ''
    lines = ['', '## Ghi chú về shop (đã kiểm tra với dữ liệu)']
    for t in v.get('tables', []):
        keys = f" — cột chính: {', '.join(t['key_columns'])}" if t.get('key_columns') else ''
        grain = f" (mỗi dòng: {t['grain']})" if t.get('grain') else ''
        lines.append(f"- tables['{t['name']}']: {t['meaning']}{grain}{keys}")
    if v.get('joins'):
        lines.append('Nối bảng: ' + '; '.join(f"{j['left']} = {j['right']} (khớp {j['match']:.0%})" for j in v['joins']))
    for title, k in (('Quy ước riêng', 'conventions'), ('Dễ sai', 'pitfalls'), ('Shop không ghi', 'not_recorded')):
        if v.get(k):
            lines.append(f'{title}: ' + ' | '.join(v[k]))
    return '\n'.join(lines) + '\n'


async def get_notes(env, cat, summary: str, cards: str) -> str:
    """Ghi chú đã render (rỗng nếu tắt / lỗi). Cache trên đĩa theo nội dung shop."""
    cache = Path(env.cfg.cache_dir) / 'notes' / f'{cat.business_id}-{_key(cat)}.json'
    if cache.exists():
        try:
            return render(json.loads(cache.read_text(encoding='utf-8')))
        except Exception:  # noqa: BLE001
            pass
    if env.llm is None:
        return ''
    c = cat.canon
    msgs = [{'role': 'system', 'content': NOTES_SYSTEM},
            {'role': 'user', 'content': fill(NOTES_USER, bid=cat.business_id, dmin=c.date_min, dmax=c.date_max, summary=summary,
                                             readme=(cat.readme.text or '(không có README)')[:9000], cards=cards)}]
    res = await env.llm.chat(msgs, json_schema=NOTES_SCHEMA, temperature=0.0, max_tokens=1800)
    data = extract_json(res.text) if res.text else None
    if not isinstance(data, dict):
        log.warning('%s: không tạo được ghi chú shop (%s)', cat.business_id, res.error or 'json lỗi')
        return ''
    v = validate(cat, data)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(v, ensure_ascii=False, indent=1), encoding='utf-8')
    log.info('%s: ghi chú shop %d bảng, %d phép nối, bỏ %d khẳng định sai', cat.business_id, len(v['tables']),
             len(v['joins']), v['dropped'])
    return render(v)
