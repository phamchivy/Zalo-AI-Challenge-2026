"""v3.3.0 — Chuẩn hoá câu hỏi bằng LLM (1 lần gọi ngắn / câu) + kiểm tra đồng thuận với tầng A.

Vì sao: trên bộ câu viết lại từ train (giữ nghĩa + đáp án), tầng A rơi từ PA 0.89 xuống 0.56 — gần đúng mức public.
Hai kiểu yếu: câu dài dòng / có từ đệm -> tầng A mất tự tin và bỏ trống; câu viết tắt -> tầng A đọc sai mà vẫn tự tin.
Cách sửa không thêm regex: LLM viết câu về DẠNG CHUẨN (giải viết tắt, thêm dấu, bỏ lời dẫn / từ đệm, giữ nguyên số và tên),
rồi tầng A đọc CẢ câu gốc lẫn câu chuẩn:
  - cùng đáp án            -> giữ quyết định tầng A (nếu câu gốc bỏ trống mà câu chuẩn chắc chắn -> dùng câu chuẩn)
  - chỉ câu chuẩn có đáp án -> dùng câu chuẩn (gỡ nhóm 'dài dòng')
  - KHÁC đáp án            -> tầng A không đáng tin cho câu này: bỏ trống, chuyển LLM (bắt nhóm 'viết tắt' sai tự tin)
  - chỉ câu gốc có đáp án  -> giữ câu gốc (như cũ)
Câu chuẩn chỉ là một cách đọc khác; luật vẫn do tầng A / rulebook quyết. LLM chuẩn hoá không trả lời câu hỏi.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Optional

from datamate.core.textutil import fold

log = logging.getLogger('canon_q')

SYS = ('Bạn chuẩn hoá câu hỏi của chủ cửa hàng về dữ liệu bán hàng thành tiếng Việt chuẩn, đầy đủ, rõ ràng. KHÔNG trả lời '
       'câu hỏi. Quy tắc:\n'
       '- Giữ NGUYÊN 100% nghĩa: cùng đại lượng được hỏi, cùng khoảng thời gian, cùng điều kiện lọc, cùng mặt hàng / '
       'khách / nhóm / kênh, cùng số phần tử (top N), cùng kiểu đáp án (số, %, danh sách, ngày...).\n'
       '- Giải viết tắt và tiếng lóng (vd dthu -> doanh thu, đh -> đơn hàng, sl -> số lượng, bn -> bao nhiêu, th3 -> '
       'tháng 3, T4/26 -> tháng 4/2026, kh -> khách hàng, sp -> sản phẩm, nhiêu -> bao nhiêu); thêm dấu cho từ thiếu dấu.\n'
       '- Bỏ lời chào, lời dẫn, từ đệm không đổi nghĩa ("cho mình hỏi", "để chốt sổ", "dzậy", "hông", "nha").\n'
       '- Giữ nguyên MỌI con số, ngày tháng, và tên riêng (tên hàng, tên khách, tên nhóm, mã) đúng như câu gốc viết.\n'
       '- Không thêm điều kiện, không bỏ điều kiện, không đổi từ chỉ luật (vd "bán chạy", "đắt khách", "còn nợ").\n'
       '- Câu đã chuẩn thì trả lại y nguyên.\n'
       'Chỉ trả về JSON {"q": "<câu chuẩn>"}.')
SCHEMA = {'type': 'object', 'properties': {'q': {'type': 'string'}}, 'required': ['q']}


def _nums(s: str) -> set[str]:
    return {x.lstrip('0') or '0' for x in re.findall(r'\d+', s)}


def numbers_kept(orig: str, new: str) -> bool:
    """Mọi số của câu gốc phải còn trong câu chuẩn (cho phép năm 2 chữ số được viết đủ: 26 -> 2026)."""
    nn = _nums(new)
    for x in _nums(orig):
        if x in nn:
            continue
        if len(x) == 2 and any(len(y) == 4 and y.endswith(x) for y in nn):
            continue
        return False
    return True


def same_text(a: str, b: str) -> bool:
    return ' '.join(fold(a).split()) == ' '.join(fold(b).split())


async def canonicalize_all(llm, questions, cache_dir: Optional[str], concurrency: int = 32, max_tokens: int = 2048,
                           temperature: float = 0.0) -> dict[str, str]:
    """{qid: câu chuẩn} cho các câu chuẩn hoá được (bỏ qua câu lỗi / mất số / trùng câu gốc). Có cache theo (model, câu)."""
    from datamate.runtime.llm_client import extract_json
    cdir = Path(cache_dir) / 'canon_q' if cache_dir else None
    if cdir:
        cdir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, str] = {}
    model = getattr(getattr(llm, 'cfg', None), 'model', '')

    async def one(q):
        key = hashlib.sha1(f'{model}\n{q.question}'.encode('utf-8')).hexdigest()
        cf = cdir / f'{key}.json' if cdir else None
        if cf is not None and cf.exists():
            try:
                v = json.loads(cf.read_text(encoding='utf-8')).get('q')
                if v:
                    out[q.id] = v
                return
            except Exception:  # noqa: BLE001
                pass
        async with sem:
            for _ in range(2):
                r = await llm.chat([{'role': 'system', 'content': SYS}, {'role': 'user', 'content': f'Câu gốc: {q.question}'}],
                                   json_schema=SCHEMA, max_tokens=max_tokens, temperature=temperature)
                o = extract_json(r.text or '') if not r.error else None
                v = (o or {}).get('q') if isinstance(o, dict) else None
                if isinstance(v, str) and len(v.strip()) >= 5 and numbers_kept(q.question, v):
                    v = v.strip()
                    if cf is not None:
                        cf.write_text(json.dumps({'orig': q.question, 'q': v}, ensure_ascii=False), encoding='utf-8')
                    if not same_text(v, q.question):
                        out[q.id] = v
                    return

    await asyncio.gather(*(one(q) for q in questions))
    return out


def rule_key(cands) -> Optional[str]:
    """Khoá đáp án của rule_plan (None nếu không có đáp án hợp lệ)."""
    from datamate.output.formatter import answer_key, normalize_answer
    for c in cands:
        if c.strategy.startswith('rule_plan') and c.status == 'ok' and c.answer_type is not None:
            try:
                t, a = normalize_answer(c.answer_type, c.answer)
                if t and t != 'abstain':
                    return answer_key(t, a)
            except Exception:  # noqa: BLE001
                return None
    return None
