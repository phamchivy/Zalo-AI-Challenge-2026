"""Chuẩn hoá đáp án về đúng format chấm (rulebook §9) và khoá so sánh dùng cho bỏ phiếu."""
from __future__ import annotations

import math
import re
from datetime import date, datetime
from typing import Any, Optional

from datamate.core.textutil import norm_answer_str, to_num
from datamate.core.types import AnswerType

TYPES = {t.value for t in AnswerType}


def _num(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return None if (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else float(x)
    return to_num(x)


def _clean_number(v: float, kind: str) -> Any:
    if kind == 'percent':
        return float(round(v, 2))
    if abs(v - round(v)) < 1e-6:
        return int(round(v))
    return float(round(v, 2))


def _date(x: Any) -> Optional[str]:
    if isinstance(x, (date, datetime)):
        return x.isoformat()[:10]
    s = str(x).strip()
    if re.fullmatch(r'\d{4}-\d{2}(-\d{2})?', s):
        return s
    m = re.fullmatch(r'(\d{4})-(\d{1,2})(?:-(\d{1,2}))?(?:[ T].*)?', s)
    if m:
        return f'{int(m[1]):04d}-{int(m[2]):02d}' + (f'-{int(m[3]):02d}' if m[3] else '')
    m = re.fullmatch(r'(\d{1,2})/(\d{1,2})/(\d{4})', s)
    if m:
        return f'{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}'
    m = re.fullmatch(r'(\d{1,2})/(\d{4})', s)
    if m:
        return f'{int(m[2]):04d}-{int(m[1]):02d}'
    return None


def normalize_answer(atype: Any, answer: Any) -> tuple[Optional[str], Any]:
    """-> (type, answer) hợp lệ hoặc (None, None) nếu không ép được."""
    t = (atype.value if isinstance(atype, AnswerType) else str(atype or '')).lower().strip()
    if t not in TYPES:
        # đoán type từ giá trị
        if isinstance(answer, bool):
            t = 'boolean'
        elif isinstance(answer, (int, float)):
            t = 'number'
        elif isinstance(answer, list):
            t = 'list'
        elif isinstance(answer, dict) and 'rows' in answer:
            t = 'table'
        elif isinstance(answer, str):
            t = 'date' if _date(answer) else 'string'
        else:
            return None, None
    if t in ('abstain', 'unanswerable'):
        return t, None
    if answer is None:
        return None, None
    if t in ('number', 'percent'):
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        v = _num(answer)
        return (t, _clean_number(v, t)) if v is not None else (None, None)
    if t == 'boolean':
        if isinstance(answer, bool):
            return t, answer
        s = str(answer).strip().lower()
        if s in ('true', 'co', 'có', 'yes', '1', 'đúng', 'dung'):
            return t, True
        if s in ('false', 'khong', 'không', 'no', '0', 'sai'):
            return t, False
        return None, None
    if t == 'date':
        d = _date(answer)
        return (t, d) if d else (None, None)
    if t == 'string':
        if isinstance(answer, list) and len(answer) == 1:
            answer = answer[0]
        if isinstance(answer, (dict, list)):
            return None, None
        s = str(answer).strip()
        return (t, s) if s else (None, None)
    if t == 'list':
        if not isinstance(answer, list):
            answer = [answer]
        out = []
        for x in answer:
            if isinstance(x, bool):
                out.append(x)
            elif isinstance(x, (int, float)):
                out.append(_clean_number(float(x), 'number'))
            elif isinstance(x, str):
                v = _num(x) if re.fullmatch(r'\s*-?[\d.,]+\s*', x) else None
                out.append(_clean_number(v, 'number') if v is not None else x.strip())
            else:
                out.append(str(x))
        return t, out
    if t == 'table':
        if not isinstance(answer, dict) or 'rows' not in answer:
            return None, None
        cols = [str(c) for c in answer.get('columns', [])]
        rows = []
        for r in answer['rows'] or []:
            if isinstance(r, dict):                     # dòng dạng {cột: giá trị} (df.to_dict('records'))
                if not cols:
                    cols = [str(c) for c in r]
                r = [r.get(c, r.get(str(c))) for c in cols]
            elif isinstance(r, tuple):
                r = list(r)
            rr = []
            for x in (r if isinstance(r, list) else [r]):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    rr.append(_clean_number(float(x), 'number'))
                elif x is None or isinstance(x, (str, bool)):
                    rr.append(x)
                else:
                    rr.append(str(x))                   # giá trị lồng / lạ -> chuỗi, không làm hỏng bỏ phiếu
            rows.append(rr)
        return t, {'columns': cols, 'rows': rows}
    return None, None


def _key_item(x: Any) -> str:
    if isinstance(x, str):
        return norm_answer_str(x)
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            return str(round(float(x), 2))
        except (OverflowError, ValueError):
            return str(x)
    return norm_answer_str(str(x))


def answer_key(t: str, a: Any) -> str:
    """Khoá so khớp theo luật chấm (chuỗi chuẩn hoá NFC/lower/bỏ dấu câu, giữ dấu tiếng Việt)."""
    if t in ('unanswerable', 'abstain'):
        return t
    if t in ('number', 'percent'):
        return f'{t}:{round(float(a), 2)}'
    if t == 'string':
        return f'{t}:{norm_answer_str(a)}'
    if t == 'list':
        return f'{t}:' + '|'.join(_key_item(x) for x in a)
    if t == 'table':
        rows = sorted('|'.join(_key_item(x) for x in (r if isinstance(r, list) else [r])) for r in a['rows'])
        return f'{t}:' + '||'.join(rows)
    return f'{t}:{a}'
