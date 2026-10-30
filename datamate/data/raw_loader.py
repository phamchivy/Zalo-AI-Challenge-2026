"""Đọc MỌI file của một doanh nghiệp về các RawTable (DataFrame toàn chuỗi), không giả định họ dữ liệu.

Xử lý: csv (nhiều encoding/sep), xlsx nhiều sheet (header không ở dòng 1, header 2 tầng, cột trùng tên,
sheet theo tháng/năm), sqlite; bỏ file khoá `~$`; gộp file tách `_1/_2`, `_2025/_2026`, `T01..T12`.
"""
from __future__ import annotations

import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from datamate.core.textutil import fold

DATA_EXT = {'.csv', '.tsv', '.txt', '.xlsx', '.xlsm', '.xls', '.sqlite', '.db', '.sqlite3'}


@dataclass
class RawTable:
    name: str                      # tên logic (đã bỏ hậu tố tách file), duy nhất trong doanh nghiệp
    kind: str                      # csv | xlsx | sqlite
    files: list[str]
    sheets: list[str]
    df: pd.DataFrame               # mọi cột kiểu object/str; thêm '__file__', '__sheet__' khi có nhiều nguồn
    header_row: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [c for c in self.df.columns if not c.startswith('__')]


# ----------------------------------------------------------------------------- helpers
def fix_mojibake(s: str) -> str:
    """'Danh s├ích h├áng h├│a' -> 'Danh sách hàng hóa' (tên file zip trên Windows)."""
    for enc in ('cp437', 'cp850'):
        try:
            t = s.encode(enc).decode('utf-8')
            if t != s:
                return t
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return s


_SPLIT_SUFFIX = re.compile(r'([ _\-]?(\(\d+\)|\d{1,2}|20\d{2}|T\d{1,2}([_\-]?20\d{2})?|thang ?\d{1,2}|part ?\d+))+$', re.I)


def logical_stem(stem: str) -> str:
    s = _SPLIT_SUFFIX.sub('', stem).strip(' _-')
    return s or stem


def _dedupe(cols: list) -> list[str]:
    seen: Counter = Counter()
    out = []
    for c in cols:
        c = '' if c is None or (isinstance(c, float) and pd.isna(c)) else str(c).strip()
        c = c or 'col'
        seen[c] += 1
        out.append(c if seen[c] == 1 else f'{c}__dup{seen[c] - 1}')
    return out


def _is_texty(x) -> bool:
    return isinstance(x, str) and bool(x.strip()) and not re.fullmatch(r'[\d.,/: \-+%]+', x.strip())


def _detect_header(raw: pd.DataFrame) -> int:
    """Chọn dòng header trong 20 dòng đầu: nhiều ô có giá trị + nhiều ô là chữ + dòng sau giống dữ liệu."""
    head = raw.head(20)
    best, best_score = 0, -1.0
    width = raw.shape[1]
    for i in range(len(head)):
        row = head.iloc[i].tolist()
        filled = sum(1 for x in row if x is not None and str(x).strip() not in ('', 'nan'))
        texty = sum(1 for x in row if _is_texty(x))
        uniq = len({str(x) for x in row if x is not None and str(x).strip() not in ('', 'nan')})
        score = filled + texty + 0.5 * uniq
        if filled < max(2, 0.5 * width):
            score -= 5
        if score > best_score + 1e-9:
            best, best_score = i, score
    return best


def _two_level_header(raw: pd.DataFrame, h: int) -> list[str]:
    """Header 2 tầng: nếu dòng trên header có nhãn nhóm (ô gộp) và header bị trùng tên -> 'Nhóm Tên'."""
    cols = [None if (x is None or str(x).strip() in ('', 'nan')) else str(x).strip() for x in raw.iloc[h].tolist()]
    if h == 0:
        return _dedupe(cols)
    dup = len([c for c in cols if c]) != len({c for c in cols if c})
    above = [None if (x is None or str(x).strip() in ('', 'nan')) else str(x).strip() for x in raw.iloc[h - 1].tolist()]
    n_above = sum(1 for x in above if x)
    if dup and 0 < n_above < len(cols):
        grp, filled = None, []
        for a in above:
            grp = a if a else grp
            filled.append(grp)
        counts = Counter(c for c in cols if c)
        cols = [f'{g} {c}' if (c and counts[c] > 1 and g) else c for c, g in zip(cols, filled)]
    return _dedupe(cols)


def _to_obj(df: pd.DataFrame) -> pd.DataFrame:
    df = df.astype(object)
    return df.where(pd.notna(df), None)


def read_csv_any(p: Path) -> pd.DataFrame:
    last = None
    for enc in ('utf-8-sig', 'utf-8', 'cp1258', 'cp1252', 'latin-1'):
        try:
            df = pd.read_csv(p, dtype=str, encoding=enc, sep=None, engine='python', keep_default_na=False)
            break
        except Exception as e:  # noqa: BLE001
            last = e
    else:
        raise RuntimeError(f'cannot read {p}: {last}')
    df.columns = _dedupe([str(c).strip().lstrip('﻿') for c in df.columns])
    # header không ở dòng đầu (nhiều cột Unnamed)
    if sum(str(c).startswith('Unnamed') for c in df.columns) > len(df.columns) / 2:
        raw = pd.read_csv(p, dtype=str, encoding=enc, sep=None, engine='python', header=None, keep_default_na=False)
        h = _detect_header(raw)
        cols = _two_level_header(raw, h)
        df = raw.iloc[h + 1:].copy()
        df.columns = cols
    df = df.replace({'': None})
    return _to_obj(df.dropna(how='all')).reset_index(drop=True)


def _read_excel_sheets(p: Path) -> dict:
    for engine in ('calamine', 'openpyxl', None):
        try:
            return pd.read_excel(p, sheet_name=None, header=None, dtype=str, engine=engine)
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError(f'cannot read {p}')


# ----------------------------------------------------------------------------- public API
def load_file(p: Path) -> list[RawTable]:
    ext = p.suffix.lower()
    name = fix_mojibake(p.name)
    stem = fix_mojibake(p.stem)
    if ext in ('.csv', '.tsv', '.txt'):
        df = read_csv_any(p)
        return [RawTable(name=stem, kind='csv', files=[name], sheets=[], df=df)]
    if ext in ('.xlsx', '.xlsm', '.xls'):
        sheets = _read_excel_sheets(p)
        groups: dict[tuple, list] = defaultdict(list)
        for sh, raw in sheets.items():
            raw = raw.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)
            if raw.empty:
                continue
            raw = _to_obj(raw)
            h = _detect_header(raw)
            cols = tuple(_two_level_header(raw, h))
            groups[cols].append((sh, raw, h))
        out = []
        multi = len(groups) > 1
        for cols, items in groups.items():
            dfs = []
            for sh, raw, h in items:
                d = raw.iloc[h + 1:].copy()
                d.columns = list(cols)
                d['__sheet__'] = str(sh)
                dfs.append(d)
            df = pd.concat(dfs, ignore_index=True).dropna(how='all', subset=list(cols))
            tname = stem if not multi else f'{stem}::{items[0][0]}' if len(items) == 1 else f'{stem}::{items[0][0]}..{items[-1][0]}'
            out.append(RawTable(name=tname, kind='xlsx', files=[name], sheets=[str(i[0]) for i in items],
                                df=df.reset_index(drop=True), header_row=items[0][2]))
        return out
    if ext in ('.sqlite', '.db', '.sqlite3'):
        con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
        out = []
        try:
            for (t,) in con.execute("select name from sqlite_master where type in ('table','view') and name not like 'sqlite_%'"):
                df = pd.read_sql(f'select * from "{t}"', con)
                df.columns = _dedupe(list(df.columns))
                out.append(RawTable(name=t, kind='sqlite', files=[name], sheets=[], df=_to_obj(df)))
        finally:
            con.close()
        return out
    return []


def load_business(biz_dir: str | Path) -> dict[str, RawTable]:
    """Đọc toàn bộ thư mục doanh nghiệp; gộp các file tách (cùng stem logic + cùng bộ cột)."""
    biz_dir = Path(biz_dir)
    files = sorted(f for f in biz_dir.rglob('*') if f.is_file() and f.suffix.lower() in DATA_EXT
                   and not f.name.startswith(('~$', '.')) and f.name.lower() != 'readme.md')
    tables: list[RawTable] = []
    for f in files:
        try:
            tables.extend(load_file(f))
        except Exception as e:  # noqa: BLE001
            tables.append(RawTable(name=f.stem, kind='error', files=[f.name], sheets=[], df=pd.DataFrame(),
                                   notes=[f'read error: {e!r}'[:300]]))
    # gộp file tách
    groups: dict[tuple, list[RawTable]] = defaultdict(list)
    for t in tables:
        if t.kind in ('csv', 'xlsx'):
            base = t.name.split('::')[0]
            key = (t.kind, fold(logical_stem(base)), tuple(sorted(t.columns)), t.name.split('::')[1] if '::' in t.name else '')
            groups[key].append(t)
        else:
            groups[(t.kind, t.name, (), id(t))].append(t)
    merged: dict[str, RawTable] = {}
    for key, items in groups.items():
        if len(items) == 1:
            t = items[0]
        else:
            dfs = []
            for it in items:
                d = it.df.copy()
                d['__file__'] = it.files[0]
                dfs.append(d)
            base = logical_stem(items[0].name.split('::')[0])
            t = RawTable(name=base, kind=items[0].kind, files=[f for it in items for f in it.files],
                         sheets=[s for it in items for s in it.sheets], df=pd.concat(dfs, ignore_index=True),
                         header_row=items[0].header_row, notes=[f'merged {len(items)} files'])
        name = t.name
        i = 2
        while name in merged:
            name = f'{t.name}#{i}'
            i += 1
        t.name = name
        merged[name] = t
    return merged
