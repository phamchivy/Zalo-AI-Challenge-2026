"""BusinessCatalog = mọi thứ đã chuẩn bị sẵn cho một doanh nghiệp (build 1 lần, cache trên đĩa).

  raw       : bảng gốc (cho tầng B sinh code)
  profiles  : profile cột
  mapping   : vai trò cột + loại bảng
  canon     : bảng canonical (cho tầng A)
  readme    : README đã parse
"""
from __future__ import annotations

import hashlib
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from datamate.core.log import get_logger
from datamate.data.canonical import Canonical, build_canonical
from datamate.data.profiler import TableProfile, profile_table, table_card
from datamate.data.raw_loader import RawTable, load_business
from datamate.data.readme import ReadmeInfo, parse_readme
from datamate.data.schema_mapper import BusinessMapping, map_business

log = get_logger('catalog')
CACHE_VERSION = 'v1.2'


def _code_version() -> str:
    """Đổi code tầng dữ liệu -> cache tự vô hiệu."""
    h = hashlib.sha1(CACHE_VERSION.encode())
    for f in sorted((Path(__file__).parent).glob('*.py')):
        h.update(f.read_bytes())
    tu = Path(__file__).parent.parent / 'core' / 'textutil.py'
    if tu.exists():
        h.update(tu.read_bytes())
    return h.hexdigest()[:10]


@dataclass
class BusinessCatalog:
    business_id: str
    root: str
    raw: dict[str, RawTable]
    profiles: dict[str, TableProfile]
    mapping: BusinessMapping
    canon: Canonical
    readme: ReadmeInfo
    build_sec: float = 0.0
    errors: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ prompt helpers
    def cards(self, max_tables: int = 25, max_cols: int = 40, max_chars: int = 24000) -> str:
        order = ['sales_flat', 'sales_lines', 'sales_orders', 'products', 'units', 'customers', 'payments', 'returns',
                 'purchases', 'suppliers', 'inventory', 'lots', 'drugs', 'price_history', 'staff', 'expenses', 'deliveries']
        names = sorted(self.profiles, key=lambda n: (order.index(self.mapping.tables[n].ttype)
                                                     if n in self.mapping.tables and self.mapping.tables[n].ttype in order else 99, n))
        out, size = [], 0
        for name in names[:max_tables]:
            tp = self.profiles[name]
            m = self.mapping.tables.get(name)
            role_of = m.col_roles if m else None
            head = f'[{m.ttype}] ' if m else ''
            card = head + table_card(tp, role_of, max_cols)
            if size + len(card) > max_chars:
                out.append(f'... (còn {len(names) - len(out)} bảng, xem tables.keys())')
                break
            out.append(card)
            size += len(card)
        return '\n\n'.join(out)

    def summary(self) -> dict:
        c = self.canon
        return {
            'business_id': self.business_id,
            'tables': {n: (m.ttype, len(self.raw[n].df)) for n, m in self.mapping.tables.items()},
            'lines': len(c.lines), 'orders': len(c.orders), 'products': len(c.products),
            'caps': sorted(c.caps), 'date_min': str(c.date_min), 'date_max': str(c.date_max),
            'notes': c.notes, 'build_sec': round(self.build_sec, 2), 'errors': self.errors,
            'health': getattr(c, 'health', {}),
        }


def _fingerprint(biz_dir: Path, llm_cfg: Optional[dict] = None) -> str:
    h = hashlib.sha1(_code_version().encode())
    if llm_cfg and llm_cfg.get('enabled'):          # v3.0.0: catalog dựng có/không LLM (và model nào) là khác nhau
        h.update(f"llm|{llm_cfg.get('model')}|{llm_cfg.get('enums')}|{llm_cfg.get('roles')}|{llm_cfg.get('min_conf')}".encode())
    for f in sorted(biz_dir.rglob('*')):
        if f.is_file():
            st = f.stat()
            h.update(f'{f.relative_to(biz_dir)}|{st.st_size}|{int(st.st_mtime)}'.encode())
    return h.hexdigest()[:16]


def build_catalog(biz_dir: str | Path, cache_dir: Optional[str] = None, llm_cfg: Optional[dict] = None) -> BusinessCatalog:
    biz_dir = Path(biz_dir)
    bid = biz_dir.name
    cache_file = None
    if cache_dir:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = Path(cache_dir) / f'{bid}-{_fingerprint(biz_dir, llm_cfg)}.pkl'
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as fh:
                    return pickle.load(fh)
            except Exception as e:  # noqa: BLE001
                log.warning('cache read failed %s: %r', cache_file, e)
    t0 = time.time()
    errors: list[str] = []
    readme_path = next((p for p in biz_dir.iterdir() if p.name.lower() == 'readme.md'), None)
    readme = parse_readme(readme_path.read_text(encoding='utf-8', errors='replace') if readme_path else '')
    raw = load_business(biz_dir)
    for t in raw.values():
        errors.extend(t.notes if t.kind == 'error' else [])
    profiles = {n: profile_table(t) for n, t in raw.items() if t.kind != 'error'}
    mapping = map_business(profiles, readme)
    # v3.0-a: bổ sung role / loại bảng còn thiếu từ quan hệ GIÁ TRỊ giữa các bảng (không dùng tên cột)
    from datamate.data.structure import infer_structure
    struct = infer_structure({n: raw[n] for n in profiles}, profiles, mapping)
    rawp = {n: raw[n] for n in profiles}
    client = None
    enum_map: dict = {}
    if llm_cfg and llm_cfg.get('enabled'):
        from datamate.llm.json_chat import JSONChat
        client = JSONChat(llm_cfg, cache_dir)
        if llm_cfg.get('enums', True):
            try:
                from datamate.data.llm_mapper import label_enums
                enum_map = label_enums(client, rawp, mapping)
            except Exception as e:  # noqa: BLE001
                errors.append(f'llm enums failed: {e!r}'[:200])
    from datamate.data.vocab import set_enum_overrides
    set_enum_overrides(enum_map)
    try:
        canon = _canon_with_checks(rawp, mapping, readme, struct.uncovered_children, errors)
        # v3.0.0: LLM bổ sung vai trò còn thiếu -> dựng lại -> giữ bản có invariants tốt hơn (ngang nhau: giữ bản cũ)
        if client is not None and llm_cfg.get('roles', True):
            try:
                from datamate.data.llm_mapper import apply_roles, inv_score, suggest_roles
                from datamate.data.structure import uncovered_children
                sugg = suggest_roles(client, profiles, mapping, float(llm_cfg.get('min_conf', 0.8)))
                if sugg:
                    m2 = apply_roles(mapping, sugg)
                    c2 = _canon_with_checks(rawp, m2, readme, uncovered_children(rawp, profiles, m2), errors)
                    s0 = inv_score(canon.health.get('invariants', {}))
                    s1 = inv_score(c2.health.get('invariants', {}))
                    if s1 < s0:
                        canon, mapping = c2, m2
                        canon.notes.append(f'llm roles accepted {sugg} (invariants {s0} -> {s1})')
                    else:
                        canon.notes.append(f'llm roles rejected {sugg} (invariants {s0} -> {s1})')
            except Exception as e:  # noqa: BLE001
                errors.append(f'llm roles failed: {e!r}'[:200])
        if enum_map.get('status') or enum_map.get('payment'):
            canon.notes.append(f'llm enums: {enum_map}')
            canon.health['llm_enums'] = enum_map
        if client is not None:
            canon.health['llm_calls'] = client.calls
            canon.health['llm_failures'] = client.failures
        if struct.notes:
            canon.notes.extend(struct.notes)
    finally:
        set_enum_overrides(None)
    cat = BusinessCatalog(business_id=bid, root=str(biz_dir), raw=raw, profiles=profiles, mapping=mapping, canon=canon,
                          readme=readme, build_sec=time.time() - t0, errors=errors)
    if cache_file:
        try:
            with open(cache_file, 'wb') as fh:
                pickle.dump(cat, fh, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:  # noqa: BLE001
            log.warning('cache write failed: %r', e)
    log.info('catalog %s built in %.1fs: %s', bid, cat.build_sec, {k: v for k, v in cat.summary().items() if k in ('lines', 'orders', 'caps')})
    return cat


def _canon_with_checks(rawp, mapping, readme, uncovered: list[str], errors: list[str]):
    """Dựng canonical + invariants (gồm độ phủ bảng con) -> canon.health['invariants']."""
    try:
        canon = build_canonical(rawp, mapping, readme)
    except Exception:  # noqa: BLE001 - tầng A hỏng không được làm sập cả shop: tầng B vẫn chạy trên bảng gốc
        import traceback
        errors.append('canonical failed: ' + traceback.format_exc()[-800:])
        from datamate.data.canonical import _empty
        canon = _empty(['canonical failed'])
    try:
        from datamate.data.invariants import check_canon
        rep = check_canon(canon, readme)
        n = len(uncovered)
        cov = {'name': 'COVERAGE_child', 'n': n + 1, 'bad': n, 'level': 'red' if n else 'green', 'rate': float(n > 0),
               'note': f'bảng con của đơn không hiểu được: {uncovered}' if n else '',
               'examples': [{'bang': t} for t in uncovered[:3]]}
        canon.health['invariants'] = {x['name']: x['level'] for x in rep['invariants']}
        canon.health['invariants'][cov['name']] = cov['level']
        canon.health['coverage'] = cov
        canon.health['inv_risk'] = rep['risk'] + (3 if n else 0)
    except Exception as e:  # noqa: BLE001
        errors.append(f'invariants failed: {e!r}'[:200])
    return canon


def _build_worker(args):
    biz_dir, cache_dir, llm_cfg = args
    cat = build_catalog(biz_dir, cache_dir, llm_cfg)
    return cat.business_id


def build_all(data_dir: str | Path, business_ids: list[str], cache_dir: str, workers: int = 4,
              llm_cfg: Optional[dict] = None) -> dict[str, BusinessCatalog]:
    """Build song song (process pool) rồi nạp lại từ cache."""
    data_dir = Path(data_dir)
    todo = [(str(data_dir / b), cache_dir, llm_cfg) for b in business_ids if (data_dir / b).is_dir()]
    if workers > 1 and len(todo) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=min(workers, len(todo))) as ex:
            list(ex.map(_build_worker, todo))
    return {Path(d).name: build_catalog(d, c, l) for d, c, l in todo}
