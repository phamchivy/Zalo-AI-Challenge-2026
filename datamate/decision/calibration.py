"""Độ tin cậy của tầng A theo ĐẶC TRƯNG (không theo id shop / id câu).

    conf = base(intent) × f_unexplained(số từ lạ) × f_health(cảnh báo dữ liệu)

  base(intent): độ chính xác đo out-of-fold trên train (cận dưới Wilson) từ file calibration.yaml
                (sinh bằng `main calibrate`); chưa có file -> dùng prior của parser.
  f_unexplained, f_health: hệ số trong config (decision.*), chỉnh bằng PA trên các shop giữ lại.
Voter trả lời khi tổng trọng số ủng hộ >= decision.min_score  =>  với tầng A đơn lẻ, min_score = độ chính xác tối thiểu.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import yaml


def wilson_lower(ok: int, n: int, z: float = 1.0) -> float:
    if n <= 0:
        return 0.0
    p = ok / n
    den = 1 + z * z / n
    centre = p + z * z / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - adj) / den)


class Calibration:
    def __init__(self, table: Optional[dict] = None, min_n: int = 8):
        self.table = table or {}
        self.min_n = min_n

    @classmethod
    def load(cls, path: Optional[str]) -> 'Calibration':
        if not path or not Path(path).exists():
            return cls()
        d = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
        return cls(d.get('intents', {}), int(d.get('min_n', 8)))

    def base(self, intent: str, prior: float) -> tuple[float, str]:
        r = self.table.get(intent)
        if r and r.get('n', 0) >= self.min_n:
            return float(r['lower']), 'calibrated'
        return prior, 'prior'


def rule_confidence(parse, canon, dcfg, cal: Calibration) -> tuple[float, dict]:
    base, src = cal.base(parse.intent, parse.confidence)
    n_unk = int((parse.features or {}).get('unexplained', 0))
    fu = dcfg.unexplained_factor[min(n_unk, len(dcfg.unexplained_factor) - 1)]
    fh = 1.0
    h = getattr(canon, 'health', {}) or {}
    warn = []
    if h.get('recon_mismatch', 0) > dcfg.recon_mismatch_max:
        fh *= dcfg.health_factor.get('recon_mismatch', 0.5)
        warn.append('recon_mismatch')
    if h.get('header_rows', 0) > 0:
        fh *= dcfg.health_factor.get('header_rows', 1.0)
        warn.append('header_rows')
    if h.get('duplicate_sources', 0) > 0:
        fh *= dcfg.health_factor.get('duplicate_sources', 1.0)
        warn.append('duplicate_sources')
    conf = base * fu * fh
    return conf, {'base': round(base, 4), 'base_src': src, 'unexplained': n_unk, 'f_unexplained': fu, 'f_health': fh,
                  'health_warn': warn}
