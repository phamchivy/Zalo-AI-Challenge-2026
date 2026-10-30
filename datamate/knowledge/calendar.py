"""Lịch âm-dương từ calendar.csv của split (không tự tính âm lịch)."""
from __future__ import annotations

import csv
from datetime import date, timedelta
from pathlib import Path
from typing import Optional


class LunarCalendar:
    def __init__(self, rows: list[dict]):
        self.by_date: dict[date, dict] = {}
        self.by_lunar: dict[tuple[int, int, int, bool], date] = {}
        for r in rows:
            d = date.fromisoformat(r['date'])
            leap = str(r.get('lunar_leap', '')).lower() in ('true', '1')
            rec = {'weekday': int(r.get('weekday') or d.isoweekday()), 'ld': int(r['lunar_day']), 'lm': int(r['lunar_month']),
                   'ly': int(r['lunar_year']), 'leap': leap, 'holiday': r.get('holiday') or ''}
            self.by_date[d] = rec
            self.by_lunar.setdefault((rec['ly'], rec['lm'], rec['ld'], leap), d)

    @classmethod
    def load(cls, path: str | Path) -> 'LunarCalendar':
        p = Path(path)
        if not p.exists():
            return cls([])
        with open(p, encoding='utf-8-sig') as f:
            return cls(list(csv.DictReader(f)))

    def solar(self, ly: int, lm: int, ld: int, leap: bool = False) -> Optional[date]:
        return self.by_lunar.get((ly, lm, ld, leap))

    def tet_window(self, year: int, start_day: int = 23, end_day: int = 10) -> Optional[tuple[date, date]]:
        """Tết năm Y: 23 tháng Chạp (năm âm Y-1) .. hết mùng 10 tháng Giêng (năm âm Y)."""
        a = self.solar(year - 1, 12, start_day)
        b = self.solar(year, 1, end_day)
        if a and b:
            return a, b
        return None

    def holiday_dates(self, keyword: str) -> list[date]:
        k = keyword.lower()
        return sorted(d for d, r in self.by_date.items() if k in r['holiday'].lower())


def before_window(start: date, n_days: int) -> tuple[date, date]:
    """N ngày ngay trước một mốc."""
    return start - timedelta(days=n_days), start - timedelta(days=1)
