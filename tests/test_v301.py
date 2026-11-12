"""v3.0.1: vai trò theo loại bảng; cột trạng thái do LLM gán -> gán nhãn lại; không gán được -> STATUS_unresolved đỏ."""
import json

import pandas as pd

from datamate.core.types import AnswerType, Decision
from datamate.data.catalog import build_catalog
from datamate.data.llm_mapper import _role_candidates, status_unresolved
from datamate.data.profiler import profile_table
from datamate.data.raw_loader import load_business
from datamate.data.readme import parse_readme
from datamate.data.schema_mapper import map_business
from datamate.decision.data_guard import guard
from datamate.llm import json_chat

LABELS = ['chốt sổ', 'bùng kèo', 'ôm về lại']
CFG = {'enabled': True, 'base_url': 'x', 'model': 'fake', 'cache': False, 'enums': True, 'roles': True,
       'min_conf': 0.8, 'api_key_env': 'NOPE'}


def _shop(tmp_path):
    d = tmp_path / 'biz'
    d.mkdir()
    orders, lines = [], []
    for o in range(240):
        orders.append({'Mã đơn': f'D{o:04d}', 'Ngày bán': f'2025-03-{1 + o % 28:02d}', 'Ghi nhận cuối': LABELS[o % 3],
                       'Kho xuất': ['K1', 'K2'][o % 2], 'Tổng tiền hàng': 20000})
        for k in range(2):
            lines.append({'Mã đơn': f'D{o:04d}', 'Tên hàng': f'Hàng {(o + k) % 7}', 'Số lượng': 1, 'Đơn giá': 10000,
                          'Thành tiền': 10000})
    pd.DataFrame(orders).to_csv(d / 'DonHang.csv', index=False)
    pd.DataFrame(lines).to_csv(d / 'ChiTiet.csv', index=False)
    return d


def _fake(monkeypatch, conf, label_ok=True):
    seen = []

    def post(self, messages, schema):
        u = messages[1]['content']
        seen.append(u)
        if 'Nhãn cần xếp' in u:
            mp = dict(zip(LABELS, ['completed', 'cancelled', 'returned'])) if label_ok else {}
            return json.dumps({'map': mp}, ensure_ascii=False)
        a = [{'column': 'Ghi nhận cuối', 'role': 'status', 'conf': conf}] if "'Ghi nhận cuối'" in u else []
        return '```json\n' + json.dumps({'assign': a}, ensure_ascii=False) + '\n```'

    monkeypatch.setattr(json_chat.JSONChat, '_post', post)
    return seen


def test_role_candidates_follow_table_type(tmp_path):
    d = _shop(tmp_path)
    raw = load_business(d)
    profiles = {n: profile_table(t) for n, t in raw.items()}
    bm = map_business(profiles, parse_readme(''))
    for t, tm in bm.tables.items():
        roles, cols = _role_candidates(tm, profiles[t])
        if tm.ttype == 'sales_orders':
            assert 'product_name' not in roles and 'qty' not in roles
        if tm.ttype == 'sales_lines':
            assert 'status' not in roles and 'paid' not in roles


def test_llm_status_column_is_relabeled_and_accepted(tmp_path, monkeypatch):
    seen = _fake(monkeypatch, 0.9)
    cat = build_catalog(_shop(tmp_path), None, CFG)
    inv = cat.canon.health['invariants']
    assert inv['I15_status_mapped'] == 'green' and 'STATUS_unresolved' not in inv
    assert any('Nhãn cần xếp' in u and 'bùng kèo' in u for u in seen)          # nhãn của cột mới được gửi đi gán
    assert cat.canon.health['llm_enums']['status']['bung keo'] == 'cancelled'


def test_status_seen_but_not_adopted_is_red(tmp_path, monkeypatch):
    _fake(monkeypatch, 0.6)                                   # dưới min_conf: không gán, nhưng LLM nói có cột trạng thái
    cat = build_catalog(_shop(tmp_path), None, CFG)
    inv = cat.canon.health['invariants']
    assert inv.get('STATUS_unresolved') == 'red'
    d = Decision(qid='q', type=AnswerType.NUMBER, answer=123, confidence=1.0)
    assert guard(d, 'revenue', inv).type == AnswerType.ABSTAIN
    assert guard(d, 'stock_qty', inv).type == AnswerType.NUMBER              # câu không dựa trên đơn: không chặn


def test_status_labels_unreadable_is_red(tmp_path, monkeypatch):
    _fake(monkeypatch, 0.9, label_ok=False)                   # gán cột được nhưng nhãn không hiểu -> I15 đỏ -> bỏ
    cat = build_catalog(_shop(tmp_path), None, CFG)
    assert cat.canon.health['invariants'].get('STATUS_unresolved') == 'red'


def test_no_status_claim_no_red(tmp_path):
    raw = load_business(_shop(tmp_path))
    profiles = {n: profile_table(t) for n, t in raw.items()}
    bm = map_business(profiles, parse_readme(''))
    assert status_unresolved({}, bm) == []
    t = next(t for t, tm in bm.tables.items() if tm.ttype == 'sales_orders')
    assert status_unresolved({t: [('Kho xuất', 'staff', 0.9)]}, bm) == []      # LLM không nói gì về trạng thái
    assert status_unresolved({t: [('Ghi nhận cuối', 'status', 0.7)]}, bm) == [t]
