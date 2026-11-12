"""v3.0.0: LLM ở bước dựng dữ liệu — client JSON (parse/kiểm/thử lại), gán nhãn, gán vai trò + Hungarian, chọn bằng invariants."""
import pandas as pd

from datamate.data.llm_mapper import apply_roles, inv_score, label_enums, suggest_roles
from datamate.data.profiler import profile_table
from datamate.data.raw_loader import load_business
from datamate.data.readme import parse_readme
from datamate.data.schema_mapper import map_business
from datamate.data.vocab import norm_status, set_enum_overrides
from datamate.llm.json_chat import extract_json


class FakeClient:
    """Thay cho model: trả lời theo hàm cho trước, chạy qua đúng hàm validate như client thật."""

    def __init__(self, fn):
        self.fn, self.calls, self.failures = fn, 0, 0

    def ask(self, system, user, schema, validate):
        self.calls += 1
        try:
            return validate(self.fn(user))
        except ValueError:
            self.failures += 1
            return None


def test_extract_json_variants():
    assert extract_json('```json\n{"a": 1}\n```') == {'a': 1}
    assert extract_json('Kết quả: {"a": {"b": "}"}} xong') == {'a': {'b': '}'}}


def _shop(tmp_path):
    d = tmp_path / 'biz'
    d.mkdir()
    rows = []
    for o in range(300):
        st = ['đã chốt đơn', 'khách bùng', 'khách gửi trả lại'][o % 3]
        rows.append({'Mã đơn': f'D{o:04d}', 'Ngày bán': '2025-03-01', 'Trạng thái': st, 'Tên hàng': f'Hàng {o % 9}',
                     'Số lượng': 1, 'Đơn giá': 10000, 'Thành tiền': 10000, 'Khách đưa tiền': 10000})
    pd.DataFrame(rows).to_csv(d / 'BanHang.csv', index=False)
    raw = load_business(d)
    profiles = {n: profile_table(t) for n, t in raw.items()}
    return raw, profiles, map_business(profiles, parse_readme(''))


def test_label_enums_only_unknown_labels(tmp_path):
    raw, profiles, bm = _shop(tmp_path)
    seen = {}

    def fn(user):
        seen['u'] = user
        return {'map': {'đã chốt đơn': 'completed', 'khách bùng': 'cancelled', 'khách gửi trả lại': 'returned'}}

    em = label_enums(FakeClient(fn), raw, bm)
    # 'khách gửi trả lại' từ điển đã hiểu (returned) -> không gửi cho LLM, không ghi đè
    assert "'khách gửi trả lại'" not in seen['u']
    assert em['status'] == {'da chot don': 'completed', 'khach bung': 'cancelled'}
    set_enum_overrides(em)
    try:
        assert norm_status('Khách bùng') == 'cancelled'
    finally:
        set_enum_overrides(None)
    assert norm_status('Khách bùng') == 'unknown'                    # ngoài bước dựng: từ điển gốc


def test_label_enums_rejects_invalid_category(tmp_path):
    raw, profiles, bm = _shop(tmp_path)
    c = FakeClient(lambda u: {'map': {'khách bùng': 'bị huỷ'}})       # loại ngoài tập cho phép
    assert label_enums(c, raw, bm)['status'] == {} and c.failures == 1


def test_suggest_roles_hungarian_and_threshold(tmp_path):
    raw, profiles, bm = _shop(tmp_path)
    tm = next(iter(bm.tables.values()))
    assert 'paid' not in tm.roles

    def fn(user):
        return {'assign': [{'column': 'Khách đưa tiền', 'role': 'paid', 'conf': 0.9},
                           {'column': 'Khách đưa tiền', 'role': 'order_total', 'conf': 0.85}]}

    sugg = suggest_roles(FakeClient(fn), profiles, bm, min_conf=0.8)
    assert sugg[tm.table] == {'paid': ('Khách đưa tiền', 0.9)}           # một cột -> một vai trò (Hungarian)
    bm2 = apply_roles(bm, sugg)
    assert bm2.tables[tm.table].roles['paid'] == 'Khách đưa tiền' and 'paid' not in tm.roles   # bản gốc không đổi
    low = suggest_roles(FakeClient(lambda u: {'assign': [{'column': 'Khách đưa tiền', 'role': 'paid', 'conf': 0.5}]}),
                        profiles, bm, min_conf=0.8)
    assert low == {}


def test_inv_score_prefers_fewer_red_then_more_green():
    assert inv_score({'a': 'green', 'b': 'red'}) > inv_score({'a': 'green', 'b': 'yellow'})
    assert inv_score({'a': 'green', 'b': 'na'}) > inv_score({'a': 'green', 'b': 'green'})
