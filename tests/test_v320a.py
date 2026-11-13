"""v3.2.0a: hiệu chỉnh tầng A trên câu viết lại — hệ số 'số đo dựa vào mặc định', script tạo split viết lại."""
from types import SimpleNamespace

from datamate.core.config import DecisionConfig
from datamate.decision.calibration import Calibration, rule_confidence
from datamate.offline.calibrate import weak_factor


def _p(weak):
    return SimpleNamespace(intent='revenue', confidence=0.9, features={'unexplained': 0, 'weak_metric': weak})


def test_weak_metric_lowers_confidence():
    cal = Calibration({}, 8, weak_metric_factor=0.5)
    c0, _ = rule_confidence(_p(0), SimpleNamespace(health={}), DecisionConfig(), cal)
    c1, m = rule_confidence(_p(1), SimpleNamespace(health={}), DecisionConfig(), cal)
    assert abs(c1 - c0 * 0.5) < 1e-9 and m['weak_metric'] and m['f_weak'] == 0.5


def test_weak_factor_from_rows():
    rows = [{'rule': {'s': 1, 'f': 1.0, 'weak': False}}] * 20 + [{'rule': {'s': 1, 'f': 1.0, 'weak': True}}] * 5 + \
           [{'rule': {'s': -1, 'f': 1.0, 'weak': True}}] * 5
    assert weak_factor(rows, 8) < 0.7
    assert weak_factor(rows[:20], 8) == 1.0                        # không đủ mẫu 'weak' -> giữ nguyên


def test_intent_marks_default_metric(tmp_path):
    from datamate.understand import intent
    assert 'weak_metric' in open(intent.__file__, encoding='utf-8').read()


def test_paraphrase_number_guard():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('mp', Path(__file__).resolve().parents[1] / 'scripts' / 'make_paraphrase.py')
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    assert mp._nums('top 3 tháng 12/2025') <= mp._nums('t12/2025 top 3 nha')
    assert not (mp._nums('top 3 tháng 12/2025') <= mp._nums('top 3 tháng 12'))
