"""End-to-end không GPU trên split tổng hợp + (tuỳ chọn) regression trên train thật nếu đặt DATAMATE_TRAIN_SPLIT."""
import asyncio
import json
import os
from pathlib import Path

import pytest

from datamate.core.config import load_config
from datamate.offline.eval import evaluate
from datamate.runtime.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def _run(split, out, cache):
    cfg = load_config(str(ROOT / 'configs' / 'no_llm.yaml'), {'cache_dir': str(cache), 'workers': 1})
    asyncio.run(Orchestrator(cfg, str(split), str(out)).run(resume=False))
    return evaluate(str(split), str(Path(out) / 'submission.jsonl'))


def test_mini_split(mini_split, tmp_path):
    rep = _run(mini_split, tmp_path / 'out', tmp_path / 'cache')
    rows = {json.loads(l)['id']: json.loads(l) for l in (tmp_path / 'out' / 'submission.jsonl').read_text().splitlines()}
    assert rep['wrong'] == 0, rows
    assert rep['correct'] >= 9, rows
    assert (tmp_path / 'out' / 'logs' / 't1.json').exists()


def test_no_rulebook_no_guess(mini_split, tmp_path):
    """Rulebook rỗng -> tầng A không có luật -> bỏ trống, không được trả lời sai câu nào (không dùng mặc định train)."""
    import shutil
    split = tmp_path / 'split'
    shutil.copytree(mini_split, split)
    (split / 'rulebook.md').write_text('# Rulebook rỗng\n', encoding='utf-8')
    rep = _run(split, tmp_path / 'out', tmp_path / 'cache')
    assert rep['wrong'] == 0
    assert rep['correct'] <= 1          # chỉ guard (ngoài khoảng dữ liệu) còn trả lời


def test_changed_rule_disables_intent(mini_split, tmp_path):
    """Đổi luật làm tròn AOV sang cách code chưa biết -> câu AOV bỏ trống, câu khác vẫn trả lời."""
    import shutil
    split = tmp_path / 'split'
    shutil.copytree(mini_split, split)
    rb = (split / 'rulebook.md').read_text(encoding='utf-8').replace('nửa đồng làm tròn lên', 'làm tròn xuống hàng nghìn')
    (split / 'rulebook.md').write_text(rb, encoding='utf-8')
    rep = _run(split, tmp_path / 'out', tmp_path / 'cache')
    rows = {json.loads(l)['id']: json.loads(l) for l in (tmp_path / 'out' / 'submission.jsonl').read_text().splitlines()}
    assert rows['t7']['type'] == 'abstain'
    assert rows['t1']['type'] == 'number' and rep['wrong'] == 0


@pytest.mark.skipif(not os.environ.get('DATAMATE_TRAIN_SPLIT'), reason='đặt DATAMATE_TRAIN_SPLIT=/path/train để chạy regression')
def test_train_regression(tmp_path):
    split = Path(os.environ['DATAMATE_TRAIN_SPLIT'])
    rep = _run(split, tmp_path / 'out', os.environ.get('DATAMATE_CACHE', tmp_path / 'cache'))
    min_pa = float(os.environ.get('DATAMATE_MIN_PA', '0.3'))
    print(json.dumps({k: rep[k] for k in ('n', 'PA', 'correct', 'wrong', 'abstain')}))
    assert rep['PA'] >= min_pa


def test_generic_primitives(mini_split, tmp_path):
    """Hàng ế + đếm ngày đạt ngưỡng + xếp hạng theo định nghĩa rulebook trên split tổng hợp."""
    from datetime import date as D
    from datamate.core.types import Plan, TimeRange
    from datamate.data.catalog import build_catalog
    from datamate.knowledge.base import KnowledgeBase
    from datamate.reasoning.executor import Executor
    kb = KnowledgeBase.load(mini_split)
    cat = build_catalog(mini_split / 'data' / 'biz_t01', str(tmp_path / 'c'))
    ex = Executor(cat.canon, kb.rulebook.params, kb.rules)
    jul = TimeRange(start=D(2025, 7, 1), end=D(2025, 7, 31))
    jun = TimeRange(start=D(2025, 6, 1), end=D(2025, 6, 30))
    assert kb.rules.get('dead_stock.def') is None
    import pytest as _pt
    from datamate.reasoning.executor import RuleMissing
    with _pt.raises(RuleMissing):                                   # rulebook fixture không định nghĩa "ế"
        ex.run(Plan(metric='dead_stock', op='list', periods=[jul]))
    kb.rules.knobs['dead_stock.def'] = 'sold_before_not_in'
    assert ex.run(Plan(metric='dead_stock', op='list', periods=[jul]))[1] == ['Bánh mì']
    kb.rules.knobs['threshold.strict'] = True
    t, a = ex.run(Plan(metric='revenue', op='count_ge', group_by='day', periods=[jun], threshold=100000, threshold_metric='revenue'))
    assert a == 1                                                    # chỉ 02/06 đạt 250.000
    t, a = ex.run(Plan(metric='active_days', op='top', group_by='customer', periods=[jun], k=1))
    assert a == 'Chị Lan'
