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


@pytest.mark.skipif(not os.environ.get('DATAMATE_TRAIN_SPLIT'), reason='đặt DATAMATE_TRAIN_SPLIT=/path/train để chạy regression')
def test_train_regression(tmp_path):
    split = Path(os.environ['DATAMATE_TRAIN_SPLIT'])
    rep = _run(split, tmp_path / 'out', os.environ.get('DATAMATE_CACHE', tmp_path / 'cache'))
    min_pa = float(os.environ.get('DATAMATE_MIN_PA', '0.3'))
    print(json.dumps({k: rep[k] for k in ('n', 'PA', 'correct', 'wrong', 'abstain')}))
    assert rep['PA'] >= min_pa
