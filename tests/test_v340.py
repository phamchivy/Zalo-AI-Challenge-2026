"""v3.4.0: log mỗi câu đúng guideline BTC (1 dòng JSON: id, answer, elapsed_ms, thinking, code, output) + manifest."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

from datamate.core.config import load_config
from datamate.runtime.orchestrator import Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def test_logs_are_one_line_with_btc_fields(mini_split, tmp_path):
    cfg = load_config(str(ROOT / 'configs' / 'no_llm.yaml'), {'cache_dir': str(tmp_path / 'c'), 'workers': 1,
                                                              'decision': {'calibration_path': None}})
    out = tmp_path / 'out'
    asyncio.run(Orchestrator(cfg, str(mini_split), str(out)).run(resume=False))
    sub = {json.loads(l)['id']: json.loads(l) for l in (out / 'submission.jsonl').read_text(encoding='utf-8').splitlines()}
    logs = list((out / 'logs').glob('*.json'))
    assert len(logs) == len(sub)
    answered = 0
    for p in logs:
        txt = p.read_text(encoding='utf-8')
        assert txt.count('\n') == 1 and txt.endswith('\n')                  # đúng 1 dòng
        r = json.loads(txt)
        assert {'id', 'answer', 'elapsed_ms', 'thinking', 'code', 'output'} <= set(r)
        assert r['id'] == p.stem and isinstance(r['elapsed_ms'], int) and r['elapsed_ms'] >= 0 and r['thinking'] is None
        assert r['answer'] == {k: v for k, v in sub[r['id']].items() if k != 'id'}
        assert r['qid'] == r['id'] and 'candidates' in r                     # trace cũ vẫn ở cùng cấp
        if r['answer']['type'] not in ('abstain', 'unanswerable'):
            answered += 1
            assert r['code'] == 'src/datamate/reasoning/executor.py::Executor.run' and r['inputs'] and r['output'] is not None
    assert answered


def test_manifest(tmp_path):
    code = tmp_path / 'code'
    (code / 'training_code' / 'data' / 'train' / 'data' / 'biz_1').mkdir(parents=True)
    (code / 'training_code' / 'data' / 'train' / 'questions.jsonl').write_text('{"id":"a"}\n{"id":"b"}\n')
    (code / 'training_code' / 'configs').mkdir(parents=True)
    (code / 'training_code' / 'configs' / 'calibration.yaml').write_text('x: 1\n')
    md = code / 'models' / 'qwen3-14b-fp8'
    (md / '.cache' / 'huggingface' / 'download').mkdir(parents=True)
    (md / 'config.json').write_text('{}')
    (md / '.cache' / 'huggingface' / 'download' / 'config.json.metadata').write_text('abc123\netag\n')
    subprocess.run([sys.executable, str(ROOT / 'training_code' / 'make_manifest.py'), str(code)], check=True)
    m = json.loads((code / 'training_code' / 'data_manifest.json').read_text(encoding='utf-8'))
    assert m['inference_model']['revision'] == 'abc123' and m['datasets'][0]['questions'] == 2
    assert m['datasets'][0]['businesses'] == 1 and len(m['inference_model']['files']) == 1
    assert len(m['calibration']['sha256']) == 64
