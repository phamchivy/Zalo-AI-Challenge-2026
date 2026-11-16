"""v3.4.0 — sinh /code/training_code/data_manifest.json lúc build image (đường dẫn, kích thước, SHA-256, số câu).
  python make_manifest.py /code
"""
import hashlib
import json
import sys
from pathlib import Path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def files(root: Path):
    out = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and '.cache' not in p.parts:
            out.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)})
    return out


def hf_revision(model_dir: Path):
    """Commit HF của bản tải về (huggingface-cli ghi trong .cache/huggingface/download/*.metadata, dòng 1)."""
    for m in sorted((model_dir / '.cache' / 'huggingface' / 'download').glob('*.metadata')):
        try:
            return m.read_text().splitlines()[0].strip()
        except Exception:  # noqa: BLE001
            pass
    return None


def count_lines(p: Path) -> int:
    return sum(1 for l in p.read_text(encoding='utf-8').splitlines() if l.strip()) if p.exists() else 0


def main(code='/code'):
    code = Path(code)
    tr = code / 'training_code' / 'data' / 'train'
    md = code / 'models' / 'qwen3-14b-fp8'
    biz = sorted(p.name for p in (tr / 'data').iterdir() if p.is_dir()) if (tr / 'data').exists() else []
    man = {
        'note': 'Không fine-tune mô hình. "Huấn luyện" duy nhất: hiệu chỉnh ngưỡng tin cậy của tầng luật (configs/calibration.yaml) '
                'trên dữ liệu train BTC cung cấp. Không dùng dữ liệu public/private test.',
        'datasets': [{
            'name': 'DataMate training split (BTC cung cấp)', 'local_path': str(tr),
            'businesses': len(biz), 'questions': count_lines(tr / 'questions.jsonl'), 'answers': count_lines(tr / 'answers.jsonl'),
            'files': files(tr)}],
        'calibration': {'local_path': str(code / 'training_code' / 'configs' / 'calibration.yaml'),
                        'sha256': sha256(code / 'training_code' / 'configs' / 'calibration.yaml')},
        'inference_model': {
            'name': 'Qwen3-14B-FP8', 'public_url': 'https://huggingface.co/Qwen/Qwen3-14B-FP8',
            'revision': hf_revision(md), 'license': 'Apache-2.0', 'parameters': '14.8B',
            'local_path': str(md), 'files': files(md)},
    }
    out = code / 'training_code' / 'data_manifest.json'
    out.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'đã ghi {out}: {len(man["datasets"][0]["files"])} file train, {len(man["inference_model"]["files"])} file model, '
          f'revision={man["inference_model"]["revision"]}')


if __name__ == '__main__':
    main(*sys.argv[1:])
