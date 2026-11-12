"""v3.0.0 — Kiểm nhanh endpoint LLM của bước dựng dữ liệu: gọi một câu gán nhãn mẫu, in kết quả đã kiểm + thời gian.
  python scripts/llm_check.py --config configs/llm_map_glm.yaml
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datamate.core.config import load_config  # noqa: E402
from datamate.data.llm_mapper import STATUS_SET, _SYS, _enum_prompt  # noqa: E402
from datamate.llm.json_chat import JSONChat  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', required=True)
a = ap.parse_args()
cfg = load_config(a.config).mapping_llm.model_dump()
cfg['cache'] = False
cli = JSONChat(cfg)
labels = [('đã chốt đơn', 120), ('khách bùng', 14), ('khách gửi trả lại', 6), ('đang đi giao', 9), ('???', 1)]
user, schema = _enum_prompt('status', labels, 'trạng thái đơn hàng')


def validate(o):
    mp = o['map']
    assert all(v in STATUS_SET for v in mp.values()), mp
    return mp


t = time.time()
print(f"endpoint={cfg['base_url']} model={cfg['model']} json_mode={cfg['json_mode']}")
res = cli.ask(_SYS, user, schema, validate)
print(f'kết quả ({time.time() - t:.1f}s, {cli.calls} lần gọi, json_mode cuối={cli.mode}):', res)
print('OK' if res else 'THẤT BẠI — xem log phía trên')
