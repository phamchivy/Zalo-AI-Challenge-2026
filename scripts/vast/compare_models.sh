#!/usr/bin/env bash
# So 2 mô hình trên cùng tập câu dev (make-subset): cùng code agent, cùng cấu hình, chỉ đổi mô hình.
#   TRAIN=/workspace/data/train SUBSET=runs/subset_dev.txt bash scripts/vast/compare_models.sh
# Kết quả: runs/cmp_<model>/ (eval_report.json, errors.jsonl, run_summary.json) + bảng tóm tắt cuối.
set -uo pipefail
cd "$(dirname "$0")/../.."
TRAIN=${TRAIN:?đường dẫn split train}
SUBSET=${SUBSET:-runs/subset_dev.txt}
MODEL_ROOT=${MODEL_ROOT:-/workspace/models}
MODELS=${MODELS:-"qwen25coder qwen3"}
MODE=${MODE:-tp}
CONFIG=${CONFIG:-configs/v2_agent.yaml}
EXTRA=${EXTRA:-}
for m in $MODELS; do
  bash scripts/vast/stop.sh > /dev/null
  eval "$(MODE=$MODE bash scripts/vast/serve.sh "$MODEL_ROOT/$m" | grep '^export')"
  echo "== $m ($DATAMATE_LLM_URLS)"
  python -m datamate.runtime.main predict --split "$TRAIN" --ids-file "$SUBSET" --out "runs/cmp_$m" \
     --config "$CONFIG" --eval --fresh $EXTRA | tail -25
done
bash scripts/vast/stop.sh > /dev/null
python - "$MODELS" <<'PY'
import json, sys
print(f"\n{'mô hình':14s} {'PA':>7s} {'đúng':>5s} {'sai':>4s} {'bỏ':>4s} {'câu LLM':>8s} {'giây':>7s} {'token':>10s}")
for m in sys.argv[1].split():
    try:
        r = json.load(open(f'runs/cmp_{m}/eval_report.json')); s = json.load(open(f'runs/cmp_{m}/run_summary.json'))
    except FileNotFoundError:
        print(m, 'chưa có kết quả'); continue
    print(f"{m:14s} {r['PA']:+7.4f} {r['correct']:5d} {r['wrong']:4d} {r['abstain']:4d} {s.get('llm_questions', 0):8d} "
          f"{s['elapsed_sec']:7.0f} {s.get('llm_tokens', 0):10d}")
PY
