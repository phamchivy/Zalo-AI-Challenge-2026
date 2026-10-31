#!/usr/bin/env bash
# Đọc test set ở /data, ghi /result/submission.jsonl + /result/logs/. Chạy offline hoàn toàn.
#   docker run --gpus all -v /path/test:/data -v /path/out:/result datamate:v1
set -uo pipefail
export DATAMATE_T0=${DATAMATE_T0:-$(date +%s)}      # mốc 6 giờ tính từ lúc container khởi động (gồm nạp mô hình)
DATA=${DATA:-/data}
OUT=${OUT:-/result}
CONFIG=${CONFIG:-/app/configs/v2_agent.yaml}
MODE=${VLLM_MODE:-tp}                 # tp: 1 bản chia 2 GPU (chắc chắn ≤15B) | dp: mỗi GPU 1 bản
QUANT=${QUANT:-fp8}
MODEL_DIR=${DATAMATE_MODEL_DIR:-/models/llm}
SERVED=${DATAMATE_LLM_MODEL:-qwen}
MAX_LEN=${MAX_MODEL_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}
mkdir -p "$OUT/logs" /tmp/vllm_logs
cd /app

NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
URLS=""
PIDS=()
QARGS=()
if [ "$QUANT" = "fp8" ] && ! grep -qi '"quant_method"' "$MODEL_DIR/config.json" 2>/dev/null; then QARGS=(--quantization fp8); fi
COMMON=(--served-model-name "$SERVED" --host 127.0.0.1 --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
        --enable-prefix-caching --max-num-seqs 128 --seed 42 "${QARGS[@]}")
if [ "$NGPU" -ge 2 ] && [ "$MODE" = "tp" ] && [ -d "$MODEL_DIR" ]; then
  python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --tensor-parallel-size "$NGPU" --port 8000 \
    "${COMMON[@]}" > /tmp/vllm_logs/vllm_tp.log 2>&1 &
  PIDS+=($!)
  URLS="http://127.0.0.1:8000/v1"
  NGPU=1                                                # chỉ 1 server để chờ
elif [ "$NGPU" -ge 1 ] && [ -d "$MODEL_DIR" ]; then
  for ((i=0; i<NGPU; i++)); do
    PORT=$((8000 + i))
    CUDA_VISIBLE_DEVICES=$i python -m vllm.entrypoints.openai.api_server --model "$MODEL_DIR" --port $PORT \
      "${COMMON[@]}" > /tmp/vllm_logs/vllm_$i.log 2>&1 &
    PIDS+=($!)
    URLS="${URLS:+$URLS,}http://127.0.0.1:$PORT/v1"
  done
fi
if [ -n "$URLS" ]; then
  export DATAMATE_LLM_URLS="$URLS"
  echo "[predict] vLLM khởi động trên $NGPU GPU: $URLS"
  READY=0
  for t in $(seq 1 180); do
    ok=1
    for ((i=0; i<NGPU; i++)); do
      curl -sf "http://127.0.0.1:$((8000 + i))/v1/models" > /dev/null || ok=0
    done
    if [ $ok -eq 1 ]; then READY=1; break; fi
    sleep 5
  done
  if [ $READY -eq 0 ]; then
    echo "[predict] vLLM không lên sau 15 phút -> chạy chỉ tầng A (không LLM)"
    export DATAMATE_NO_LLM=1
  fi
else
  echo "[predict] Không có GPU hoặc model -> chạy chỉ tầng A"
  export DATAMATE_NO_LLM=1
fi

python -m datamate.runtime.main predict --split "$DATA" --out "$OUT" --config "$CONFIG"
RC=$?
cp /tmp/vllm_logs/*.log "$OUT/logs/" 2>/dev/null || true
for p in "${PIDS[@]:-}"; do [ -n "$p" ] && kill "$p" 2>/dev/null; done
# luôn có file nộp (câu thiếu = abstain)
if [ ! -s "$OUT/submission.jsonl" ]; then
  python - "$DATA" "$OUT" <<'PY'
import json, sys
d, o = sys.argv[1], sys.argv[2]
with open(f'{o}/submission.jsonl', 'w') as f:
    for l in open(f'{d}/questions.jsonl', encoding='utf-8'):
        if l.strip():
            f.write(json.dumps({'id': json.loads(l)['id'], 'type': 'abstain'}) + '\n')
PY
fi
exit $RC
