#!/usr/bin/env bash
# Khởi động vLLM cho DataMate.
#   bash scripts/vast/serve.sh /workspace/models/qwen25coder            # mặc định: tp (1 bản, chia 2 GPU), fp8
#   MODE=dp bash scripts/vast/serve.sh /workspace/models/qwen25coder    # 2 bản, mỗi GPU 1 bản (cổng 8000, 8001)
#   QUANT=none bash scripts/vast/serve.sh ...                           # bf16
# In ra biến DATAMATE_LLM_URLS cần export cho lần predict.
set -euo pipefail
MODEL=${1:?đường dẫn mô hình}
MODE=${MODE:-tp}
QUANT=${QUANT:-fp8}
MAX_LEN=${MAX_LEN:-32768}
GPU_UTIL=${GPU_UTIL:-0.90}
MAX_SEQS=${MAX_SEQS:-128}
LOG_DIR=${LOG_DIR:-/workspace/vllm_logs}
mkdir -p "$LOG_DIR"
QARGS=()
if [ "$QUANT" = "fp8" ] && ! grep -qi '"quant_method"' "$MODEL/config.json" 2>/dev/null; then
  QARGS=(--quantization fp8)          # lượng tử hoá FP8 lúc nạp (mô hình bf16); checkpoint FP8 sẵn thì không cần
fi
COMMON=(--served-model-name qwen --max-model-len "$MAX_LEN" --gpu-memory-utilization "$GPU_UTIL"
        --enable-prefix-caching --max-num-seqs "$MAX_SEQS" --seed 42 "${QARGS[@]}")
if [ "$MODE" = "tp" ]; then
  nohup python -m vllm.entrypoints.openai.api_server --model "$MODEL" --tensor-parallel-size 2 --port 8000 \
     --host 127.0.0.1 "${COMMON[@]}" > "$LOG_DIR/vllm_tp.log" 2>&1 &
  URLS="http://127.0.0.1:8000/v1"
else
  for i in 0 1; do
    CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server --model "$MODEL" --port $((8000 + i)) \
       --host 127.0.0.1 "${COMMON[@]}" > "$LOG_DIR/vllm_dp$i.log" 2>&1 &
  done
  URLS="http://127.0.0.1:8000/v1,http://127.0.0.1:8001/v1"
fi
T0=$(date +%s)
for u in ${URLS//,/ }; do
  until curl -sf "$u/models" > /dev/null; do
    sleep 5
    if [ $(( $(date +%s) - T0 )) -gt 1500 ]; then echo "vLLM không lên sau 25 phút, xem $LOG_DIR"; exit 1; fi
  done
done
echo "vLLM sẵn sàng sau $(( $(date +%s) - T0 ))s (mode=$MODE quant=$QUANT)"
echo "export DATAMATE_LLM_URLS=$URLS"
