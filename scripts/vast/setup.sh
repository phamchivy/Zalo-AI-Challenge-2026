#!/usr/bin/env bash
# Chuẩn bị máy Vast.ai (2x RTX 5090): cài vLLM + DataMate, tải mô hình, kiểm tra GPU Blackwell.
#   bash scripts/vast/setup.sh            # cài + tải cả 2 mô hình ứng viên
#   MODELS="qwen25coder" bash scripts/vast/setup.sh
set -euo pipefail
MODELS=${MODELS:-"qwen25coder qwen3"}
MODEL_ROOT=${MODEL_ROOT:-/workspace/models}
cd "$(dirname "$0")/../.."

echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -i "CUDA Version" || true

echo "== Python packages (vLLM bản mới nhất có wheel CUDA >= 12.8 cho Blackwell; ghi lại phiên bản để pin vào Docker)"
python -m pip install -U pip
python -m pip install -U vllm "huggingface_hub[cli]"
python -m pip install -r requirements.txt
python - <<'PY'
import torch, vllm
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'vllm', vllm.__version__)
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), 'capability', torch.cuda.get_device_capability(i))
assert torch.cuda.is_available(), 'không thấy GPU'
x = torch.randn(1024, 1024, device='cuda', dtype=torch.bfloat16)
print('phép nhân thử trên GPU:', float((x @ x).float().abs().mean()) > 0, '(lỗi "no kernel image" = torch chưa hỗ trợ sm_120)')
PY

mkdir -p "$MODEL_ROOT"
for m in $MODELS; do
  case $m in
    qwen25coder) repo=Qwen/Qwen2.5-Coder-14B-Instruct ;;
    qwen3)       repo=Qwen/Qwen3-14B ;;
    qwen3fp8)    repo=Qwen/Qwen3-14B-FP8 ;;
    *) echo "mô hình lạ: $m"; exit 1 ;;
  esac
  echo "== tải $repo -> $MODEL_ROOT/$m"
  huggingface-cli download "$repo" --local-dir "$MODEL_ROOT/$m"
done
du -sh "$MODEL_ROOT"/* || true
echo "XONG. Tiếp theo: bash scripts/vast/serve.sh $MODEL_ROOT/qwen25coder"
