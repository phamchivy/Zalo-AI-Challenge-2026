#!/usr/bin/env bash
# Dừng mọi server vLLM (trước khi đổi mô hình / chế độ).
pkill -f vllm.entrypoints.openai.api_server || true
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv
