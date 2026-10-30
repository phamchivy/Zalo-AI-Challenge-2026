# Một image duy nhất: vLLM (OpenAI server) + DataMate agent + trọng số model.
# Pin đúng tag vLLM đã test trên RTX 5090 (Blackwell cần CUDA >= 12.8), ví dụ: --build-arg VLLM_IMAGE=vllm/vllm-openai:v0.11.0
ARG VLLM_IMAGE=vllm/vllm-openai:latest
FROM ${VLLM_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    DATAMATE_MODEL_DIR=/models/llm DATAMATE_LLM_MODEL=qwen

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# trọng số model (đặt sẵn ở models/llm khi build: huggingface-cli download ... --local-dir models/llm)
COPY models/llm /models/llm
COPY datamate /app/datamate
COPY configs /app/configs
COPY docker/predict.sh /app/predict.sh
COPY README.md /app/README.md
RUN chmod +x /app/predict.sh && python -c "import datamate.reasoning.strategies"

ENTRYPOINT []
CMD ["/app/predict.sh"]
