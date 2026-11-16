# v3.4.0 — image nộp DataMate theo "Docker Submission Guide" của BTC (mục 4, 9).
# Build (2 thư mục lớn đưa vào bằng build-context, không cần chép vào repo):
#   docker build --platform linux/amd64 \
#     --build-context models=/đường/dẫn/models/llm \
#     --build-context traindata=/đường/dẫn/train \
#     -t zaic2026_datamate_beyond_models:final .
FROM vllm/vllm-openai:v0.10.1.1
ENTRYPOINT []
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PIP_NO_CACHE_DIR=1 NCCL_P2P_DISABLE=1 \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 \
    PYTHONPATH=/code/src DATAMATE_MODEL_DIR=/code/models/qwen3-14b-fp8 DATAMATE_LLM_MODEL=qwen3-14b
WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt
RUN pip install --no-cache-dir --no-deps nvidia-nccl-cu12==2.27.7
# trọng số (lớp riêng, giữ cache khi sửa code)
COPY --from=models . /code/models/qwen3-14b-fp8/
COPY --from=traindata . /code/training_code/data/train/
COPY training_code/ /code/training_code/
COPY datamate/ /code/src/datamate/
COPY configs/ /code/src/configs/
COPY docker/predict.sh /code/predict.sh
COPY docker/README_SUBMIT.md /code/README.md
RUN chmod +x /code/predict.sh \
 && test -f /code/models/qwen3-14b-fp8/config.json \
 && test -f /code/src/configs/calibration.yaml \
 && test -f /code/training_code/data/train/questions.jsonl \
 && cp /code/src/configs/calibration.yaml /code/src/configs/qa_qwen14b.yaml /code/training_code/configs/ \
 && cd /code/src && python -c "import datamate.reasoning.strategies, datamate.agent.agent, scipy" \
 && python /code/training_code/make_manifest.py /code
CMD ["bash", "/code/predict.sh"]
