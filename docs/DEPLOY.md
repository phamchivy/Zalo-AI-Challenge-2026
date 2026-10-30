# Triển khai lên máy GPU thật (vLLM)

## 1. Thuê máy
* 2 × RTX 5090 (32 GB/GPU), RAM ≥ 64 GB, disk ≥ 200 GB, driver NVIDIA hỗ trợ CUDA ≥ 12.8 (Blackwell). 1 GPU vẫn chạy được (xem §6).
* vast.ai / RunPod: chọn template có Docker + CUDA 12.8. Mở SSH, không cần mở cổng ra ngoài (vLLM chỉ nghe 127.0.0.1).

## 2. Cài đặt
```bash
git clone <repo> datamate && cd datamate
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install "vllm==<bản đã kiểm tra chạy trên 5090>"      # ghi lại đúng phiên bản vào docker/Dockerfile (VLLM_IMAGE)
huggingface-cli download Qwen/Qwen3-14B-AWQ --local-dir models/llm   # hoặc model ≤ 15B khác; tổng tham số mọi model ≤ 15B
```
Giải nén dữ liệu: `/data/train`, `/data/public_test` (mỗi split có `data/`, `rulebook.md`, `calendar.csv`, `questions.jsonl`).

## 3. Chạy vLLM khi phát triển (mỗi GPU một server)
```bash
for i in 0 1; do
  CUDA_VISIBLE_DEVICES=$i nohup python -m vllm.entrypoints.openai.api_server \
    --model models/llm --served-model-name qwen --host 127.0.0.1 --port $((8000+i)) \
    --max-model-len 32768 --gpu-memory-utilization 0.90 --enable-prefix-caching --seed 42 \
    > vllm_$i.log 2>&1 &
done
curl -s localhost:8000/v1/models && curl -s localhost:8001/v1/models
```
`configs/baseline.yaml` đã trỏ 2 URL này; đổi bằng `export DATAMATE_LLM_URLS=http://127.0.0.1:8000/v1,...`.
Prefix caching quan trọng: prompt = [rulebook] + [README/bảng của shop] + [câu hỏi]; orchestrator xử lý câu theo từng shop nên
phần đầu được cache, mỗi câu chỉ tốn phần câu hỏi + code sinh ra.

## 4. Vòng đánh giá
```bash
python -m datamate.runtime.main ask --split /data/train --id q_0c8f3e07768f --show-program     # smoke test 1 câu
make eval-train TRAIN=/data/train OUT=runs/full_v1                                               # toàn bộ train + chấm
make eval-train-nollm TRAIN=/data/train OUT=runs/a_v1                                            # chỉ tầng A (CPU)
make predict-public PUBLIC=/data/public_test OUT=runs/pub_v1                                     # tạo file nộp public
```
Ước lượng thời gian (cần đo lại trên máy thật):

| Bước | 2 × 5090 | 1 × 5090 |
|---|---|---|
| vLLM nạp model 14B | 2–5 phút | 2–5 phút |
| Build catalog (cache lại cho lần sau) | 1–3 phút | 1–3 phút |
| Train 3000 câu, tầng A + B | ~30–50 phút | ~60–90 phút |
| Public ~1000 câu | ~12–20 phút | ~25–40 phút |
| Private ~500 câu | ~8–12 phút | ~15–25 phút |
| Chỉ tầng A (CPU), train | 2–5 phút | 2–5 phút |

Chạy lại sau khi sửa code: catalog tự rebuild khi code tầng `data/` đổi (cache theo hash code + file), nếu chỉ sửa prompt / luật
câu hỏi thì catalog dùng cache. `--fresh` để không resume submission cũ.

## 5. Docker để nộp (hạn 16/11)
```bash
# trọng số phải nằm trong thư mục build: ./models/llm (đã tải ở bước 2)
docker build -f docker/Dockerfile --build-arg VLLM_IMAGE=vllm/vllm-openai:<tag đã test> -t datamate:v1 .
# chạy thử y như BTC
docker run --rm --gpus all --network none -v /data/public_test:/data:ro -v $PWD/out_docker:/result datamate:v1
# đóng gói
docker save datamate:v1 | gzip > datamate_v1.tar.gz && sha256sum datamate_v1.tar.gz > datamate_v1.sha256
```
`predict.sh`: tự đếm GPU → mỗi GPU một vLLM → chờ sẵn sàng (tối đa 15 phút) → chạy agent → luôn ghi `submission.jsonl`
(nếu vLLM hỏng thì chạy tầng A; nếu agent hỏng thì mọi câu abstain) → log từng câu ở `/result/logs/`.
Kiểm tra `--network none` để chắc chắn không có gọi mạng.

## 6. Một GPU
`predict.sh` tự dùng 1 server; khi dev dùng `--config configs/single_gpu.yaml` (1 mẫu code/câu, ít song song hơn).
Rủi ro chính là thời gian: theo dõi `run_summary.json`; `time_budget_sec` đảm bảo quá giờ thì chỉ chạy tầng A.

## 7. Tính ổn định khi BTC chạy lại (|ΔPA| ≤ 0.03)
* temperature 0 cho mẫu đầu, seed cố định cho các mẫu sau; `--seed 42` cho vLLM; tầng A deterministic.
* Không phụ thuộc thời gian thực (mọi "hôm nay" lấy từ `as_of_date`).
* Nộp kèm `logs/` để BTC đối chiếu khi khác biệt.
