# V2 — chạy trên Vast.ai (2× RTX 5090)

Mọi lệnh chạy trong thư mục repo trên máy Vast. Dữ liệu đặt ở `/workspace/data/train` và `/workspace/data/public_test`.

## 0. Chuẩn bị máy

1. Thuê máy (on-demand), template PyTorch có **CUDA ≥ 12.8**, đĩa ≥ 200 GB.
2. Đưa code lên: `git clone` repo của team, hoặc chép code v1.3 rồi `git am datamate_v2.0.patch`.
3. Đưa dữ liệu lên: `rsync -avz train public_test root@<ip>:/workspace/data/ -e "ssh -p <port>"`.
4. Cài đặt và tải mô hình (khoảng 15–30 phút):
   ```bash
   bash scripts/vast/setup.sh            # cài vLLM, kiểm tra GPU, tải Qwen2.5-Coder-14B + Qwen3-14B vào /workspace/models
   ```
   Ghi lại các phiên bản `torch / cuda / vllm` mà script in ra. Sau này pin đúng phiên bản đó vào Docker.

## 1. Đo thông lượng (khoảng 10 phút / chế độ)

```bash
bash scripts/vast/serve.sh /workspace/models/qwen25coder                  # tp + fp8 (mặc định)
export DATAMATE_LLM_URLS=http://127.0.0.1:8000/v1
python scripts/bench_llm.py --split /workspace/data/public_test --concurrency 16 64 128
bash scripts/vast/stop.sh
MODE=dp bash scripts/vast/serve.sh /workspace/models/qwen25coder          # chỉ để so, dùng khi BTC cho phép
```
Gửi lại: số `tok/s` và `~câu agent / 5h` của từng chế độ.

## 2. Tập câu dev để so mô hình

```bash
# tầng A trên dev (CPU, vài phút) — bỏ qua nếu đã có runs/dev_raw từ máy team
python -m datamate.runtime.main predict --split /workspace/data/train --split-file configs/split.yaml --part dev \
   --out runs/dev_raw --config configs/no_llm_raw.yaml --eval --fresh
python -m datamate.runtime.main make-subset --split /workspace/data/train --run runs/dev_raw \
   --split-file configs/split.yaml --n 200 --out runs/subset_dev.txt
```
Tập này ưu tiên câu tầng A bỏ trống (55%), câu tầng A sai (25%), và thêm câu tầng A đúng (20%) để thấy LLM có làm hỏng câu dễ không.

## 3. So 2 mô hình (khoảng 1–2 giờ)

```bash
TRAIN=/workspace/data/train SUBSET=runs/subset_dev.txt bash scripts/vast/compare_models.sh
```
In ra bảng PA / đúng / sai / bỏ / thời gian / token cho từng mô hình. **Chọn theo PA**, sau đó kiểm tra thời gian.

## 4. Bật / tắt từng phần trên tập con (mô hình đã chọn)

```bash
bash scripts/vast/serve.sh /workspace/models/<mô hình>; export DATAMATE_LLM_URLS=http://127.0.0.1:8000/v1
P="python -m datamate.runtime.main predict --split /workspace/data/train --ids-file runs/subset_dev.txt --config configs/v2_agent.yaml --eval --fresh"
$P --out runs/ab_full                                                         # đủ cả v2.0 + v2.1 + v2.2
$P --out runs/ab_nonotes --set strategy.agent.options.shop_notes=false        # tắt ghi chú shop (v2.1)
$P --out runs/ab_norules --set strategy.agent.options.rule_hints=false        # tắt luật liên quan (v2.2)
$P --out runs/ab_k6      --set strategy.agent.samples=6                       # 6 lời giải
$P --out runs/ab_think   --set llm.disable_thinking=false                     # chỉ khi dùng Qwen3: bật suy nghĩ
```
Phần nào làm PA giảm thì tắt trong `configs/v2_agent.yaml`.

## 5. Chạy dev đầy đủ rồi chọn ngưỡng (v2.3)

```bash
python -m datamate.runtime.main predict --split /workspace/data/train --split-file configs/split.yaml --part dev \
   --out runs/v2_dev --config configs/v2_agent.yaml --eval --fresh
python -m datamate.runtime.main sweep --split /workspace/data/train --run runs/v2_dev --split-file configs/split.yaml
```
Sửa `decision.min_score / min_share` theo dòng đầu tiên, nên chọn chặt hơn một nấc. Không chỉnh ngưỡng theo holdout hay public.

## 6. Holdout (một con số) rồi public

```bash
python -m datamate.runtime.main predict --split /workspace/data/train --split-file configs/split.yaml --part holdout \
   --out runs/v2_holdout --config configs/v2_agent.yaml --eval --fresh
python -m datamate.runtime.main predict --split /workspace/data/public_test --out runs/v2_public \
   --config configs/v2_agent.yaml --fresh
```
Nộp `runs/v2_public/submission.jsonl`.

Gửi lại cho Claude:
- PA holdout và số câu sai;
- PA public;
- `runs/v2_public/run_summary.json` (thời gian, số câu đã gọi LLM);
- nén `runs/v2_public/logs`.

**Thời gian:** `elapsed_sec` của public (1000 câu) dùng để ước tính private. Nếu > 4 giờ thì giảm `samples` / `max_steps` hoặc tăng `llm_skip_support`.

## Lưu ý
- **Giới hạn 15B tính tổng:** mặc định chạy 1 bản mô hình chia qua 2 GPU (`MODE=tp`). Chế độ 2 bản (`MODE=dp`) chỉ dùng khi BTC xác nhận hợp lệ.
- **Cache:** lưu ở `/tmp/datamate_cache` (catalog, ghi chú shop). Đổi code phần dữ liệu thì xoá thư mục này.
- **Dừng giữa chừng:** chạy lại cùng lệnh nhưng bỏ `--fresh` để tiếp tục, các câu đã có sẽ được bỏ qua.
