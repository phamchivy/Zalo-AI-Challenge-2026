# DataMate v3.0.0 — LLM ở bước dựng dữ liệu (bước 4 kế hoạch v3): gán nhãn + gán vai trò cột, dữ liệu quyết định cuối

Tiếp v3.0.0-a (suy cấu trúc từ giá trị). Phần này cần LLM; **mặc định TẮT** (`mapping_llm.enabled: false`) nên
`configs/no_llm.yaml` chạy y như v3.0.0-a.

## Kiến trúc (khung Magneto, LLM không có quyết định cuối)
| Bước | Làm gì | Ai quyết |
|---|---|---|
| Gán nhãn | Nhãn trạng thái / thanh toán mà **từ điển không hiểu** (đếm theo tần suất, ≤ 60 nhãn) -> LLM xếp vào tập đóng + `unknown` | LLM xếp; `unknown` giữ nguyên -> I15 đỏ -> chặn câu liên quan. Nhãn từ điển đã hiểu **không bị gửi, không bị ghi đè** |
| Gán vai trò | Bảng bán hàng còn thiếu vai trò cốt lõi -> **profile nén** của cột chưa gán (kiểu, % trống, số giá trị khác nhau, ~8 mẫu, mô tả README) + danh sách vai trò còn thiếu -> LLM chấm 0–1 | **Hungarian** gán một-một; chỉ nhận cặp ≥ `min_conf` (0.8) và hợp kiểu giá trị |
| Chọn | Dựng canonical **có** và **không có** vai trò LLM, so invariants (đỏ, vàng, số invariant kiểm được và xanh) | **Dữ liệu**: chỉ nhận khi tốt hơn hẳn; ngang nhau giữ bản không LLM |

- Không gửi bảng thô, chỉ profile. Không tên shop, không luật riêng.
- Cache câu trả lời theo hash (model + prompt) trong `<cache_dir>/llm_map/`: chạy lại không tốn lượt gọi.

## Đổi model (không sửa code)
| | Phát triển (bây giờ) | Nộp bài |
|---|---|---|
| Config | `configs/llm_map_glm.yaml` | `configs/llm_map_qwen14b.yaml` |
| Endpoint | API nội bộ GLM-4.7 | vLLM Qwen3-14B trong container (`http://127.0.0.1:8000/v1`) |
| `json_mode` | `prompt` (endpoint không ép được schema: schema trong prompt, bóc ```json```, kiểm, thử lại ≤ 3 lần) | `schema` (guided decoding ép đúng JSON) |
| Khoá | `export DM_LLM_KEY=...` (không bao giờ ghi vào file) | không cần |

Ghi đè nhanh bằng biến môi trường: `DM_LLM_URL`, `DM_LLM_MODEL`, `DM_LLM_KEY`. Module: `datamate/llm/json_chat.py`
(client chuẩn OpenAI), `datamate/data/llm_mapper.py`, cấu hình `MappingLLMConfig` trong `core/config.py`.
**Lưu ý:** API ngoài/nội bộ chỉ dùng với dữ liệu **train**. Public/private chỉ chạy bằng model local.

## Kiểm (không có model thật trong môi trường dev của Claude)
- Server giả lập trả lời bọc ```json``` và từ chối json_schema (như GLM) -> client tự chuyển sang `prompt`, parse đúng.
- Shop POS 17 bảng, bộ làm bẩn, LLM giả lập gán nhãn: `status` 17/0/101 -> **111/0/7** (= bản sạch). Con số thật phụ
  thuộc model; đây chỉ kiểm đường ống.
- `no_llm.yaml`: train mini 264/0/13 không đổi. pytest 85 passed (+`tests/test_v300.py`).

## Chạy (máy công ty)
```bash
git am datamate_v3.0.0.patch && git tag v3.0.0
export DM_LLM_KEY='<key>'
T=/home/vht/phamchivy/datamate/train
# 0) endpoint
python scripts/llm_check.py --config configs/llm_map_glm.yaml
# 1) bộ làm bẩn dev (đã có runs/dirty_dev_status, runs/dirty_dev_ren)
for P in status ren; do
  python -m datamate.runtime.main predict --split runs/dirty_dev_$P --out runs/dirty_dev_${P}_pred_300 --config configs/llm_map_glm.yaml --fresh --set workers=2
  python scripts/make_dirty.py report --split runs/dirty_dev_$P --pred runs/dirty_dev_${P}_pred_300 --errors --shops
done
# 2) train dev sạch có LLM: không được tụt so với no_llm (2105/1/260)
python -m datamate.runtime.main predict --split $T --split-file configs/split.yaml --part dev --out runs/v300_dev_glm --config configs/llm_map_glm.yaml --eval --fresh
```
Mục tiêu:
- `status`: số câu bỏ giảm mạnh từ 1967, sai giữ ≈ 0.
- `ren`: sai 0, số câu bỏ giảm.
- dev sạch: không tụt.

Chưa nộp public bằng GLM (không gửi dữ liệu public ra API). Nộp khi chạy được Qwen3-14B local (`llm_map_qwen14b.yaml`).
