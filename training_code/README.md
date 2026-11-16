# training_code — DataMate (đội Beyond Models)

**Không fine-tune mô hình, không sinh dữ liệu huấn luyện.** Mô hình suy luận là Qwen3-14B-FP8 gốc (xem `/code/README.md`).

Bước "huấn luyện" duy nhất là **hiệu chỉnh ngưỡng tin cậy** của tầng luật (tầng A). Bước này chỉ dùng **dữ liệu train do BTC cung cấp**, không dùng public hay private test. Kết quả là file `configs/calibration.yaml`, được dùng lúc suy luận tại `/code/src/configs/calibration.yaml`. Cả hai bản là cùng một file, so được bằng SHA-256 trong `data_manifest.json`.

## Thành phần

| Đường dẫn | Nội dung |
|---|---|
| `data/train/` | Dữ liệu train BTC cung cấp, nguyên bản: `data/<biz>/`, `questions.jsonl`, `answers.jsonl`, `rulebook.md`, `calendar.csv` |
| `configs/calibration.yaml` | Kết quả hiệu chỉnh (Wilson lower bound theo intent, ngưỡng `min_score`) |
| `configs/qa_qwen14b.yaml` | Cấu hình suy luận đã nộp (bản sao của `/code/src/configs/qa_qwen14b.yaml`) |
| `data_manifest.json` | Đường dẫn, kích thước, SHA-256 mọi file train và model; số shop / câu; revision HF của model. Sinh lúc build bằng `make_manifest.py` |
| Mã nguồn | Dùng chung với suy luận: `/code/src/datamate/offline/calibrate.py` (hiệu chỉnh), `/code/src/configs/no_llm_raw.yaml` (lần chạy thô) |

## Tái lập hiệu chỉnh (CPU, không cần GPU hay mạng; khoảng 10–20 phút)

```bash
cd /code/src
export PYTHONPATH=/code/src
TRAIN=/code/training_code/data/train
# 1) Tầng A trả lời mọi câu có kế hoạch (ngưỡng 0, chưa hiệu chỉnh)
python -m datamate.runtime.main predict --split $TRAIN --out /tmp/raw --config configs/no_llm_raw.yaml --fresh
# 2) Hiệu chỉnh theo intent (k-fold theo shop), ghi ngưỡng
python -m datamate.runtime.main calibrate --split $TRAIN --run /tmp/raw --out /tmp/calibration.yaml
diff <(grep -v '^#' /tmp/calibration.yaml) <(grep -v '^#' configs/calibration.yaml)
```

- **Phần cứng gốc:** máy CPU, không GPU.
- **Tất định:** tầng A là pandas tất định, không có ngẫu nhiên.
- **Không dùng LLM** trong bước này.
- **Sai khác có thể gặp:** nếu file nộp được hiệu chỉnh trên một phần train (tập dev chia theo shop), số trong file có thể lệch nhẹ so với khi chạy lại trên toàn bộ train. Ngưỡng nộp là file `configs/calibration.yaml` kèm theo.
