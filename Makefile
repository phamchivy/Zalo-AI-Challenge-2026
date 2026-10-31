# Lệnh tắt cho nhóm. TRAIN/PUBLIC trỏ tới thư mục split đã giải nén.
TRAIN ?= /data/train
PUBLIC ?= /data/public_test
OUT ?= runs/$(shell date +%m%d_%H%M)
CONFIG ?= configs/baseline.yaml

test:
	python -m pytest -q tests

regress:                       ## tầng A trên train thật (CPU)
	DATAMATE_TRAIN_SPLIT=$(TRAIN) python -m pytest -q tests/test_end_to_end.py -s

build:
	python -m datamate.runtime.main build --split $(TRAIN) -v

eval-train:                    ## chạy toàn bộ train + chấm
	python -m datamate.runtime.main predict --split $(TRAIN) --out $(OUT) --config $(CONFIG) --eval --fresh

rules:                         ## luật đọc được từ rulebook của split
	python -m datamate.runtime.main rules --split $(PUBLIC)

calibrate:                     ## chạy thô trên train + hiệu chỉnh out-of-fold -> configs/calibration.yaml
	python -m datamate.runtime.main predict --split $(TRAIN) --out runs/raw --config configs/no_llm_raw.yaml --eval --fresh
	python -m datamate.runtime.main calibrate --split $(TRAIN) --run runs/raw --out configs/calibration.yaml

SEEN ?= biz_101 biz_102 biz_104 biz_105 biz_106 biz_109 biz_110 biz_112 biz_116 biz_117 biz_122 biz_123 biz_124 biz_127 biz_129 biz_130

split:                         ## chia dev / holdout (holdout = shop chưa ai xem lỗi) -> configs/split.yaml
	python -m datamate.runtime.main split-info --split $(TRAIN)
	python -m datamate.runtime.main make-split --split $(TRAIN) --seen $(SEEN) --out configs/split.yaml

dev:                           ## chạy thô trên dev + hiệu chỉnh chỉ trên dev
	python -m datamate.runtime.main predict --split $(TRAIN) --split-file configs/split.yaml --part dev --out runs/dev_raw --config configs/no_llm_raw.yaml --eval --fresh
	python -m datamate.runtime.main calibrate --split $(TRAIN) --run runs/dev_raw --split-file configs/split.yaml --out configs/calibration.yaml

holdout:                       ## 1 con số PA trên holdout (không xuất lỗi)
	python -m datamate.runtime.main predict --split $(TRAIN) --split-file configs/split.yaml --part holdout --out runs/holdout --config configs/no_llm.yaml --eval --fresh

v2-sweep:                      ## v2.3: chọn ngưỡng bỏ phiếu trên dev từ logs của lần chạy v2 (RUN=runs/v2_dev)
	python -m datamate.runtime.main sweep --split $(TRAIN) --run $(RUN) --split-file configs/split.yaml

eval-train-nollm:
	python -m datamate.runtime.main predict --split $(TRAIN) --out $(OUT) --config configs/no_llm.yaml --eval --fresh

predict-public:
	python -m datamate.runtime.main predict --split $(PUBLIC) --out $(OUT) --config $(CONFIG) --fresh

docker:
	docker build -f docker/Dockerfile -t datamate:v2.0 .
