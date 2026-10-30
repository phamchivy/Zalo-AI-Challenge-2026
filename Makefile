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

eval-train-nollm:
	python -m datamate.runtime.main predict --split $(TRAIN) --out $(OUT) --config configs/no_llm.yaml --eval --fresh

predict-public:
	python -m datamate.runtime.main predict --split $(PUBLIC) --out $(OUT) --config $(CONFIG) --fresh

docker:
	docker build -f docker/Dockerfile -t datamate:v1.1 .
