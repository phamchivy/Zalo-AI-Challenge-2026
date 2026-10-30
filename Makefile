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

eval-train-nollm:
	python -m datamate.runtime.main predict --split $(TRAIN) --out $(OUT) --config configs/no_llm.yaml --eval --fresh

predict-public:
	python -m datamate.runtime.main predict --split $(PUBLIC) --out $(OUT) --config $(CONFIG) --fresh

docker:
	docker build -f docker/Dockerfile -t datamate:v1 .
