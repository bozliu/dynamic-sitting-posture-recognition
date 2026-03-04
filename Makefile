PYTHON ?= python

.PHONY: install test image-demo evaluate calibrate benchmark prefetch-models build-macos-app build-dmg

install:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[dev]

test:
	pytest

image-demo:
	posture-recognition image \
		--input sample_images/straight_hf_flip.jpg \
		--output-image artifacts/demo_overlay.jpg \
		--output-json artifacts/demo_result.json

evaluate:
	posture-recognition evaluate --images-dir sample_images --report artifacts/eval.json

calibrate:
	posture-recognition calibrate --images-dir sample_images --write-config configs/calibration.yaml --report artifacts/calibration_report.json

benchmark:
	posture-recognition benchmark --images-dir sample_images --num-frames 300 --report artifacts/benchmark.json

prefetch-models:
	python scripts/prefetch_models.py --output-dir packaging/models

build-macos-app:
	bash scripts/build_macos_app.sh

build-dmg:
	bash scripts/build_dmg.sh v2.0.0
