.PHONY: setup data validate baseline train train-lowmem eval serve

setup:
	bash scripts/bootstrap_mac.sh

data:
	python scripts/generate_dataset.py --train 3000 --valid 300 --test 500

validate:
	python scripts/validate_dataset.py

baseline:
	python scripts/eval_mlx.py --limit 50

train:
	bash scripts/train_mac.sh

train-lowmem:
	bash scripts/train_mac_lowmem.sh

eval:
	python scripts/eval_mlx.py --adapter adapters/jase-v1 --limit 100

serve:
	python scripts/serve_mlx.py --adapter adapters/jase-v1

# ------------------------------------------------------------------ v2
.PHONY: test gold data-v2 train-v2 train-v2-lowmem eval-base eval-v1 eval-v2 compare runner

test:
	python tests/test_guard.py

gold:
	python benchmarks/build_gold.py

data-v2:
	python scripts/generate_dataset_v2.py --n 14000 --check 1200

train-v2:
	mlx_lm.lora --config configs/mlx_lora_v2.yaml

train-v2-lowmem:
	mlx_lm.lora --config configs/mlx_lora_v2_lowmem.yaml

eval-base:
	python scripts/eval_v2.py --label base --report reports/000-base.json

eval-v1:
	python scripts/eval_v2.py --adapter adapters/jase-v1-iter400 --label jase-v1-iter400 --report reports/001-v1-iter400.json

eval-v2:
	python scripts/eval_v2.py --adapter adapters/jase-v2 --label jase-v2 --report reports/002-v2.json

compare:
	python scripts/compare_reports.py

runner:
	.venv/bin/python scripts/jase_runner.py

coverage:
	python scripts/coverage_check.py data/v9/train.jsonl
