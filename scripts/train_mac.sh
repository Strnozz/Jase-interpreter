#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
mlx_lm.lora --config configs/mlx_lora.yaml
