#!/usr/bin/env bash
set -euo pipefail
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "This Mac setup requires Apple Silicon (arm64 / M1 or newer)."
  exit 1
fi
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-mac.txt
python scripts/generate_dataset.py --train 3000 --valid 300 --test 500
python scripts/validate_dataset.py
cat <<'EOF'

Ready.
1) Baseline: python scripts/infer_mlx.py "Trovami un panino con la mortadella a Milano sotto i 10 euro"
2) Train:    mlx_lm.lora --config configs/mlx_lora.yaml
3) Test:     python scripts/infer_mlx.py --adapter adapters/jase-v1 "Trovami uno Sphynx femmina a Milano sotto i 1200 euro"
EOF
