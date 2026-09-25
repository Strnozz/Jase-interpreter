"""Smoke train, then reload its saved adapter in a fresh process for inference."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs" / "qwen35-9b"


def main():
    before = set(RUNS.glob("*/manifest.json"))
    subprocess.run([sys.executable, "training/train.py", "--model", "qwen35-9b", "--smoke"],
                   cwd=ROOT, check=True)
    created = set(RUNS.glob("*/manifest.json")) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected one new smoke run, got {len(created)}")
    manifest = json.loads(created.pop().read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("Smoke training did not complete")
    adapter = ROOT / manifest["adapter"]
    subprocess.run([sys.executable, "evaluation/compare.py",
                    "Trovami un ristorante giapponese stasera sotto i 40 euro",
                    "--models", "qwen35-9b", "--adapter", f"qwen35-9b={adapter}",
                    "--output", str(adapter.parent / "post_train_inference")], cwd=ROOT, check=True)
    print(f"Adapter reload and inference passed: {adapter}")


if __name__ == "__main__":
    main()
