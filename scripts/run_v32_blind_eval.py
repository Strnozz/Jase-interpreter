"""Sequential, restartable V32 evaluation of frozen 9B adapters."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import sha256  # noqa: E402

PANEL = ROOT / "benchmarks/v32/blind.jsonl"
MANIFEST = ROOT / "benchmarks/v32/manifest.json"
STATUS = ROOT / "benchmarks/outputs/v32_eval_launch.json"
RUNNER = ROOT / "evaluation/run_semantic_v1_3_9b.py"
RUNS = [
    ("v24", "training/runs/qwen35-9b-v24/20260926T180548Z-28e20ea2",
     None, "benchmarks/outputs/v24-9b-v32-blind"),
    ("v25", "training/runs/qwen35-9b-v25/20260926T230452Z-54b5d764",
     None, "benchmarks/outputs/v25-9b-v32-blind"),
    ("v26_best", "training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c",
     "training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/best_adapter",
     "benchmarks/outputs/v26-9b-v32-blind"),
]


def write(stage, **extra):
    old = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    old.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    temp = STATUS.with_suffix(".tmp")
    temp.write_text(json.dumps(old, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATUS)


def main():
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["panel_sha256"]
    if sha256(PANEL) != expected:
        raise RuntimeError("V32 panel changed after freeze")
    for name, run, adapter, output in RUNS:
        summary = ROOT / output / "summary.json"
        if summary.exists():
            saved = json.loads(summary.read_text(encoding="utf-8"))
            if saved.get("panel_sha256") != expected:
                raise RuntimeError(f"{name} output has a different panel hash")
            continue
        write(f"{name}_running", panel_sha256=expected)
        cmd = [sys.executable, str(RUNNER), "--run", run,
               "--panel", "benchmarks/v32/blind.jsonl", "--output", output]
        if adapter:
            cmd.extend(["--adapter", adapter])
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode:
            write("failed", failed_model=name, exit_code=result.returncode)
            return result.returncode
    write("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
