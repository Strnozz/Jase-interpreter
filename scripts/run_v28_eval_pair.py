"""Detached, restartable V28 evaluation of the frozen V25 and V24 adapters."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PANEL = ROOT / "benchmarks/v28/blind.jsonl"
MANIFEST = ROOT / "benchmarks/v28/manifest.json"
STATUS = ROOT / "benchmarks/outputs/v28_eval_launch.json"
RUNNER = ROOT / "evaluation/run_semantic_v1_3_9b.py"
RUNS = [
    ("v25", "training/runs/qwen35-9b-v25/20260926T230452Z-54b5d764",
     "benchmarks/outputs/v25-9b-v28-blind"),
    ("v24", "training/runs/qwen35-9b-v24/20260926T180548Z-28e20ea2",
     "benchmarks/outputs/v24-9b-v28-blind"),
]


def write(stage: str, **extra):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    old.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **extra)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(old, indent=2) + "\n", encoding="utf-8")
    tmp.replace(STATUS)


def main():
    from jase.multimodel import sha256

    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["panel_sha256"]
    if sha256(PANEL) != expected:
        raise RuntimeError("V28 frozen panel hash changed")
    for name, run, output in RUNS:
        summary = ROOT / output / "summary.json"
        if summary.exists():
            record = json.loads(summary.read_text(encoding="utf-8"))
            if record.get("panel_sha256") != expected:
                raise RuntimeError(f"{name} output is for another panel")
            continue
        write(f"{name}_running", panel_sha256=expected)
        result = subprocess.run([sys.executable, str(RUNNER), "--run", run,
                                 "--panel", "benchmarks/v28/blind.jsonl",
                                 "--output", output], cwd=ROOT)
        if result.returncode:
            write("failed", failed_model=name, exit_code=result.returncode)
            return result.returncode
    write("complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
