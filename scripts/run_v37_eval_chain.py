"""Detached sequential V37 V27/V26 comparison; never overlaps two 9B models."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "evaluation/v37_eval_chain.json"
LOG = ROOT / "evaluation/v37_eval_chain.log"
V27 = ROOT / "benchmarks/outputs/v27-conversational-v37"
V26 = ROOT / "benchmarks/outputs/v26-legacy-v37"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-pid", type=int, required=True)
    args = ap.parse_args()
    import psutil

    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}

    def write(stage: str, **extra) -> None:
        state.update(extra)
        state["stage"] = stage
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        write("waiting_v27", v27_pid=args.wait_pid)
        deadline = time.monotonic() + 7200
        while not (V27 / "summary.json").is_file():
            if not psutil.pid_exists(args.wait_pid):
                raise RuntimeError("V27 evaluator ended without summary")
            if time.monotonic() > deadline:
                raise TimeoutError("V27 evaluation exceeded 2 hours")
            time.sleep(20)
        if not (V26 / "summary.json").is_file():
            write("v26_baseline_running")
            with LOG.open("a", encoding="utf-8") as log:
                subprocess.run([sys.executable, "evaluation/run_v37_conversational.py",
                    "--run", "training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c",
                    "--adapter", "training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/best_adapter",
                    "--output", "benchmarks/outputs/v26-legacy-v37", "--legacy-v26"],
                    cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
        write("v37_complete", v27_summary=str((V27 / "summary.json").relative_to(ROOT)),
              v26_summary=str((V26 / "summary.json").relative_to(ROOT)))
        return 0
    except Exception as exc:
        write("failed", error=repr(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
