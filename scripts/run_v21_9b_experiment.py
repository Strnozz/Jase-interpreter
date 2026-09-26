"""Detached V21 warm-start smoke, training, fresh reload and benchmark."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training/runs/qwen35-9b-v21"
LAUNCH = ROOT / "training/v21_9b_launch.json"
LOG = ROOT / "training/v21_9b_launch.log"
EVAL = ROOT / "benchmarks/outputs/v21-9b-posttrain"


def write(stage: str, **fields):
    data = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    temp = LAUNCH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(LAUNCH)


def call(*args: str):
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(args)}\n")
        log.flush()
        subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def new_run(before: set[Path]):
    created = set(RUNS.glob("*/manifest.json")) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected one new run, found {len(created)}")
    return created.pop().parent


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        write("preflight")
        cfg = json.loads((ROOT / "configs/models/qwen35-9b-v21.json").read_text(encoding="utf-8"))
        audit = json.loads((ROOT / "data/v21_9b/token_audit.json").read_text(encoding="utf-8"))
        source = json.loads((ROOT / "data/v21_9b/manifest.json").read_text(encoding="utf-8"))
        for split, key in (("train", "expected_train_rows"), ("valid", "expected_validation_rows")):
            path = ROOT / f"data/v21_9b/{split}.jsonl"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if (digest != source["output_sha256"][split] or digest != audit["splits"][split]["sha256"]
                    or audit["splits"][split]["rows"] != cfg[key]
                    or audit["splits"][split]["over_512"]):
                raise RuntimeError(f"V21 {split} changed after dataset/tokenizer audit")
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free; preserve checkpoints and inspect disk")
        before = set(RUNS.glob("*/manifest.json"))
        write("smoke_running")
        call("training/train.py", "--model", "qwen35-9b-v21", "--smoke")
        smoke = new_run(before)
        write("adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
        call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))
        before = set(RUNS.glob("*/manifest.json"))
        write("full_running", smoke_run=str(smoke.relative_to(ROOT)))
        call("training/train.py", "--model", "qwen35-9b-v21", "--full")
        full = new_run(before)
        manifest = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise RuntimeError("Full training process exited without complete manifest")
        write("evaluation_running", smoke_run=str(smoke.relative_to(ROOT)),
              full_run=str(full.relative_to(ROOT)))
        call("evaluation/run_v20_9b.py", "--run", str(full.relative_to(ROOT)),
             "--output", str(EVAL.relative_to(ROOT)))
        if not (EVAL / "summary.json").is_file():
            raise RuntimeError("Post-training evaluation did not produce summary")
        write("complete", smoke_run=str(smoke.relative_to(ROOT)),
              full_run=str(full.relative_to(ROOT)), evaluation=str(EVAL.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
