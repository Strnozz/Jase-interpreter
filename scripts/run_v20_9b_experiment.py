"""Detached smoke → adapter reload → full QLoRA run. See launch status JSON."""
from __future__ import annotations

import json
import hashlib
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "training" / "runs" / "qwen35-9b-v20"
LAUNCH = ROOT / "training" / "v20_9b_launch.json"
LOG = ROOT / "training" / "v20_9b_launch.log"


def write(stage: str, **fields):
    data = {"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(), **fields}
    tmp = LAUNCH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(LAUNCH)


def call(*args: str):
    with LOG.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(args)}\n")
        log.flush()
        subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def new_run(before: set[Path]) -> Path:
    created = set(RUNS.glob("*/manifest.json")) - before
    if len(created) != 1:
        raise RuntimeError(f"Expected one new run, found {len(created)}")
    return created.pop().parent


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    audit = json.loads((ROOT / "data/v20_9b/token_audit.json").read_text(encoding="utf-8"))
    for part in ("train", "valid"):
        path = ROOT / f"data/v20_9b/{part}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["splits"][part]["source_sha256"]:
            raise RuntimeError(f"Dataset changed after tokenizer audit: {part}")
    if any(audit["splits"][part]["over_512"] for part in ("train", "valid")):
        raise RuntimeError("V20 training example exceeds 512 tokens")
    try:
        before = set(RUNS.glob("*/manifest.json"))
        write("smoke_running")
        call("training/train.py", "--model", "qwen35-9b-v20", "--smoke")
        smoke = new_run(before)
        write("adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
        call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))
        before = set(RUNS.glob("*/manifest.json"))
        write("full_running", smoke_run=str(smoke.relative_to(ROOT)))
        call("training/train.py", "--model", "qwen35-9b-v20", "--full")
        full = new_run(before)
        manifest = json.loads((full / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise RuntimeError("Full training process exited without complete manifest")
        write("training_complete", smoke_run=str(smoke.relative_to(ROOT)),
              full_run=str(full.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
