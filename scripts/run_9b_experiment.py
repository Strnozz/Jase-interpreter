"""Restartable detached smoke, full QLoRA run and paired benchmark evaluation."""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT))
from jase.multimodel import load_config, sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    cfg = load_config(args.model)
    if not cfg["name"].startswith("qwen35-9b-v"):
        ap.error("Only versioned Qwen 3.5 9B profiles are supported")
    version = cfg["name"].rsplit("-", 1)[-1]
    launch = ROOT / f"training/{version}_9b_launch.json"
    log_path = ROOT / f"training/{version}_9b_launch.log"
    runs = ROOT / "training/runs" / cfg["name"]
    best_out = ROOT / f"benchmarks/outputs/{version}-9b-best"
    final_out = ROOT / f"benchmarks/outputs/{version}-9b-final"
    extra = (["--extra-panel", "transfer=" + cfg["extra_benchmark"]]
             if cfg.get("extra_benchmark") else [])
    state = json.loads(launch.read_text(encoding="utf-8")) if launch.exists() else {}
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def write(stage: str, **fields):
        state.update(fields)
        state.update({"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat(),
                      "model": cfg["name"]})
        temp = launch.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(launch)

    def call(*params: str):
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(params)}\n")
            log.flush()
            subprocess.run([sys.executable, *params], cwd=ROOT, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def new_run(before: set[Path]) -> Path | None:
        created = set(runs.glob("*/manifest.json")) - before
        if len(created) > 1:
            raise RuntimeError(f"Ambiguous run creation: {created}")
        return created.pop().parent if created else None

    def load_run(path_text: str):
        path = (ROOT / path_text).resolve()
        if runs.resolve() not in path.parents:
            raise RuntimeError("Recorded run outside expected directory")
        return path, json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    try:
        if state.get("stage") == "complete" and best_out.joinpath("summary.json").is_file() and final_out.joinpath("summary.json").is_file():
            return 0
        write("preflight")
        data = ROOT / cfg["dataset"]
        source = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((data / "token_audit.json").read_text(encoding="utf-8"))
        for split, count_key in (("train", "expected_train_rows"), ("valid", "expected_validation_rows")):
            digest = sha256(data / f"{split}.jsonl")
            if (digest != source["output_sha256"][split] or digest != audit["splits"][split]["sha256"]
                    or audit["splits"][split]["rows"] != cfg[count_key]
                    or audit["splits"][split]["max_total_tokens"] > cfg["max_length"]):
                raise RuntimeError(f"{split} changed after audit")
        for relative, digest in source.get("excluded_benchmark_sha256", {}).items():
            if sha256(ROOT / relative) != digest:
                raise RuntimeError(f"Excluded benchmark changed after corpus build: {relative}")
        builder = ROOT / f"scripts/build_{version}_9b_data.py"
        if builder.is_file() and source.get("generator_sha256") != sha256(builder):
            raise RuntimeError("Corpus generator changed after dataset build")
        if source["system_prompt_sha256"] != sha256(ROOT / cfg["system_prompt_file"]):
            raise RuntimeError("System prompt changed after corpus build")
        if shutil.disk_usage(ROOT).free < 10 * 1024**3:
            raise RuntimeError("Less than 10 GiB free; inspect disk without deleting checkpoints")
        if cfg.get("baseline_transfer_run"):
            baseline_out = ROOT / cfg["baseline_transfer_output"]
            baseline_run = ROOT / cfg["baseline_transfer_run"]
            if not baseline_out.joinpath("summary.json").is_file():
                baseline_manifest = json.loads((baseline_run / "manifest.json").read_text(encoding="utf-8"))
                if baseline_manifest["status"] != "complete" or not extra:
                    raise RuntimeError("Baseline transfer evaluation needs a complete run and extra panel")
                write("baseline_transfer_running")
                call("evaluation/run_v20_9b.py", "--run", cfg["baseline_transfer_run"],
                     "--adapter", baseline_manifest["final_adapter"],
                     "--output", cfg["baseline_transfer_output"], "--only-extra", *extra)
        if state.get("smoke_run"):
            smoke, manifest = load_run(state["smoke_run"])
            if manifest["status"] != "complete" or not smoke.joinpath("post_train_inference.json").is_file():
                state.pop("smoke_run")
        if not state.get("smoke_run"):
            before = set(runs.glob("*/manifest.json"))
            write("smoke_running")
            try:
                call("training/train.py", "--model", cfg["name"], "--smoke")
            finally:
                smoke = new_run(before)
                if smoke:
                    write("adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
            if not smoke:
                raise RuntimeError("Smoke run not created")
            call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))
        if state.get("full_run"):
            full, manifest = load_run(state["full_run"])
        else:
            full = None
            manifest = None
        if manifest is None or manifest["status"] != "complete":
            write("full_running")
            if full:
                checkpoints = [p for p in full.glob("checkpoint-step-*")
                               if (p / "state.pt").is_file() and (p / "adapter/adapter_config.json").is_file()]
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
                    call("training/train.py", "--model", cfg["name"], "--full",
                         "--resume-from-checkpoint", str(latest.relative_to(ROOT)))
                else:
                    full = None
            if full is None:
                before = set(runs.glob("*/manifest.json"))
                try:
                    call("training/train.py", "--model", cfg["name"], "--full")
                finally:
                    created = new_run(before)
                    if created:
                        full = created
                        write("full_running", full_run=str(full.relative_to(ROOT)))
            if not full:
                raise RuntimeError("Full run not created")
            full, manifest = load_run(str(full.relative_to(ROOT)))
            if manifest["status"] != "complete":
                raise RuntimeError("Full process exited without complete manifest")
        if not best_out.joinpath("summary.json").is_file():
            write("evaluation_best_running", full_run=str(full.relative_to(ROOT)))
            call("evaluation/run_v20_9b.py", "--run", str(full.relative_to(ROOT)),
                 "--output", str(best_out.relative_to(ROOT)), *extra)
        if not final_out.joinpath("summary.json").is_file():
            write("evaluation_final_running", evaluation_best=str(best_out.relative_to(ROOT)))
            call("evaluation/run_v20_9b.py", "--run", str(full.relative_to(ROOT)),
                 "--adapter", manifest["final_adapter"],
                 "--output", str(final_out.relative_to(ROOT)), *extra)
        if not best_out.joinpath("summary.json").is_file() or not final_out.joinpath("summary.json").is_file():
            raise RuntimeError("Incomplete benchmark summaries")
        write("complete", full_run=str(full.relative_to(ROOT)),
              evaluation_best=str(best_out.relative_to(ROOT)),
              evaluation_final=str(final_out.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with log_path.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
