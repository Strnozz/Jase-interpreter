"""Detached, restartable smoke/train/benchmark pipeline for frozen V26."""
from __future__ import annotations

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

MODEL = "qwen35-9b-v26"
CFG = load_config(MODEL)
RUNS = ROOT / "training/runs" / MODEL
STATE = ROOT / "training/v26_9b_launch.json"
LOG = ROOT / "training/v26_9b_launch.log"
PANEL = ROOT / "benchmarks/v29/blind.jsonl"
BEST = ROOT / "benchmarks/outputs/v26-9b-v29-best"
FINAL = ROOT / "benchmarks/outputs/v26-9b-v29-final"
BASELINES = [ROOT / "benchmarks/outputs/v24-9b-v29-blind/summary.json",
             ROOT / "benchmarks/outputs/v25-9b-v29-blind/summary.json"]


def main():
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def write(stage, **fields):
        state.update(fields)
        state.update({"stage": stage, "model": MODEL,
                      "updated_at": datetime.now(timezone.utc).isoformat()})
        temp = STATE.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(STATE)

    def call(*args):
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(args)}\n")
            log.flush()
            subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def manifest(run):
        return json.loads((run / "manifest.json").read_text(encoding="utf-8"))

    def new_run(before):
        created = set(RUNS.glob("*/manifest.json")) - before
        if len(created) != 1:
            raise RuntimeError(f"Expected exactly one new run, got {created}")
        return created.pop().parent

    def evaluate(run, adapter, output):
        if not (output / "summary.json").is_file():
            call("evaluation/run_semantic_v1_3_9b.py", "--run", str(run),
                 "--adapter", str(adapter), "--panel", str(PANEL.relative_to(ROOT)),
                 "--output", str(output.relative_to(ROOT)))

    try:
        if state.get("stage") == "complete" and all((p / "summary.json").is_file()
                                                   for p in (BEST, FINAL)):
            return 0
        write("preflight")
        source = json.loads((ROOT / CFG["dataset"] / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((ROOT / CFG["dataset"] / "token_audit.json").read_text(encoding="utf-8"))
        panel_manifest = json.loads((ROOT / "benchmarks/v29/manifest.json").read_text(encoding="utf-8"))
        if source["panel_frozen_before_data"]["sha256"] != panel_manifest["panel_sha256"] or \
                sha256(PANEL) != panel_manifest["panel_sha256"]:
            raise RuntimeError("V29 frozen panel changed")
        if source["generator_sha256"] != sha256(ROOT / "scripts/build_v26_9b_data.py") or \
                source["system_prompt_sha256"] != sha256(ROOT / CFG["system_prompt_file"]):
            raise RuntimeError("V26 generator or prompt changed after data freeze")
        for split, key in (("train", "expected_train_rows"), ("valid", "expected_validation_rows")):
            path = ROOT / CFG["dataset"] / f"{split}.jsonl"
            if sha256(path) != source["output_sha256"][split] or \
                    sha256(path) != audit["splits"][split]["sha256"] or \
                    audit["splits"][split]["rows"] != CFG[key] or \
                    audit["splits"][split]["max_total_tokens"] > 512:
                raise RuntimeError(f"V26 {split} no longer matches audit")
        for relative, digest in source["source_sha256"].items():
            version, split = relative.split("_")
            if sha256(ROOT / f"data/{version}_9b/{split}.jsonl") != digest:
                raise RuntimeError(f"Source changed: {relative}")
        for relative, digest in source["excluded_benchmark_sha256"].items():
            if sha256(ROOT / relative) != digest:
                raise RuntimeError(f"Benchmark changed: {relative}")
        for baseline in BASELINES:
            if not baseline.is_file() or json.loads(baseline.read_text(encoding="utf-8"))["panel_sha256"] != panel_manifest["panel_sha256"]:
                raise RuntimeError(f"Baseline missing or mismatched: {baseline}")
        if shutil.disk_usage(ROOT).free < 15 * 1024**3:
            raise RuntimeError("Less than 15 GiB free disk")
        if not (ROOT / CFG["initial_adapter"] / "adapter_model.safetensors").is_file():
            raise RuntimeError("Initial V25 adapter missing")

        smoke = ROOT / state["smoke_run"] if state.get("smoke_run") else None
        valid_smoke = smoke and (smoke / "manifest.json").is_file() and \
            manifest(smoke)["status"] == "complete" and (smoke / "post_train_inference.json").is_file()
        if not valid_smoke:
            before = set(RUNS.glob("*/manifest.json"))
            write("smoke_running")
            call("training/train.py", "--model", MODEL, "--smoke")
            smoke = new_run(before)
            write("adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
            call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))

        full = ROOT / state["full_run"] if state.get("full_run") else None
        full_manifest = manifest(full) if full and (full / "manifest.json").is_file() else None
        if not full_manifest or full_manifest["status"] != "complete":
            write("full_running")
            if full:
                checkpoints = [p for p in full.glob("checkpoint-step-*")
                               if (p / "state.pt").is_file() and
                               (p / "adapter/adapter_config.json").is_file()]
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
                    call("training/train.py", "--model", MODEL, "--full",
                         "--resume-from-checkpoint", str(latest.relative_to(ROOT)))
                else:
                    full = None
            if full is None:
                before = set(RUNS.glob("*/manifest.json"))
                try:
                    call("training/train.py", "--model", MODEL, "--full")
                finally:
                    created = set(RUNS.glob("*/manifest.json")) - before
                    if len(created) == 1:
                        full = created.pop().parent
                        write("full_running", full_run=str(full.relative_to(ROOT)))
            if full is None or manifest(full)["status"] != "complete":
                raise RuntimeError("Full run did not complete")
        full_manifest = manifest(full)
        write("evaluation_best_running", full_run=str(full.relative_to(ROOT)))
        evaluate(full, ROOT / full_manifest["adapter"], BEST)
        chosen = (ROOT / full_manifest["adapter"]).resolve()
        last = (ROOT / full_manifest["final_adapter"]).resolve()
        if chosen == last:
            write("evaluation_final_copying")
            FINAL.mkdir(parents=True, exist_ok=True)
            for src in BEST.iterdir():
                dst = FINAL / src.name
                if dst.exists() and sha256(dst) != sha256(src):
                    raise RuntimeError(f"Conflicting final evaluation {dst}")
                shutil.copy2(src, dst)
        else:
            write("evaluation_final_running")
            evaluate(full, last, FINAL)
        if not all((p / "summary.json").is_file() for p in (BEST, FINAL)):
            raise RuntimeError("Evaluation summary missing")
        write("complete", full_run=str(full.relative_to(ROOT)),
              best_output=str(BEST.relative_to(ROOT)), final_output=str(FINAL.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
