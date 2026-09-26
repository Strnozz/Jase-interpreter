"""Detached, restartable V25 smoke/train/eval pipeline with frozen V26 baseline."""
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

MODEL = "qwen35-9b-v25"
CFG = load_config(MODEL)
RUNS = ROOT / "training/runs" / MODEL
STATE = ROOT / "training/v25_9b_launch.json"
LOG = ROOT / "training/v25_9b_launch.log"
BASELINE = ROOT / "benchmarks/outputs/v24-9b-v26-blind"
BEST = ROOT / "benchmarks/outputs/v25-9b-v26-best"
FINAL = ROOT / "benchmarks/outputs/v25-9b-v26-final"
V25 = ROOT / "benchmarks/outputs/v25-9b-v25-diagnostic"
PANEL26 = "benchmarks/v26/blind.jsonl"
PANEL25 = "benchmarks/v25/blind.jsonl"
V24_RUN = "training/runs/qwen35-9b-v24/20260926T180548Z-28e20ea2"


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def write(stage, **fields):
        state.update(fields)
        state.update({"stage": stage, "model": MODEL,
                      "updated_at": datetime.now(timezone.utc).isoformat()})
        staged = STATE.with_suffix(".json.tmp")
        staged.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.replace(STATE)

    def call(*params):
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(params)}\n")
            log.flush()
            subprocess.run([sys.executable, *params], cwd=ROOT, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def new_run(before):
        created = set(RUNS.glob("*/manifest.json")) - before
        if len(created) != 1:
            raise RuntimeError(f"Expected one new run, got {created}")
        return created.pop().parent

    def manifest(path):
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    def evaluate(run, adapter, panel, output):
        if not output.joinpath("summary.json").is_file():
            call("evaluation/run_semantic_v1_3_9b.py", "--run", str(run),
                 "--adapter", str(adapter), "--panel", panel, "--output", str(output.relative_to(ROOT)))

    try:
        if state.get("stage") == "complete" and all(p.joinpath("summary.json").is_file()
                                                     for p in (BASELINE, BEST, FINAL, V25)):
            return 0
        write("preflight")
        data = ROOT / CFG["dataset"]
        source = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((data / "token_audit.json").read_text(encoding="utf-8"))
        panel_manifest = json.loads((ROOT / "benchmarks/v26/manifest.json").read_text(encoding="utf-8"))
        if (source["panel_frozen_before_data"]["sha256"] != panel_manifest["panel_sha256"] or
                sha256(ROOT / PANEL26) != panel_manifest["panel_sha256"]):
            raise RuntimeError("V26 panel provenance changed")
        if source["generator_sha256"] != sha256(ROOT / "scripts/build_v25_9b_data.py"):
            raise RuntimeError("V25 dataset generator changed")
        if source["system_prompt_sha256"] != sha256(ROOT / CFG["system_prompt_file"]):
            raise RuntimeError("System prompt changed")
        for split, key in (("train", "expected_train_rows"), ("valid", "expected_validation_rows")):
            path = data / f"{split}.jsonl"
            if (sha256(path) != source["output_sha256"][split] or
                    sha256(path) != audit["splits"][split]["sha256"] or
                    audit["splits"][split]["rows"] != CFG[key] or
                    audit["splits"][split]["max_total_tokens"] > CFG["max_length"]):
                raise RuntimeError(f"V25 {split} changed after audit")
        for relative, digest in source["excluded_benchmark_sha256"].items():
            if sha256(ROOT / relative) != digest:
                raise RuntimeError(f"Frozen benchmark changed: {relative}")
        if shutil.disk_usage(ROOT).free < 12 * 1024**3:
            raise RuntimeError("Less than 12 GiB free")
        v24 = ROOT / V24_RUN
        v24_manifest = manifest(v24)
        if v24_manifest["status"] != "complete":
            raise RuntimeError("V24 baseline incomplete")
        write("baseline_v26_running")
        evaluate(v24, ROOT / v24_manifest["adapter"], PANEL26, BASELINE)
        if state.get("smoke_run"):
            smoke = ROOT / state["smoke_run"]
            valid_smoke = (smoke / "manifest.json").is_file() and manifest(smoke)["status"] == "complete" \
                and (smoke / "post_train_inference.json").is_file()
        else:
            valid_smoke = False
        if not valid_smoke:
            before = set(RUNS.glob("*/manifest.json"))
            write("smoke_running")
            call("training/train.py", "--model", MODEL, "--smoke")
            smoke = new_run(before)
            write("adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
            call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))
        if state.get("full_run"):
            full = ROOT / state["full_run"]
            full_manifest = manifest(full)
        else:
            full = None
            full_manifest = None
        if full_manifest is None or full_manifest["status"] != "complete":
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
                if full is None:
                    raise RuntimeError("Full run was not created")
            full_manifest = manifest(full)
            if full_manifest["status"] != "complete":
                raise RuntimeError("Training process exited without complete manifest")
        write("evaluation_best_running", full_run=str(full.relative_to(ROOT)))
        evaluate(full, ROOT / full_manifest["adapter"], PANEL26, BEST)
        chosen = (ROOT / full_manifest["adapter"]).resolve()
        last = (ROOT / full_manifest["final_adapter"]).resolve()
        if chosen == last:
            write("evaluation_final_copying")
            FINAL.mkdir(parents=True, exist_ok=True)
            for src in BEST.iterdir():
                dst = FINAL / src.name
                if dst.exists() and sha256(dst) != sha256(src):
                    raise RuntimeError(f"Refusing to replace differing evaluation {dst}")
                shutil.copy2(src, dst)
        else:
            write("evaluation_final_running")
            evaluate(full, last, PANEL26, FINAL)
        write("evaluation_v25_diagnostic_running")
        evaluate(full, chosen, PANEL25, V25)
        if not all(p.joinpath("summary.json").is_file() for p in (BASELINE, BEST, FINAL, V25)):
            raise RuntimeError("Evaluation summary missing")
        write("complete", full_run=str(full.relative_to(ROOT)),
              baseline_output=str(BASELINE.relative_to(ROOT)), best_output=str(BEST.relative_to(ROOT)),
              final_output=str(FINAL.relative_to(ROOT)), diagnostic_output=str(V25.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
