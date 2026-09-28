"""Detached, restartable V27 pilot, smoke and full QLoRA training pipeline.

V37 is not opened for model selection or training. Evaluation is a later stage.
"""
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

STATE = ROOT / "training/v27_conversational_launch.json"
LOG = ROOT / "training/v27_conversational_launch.log"
PILOTS = ("qwen35-9b-v27-pilot-a", "qwen35-9b-v27-pilot-b")
FINAL = ("qwen35-9b-v27", "qwen35-9b-v27-base")


def main() -> int:
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    def write(stage: str, **fields) -> None:
        state.update(fields)
        state.update({"stage": stage, "updated_at": datetime.now(timezone.utc).isoformat()})
        temp = STATE.with_suffix(".json.tmp")
        temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(STATE)

    def call(*args: str) -> None:
        with LOG.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(args)}\n")
            log.flush()
            subprocess.run([sys.executable, *args], cwd=ROOT, env=os.environ.copy(),
                           stdout=log, stderr=subprocess.STDOUT, check=True)

    def manifest(path: Path) -> dict:
        return json.loads((path / "manifest.json").read_text(encoding="utf-8"))

    def ensure_run(model: str, mode: str) -> Path:
        key = f"{model}_{mode}_run"
        runs = ROOT / "training/runs" / model
        previous = ROOT / state[key] if state.get(key) else None
        if previous is None and runs.exists():
            # Recover an interrupted process even when it never returned to the
            # runner's finally block to write its run path.
            incomplete = [p.parent for p in runs.glob("*/manifest.json")
                          if (record := manifest(p.parent)).get("mode") == mode and
                          record.get("model") == model and record.get("status") in {"running", "failed"}]
            if incomplete:
                previous = max(incomplete, key=lambda p: p.name)
                write(f"{model}_{mode}_recovered", **{key: str(previous.relative_to(ROOT))})
        if previous and (previous / "manifest.json").is_file():
            record = manifest(previous)
            if record["status"] == "complete":
                if mode == "smoke" and not (previous / "post_train_inference.json").is_file():
                    write(f"{model}_adapter_reload_running")
                    call("scripts/verify_v27_adapter.py", "--run", str(previous.relative_to(ROOT)))
                return previous
            if mode == "full" and record["status"] in {"running", "failed"}:
                checkpoints = [p for p in previous.glob("checkpoint-step-*")
                               if (p / "state.pt").is_file() and
                               (p / "adapter/adapter_config.json").is_file()]
                if checkpoints:
                    latest = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[-1]))
                    write(f"{model}_full_resuming", checkpoint=str(latest.relative_to(ROOT)))
                    call("training/train.py", "--model", model, "--full",
                         "--resume-from-checkpoint", str(latest.relative_to(ROOT)))
                    if manifest(previous)["status"] == "complete":
                        return previous
            # Preserve failed runs; a new run is a distinct attempt.
        before = set(runs.glob("*/manifest.json"))
        write(f"{model}_{mode}_running")
        try:
            call("training/train.py", "--model", model, "--" + mode)
        finally:
            created = set(runs.glob("*/manifest.json")) - before
            if len(created) == 1:
                previous = created.pop().parent
                write(f"{model}_{mode}_running", **{key: str(previous.relative_to(ROOT))})
        if previous is None or manifest(previous)["status"] != "complete":
            raise RuntimeError(f"{model} {mode} failed")
        if mode == "smoke":
            write(f"{model}_adapter_reload_running")
            call("scripts/verify_v27_adapter.py", "--run", str(previous.relative_to(ROOT)))
        return previous

    try:
        if state.get("stage") == "training_complete":
            return 0
        write("preflight")
        panel_path = ROOT / "benchmarks/v37_conversational/blind.jsonl"
        panel = json.loads((panel_path.parent / "manifest.json").read_text(encoding="utf-8"))
        if sha256(panel_path) != panel["panel_sha256"]:
            raise RuntimeError("Frozen V37 panel changed")
        data_dir = ROOT / "data/v27_conversational"
        source = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
        audit = json.loads((data_dir / "tokenizer_audit.json").read_text(encoding="utf-8"))
        if source["v37_sha256"] != panel["panel_sha256"] or source["generator_sha256"] != sha256(
                ROOT / "data/build_v27_conversational.py"):
            raise RuntimeError("V27 source or V37 provenance changed")
        if source["leakage"]["exact_benchmark_overlap"] != 0 or source["leakage"]["max_benchmark_token_jaccard"] >= .85:
            raise RuntimeError("Benchmark leakage audit failed")
        for split in ("train", "valid"):
            path = data_dir / f"{split}.jsonl"
            if (sha256(path) != source["splits"][split]["sha256"] or
                    sha256(path) != audit["splits"][split]["sha256"] or
                    audit["splits"][split]["over_configured_limit"] or
                    audit["splits"][split]["completion_mask_verified_rows"] != source["splits"][split]["rows"]):
                raise RuntimeError(f"V27 {split} differs from audited source")
        if shutil.disk_usage(ROOT).free < 20 * 1024**3:
            raise RuntimeError("Less than 20 GiB free disk")
        for name in (*PILOTS, *FINAL):
            cfg = load_config(name)
            if (cfg["revision"] != "c202236235762e1c871ad0ccb60c8ee5ba337b9a" or
                    cfg["expected_train_rows"] != source["splits"]["train"]["rows"] or
                    cfg["expected_validation_rows"] != source["splits"]["valid"]["rows"] or
                    cfg["max_length"] != 512):
                raise RuntimeError(f"Invalid pinned configuration {name}")
        frozen_adapter = ROOT / load_config(PILOTS[0])["initial_adapter"] / "adapter_model.safetensors"
        if not frozen_adapter.is_file():
            raise RuntimeError("Frozen V26 adapter missing")

        pilot_records = {}
        for model in PILOTS:
            ensure_run(model, "smoke")
            run = ensure_run(model, "full")
            info = manifest(run)
            pilot_records[model] = {"run": str(run.relative_to(ROOT)),
                                    "best_validation_loss": info["best_validation_loss"],
                                    "best_step": info["best_step"],
                                    "peak_vram_bytes": info.get("peak_vram_bytes")}
            write("pilot_running", pilot_results=pilot_records)
        warm = pilot_records[PILOTS[0]]["best_validation_loss"]
        fresh = pilot_records[PILOTS[1]]["best_validation_loss"]
        # Preserve V26 semantics unless base-init wins convincingly on the same
        # internal validation sample. V37 is never used for this choice.
        selected = FINAL[1] if fresh < warm * .90 else FINAL[0]
        decision = {"selected_model": selected, "warm_start_loss": warm,
                    "base_init_loss": fresh, "selection_rule": "base only if validation loss < 0.90 * warm-start loss",
                    "blind_panel_used_for_selection": False}
        (ROOT / "training/v27_pilot_comparison.json").write_text(
            json.dumps({"pilots": pilot_records, **decision}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        write("pilot_complete", pilot_results=pilot_records, **decision)
        ensure_run(selected, "smoke")
        final_run = ensure_run(selected, "full")
        write("training_complete", selected_model=selected, full_run=str(final_run.relative_to(ROOT)),
              adapter=manifest(final_run)["adapter"])
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write("failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
