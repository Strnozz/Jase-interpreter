"""Detached, restartable Granite 4.1 challenger: vanilla, smoke, train, evaluate.

Run only after the pinned snapshot is complete. Any failed compatibility stage
stops the pipeline before full training. Benchmark gold and semantic data are
read-only throughout; evaluations are never used to tune the run.
"""
from __future__ import annotations

import importlib.metadata
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

MODEL = "granite41-8b-v26-challenger"
CFG = load_config(MODEL)
RUNS = ROOT / "training/runs" / MODEL
STATUS = ROOT / "training/granite41_challenger_launch.json"
LOG = ROOT / "training/granite41_challenger_launch.log"
V32 = ROOT / "benchmarks/v32/blind.jsonl"
V33 = ROOT / "benchmarks/v33/blind.jsonl"
VANILLA = {"v32": "benchmarks/outputs/granite41-vanilla-v32",
           "v33": "benchmarks/outputs/granite41-vanilla-v33"}
ADAPTED = {kind: {panel: f"benchmarks/outputs/granite41-v26-{kind}-{panel}"
                  for panel in ("v33", "v32")}
           for kind in ("best", "final")}


def read_state():
    return json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}


def write(state, stage, **fields):
    state.update(fields)
    state.update(stage=stage, model=MODEL,
                 updated_at=datetime.now(timezone.utc).isoformat())
    temp = STATUS.with_suffix(".json.tmp")
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(STATUS)


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
        raise RuntimeError(f"Expected one new run, found {created}")
    return created.pop().parent


def evaluate(panel_name, output, *, run=None, adapter=None):
    panel = V33 if panel_name == "v33" else V32
    expected = sha256(panel)
    summary = ROOT / output / "summary.json"
    if summary.is_file():
        saved = json.loads(summary.read_text(encoding="utf-8"))
        if saved.get("panel_sha256") != expected or saved.get("model_config") != MODEL:
            raise RuntimeError(f"Conflicting evaluation output: {summary}")
        return saved
    args = ["evaluation/run_semantic_v1_3_9b.py"]
    if run is None:
        args += ["--model-config", MODEL]
    else:
        args += ["--run", str(run.relative_to(ROOT)), "--adapter", str(adapter.relative_to(ROOT))]
    args += ["--panel", str(panel.relative_to(ROOT)), "--output", output]
    call(*args)
    saved = json.loads(summary.read_text(encoding="utf-8"))
    if saved["panel_sha256"] != expected:
        raise RuntimeError("Evaluator reported a different panel hash")
    return saved


def preflight():
    import torch
    from transformers import AutoTokenizer

    frozen = json.loads((ROOT / "benchmarks/v33/manifest.json").read_text(encoding="utf-8"))
    if frozen["panel_sha256"] != sha256(V33) or frozen["cases"] < 100:
        raise RuntimeError("V33 changed or is too small")
    v32_manifest = json.loads((ROOT / "benchmarks/v32/manifest.json").read_text(encoding="utf-8"))
    if v32_manifest["panel_sha256"] != sha256(V32):
        raise RuntimeError("V32 historical panel changed")
    qwen = json.loads((ROOT / "training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/manifest.json")
                      .read_text(encoding="utf-8"))
    audit = json.loads((ROOT / "training/audits/granite41-8b-v26-token-audit.json")
                       .read_text(encoding="utf-8"))
    for split in ("train", "valid"):
        path = ROOT / CFG["dataset"] / f"{split}.jsonl"
        digest = sha256(path)
        if (digest != qwen["dataset_sha256"][split] or
                digest != audit["splits"][split]["sha256"] or
                audit["splits"][split]["over_512"] or
                audit["splits"][split]["max_total_tokens"] > CFG["max_length"]):
            raise RuntimeError(f"Dataset {split} differs from frozen Qwen V26 or exceeds 512")
    snapshot = (ROOT / "hf-cache/hub/models--ibm-granite--granite-4.1-8b/snapshots" /
                CFG["revision"])
    if len(list(snapshot.glob("model-*-of-*.safetensors"))) != 4:
        raise RuntimeError("Pinned Granite weight snapshot is incomplete")
    if shutil.disk_usage(ROOT).free < 25 * 1024**3:
        raise RuntimeError("Less than 25 GiB free disk before training")
    if not torch.cuda.is_available() or "4080 SUPER" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Expected RTX 4080 SUPER CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 unavailable; do not silently change the comparison")
    tok = AutoTokenizer.from_pretrained(CFG["tokenizer_id"],
                                        revision=CFG["tokenizer_revision"],
                                        local_files_only=True, trust_remote_code=False)
    sample = "Prenotami un treno per Bologna domani."
    if tok.decode(tok.encode(sample, add_special_tokens=False), skip_special_tokens=True) != sample:
        raise RuntimeError("Granite tokenizer round-trip changed the user text")
    rendered = tok.apply_chat_template([
        {"role": "system", "content": "You are Jase Interpreter."},
        {"role": "user", "content": sample}], tokenize=False, add_generation_prompt=True)
    if "assistant" not in rendered or sample not in rendered:
        raise RuntimeError("Granite instruction template is incomplete")
    return {"panel_v33_sha256": frozen["panel_sha256"],
            "panel_v32_sha256": v32_manifest["panel_sha256"],
            "dataset_sha256": qwen["dataset_sha256"],
            "model_revision": CFG["revision"],
            "config_sha256": sha256(ROOT / "configs/models" / f"{MODEL}.json"),
            "gpu": torch.cuda.get_device_name(0),
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
            "cuda": torch.version.cuda,
            "versions": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "peft", "bitsandbytes", "accelerate")}}


def main():
    state = read_state()
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if state.get("stage") == "complete":
            return 0
        write(state, "preflight")
        provenance = preflight()
        write(state, "vanilla_v32_running", provenance=provenance)
        vanilla32 = evaluate("v32", VANILLA["v32"])
        write(state, "vanilla_v33_running")
        vanilla33 = evaluate("v33", VANILLA["v33"])
        if vanilla32["json_valid"] + vanilla33["json_valid"] == 0:
            raise RuntimeError("Vanilla Granite produced no parseable JSON in the compatibility panels")

        smoke = ROOT / state["smoke_run"] if state.get("smoke_run") else None
        valid_smoke = smoke and (smoke / "manifest.json").is_file() and \
            manifest(smoke)["status"] == "complete"
        if not valid_smoke:
            before = set(RUNS.glob("*/manifest.json"))
            write(state, "smoke_running")
            call("training/train.py", "--model", MODEL, "--smoke")
            smoke = new_run(before)
            write(state, "smoke_complete", smoke_run=str(smoke.relative_to(ROOT)))
        if not (smoke / "post_train_inference.json").is_file():
            write(state, "adapter_reload_running", smoke_run=str(smoke.relative_to(ROOT)))
            call("scripts/verify_v20_adapter.py", "--run", str(smoke.relative_to(ROOT)))
            if not (smoke / "post_train_inference.json").is_file():
                raise RuntimeError("Smoke adapter reload report missing")

        full = ROOT / state["full_run"] if state.get("full_run") else None
        full_manifest = manifest(full) if full and (full / "manifest.json").is_file() else None
        if not full_manifest or full_manifest["status"] != "complete":
            write(state, "full_running")
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
                        write(state, "full_running", full_run=str(full.relative_to(ROOT)))
            if full is None or manifest(full)["status"] != "complete":
                raise RuntimeError("Full Granite run did not complete")

        full_manifest = manifest(full)
        best = ROOT / full_manifest["adapter"]
        final = ROOT / full_manifest["final_adapter"]
        for kind, adapter in (("best", best), ("final", final)):
            for panel in ("v33", "v32"):
                write(state, f"evaluation_{kind}_{panel}_running",
                      full_run=str(full.relative_to(ROOT)))
                evaluate(panel, ADAPTED[kind][panel], run=full, adapter=adapter)
        write(state, "complete", full_run=str(full.relative_to(ROOT)),
              best_adapter=str(best.relative_to(ROOT)),
              final_adapter=str(final.relative_to(ROOT)))
        return 0
    except Exception as exc:
        with LOG.open("a", encoding="utf-8") as log:
            traceback.print_exc(file=log)
        write(state, "failed", error=repr(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
