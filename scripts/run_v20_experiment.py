#!/usr/bin/env python3
"""Durable, fail-fast v20 train/eval sequence for the native Mac runner."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/v20"
STATE = OUT / "status.json"
SPEC = ROOT / "runs/v20_experiment.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def verify_inputs(spec):
    for relative, expected in spec["input_hashes"].items():
        if sha(ROOT / relative) != expected:
            raise RuntimeError(f"Frozen input changed: {relative}")


def preflight(spec):
    import mlx.core as mx
    import yaml
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    if not mx.metal.is_available():
        raise RuntimeError("Metal GPU is not available to this process")
    cfg = yaml.safe_load((ROOT / spec["config"]).read_text())
    snapshot = snapshot_download(cfg["model"], local_files_only=True,
        allow_patterns=["*.json", "model*.safetensors", "*.py", "tokenizer.model",
                        "*.tiktoken", "tiktoken.model", "*.txt", "*.jsonl", "*.jinja"])
    tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    stats = {}
    for split in ("train", "valid"):
        lengths = []
        for line in (ROOT / cfg["data"] / f"{split}.jsonl").read_text().splitlines():
            messages = json.loads(line)["messages"]
            ids = tok.apply_chat_template(messages, tokenize=True, return_dict=False)
            prefix = tok.apply_chat_template(messages[:-1], tokenize=True,
                add_generation_prompt=True, return_dict=False)
            inference = tok.apply_chat_template(messages[:-1], tokenize=True,
                add_generation_prompt=True, return_dict=False, enable_thinking=False)
            if ids[:len(prefix)] != prefix or prefix != inference:
                raise RuntimeError("Training loss mask and inference prefix differ")
            lengths.append(len(ids))
        lengths.sort()
        stats[split] = {"rows": len(lengths), "min": lengths[0],
            "median": lengths[len(lengths) // 2], "p99": lengths[int(len(lengths) * .99)],
            "max": lengths[-1],
            "truncated": sum(n > cfg["max_seq_length"] for n in lengths)}
    if any(s["truncated"] for s in stats.values()):
        raise RuntimeError(f"Examples would be truncated: {stats}")
    report = {"metal_available": True, "model_snapshot": snapshot,
              "model_revision": Path(snapshot).name, "token_lengths": stats,
              "mask_and_inference_prefix_match": True,
              "microbatches": cfg["iters"],
              "optimizer_updates": cfg["iters"] // cfg["grad_accumulation_steps"],
              "epochs": cfg["iters"] * cfg["batch_size"] / stats["train"]["rows"],
              "versions": {p: importlib.metadata.version(p) for p in
                           ("mlx", "mlx-lm", "transformers", "jsonschema", "numpy")}}
    save_json(OUT / "preflight.json", report)
    print(json.dumps(report, indent=2), flush=True)


def command_phase(name, command, env):
    log = OUT / f"{name}.log"
    print(f"START {name}: {' '.join(command)}; log: {log.relative_to(ROOT)}", flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)
    print(f"DONE {name}", flush=True)


def compare_reports():
    old = json.loads((OUT / "baseline-dev.json").read_text())
    new = json.loads((OUT / "candidate-dev.json").read_text())
    old_gold = json.loads((OUT / "baseline-gold.json").read_text())
    new_gold = json.loads((OUT / "candidate-gold.json").read_text())
    if old["data_sha256"] != new["data_sha256"] or old["guard_sha256"] != new["guard_sha256"]:
        raise RuntimeError("Reports do not refer to the same data and Guard")
    a, b = old["by_tag"]["v19-exact-unseen"], new["by_tag"]["v19-exact-unseen"]
    # These thresholds are declared in the protocol before any inference.
    evidence = {
        "accepted_errors_reduced": b["accepted_but_wrong"] < a["accepted_but_wrong"],
        "exact_not_worse": b["semantic_exact_rate"] >= a["semantic_exact_rate"],
        "recall_not_worse": b["facts"]["recall"] >= a["facts"]["recall"],
        "precision_loss_at_most_one_point": b["facts"]["precision"] >= a["facts"]["precision"] - .01,
        "unnecessary_clarifications_not_increased":
            new["decisions"]["clarification_false_positive"] <= old["decisions"]["clarification_false_positive"],
        "gold_regression_at_most_two_cases":
            new_gold["release_gates"]["mandatory_passed"] >= old_gold["release_gates"]["mandatory_passed"] - 2,
    }
    result = {"baseline": old["adapter"], "candidate": new["adapter"],
              "benchmark_role": "dev_regression", "primary_panel": "v19-exact-unseen",
              "baseline_unseen": a, "candidate_unseen": b, "criteria": evidence,
              "directionally_promising": all(evidence.values()),
              "production_ready": False,
              "limitations": ["One training seed", "Synthetic shared-generator development data",
                              "Exact-text exclusion does not imply structural independence",
                              "Seen clarification cases are diagnostics, not comparative evidence",
                              "Release holdout not evaluated; no automatic promotion"]}
    save_json(OUT / "comparison.json", result)
    rows = [
        ("Casi nel pannello principale", a["n"], b["n"]),
        ("Semantic exact", a["semantic_exact_rate"], b["semantic_exact_rate"]),
        ("Accepted but wrong", a["accepted_but_wrong"], b["accepted_but_wrong"]),
        ("Recall fatti", a["facts"]["recall"], b["facts"]["recall"]),
        ("Precision fatti", a["facts"]["precision"], b["facts"]["precision"]),
        ("Chiarimenti inutili, pannello completo", old["decisions"]["clarification_false_positive"],
         new["decisions"]["clarification_false_positive"]),
        ("Mandatory gold superati", old_gold["release_gates"]["mandatory_passed"],
         new_gold["release_gates"]["mandatory_passed"]),
    ]
    verdict = ("Risultato promettente secondo i criteri predefiniti. Da confermare su altri seed e dati indipendenti."
               if result["directionally_promising"] else
               "I criteri predefiniti non sono tutti soddisfatti. Analizzare il DEV prima di estendere il training.")
    text = "# Esperimento v20 — risultato\n\n" + verdict + "\n\n"
    text += "| Metrica | v19-cap16 | v20-quality |\n|---|---:|---:|\n"
    text += "\n".join(f"| {name} | {av} | {bv} |" for name, av, bv in rows)
    text += "\n\nConfronto principale sui testi assenti dal training v19. "
    text += "I dieci chiarimenti già visti da v19 sono diagnostica separata. "
    text += "Dati sintetici dello stesso generatore, un solo seed: non è una misura di generalizzazione su utenti reali. "
    text += "Holdout di release non eseguito; candidato non promosso automaticamente.\n"
    (OUT / "SUMMARY.md").write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "experiment.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        spec = json.loads(SPEC.read_text())
        state = json.loads(STATE.read_text()) if STATE.exists() else {"completed_phases": []}
        spec_hash = sha(SPEC)
        if state.get("spec_sha256", spec_hash) != spec_hash:
            raise RuntimeError("Cannot resume with a different experiment specification")
        state["spec_sha256"] = spec_hash
        env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        os.environ.update({k: env[k] for k in ("HF_HUB_OFFLINE", "TOKENIZERS_PARALLELISM")})
        python = str(ROOT / ".venv/bin/python")

        def phase(name, callback):
            verify_inputs(spec)
            if name in state["completed_phases"]:
                return
            state.update(status="running", phase=name, updated=datetime.now(timezone.utc).isoformat())
            save_json(STATE, state)
            callback()
            state["completed_phases"].append(name)
            save_json(STATE, state)

        def train():
            if (ROOT / spec["candidate_adapter"]).exists():
                raise RuntimeError("Candidate adapter already exists; refusing to overwrite or silently restart")
            command_phase("train", [str(ROOT / ".venv/bin/mlx_lm.lora"), "--config", spec["config"]], env)
            save_json(OUT / "candidate_hashes.json", {
                p.name: sha(p) for p in (ROOT / spec["candidate_adapter"]).iterdir()
                if p.name in ("adapters.safetensors", "adapter_config.json")})

        def evaluate(label, adapter, data):
            command_phase(label, [python, "scripts/eval_v2.py", "--adapter", adapter,
                "--data", data, "--pipeline", "--seed", "2026", "--max-tokens", "420",
                "--benchmark-role", "dev_regression", "--label", label,
                "--report", f"reports/v20/{label}.json"], env)

        try:
            phase("preflight", lambda: preflight(spec))
            if args.preflight_only:
                state.update(status="preflight_passed", phase="awaiting_training")
            else:
                phase("train", train)
                for label, adapter, data in (
                    ("baseline-dev", spec["baseline_adapter"], spec["dev"]),
                    ("candidate-dev", spec["candidate_adapter"], spec["dev"]),
                    ("baseline-gold", spec["baseline_adapter"], "benchmarks/gold_v1.jsonl"),
                    ("candidate-gold", spec["candidate_adapter"], "benchmarks/gold_v1.jsonl"),
                ):
                    phase(label, lambda label=label, adapter=adapter, data=data: evaluate(label, adapter, data))
                phase("comparison", compare_reports)
                state.update(status="complete", phase="complete")
            state["updated"] = datetime.now(timezone.utc).isoformat()
            save_json(STATE, state)
        except Exception as exc:
            state.update(status="failed", error=str(exc), updated=datetime.now(timezone.utc).isoformat())
            save_json(STATE, state)
            raise


if __name__ == "__main__":
    main()
