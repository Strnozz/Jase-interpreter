"""Configuration-driven QLoRA training. Run smoke before full training."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import clean_validation, encode_supervised, load_config, read_jsonl, sha256  # noqa: E402


def git_head() -> str | None:
    try:
        git = shutil.which("git") or r"C:\Program Files\Git\cmd\git.exe"
        return subprocess.check_output([git, "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def select_language_linears(model, last_n_layers: int):
    """Use exact text-model module paths; never attach LoRA to the vision tower."""
    import bitsandbytes as bnb
    names = [name for name, module in model.named_modules()
             if isinstance(module, bnb.nn.Linear4bit)
             and any(f".{part}." in f".{name}." for part in ("language_model", "text_model"))]
    if not names or any("visual" in n or "vision" in n for n in names):
        raise RuntimeError("Could not isolate quantized language-model layers; inspect architecture before training")
    by_layer = {int(match.group(1)) for name in names
                if (match := re.search(r"(?:^|\.)(?:language_model|text_model)\.layers\.(\d+)\.", name))}
    if len(by_layer) < last_n_layers:
        raise RuntimeError(f"Expected at least {last_n_layers} quantized text layers, found {sorted(by_layer)}")
    selected_layers = set(sorted(by_layer)[-last_n_layers:])
    selected = [name for name in names if (match := re.search(
        r"(?:^|\.)(?:language_model|text_model)\.layers\.(\d+)\.", name))
        and int(match.group(1)) in selected_layers]
    if not selected:
        raise RuntimeError("No trainable text linear modules selected")
    return selected


def load_base(cfg):
    import torch
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RuntimeError("At least 14 GiB VRAM required for this 9B profile")
    q = cfg["quantization"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if q["compute_dtype"] == "bfloat16" and dtype != torch.bfloat16:
        raise RuntimeError("This profile requires bf16; change config only after a validated fp16 smoke test")
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForMultimodalLM.from_pretrained(
        cfg["model_id"], revision=cfg["revision"], device_map={"": 0},
        dtype=dtype, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type=q["type"], bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=dtype))
    return model, tok


def checkpoint_parameter_count(cfg):
    """Count original checkpoint tensor shapes, not packed 4-bit storage elements."""
    from safetensors import safe_open

    snapshot = (ROOT / "hf-cache" / "hub" /
                ("models--" + cfg["model_id"].replace("/", "--")) /
                "snapshots" / cfg["revision"])
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"Pinned weight snapshot missing: {snapshot}")
    total = 0
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            total += sum(math.prod(handle.get_slice(key).get_shape()) for key in handle.keys())
    return total


def checkpoint_dir(path_text: str, cfg: dict) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    base = (ROOT / "training" / "runs" / cfg["name"]).resolve()
    if base not in path.parents or not path.name.startswith("checkpoint-step-"):
        raise ValueError(f"Checkpoint must be inside {base}: {path}")
    if not (path / "state.pt").is_file() or not (path / "adapter" / "adapter_config.json").is_file():
        raise ValueError(f"Incomplete checkpoint: {path}")
    return path


def write_checkpoint(out, step, model, optimizer, scheduler, elapsed_seconds,
                     best_loss, best_step, validation_history, torch):
    name = f"checkpoint-step-{step:06d}"
    final = out / name
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint {final}")
    staged = out / f"{name}.tmp-{uuid.uuid4().hex[:8]}"
    staged.mkdir(parents=False, exist_ok=False)
    model.save_pretrained(staged / "adapter", safe_serialization=True)
    torch.save({"global_step": step, "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(),
                "elapsed_seconds": elapsed_seconds,
                "best_validation_loss": best_loss if math.isfinite(best_loss) else None,
                "best_step": best_step, "validation_history": validation_history},
               staged / "state.pt")
    staged.rename(final)
    return final


def verified_smoke(cfg, dataset_hashes):
    base = ROOT / "training" / "runs" / cfg["name"]
    for path in sorted(base.glob("*/manifest.json"), reverse=True):
        try:
            prior = json.loads(path.read_text(encoding="utf-8"))
            inference = json.loads((path.parent / "post_train_inference.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (prior.get("mode") == "smoke" and prior.get("status") == "complete"
                and prior.get("config") == cfg and prior.get("dataset_sha256") == dataset_hashes
                and prior.get("trainer_sha256") == sha256(Path(__file__))
                and prior.get("pipeline_sha256") == sha256(ROOT / "jase" / "multimodel.py")
                and prior.get("system_prompt_sha256") == sha256(ROOT / cfg.get("system_prompt_file", "TRAIN_SYSTEM_PROMPT.txt"))
                and prior.get("initial_adapter_sha256") == initial_adapter_sha256(cfg)
                and prior.get("schema_sha256") == sha256(ROOT / cfg["schema"])
                and isinstance(prior.get("smoke_max_tokens"), int)
                and prior["smoke_max_tokens"] > 0
                and prior["smoke_max_tokens"] == prior.get("corpus_max_tokens")
                and inference.get("status") == "complete"):
            return prior["run_id"]
    raise RuntimeError("Full training requires a completed smoke run with saved adapter reload/inference on this exact config and corpus")


def initial_adapter_sha256(cfg):
    """Fingerprint warm-start weights and PEFT topology, if configured."""
    adapter_text = cfg.get("initial_adapter")
    if not adapter_text:
        return None
    adapter = (ROOT / adapter_text).resolve()
    if ROOT not in adapter.parents:
        raise ValueError("Initial adapter must be inside the repository")
    import hashlib
    digest = hashlib.sha256()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        path = adapter / name
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def loss_on(model, examples, torch):
    model.eval()
    losses = []
    with torch.no_grad():
        for row in examples:
            batch = {k: torch.tensor([v], device="cuda") for k, v in row.items()}
            value = float(model(**batch).loss.detach())
            if not (0 <= value < float("inf")):
                raise RuntimeError("NaN or infinite validation loss")
            losses.append(value)
    model.train()
    return sum(losses) / len(losses)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-9b")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    ap.add_argument("--resume-from-checkpoint", help="Resume the same full run from a checkpoint-step-N directory")
    args = ap.parse_args()
    if args.resume_from_checkpoint and not args.full:
        ap.error("--resume-from-checkpoint requires --full")
    cfg = load_config(args.model)
    system_prompt = (ROOT / cfg.get("system_prompt_file", "TRAIN_SYSTEM_PROMPT.txt")).read_text(encoding="utf-8").strip()
    adapter_hash = initial_adapter_sha256(cfg)
    if cfg["backend"] != "transformers":
        ap.error("Training this backend is not supported by this runner")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    import torch
    import psutil
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch unavailable; run scripts/hardware_probe.py first")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    profile = cfg["smoke"] if args.smoke else cfg["training"]
    data = ROOT / cfg["dataset"]
    dataset_hashes = {split: sha256(data / f"{split}.jsonl") for split in ("train", "valid")}
    if args.full:
        prerequisite = verified_smoke(cfg, dataset_hashes)
    else:
        prerequisite = None
    resume = checkpoint_dir(args.resume_from_checkpoint, cfg) if args.resume_from_checkpoint else None
    if resume:
        out = resume.parent
        metadata = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        if (metadata.get("mode") != "full" or metadata.get("status") == "complete"
                or metadata.get("config") != cfg or metadata.get("dataset_sha256") != dataset_hashes
                or metadata.get("trainer_sha256") != sha256(Path(__file__))
                or metadata.get("pipeline_sha256") != sha256(ROOT / "jase" / "multimodel.py")
                or metadata.get("system_prompt_sha256") != sha256(ROOT / cfg.get("system_prompt_file", "TRAIN_SYSTEM_PROMPT.txt"))
                or metadata.get("initial_adapter_sha256") != adapter_hash
                or metadata.get("schema_sha256") != sha256(ROOT / cfg["schema"])):
            raise RuntimeError("Resume checkpoint provenance differs from current run inputs or code")
        metadata.setdefault("resume_events", []).append({
            "at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(resume.relative_to(ROOT))})
        metadata["status"] = "running"
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        out = ROOT / "training" / "runs" / cfg["name"] / run_id
        out.mkdir(parents=True, exist_ok=False)
        gpu_free, _gpu_total = torch.cuda.mem_get_info()
        metadata = {"run_id": run_id, "model": cfg["name"], "model_id": cfg["model_id"],
                    "mode": "smoke" if args.smoke else "full", "status": "running",
                    "created_at": datetime.now(timezone.utc).isoformat(), "git_commit": git_head(),
                    "seed": cfg["seed"], "config": cfg,
                    "dataset_sha256": dataset_hashes,
                    "verified_smoke_run": prerequisite,
                    "trainer_sha256": sha256(Path(__file__)),
                    "pipeline_sha256": sha256(ROOT / "jase" / "multimodel.py"),
                    "system_prompt_sha256": sha256(ROOT / cfg.get("system_prompt_file", "TRAIN_SYSTEM_PROMPT.txt")),
                    "initial_adapter_sha256": adapter_hash,
                    "schema_sha256": sha256(ROOT / cfg["schema"]),
                    "checkpoint_parameter_count": checkpoint_parameter_count(cfg),
                    "versions": {p: importlib.metadata.version(p) for p in
                                 ("torch", "transformers", "peft", "bitsandbytes", "accelerate")},
                    "hardware": {"gpu": torch.cuda.get_device_name(0),
                                 "vram_bytes": torch.cuda.get_device_properties(0).total_memory,
                                 "cuda": torch.version.cuda,
                                 "ram_bytes": psutil.virtual_memory().total,
                                 "vram_free_before_load_bytes": gpu_free,
                                 "vram_idle_bytes": torch.cuda.memory_allocated()}}
    manifest = out / "manifest.json"

    def save():
        staged = manifest.with_suffix(".json.tmp")
        staged.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        staged.replace(manifest)

    save()
    started = time.monotonic()
    try:
        load_started = time.monotonic()
        model, tokenizer = load_base(cfg)
        metadata["model_load_seconds"] = round(time.monotonic() - load_started, 2)
        metadata["resolved_model_revision"] = getattr(model.config, "_commit_hash", None)
        if metadata["resolved_model_revision"] != cfg["revision"]:
            raise RuntimeError("Loaded model revision differs from pinned revision")
        metadata["vram_after_base_load_bytes"] = torch.cuda.memory_allocated()
        metadata["vram_free_after_base_load_bytes"] = torch.cuda.mem_get_info()[0]
        train_rows = read_jsonl(data / "train.jsonl")
        valid_rows = read_jsonl(data / "valid.jsonl")
        if any(row["messages"][0] != {"role": "system", "content": system_prompt}
               for row in train_rows + valid_rows):
            raise RuntimeError("Dataset system prompt differs from configured training prompt")
        if (len(train_rows) != cfg["expected_train_rows"]
                or len(valid_rows) != cfg["expected_validation_rows"]):
            raise RuntimeError(f"Dataset split counts changed: train={len(train_rows)}, validation={len(valid_rows)}")
        metadata["train_rows"] = len(train_rows)
        metadata["validation_source_rows"] = len(valid_rows)
        valid_rows, metadata["validation_cleaning"] = clean_validation(train_rows, valid_rows)
        if args.smoke:
            longest = max(train_rows, key=lambda row: len(
                encode_supervised(tokenizer, row, cfg["max_length"])["input_ids"]))
            metadata["corpus_max_tokens"] = len(
                encode_supervised(tokenizer, longest, cfg["max_length"])["input_ids"])
            rng = random.Random(cfg["seed"])
            train_rows = rng.sample(train_rows, profile["train_examples"])
            if longest not in train_rows:
                train_rows[0] = longest
        random.Random(cfg["seed"]).shuffle(train_rows)
        if args.smoke:
            train_rows.remove(longest)
            train_rows.insert(0, longest)
        metadata["train_order"] = "Python random.Random(seed).shuffle, once before tokenization"
        valid_rows = random.Random(cfg["seed"] + 1).sample(
            valid_rows, profile["validation_examples"])
        train = [encode_supervised(tokenizer, row, cfg["max_length"]) for row in train_rows]
        if args.smoke:
            metadata["smoke_max_tokens"] = max(len(row["input_ids"]) for row in train)
        valid = [encode_supervised(tokenizer, row, cfg["max_length"]) for row in valid_rows]
        metadata["validation_sample_rows"] = len(valid)
        example = train[0]
        masked = sum(label == -100 for label in example["labels"])
        active = len(example["labels"]) - masked
        if (masked == 0 or active == 0 or example["labels"][:masked] != [-100] * masked
                or example["labels"][masked:] != example["input_ids"][masked:]):
            raise RuntimeError("Completion-only loss mask did not match assistant target")
        inspection = {"example_index_after_shuffle": 0, "total_tokens": len(example["input_ids"]),
                      "masked_prompt_tokens": masked, "assistant_target_tokens": active,
                      "masked_token_label": -100,
                      "masked_prefix_preview": tokenizer.decode(example["input_ids"][:min(masked, 24)]),
                      "active_target_preview": tokenizer.decode(example["input_ids"][masked:masked + 24])}
        (out / "loss_mask_inspection.json").write_text(
            json.dumps(inspection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model.config.use_cache = False
        lc = cfg["lora"]
        targets = select_language_linears(model, lc["last_n_layers"])
        if resume:
            model = PeftModel.from_pretrained(model, str(resume / "adapter"), is_trainable=True)
        elif cfg.get("initial_adapter"):
            initial = ROOT / cfg["initial_adapter"]
            peft_cfg = json.loads((initial / "adapter_config.json").read_text(encoding="utf-8"))
            if (peft_cfg.get("r") != lc["r"] or peft_cfg.get("lora_alpha") != lc["alpha"]
                    or peft_cfg.get("lora_dropout") != lc["dropout"]
                    or set(peft_cfg.get("target_modules", [])) != set(targets)):
                raise RuntimeError("Initial adapter LoRA topology differs from V21 config")
            model = PeftModel.from_pretrained(model, str(initial), is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=lc["r"], lora_alpha=lc["alpha"],
                lora_dropout=lc["dropout"], bias="none", target_modules=targets))
        metadata["lora_modules"] = len(targets)
        metadata["lora_target_modules"] = targets
        metadata["lora_target_leaf_names"] = sorted({name.rsplit(".", 1)[-1] for name in targets})
        metadata["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        metadata["trainable_percent_of_checkpoint"] = round(
            100 * metadata["trainable_parameters"] / metadata["checkpoint_parameter_count"], 4)
        save()
        accum = cfg["training"]["gradient_accumulation_steps"]
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=cfg["training"]["learning_rate"],
                                      weight_decay=cfg["training"]["weight_decay"])
        if args.full:
            warmup = cfg["training"]["warmup_steps"]
            total = cfg["training"]["max_steps"]
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: (
                (step + 1) / warmup if step < warmup else
                0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) / 2))
        else:
            scheduler = None
        start_step = 0
        elapsed_offset = 0.0
        best_validation_loss = float("inf")
        best_step = None
        validation_history = []
        if resume:
            state = torch.load(resume / "state.pt", map_location="cpu", weights_only=True)
            if state["global_step"] != int(resume.name.rsplit("-", 1)[-1]):
                raise RuntimeError("Checkpoint step and state disagree")
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            random.setstate(state["python_rng"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state_all(state["cuda_rng"])
            start_step = state["global_step"]
            elapsed_offset = state["elapsed_seconds"]
            best_validation_loss = (state["best_validation_loss"] if state["best_validation_loss"] is not None
                                    else float("inf"))
            best_step = state["best_step"]
            validation_history = state["validation_history"]
            if start_step >= profile["max_steps"]:
                raise RuntimeError("Checkpoint already reached configured maximum step")
        summary = {"MODEL": cfg["model_id"], "REVISION": cfg["revision"],
                   "DATASET": cfg["dataset"], "TRAIN_SAMPLES": len(train_rows),
                   "VALIDATION_SAMPLES": len(valid_rows), "VALIDATION_USED": len(valid),
                   "QUANTIZATION": cfg["quantization"], "COMPUTE_DTYPE": cfg["quantization"]["compute_dtype"],
                   "MAX_LENGTH": cfg["max_length"], "BATCH_SIZE": cfg["training"]["batch_size"],
                   "GRADIENT_ACCUMULATION": accum,
                   "EFFECTIVE_BATCH_SIZE": cfg["training"]["batch_size"] * accum,
                   "EPOCHS": profile["max_steps"] * accum / len(train_rows),
                   "LEARNING_RATE": cfg["training"]["learning_rate"],
                   "LORA_R": lc["r"], "LORA_ALPHA": lc["alpha"],
                   "LORA_TARGET_MODULES": metadata["lora_target_leaf_names"],
                   "TOTAL_PARAMETERS": metadata["checkpoint_parameter_count"],
                   "TRAINABLE_PARAMETERS": metadata["trainable_parameters"],
                   "TRAINABLE_PERCENT": metadata["trainable_percent_of_checkpoint"],
                   "EXPECTED_OUTPUT_DIRECTORY": str(out), "START_STEP": start_step,
                   "LOSS_MASK": inspection}
        (out / "run_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        index = start_step * accum
        with (out / "steps.jsonl").open("a" if resume else "w", encoding="utf-8") as log:
            for step in range(start_step + 1, profile["max_steps"] + 1):
                step_started = time.monotonic()
                step_loss = 0.0
                step_tokens = 0
                for _ in range(accum):
                    row = train[index % len(train)]
                    index += 1
                    step_tokens += len(row["input_ids"])
                    batch = {k: torch.tensor([v], device="cuda") for k, v in row.items()}
                    loss = model(**batch).loss
                    value = float(loss.detach())
                    if not (0 <= value < float("inf")):
                        raise RuntimeError(f"NaN or infinite training loss at step {step}")
                    (loss / accum).backward()
                    step_loss += value / accum
                grad_norm = float(torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg["training"]["max_grad_norm"]))
                if not math.isfinite(grad_norm):
                    raise RuntimeError(f"NaN or infinite gradient norm at step {step}")
                optimizer.step()
                lr = optimizer.param_groups[0]["lr"]
                if scheduler:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                step_seconds = time.monotonic() - step_started
                record = {"step": step, "epoch": round(step * accum / len(train_rows), 6),
                          "train_loss": round(step_loss, 6), "learning_rate": lr,
                          "gradient_norm": grad_norm,
                          "input_tokens": step_tokens,
                          "input_tokens_per_second": round(step_tokens / step_seconds, 3),
                          "samples_per_second": round(accum / step_seconds, 4),
                          "step_seconds": round(step_seconds, 3),
                          "seconds": round(elapsed_offset + time.monotonic() - started, 2),
                          "vram_allocated_bytes": torch.cuda.memory_allocated(),
                          "vram_reserved_bytes": torch.cuda.memory_reserved(),
                          "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                          "process_ram_bytes": psutil.Process().memory_info().rss}
                log.write(json.dumps(record) + "\n")
                log.flush()
                if args.smoke or step % 50 == 0:
                    print(json.dumps(record), flush=True)
                if args.full and step % profile["eval_every_steps"] == 0 and step != profile["max_steps"]:
                    value = loss_on(model, valid, torch)
                    validation_history.append({"step": step, "loss": value})
                    if value < best_validation_loss:
                        best_validation_loss, best_step = value, step
                        model.save_pretrained(out / "best_adapter", safe_serialization=True)
                    metadata["validation_history"] = validation_history
                    metadata["best_validation_loss"] = best_validation_loss
                    metadata["best_step"] = best_step
                    if (len(validation_history) >= 3
                            and validation_history[-1]["loss"] > validation_history[-2]["loss"]
                            > validation_history[-3]["loss"]):
                        metadata.setdefault("warnings", []).append(
                            f"Validation loss rose at 3 consecutive evaluations ending at step {step}; inspect overfitting")
                    save()
                    print(json.dumps({"validation_step": step, "validation_loss": value,
                                      "best_step": best_step, "best_validation_loss": best_validation_loss}), flush=True)
                if args.full and (step in profile.get("early_checkpoint_steps", [])
                                  or step % profile["checkpoint_every_steps"] == 0):
                    checkpoint = write_checkpoint(out, step, model, optimizer, scheduler,
                        elapsed_offset + time.monotonic() - started,
                        best_validation_loss, best_step, validation_history, torch)
                    metadata["last_checkpoint"] = str(checkpoint.relative_to(ROOT))
                    metadata["last_checkpoint_step"] = step
                    save()
                    print(json.dumps({"checkpoint_step": step, "checkpoint": str(checkpoint)}), flush=True)
        metadata["train_loss_last"] = step_loss
        metadata["validation_loss"] = loss_on(model, valid, torch)
        validation_history.append({"step": profile["max_steps"], "loss": metadata["validation_loss"]})
        metadata["validation_history"] = validation_history
        if metadata["validation_loss"] < best_validation_loss:
            best_validation_loss, best_step = metadata["validation_loss"], profile["max_steps"]
        metadata["best_validation_loss"] = best_validation_loss
        metadata["best_step"] = best_step
        if args.full and profile["max_steps"] % profile["checkpoint_every_steps"] != 0:
            checkpoint = write_checkpoint(out, profile["max_steps"], model, optimizer, scheduler,
                elapsed_offset + time.monotonic() - started,
                best_validation_loss, best_step, validation_history, torch)
            metadata["last_checkpoint"] = str(checkpoint.relative_to(ROOT))
            metadata["last_checkpoint_step"] = profile["max_steps"]
        metadata["steps"] = profile["max_steps"]
        metadata["examples_seen"] = profile["max_steps"] * accum
        metadata["effective_epochs"] = metadata["examples_seen"] / len(train_rows)
        metadata["checkpoint_selection"] = "final; smoke run has no model selection" if args.smoke else "lowest loss on fixed 64-case validation sample"
        by_step = {}
        for line in (out / "steps.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                record = json.loads(line)
                by_step[record["step"]] = record
        measured = list(by_step.values())
        active_seconds = sum(r["step_seconds"] for r in measured)
        metadata["input_tokens_per_second"] = round(
            sum(r["input_tokens"] for r in measured) / active_seconds, 3)
        metadata["samples_per_second"] = round(len(measured) * accum / active_seconds, 4)
        metadata["mean_vram_allocated_bytes"] = round(
            sum(r["vram_allocated_bytes"] for r in measured) / len(measured))
        metadata["peak_vram_reserved_bytes"] = max(r["vram_reserved_bytes"] for r in measured)
        metadata["peak_vram_bytes"] = max(r["peak_vram_bytes"] for r in measured)
        metadata["peak_process_ram_bytes"] = max(r["process_ram_bytes"] for r in measured)
        metadata["process_ram_bytes_at_end"] = psutil.Process().memory_info().rss
        metadata["duration_seconds"] = round(elapsed_offset + time.monotonic() - started, 2)
        model.save_pretrained(out / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(out / "adapter")
        selected = out / ("best_adapter" if args.full and best_step != profile["max_steps"] else "adapter")
        if selected.name == "best_adapter":
            tokenizer.save_pretrained(selected)
        metadata["final_adapter"] = str((out / "adapter").relative_to(ROOT))
        metadata["adapter"] = str(selected.relative_to(ROOT))
        metadata["adapter_bytes"] = sum(path.stat().st_size for path in selected.rglob("*") if path.is_file())
        metadata["best_checkpoint"] = metadata["adapter"]
        with (out / "pip_freeze.txt").open("w", encoding="utf-8") as frozen:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], check=True,
                           stdout=frozen)
        metadata["status"] = "complete"
        save()
        print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
        return 0
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = repr(exc)
        metadata["oom"] = isinstance(exc, torch.cuda.OutOfMemoryError)
        save()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
