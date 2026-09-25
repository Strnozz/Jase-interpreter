"""Run identical Jase requests through configured models, one GPU model at a time."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.canon import ExtractionError, extract_contract  # noqa: E402
from jase.guard import ACCEPT, REPAIRED, check_contract  # noqa: E402
from jase.metrics import aggregate, compare, failure_taxonomy  # noqa: E402
from jase.multimodel import case, load_config, messages, read_jsonl, sha256  # noqa: E402


def process_ram():
    try:
        import psutil
        return psutil.Process().memory_info().rss
    except ImportError:
        return None


def load_backend(cfg, adapter):
    if cfg["backend"] == "mlx":
        if not adapter or not Path(adapter).is_dir():
            raise RuntimeError(f"MLX baseline adapter missing: {adapter}. Copy the original adapter without modifying it.")
        from mlx_lm import generate, load
        from mlx_lm.sample_utils import make_sampler
        model, tokenizer = load(cfg["model_id"], adapter_path=str(adapter))

        def infer(text):
            prompt = tokenizer.apply_chat_template(messages(text), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            start = time.perf_counter()
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=420,
                           sampler=make_sampler(temp=0.0), verbose=False)
            elapsed = time.perf_counter() - start
            return {"output": raw, "latency_s": elapsed, "ttft_s": None,
                    "input_tokens": len(tokenizer.encode(prompt)),
                    "output_tokens": len(tokenizer.encode(raw)), "peak_vram_bytes": None,
                    "process_ram_bytes_after": process_ram()}
        return infer

    import torch
    from peft import PeftModel
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteria, StoppingCriteriaList
    if not adapter or not Path(adapter).is_dir():
        raise RuntimeError(f"PEFT adapter missing: {adapter}. Complete the smoke test first.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch unavailable")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    q = cfg["quantization"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"])
    model = AutoModelForMultimodalLM.from_pretrained(cfg["model_id"], revision=cfg["revision"],
        device_map={"": 0}, dtype=dtype, attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_quant_type=q["type"], bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=dtype))
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()

    class FirstToken(StoppingCriteria):
        def __init__(self):
            self.at = None

        def __call__(self, input_ids, scores, **kwargs):
            if self.at is None:
                self.at = time.perf_counter()
            return False

    def infer(text):
        prompt = tokenizer.apply_chat_template(messages(text), tokenize=True,
            return_dict=True, return_tensors="pt", add_generation_prompt=True,
            enable_thinking=False).to("cuda")
        first = FirstToken()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**prompt, max_new_tokens=cfg["max_new_tokens"],
                do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([first]))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        output = ids[0, prompt["input_ids"].shape[-1]:]
        return {"output": tokenizer.decode(output, skip_special_tokens=True),
                "latency_s": elapsed, "ttft_s": first.at - start if first.at else None,
                "input_tokens": prompt["input_ids"].shape[-1],
                "output_tokens": len(output),
                "peak_vram_bytes": torch.cuda.max_memory_allocated(),
                "process_ram_bytes_after": process_ram()}
    return infer


def evaluate_output(raw, expected, text, schema):
    import jsonschema
    result = {"json_valid": False, "schema_valid": False, "guard": None,
              "has_expected": expected is not None,
              "semantic_exact": False, "metrics": None}
    try:
        result["json_valid"] = isinstance(json.loads(raw), dict)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        obj = extract_contract(raw)
    except ExtractionError as exc:
        result["error"] = exc.code
        return result
    try:
        jsonschema.validate(obj, schema)
        result["schema_valid"] = True
    except jsonschema.ValidationError as exc:
        result["error"] = f"schema:{exc.validator}"
    guard = check_contract(obj, text)
    result["guard"] = guard.status
    if expected is not None and result["schema_valid"]:
        delivered = guard.contract if guard.status in (ACCEPT, REPAIRED) else None
        metrics = compare(expected, delivered or obj, text)
        result["metrics"] = metrics
        result["semantic_exact"] = delivered is not None and bool(metrics["semantic_exact"])
        result["accepted_but_wrong"] = guard.status in (ACCEPT, REPAIRED) and not result["semantic_exact"]
        expected_missing = bool(expected.get("clarification", {}).get("required"))
        requested_missing = guard.status not in (ACCEPT, REPAIRED) or bool(obj.get("clarification", {}).get("required"))
        result["missing_info_correct"] = expected_missing == requested_missing
        result["unnecessary_clarification"] = not expected_missing and requested_missing
    result["contract"] = obj
    return result


def summarize(rows):
    metrics = [r["evaluation"]["metrics"] for r in rows if r["evaluation"]["metrics"]]
    n = len(rows)
    labeled = sum(r["evaluation"].get("has_expected", False) for r in rows)
    agg = aggregate(metrics) if metrics else None
    output_tokens = sum(r["performance"]["output_tokens"] for r in rows)
    latency = sum(r["performance"]["latency_s"] for r in rows)
    ttft_values = [r["performance"]["ttft_s"] for r in rows
                   if r["performance"]["ttft_s"] is not None]
    return {"cases": n, "json_valid_rate": sum(r["evaluation"]["json_valid"] for r in rows) / n,
            "schema_valid_rate": sum(r["evaluation"]["schema_valid"] for r in rows) / n,
            "quality_evaluable_cases": len(metrics),
            "quality_scope": "conditional on schema-valid output; complete success uses all labeled cases",
            "semantic_success_rate": sum(r["evaluation"]["semantic_exact"] for r in rows) / labeled if labeled else None,
            "complete_success_rate": sum(r["evaluation"]["semantic_exact"] and
                                         r["evaluation"]["json_valid"] and
                                         r["evaluation"]["schema_valid"] for r in rows) / labeled if labeled else None,
            "accepted_but_wrong": sum(r["evaluation"].get("accepted_but_wrong", False) for r in rows),
            "missing_info_accuracy": sum(r["evaluation"].get("missing_info_correct", False) for r in rows) / labeled if labeled else None,
            "unnecessary_clarification_rate": sum(r["evaluation"].get("unnecessary_clarification", False) for r in rows) / labeled if labeled else None,
            "latency_mean_s": latency / n, "tokens_per_second": output_tokens / latency if latency else None,
            "ttft_mean_s": sum(ttft_values) / len(ttft_values) if ttft_values else None,
            "peak_vram_bytes": max((r["performance"]["peak_vram_bytes"] or 0 for r in rows), default=0),
            "max_process_ram_bytes_after_inference": max((r["performance"]["process_ram_bytes_after"] or 0 for r in rows), default=0),
            "quality": agg,
            "unavailable_metrics": ["tool_selection", "plan_correctness"],
            "metric_note": "GoalContract 1.1 has no selected tool or executable plan; action is the intent."}


def main():
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="Direct A/B request; omit for benchmark")
    ap.add_argument("--models", nargs="+", default=["qwen35-2b", "qwen35-9b"])
    ap.add_argument("--adapter", action="append", default=[], metavar="MODEL=PATH")
    ap.add_argument("--data", default="data/v20_quality/dev.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="Deterministic random subset of eligible cases")
    ap.add_argument("--sample-seed", type=int, default=2026)
    ap.add_argument("--include-seen", action="store_true", help="Include the 10 v20 dev cases tagged as seen by v19")
    ap.add_argument("--include-outputs", action="store_true", help="Store dev prompts and raw outputs for inspection")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.include_outputs and args.data != "data/v20_quality/dev.jsonl":
        ap.error("--include-outputs is only available for the development panel")
    adapters = dict(a.split("=", 1) for a in args.adapter)
    configs = [load_config(name) for name in args.models]
    cases = ([{"text": args.text, "contract": None}] if args.text else
             [case(r) for r in read_jsonl(ROOT / args.data)])
    if not args.text and args.data == "data/v20_quality/dev.jsonl" and not args.include_seen:
        cases = [c for c in cases if "v19-exact-seen" not in c.get("tags", [])]
    if args.sample:
        if args.text or args.sample > len(cases) or args.sample < 0:
            ap.error("--sample requires benchmark data and a size within the eligible case count")
        selected = set(random.Random(args.sample_seed).sample(range(len(cases)), args.sample))
        cases = [item for index, item in enumerate(cases) if index in selected]
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        ap.error("No benchmark cases")
    # A single frozen case sequence, hashed before any model is loaded.
    data_hash = None if args.text else sha256(ROOT / args.data)
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "data": args.data,
              "data_sha256": data_hash, "prompt_sha256": sha256(ROOT / "TRAIN_SYSTEM_PROMPT.txt"),
              "evaluator_sha256": sha256(Path(__file__)),
              "metrics_sha256": sha256(ROOT / "jase" / "metrics.py"),
              "sample": args.sample or None, "sample_seed": args.sample_seed if args.sample else None,
              "case_ids": [item.get("id") for item in cases] if not args.text else None,
              "filter": "v19-exact-unseen" if args.data == "data/v20_quality/dev.jsonl" and not args.include_seen and not args.text else None,
              "models": {}, "status": "complete"}
    for cfg in configs:
        name = cfg["name"]
        adapter_text = adapters.get(name) or cfg.get("adapter")
        adapter = Path(adapter_text) if adapter_text else None
        if adapter and not adapter.is_absolute():
            adapter = ROOT / adapter
        model_record = {"model_id": cfg["model_id"], "adapter": str(adapter), "backend": cfg["backend"],
                        "params_billion": cfg["params_billion"], "quantization": cfg["quantization_label"]}
        report["models"][name] = model_record
        try:
            start = time.perf_counter()
            infer = load_backend(cfg, adapter)
            model_record["load_time_s"] = round(time.perf_counter() - start, 3)
            schema = json.loads((ROOT / cfg["schema"]).read_text(encoding="utf-8"))
            rows = []
            for index, item in enumerate(cases):
                performance = infer(item["text"])
                ev = evaluate_output(performance["output"], item.get("contract"), item["text"], schema)
                row = {"id": item.get("id", index), "performance": {k: v for k, v in performance.items() if k != "output"},
                       "evaluation": ev}
                if args.text:
                    row["output"] = performance["output"]
                else:
                    # Keep the release holdout sealed: no per-case text or predictions in benchmark reports.
                    row["evaluation"].pop("contract", None)
                    if args.include_outputs:
                        row["text"] = item["text"]
                        row["output"] = performance["output"]
                rows.append(row)
            model_record["summary"] = summarize(rows)
            model_record["failure_taxonomy"] = failure_taxonomy([
                row["evaluation"]["metrics"] for row in rows if row["evaluation"]["metrics"]])
            if not args.text:
                model_record["by_tag"] = {tag: summarize([row for row, item in zip(rows, cases)
                                                           if tag in item.get("tags", [])])
                                          for tag in sorted({tag for item in cases for tag in item.get("tags", [])})}
            if args.text:
                model_record["cases"] = rows
        except Exception as exc:
            model_record["error"] = str(exc)
            report["status"] = "incomplete"
        finally:
            if "infer" in locals():
                del infer
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
    out = Path(args.output) if args.output else ROOT / "benchmarks" / "outputs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    header = ["model", "params_B", "quantization", "status", "cases", "intent", "facts_precision", "facts_recall", "json", "schema", "missing_info", "unnecessary_clarification", "success", "accepted_but_wrong", "tokens_per_second", "peak_vram_gib"]
    table = []
    for name, rec in report["models"].items():
        s = rec.get("summary", {})
        q = s.get("quality") or {}
        table.append([name, rec["params_billion"], rec["quantization"], rec.get("error", "ok"), s.get("cases"), q.get("goals", {}).get("action_accuracy"),
                      q.get("facts", {}).get("precision"), q.get("facts", {}).get("recall"),
                      s.get("json_valid_rate"), s.get("schema_valid_rate"), s.get("missing_info_accuracy"),
                      s.get("unnecessary_clarification_rate"), s.get("complete_success_rate"),
                      s.get("accepted_but_wrong"), s.get("tokens_per_second"),
                      round(s["peak_vram_bytes"] / 1024**3, 2) if s.get("peak_vram_bytes") else None])
    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(table)
    md = "# Jase multi-model comparison\n\nDataset: `" + str(args.data) + "`\n\n"
    md += "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    md += "".join("| " + " | ".join("—" if v is None else str(v) for v in row) + " |\n" for row in table)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"JSON/CSV/Markdown: {out}")
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
