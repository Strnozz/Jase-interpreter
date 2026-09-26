"""Reload the V20 adapter and score new V1.2 panels plus convertible V19 dev."""
from __future__ import annotations

import json
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.canon import ExtractionError  # noqa: E402
from jase.goal_contract_v1_2 import parse_contract  # noqa: E402
from jase.metrics_v1_2 import aggregate_v12, compare_v12  # noqa: E402
from jase.multimodel import load_config, read_jsonl, sha256  # noqa: E402
from scripts.migrate_v19_to_v20 import convert  # noqa: E402

RUN = ROOT / "training/runs/qwen35-9b-v20/20260925T190927Z-0468b634"
OUT = ROOT / "benchmarks/outputs/v20-9b-posttrain"


def historical_cases():
    rows = read_jsonl(ROOT / "data/v20_quality/dev.jsonl")
    out = []
    for row in rows:
        if "v19-exact-unseen" not in row.get("tags", []):
            continue
        gold, reasons = convert(row["contract"], row["text"], include_nonfind=True)
        if not reasons:
            out.append({"id": row["id"], "text": row["text"], "contract": gold})
    return out


def baseline_2b(historical):
    report = json.loads((ROOT / "reports/v20/baseline-dev.json").read_text(encoding="utf-8"))
    failures = {row["id"]: row for row in report["failures"]}
    results = []
    for row in historical:
        old = failures.get(row["id"])
        if old is None:
            predicted = row["contract"]
        else:
            predicted, reasons = convert(old["got"], row["text"], include_nonfind=True)
            if reasons:
                predicted = None
        metric = compare_v12(row["contract"], predicted)
        metric["id"] = row["id"]
        results.append(metric)
    return {"source": "reports/v20/baseline-dev.json", "cases": len(results),
            "projection": "V1.1 success -> gold; failures -> V1.1 got, conservatively converted to V1.2",
            "summary": aggregate_v12(results)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN.relative_to(ROOT)))
    ap.add_argument("--output", default=str(OUT.relative_to(ROOT)))
    ap.add_argument("--adapter", help="Override manifest-selected adapter for checkpoint comparison")
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    out = (ROOT / args.output).resolve()
    if ROOT not in run.parents or ROOT not in out.parents:
        ap.error("run and output must be inside the repository")
    adapter = (ROOT / (args.adapter or "")).resolve() if args.adapter else None
    if adapter and (ROOT not in adapter.parents or not (adapter / "adapter_config.json").is_file()):
        ap.error("adapter override must be a repository PEFT adapter directory")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from peft import PeftModel
    from training.train import load_base

    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("Training is not complete")
    cfg = load_config(manifest["model"])
    prompt = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()
    panels = {
        "hard_dev": read_jsonl(ROOT / "benchmarks/v20/hard_dev.jsonl"),
        "holdout": read_jsonl(ROOT / "benchmarks/v20/holdout.jsonl"),
        "v19_common": historical_cases(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "baseline_2b_common.json").write_text(
        json.dumps(baseline_2b(panels["v19_common"]), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    model, tokenizer = load_base(cfg)
    adapter = adapter or ROOT / manifest["adapter"]
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    summary = {"run": str(run.relative_to(ROOT)), "model": cfg["model_id"],
               "revision": cfg["revision"], "adapter": str(adapter.relative_to(ROOT)), "panels": {}}
    for panel, rows in panels.items():
        path = out / f"{panel}.jsonl"
        completed = {r["id"]: r for r in read_jsonl(path)} if path.exists() else {}
        with path.open("a", encoding="utf-8") as handle:
            for index, row in enumerate(rows, 1):
                if row["id"] in completed:
                    continue
                turns = [{"role": "system", "content": prompt},
                         {"role": "user", "content": row["text"]}]
                encoded = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=True,
                    return_tensors="pt", add_generation_prompt=True, enable_thinking=False).to("cuda")
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                with torch.inference_mode():
                    ids = model.generate(**encoded, max_new_tokens=cfg["max_new_tokens"],
                        do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - start
                output_ids = ids[0, encoded["input_ids"].shape[-1]:]
                raw = tokenizer.decode(output_ids, skip_special_tokens=True)
                try:
                    json_valid = isinstance(json.loads(raw), dict)
                except (ValueError, TypeError):
                    json_valid = False
                try:
                    predicted = parse_contract(raw)
                    metric = compare_v12(row["contract"], predicted)
                except (ValueError, TypeError, KeyError, ExtractionError) as exc:
                    metric = {"schema_valid": False, "complete_success": False,
                              "issues": [type(exc).__name__]}
                metric["json_valid"] = json_valid
                metric["complete_success"] = bool(metric["complete_success"] and json_valid)
                record = {"id": row["id"], "text": row["text"], "raw": raw,
                          "metric": metric, "output_tokens": len(output_ids),
                          "latency_seconds": round(seconds, 3),
                          "peak_vram_bytes": torch.cuda.max_memory_allocated()}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 20 == 0:
                    print(f"{panel}: {index}/{len(rows)}", flush=True)
        produced = read_jsonl(path)
        if len(produced) != len(rows) or {r["id"] for r in produced} != {r["id"] for r in rows}:
            raise RuntimeError(f"Incomplete panel {panel}")
        metrics = [r["metric"] for r in produced]
        seconds = sum(r["latency_seconds"] for r in produced)
        report = {"cases": len(rows), "gold_sha256": (sha256(ROOT / f"benchmarks/v20/{panel}.jsonl")
                  if panel != "v19_common" else sha256(ROOT / "data/v20_quality/dev.jsonl")),
                  "json_valid_rate": sum(r["json_valid"] for r in metrics) / len(rows),
                  "metrics": aggregate_v12(metrics),
                  "tokens_per_second": sum(r["output_tokens"] for r in produced) / seconds,
                  "peak_vram_bytes": max(r["peak_vram_bytes"] for r in produced)}
        summary["panels"][panel] = report
        (out / f"{panel}_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({panel: report}, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
