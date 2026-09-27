"""Restartable 9B benchmark for V1.3 gold and conservatively projectable V1.2 gold."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.metrics_v1_3 import aggregate, score  # noqa: E402
from jase.multimodel import load_config, read_jsonl, sha256  # noqa: E402
from jase.foundation import chat_template_kwargs, load_base  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", help="Completed training run with PEFT adapter")
    source.add_argument("--model-config", help="Config name for vanilla base-model evaluation")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--adapter")
    args = ap.parse_args()
    panel, output = ((ROOT / x).resolve() for x in (args.panel, args.output))
    if any(ROOT not in x.parents for x in (panel, output)):
        ap.error("panel and output must be in repository")
    if args.run:
        run = (ROOT / args.run).resolve()
        if ROOT not in run.parents:
            ap.error("run must be in repository")
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "complete":
            raise RuntimeError("Run is not complete")
        cfg = load_config(manifest["model"])
        adapter = (ROOT / (args.adapter or manifest["adapter"])).resolve()
        if ROOT not in adapter.parents or not (adapter / "adapter_config.json").is_file():
            ap.error("Adapter must be a repository PEFT directory")
    else:
        if args.adapter:
            ap.error("--adapter requires --run")
        run, adapter = None, None
        cfg = load_config(args.model_config)
    rows = read_jsonl(panel)
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise RuntimeError("Panel must be nonempty with unique IDs")
    prompt = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel

    output.mkdir(parents=True, exist_ok=True)
    cases_path = output / "cases.jsonl"
    completed = {r["id"]: r for r in read_jsonl(cases_path)} if cases_path.exists() else {}
    model, tokenizer = load_base(cfg)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    with cases_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, row in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            turns = [{"role": "system", "content": prompt},
                     {"role": "user", "content": row["text"]}]
            encoded = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True,
                **chat_template_kwargs(cfg)).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**encoded,
                                     **cfg.get("decoding", {"max_new_tokens": cfg["max_new_tokens"],
                                                            "do_sample": False}),
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            generated = ids[0, encoded["input_ids"].shape[-1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            try:
                parsed = json.loads(raw)
                json_valid = isinstance(parsed, dict)
            except (TypeError, ValueError):
                parsed, json_valid = None, False
            metric = score(row["contract"], parsed, row["text"])
            record = {"id": row["id"], "family": row.get("family", row.get("category")),
                      "text": row["text"], "raw": raw, "json_valid": json_valid,
                      "metric": metric, "output_tokens": len(generated),
                      "latency_seconds": round(seconds, 3),
                      "peak_vram_bytes": torch.cuda.max_memory_allocated()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0:
                print(f"{index}/{len(rows)}", flush=True)
    produced = read_jsonl(cases_path)
    if len(produced) != len(rows) or {r["id"] for r in produced} != {r["id"] for r in rows}:
        raise RuntimeError("Benchmark incomplete")
    summary = {"run": str(run.relative_to(ROOT)) if run else None,
               "adapter": str(adapter.relative_to(ROOT)) if adapter else None,
               "model_config": cfg["name"], "model_id": cfg["model_id"],
               "revision": cfg["revision"],
               "system_prompt_sha256": sha256(ROOT / cfg["system_prompt_file"]),
               "panel": str(panel.relative_to(ROOT)), "panel_sha256": sha256(panel),
               "cases_sha256": sha256(cases_path),
               "json_valid": sum(r["json_valid"] for r in produced),
               "schema_valid": sum(r["json_valid"] and not r["metric"].get("candidate_projection_hold")
                                   for r in produced),
               "metrics": aggregate([r["metric"] for r in produced]),
               "tokens_per_second": round(sum(r["output_tokens"] for r in produced) /
                                          sum(r["latency_seconds"] for r in produced), 2),
               "peak_vram_bytes": max(r["peak_vram_bytes"] for r in produced),
               "limitations": ["GoalContract V1.2 projection can hold ambiguous outputs",
                               "Exact mismatch does not prove material risk",
                               "Targeted AI-authored panel; manual review still required"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
