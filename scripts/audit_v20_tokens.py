"""Read-only tokenizer audit of V20 candidates and matched V19 targets."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.multimodel import read_jsonl, sha256  # noqa: E402


def lengths(tokenizer, row: dict) -> tuple[int, int, int]:
    turns = row["messages"]
    kwargs = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(turns[:-1], add_generation_prompt=True, **kwargs)
    full = tokenizer.apply_chat_template(turns, **kwargs)
    if full[:len(prefix)] != prefix:
        raise ValueError("prefix mismatch")
    return len(prefix), len(full) - len(prefix), len(full)


def stats(values: list[int]) -> dict:
    values = sorted(values)
    if not values:
        return {"count": 0}
    def pct(p):
        return values[min(len(values) - 1, int((len(values) - 1) * p))]
    return {"count": len(values), "min": values[0], "mean": round(statistics.mean(values), 2),
            "median": statistics.median(values), "p90": pct(.9), "p95": pct(.95),
            "p99": pct(.99), "max": values[-1]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/v20")
    args = parser.parse_args()
    dataset = ROOT / args.dataset
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    cfg = json.loads((ROOT / "configs/models/qwen35-9b.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    report = {"model": cfg["model_id"], "revision": cfg["revision"],
              "method": "exact chat template, thinking disabled; no truncation",
              "configured_v19_length": 512, "splits": {}}
    for split in ("train", "valid"):
        old = {}
        for index, row in enumerate(read_jsonl(ROOT / f"data/v19_clean/{split}.jsonl"), 1):
            text = next(m["content"] for m in row["messages"] if m["role"] == "user")
            key = hashlib.sha256(f"{split}:{index}:{text}".encode()).hexdigest()[:20]
            old[key] = row
        rows = read_jsonl(dataset / f"{split}.jsonl")
        current = {part: [] for part in ("input", "target", "total")}
        old_target = []
        for row in rows:
            i, t, full = lengths(tokenizer, row)
            for key, value in (("input", i), ("target", t), ("total", full)):
                current[key].append(value)
            old_target.append(lengths(tokenizer, old[row["source_id"]])[1])
        report["splits"][split] = {part: stats(values) for part, values in current.items()}
        report["splits"][split]["matched_v19_target"] = stats(old_target)
        report["splits"][split]["over_512"] = sum(x > 512 for x in current["total"])
        report["splits"][split]["source_sha256"] = sha256(dataset / f"{split}.jsonl")
        print(split, report["splits"][split]["total"], flush=True)
    report["panels"] = {}
    system = (ROOT / "V20_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
    for name in ("hard_dev", "holdout"):
        path = ROOT / f"benchmarks/v20/{name}.jsonl"
        current = {part: [] for part in ("input", "target", "total")}
        for case in read_jsonl(path):
            row = {"messages": [{"role": "system", "content": system},
                                {"role": "user", "content": case["text"]},
                                {"role": "assistant", "content": json.dumps(case["contract"],
                                    ensure_ascii=False, separators=(",", ":"))}]}
            i, t, full = lengths(tokenizer, row)
            for key, value in (("input", i), ("target", t), ("total", full)):
                current[key].append(value)
        report["panels"][name] = {part: stats(values) for part, values in current.items()}
        report["panels"][name]["over_512"] = sum(x > 512 for x in current["total"])
        report["panels"][name]["sha256"] = sha256(path)
        print(name, report["panels"][name]["total"], flush=True)
    (dataset / "token_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
