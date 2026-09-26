"""Pinned-tokenizer, schema, mask and split audit for the frozen V21 corpus."""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.leakage import normalize_text  # noqa: E402
from jase.multimodel import encode_supervised, read_jsonl, sha256  # noqa: E402


def main() -> int:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    cfg = json.loads((ROOT / "configs/models/qwen35-9b-v21.json").read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"], local_files_only=True)
    dataset = ROOT / cfg["dataset"]
    prompt = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()
    report = {"model": cfg["model_id"], "revision": cfg["revision"],
              "max_length": cfg["max_length"], "splits": {}}
    seen = set()
    for split in ("train", "valid"):
        path = dataset / f"{split}.jsonl"
        rows = read_jsonl(path)
        lengths, target_lengths = [], []
        categories = Counter()
        features = Counter()
        for row in rows:
            if row["messages"][0] != {"role": "system", "content": prompt}:
                raise RuntimeError("System prompt mismatch")
            text = row["messages"][1]["content"]
            key = normalize_text(text)
            if key in seen:
                raise RuntimeError(f"Duplicate/cross-split prompt: {text}")
            seen.add(key)
            gold = json.loads(row["messages"][-1]["content"])
            issues = validate_contract(gold)
            if issues:
                raise RuntimeError(f"Invalid gold {text}: {issues}")
            encoded = encode_supervised(tok, row, cfg["max_length"])
            labels = encoded["labels"]
            masked = sum(value == -100 for value in labels)
            if masked <= 0 or masked >= len(labels) or labels[masked:] != encoded["input_ids"][masked:]:
                raise RuntimeError("Completion-only masking failed")
            lengths.append(len(labels))
            target_lengths.append(len(labels) - masked)
            categories[row["category"]] += 1
            features["non_actionable"] += gold["kind"] == "non_actionable"
            features["multi_goal"] += len(gold["goals"]) > 1
            features["context"] += bool(gold.get("context"))
            for goal in gold["goals"]:
                features["missing"] += bool(goal.get("missing"))
                features["policy"] += bool(goal.get("policy"))
                features["dependency"] += bool(goal.get("depends_on"))
                features["condition"] += bool(goal.get("condition"))
                features["ranking"] += bool(goal.get("ranking"))
                features["value_reference"] += any("value_ref" in f for f in goal["facts"])
        values = sorted(lengths)
        report["splits"][split] = {"rows": len(rows), "sha256": sha256(path),
            "mean_total_tokens": round(statistics.mean(values), 2),
            "p99_total_tokens": values[int(.99 * (len(values) - 1))],
            "max_total_tokens": values[-1],
            "mean_target_tokens": round(statistics.mean(target_lengths), 2),
            "over_512": sum(x > 512 for x in values),
            "categories": dict(categories), "features": dict(features)}
    (dataset / "token_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
