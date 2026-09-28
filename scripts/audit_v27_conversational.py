"""Full V27 tokenizer, schema and completion-mask audit; no model weights loaded."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.conversation_v1 import validate_decision  # noqa: E402
from jase.multimodel import encode_supervised, read_jsonl  # noqa: E402

DATA = ROOT / "data/v27_conversational"
CFG = json.loads((ROOT / "configs/models/qwen35-9b-v27.json").read_text(encoding="utf-8"))


def main() -> None:
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(CFG["model_id"], revision=CFG["revision"], local_files_only=True)
    prompt = (ROOT / CFG["system_prompt_file"]).read_text(encoding="utf-8").strip()
    report = {"model_id": CFG["model_id"], "revision": CFG["revision"],
              "tokenizer": tokenizer.__class__.__name__, "max_length": CFG["max_length"], "splits": {}}
    keys = {}
    for split in ("train", "valid"):
        path = DATA / f"{split}.jsonl"
        rows = read_jsonl(path)
        lengths, target_lengths = [], []
        categories = Counter()
        keys[split] = set()
        first_inspection = None
        for index, row in enumerate(rows):
            if row["messages"][0] != {"role": "system", "content": prompt}:
                raise ValueError(f"System prompt drift at {split}:{index}")
            input_json = json.loads(row["messages"][1]["content"])
            target_json = json.loads(row["messages"][2]["content"])
            if input_json["task"] == "semantic_update":
                validate_decision(target_json)
            elif input_json["task"] == "verbalize_plan":
                if set(target_json) != {"reply"} or not isinstance(target_json["reply"], str):
                    raise ValueError("Invalid Stage-B target")
            else:
                raise ValueError("Unknown model task")
            key = row["messages"][1]["content"]
            if key in keys[split]:
                raise ValueError(f"Duplicate prompt within {split}")
            keys[split].add(key)
            encoded = encode_supervised(tokenizer, row, 4096, {"enable_thinking": False})
            size = len(encoded["input_ids"])
            active = sum(label != -100 for label in encoded["labels"])
            prefix = size - active
            if (not 0 < prefix < size or encoded["labels"][:prefix] != [-100] * prefix or
                    encoded["labels"][prefix:] != encoded["input_ids"][prefix:]):
                raise ValueError(f"Completion mask error at {split}:{index}")
            lengths.append(size)
            target_lengths.append(active)
            categories[row["category"]] += 1
            if first_inspection is None:
                first_inspection = {"index": index, "category": row["category"],
                                    "total_tokens": size, "masked_tokens": prefix, "target_tokens": active,
                                    "masked_prefix_preview": tokenizer.decode(encoded["input_ids"][:20]),
                                    "active_target_preview": tokenizer.decode(encoded["input_ids"][prefix:prefix + 25])}
        ordered = sorted(lengths)
        report["splits"][split] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "categories": dict(categories), "max_tokens": max(lengths),
            "p50_tokens": ordered[len(ordered) // 2], "p95_tokens": ordered[int(len(ordered) * .95)],
            "over_512": sum(x > 512 for x in lengths),
            "over_configured_limit": sum(x > CFG["max_length"] for x in lengths),
            "min_target_tokens": min(target_lengths), "max_target_tokens": max(target_lengths),
            "completion_mask_verified_rows": len(rows), "first_inspection": first_inspection}
    report["cross_split_exact_prompt_overlap"] = len(keys["train"] & keys["valid"])
    if report["cross_split_exact_prompt_overlap"] or any(
            report["splits"][split]["over_configured_limit"] for split in ("train", "valid")):
        raise ValueError(json.dumps(report, ensure_ascii=False))
    out = DATA / "tokenizer_audit.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": report["splits"]["train"]["rows"],
                      "valid": report["splits"]["valid"]["rows"],
                      "max_train_tokens": report["splits"]["train"]["max_tokens"],
                      "max_valid_tokens": report["splits"]["valid"]["max_tokens"],
                      "over_512_train": report["splits"]["train"]["over_512"]}, indent=2))


if __name__ == "__main__":
    main()
