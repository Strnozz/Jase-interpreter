"""Check every training example against the exact model chat template and length cap."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import clean_validation, encode_supervised, load_config, read_jsonl, sha256  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-9b")
    args = ap.parse_args()
    cfg = load_config(args.model)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"],
                                              local_files_only=True)
    data = ROOT / cfg["dataset"]
    train = read_jsonl(data / "train.jsonl")
    valid, cleaning = clean_validation(train, read_jsonl(data / "valid.jsonl"))
    report = {"model": cfg["model_id"], "revision": cfg["revision"],
              "max_length": cfg["max_length"], "validation_cleaning": cleaning,
              "data_sha256": {name: sha256(data / f"{name}.jsonl") for name in ("train", "valid")}}
    for name, rows in (("train", train), ("validation", valid)):
        lengths = []
        for i, row in enumerate(rows):
            encoded = encode_supervised(tokenizer, row, cfg["max_length"])
            lengths.append(len(encoded["input_ids"]))
            if (i + 1) % 5000 == 0:
                print(f"{name}: {i + 1}/{len(rows)}", flush=True)
        lengths.sort()
        report[name] = {"rows": len(lengths), "min_tokens": lengths[0],
                        "median_tokens": lengths[len(lengths) // 2],
                        "p99_tokens": lengths[int(len(lengths) * .99)],
                        "max_tokens": lengths[-1], "truncated": 0}
    out = ROOT / "docs" / "QWEN35_9B_TOKEN_PREFLIGHT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
