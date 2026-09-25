"""Shared, dependency-light contracts for multi-model experiments."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "models"


def load_config(name: str) -> dict:
    path = CONFIGS / f"{name}.json"
    if name != path.stem or not path.is_file():
        raise ValueError(f"Unknown model {name!r}; available: {', '.join(p.stem for p in CONFIGS.glob('*.json'))}")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("name") != name:
        raise ValueError(f"Config name mismatch: {path}")
    return cfg


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(path: Path) -> str:
    """Hash Git-style LF content, stable across macOS and core.autocrlf Windows."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def case(row: dict) -> dict:
    if "messages" in row:
        messages = row["messages"]
        return {"text": next(m["content"] for m in messages if m["role"] == "user"),
                "contract": json.loads(messages[-1]["content"]), "tags": row.get("tags", []),
                "family": row.get("source_scenario_id")}
    return row


def dataset_stats(path: Path) -> dict:
    from .leakage import normalize_text

    rows = [case(row) for row in read_jsonl(path)]
    unique_prompts = len({normalize_text(r["text"]) for r in rows})
    actions = Counter(g["action"] for row in rows for g in row["contract"].get("goals", []))
    tags = Counter(t for row in rows for t in row.get("tags", []))
    total_actions = sum(actions.values())
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "rows": len(rows),
            "unique_prompts": unique_prompts, "duplicate_prompt_rows": len(rows) - unique_prompts,
            "actions": dict(sorted(actions.items())), "tags": dict(sorted(tags.items())),
            "rare_actions_under_1pct": sorted(a for a, count in actions.items()
                                               if count / max(1, total_actions) < .01),
            "mean_input_chars": round(sum(len(r["text"]) for r in rows) / max(1, len(rows)), 2),
            "mean_output_chars": round(sum(len(json.dumps(r["contract"], ensure_ascii=False)) for r in rows) / max(1, len(rows)), 2),
            "families": len({r.get("family") for r in rows if r.get("family")})}


def messages(text: str) -> list[dict]:
    system = (ROOT / "TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": text}]


def encode_supervised(tokenizer, row: dict, max_length: int) -> dict:
    turns = row["messages"]
    common = dict(tokenize=True, return_dict=False, enable_thinking=False)
    prefix = tokenizer.apply_chat_template(turns[:-1], add_generation_prompt=True, **common)
    full = tokenizer.apply_chat_template(turns, **common)
    if full[:len(prefix)] != prefix:
        raise ValueError("Training prefix differs from inference prompt")
    if len(full) > max_length:
        raise ValueError(f"Example has {len(full)} tokens, exceeds {max_length}; refusing truncation")
    if len(full) == len(prefix):
        raise ValueError("Assistant answer has no tokens")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prefix) + full[len(prefix):]}


def clean_validation(train_rows: list[dict], valid_rows: list[dict]) -> tuple[list[dict], dict]:
    from .leakage import normalize_text

    def key(row):
        return normalize_text(next(m["content"] for m in row["messages"] if m["role"] == "user"))

    train_texts = {key(row) for row in train_rows}
    labels: dict[str, set[str]] = {}
    for row in valid_rows:
        labels.setdefault(key(row), set()).add(row["messages"][-1]["content"])
    conflicting = {name for name, values in labels.items() if len(values) > 1}
    seen = set()
    kept = []
    for row in valid_rows:
        name = key(row)
        if name in train_texts or name in conflicting or name in seen:
            continue
        seen.add(name)
        kept.append(row)
    return kept, {"source_rows": len(valid_rows), "effective_rows": len(kept),
                  "overlap_prompts_removed": len(train_texts & labels.keys()),
                  "conflicting_prompts_removed": len(conflicting),
                  "duplicate_rows_removed": len(valid_rows) - len(kept) -
                                            sum(key(r) in train_texts or key(r) in conflicting for r in valid_rows)}
