"""Conservative, deterministic V1.1→V1.2 candidate migration. No training.

Only a one-goal, read-only find request with no gate/clarification/reference
can be auto-labeled. Everything else is queued for human semantic review.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.guard import ACCEPT, check_contract  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402

SOURCE = ROOT / "data" / "v19_clean"
DEST = ROOT / "data" / "v20"
SYSTEM = (ROOT / "V20_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle, 1):
            yield index, json.loads(line)


def source_text(row: dict) -> str:
    return next(m["content"] for m in row["messages"] if m["role"] == "user")


def source_contract(row: dict) -> dict:
    return json.loads(next(m["content"] for m in row["messages"] if m["role"] == "assistant"))


def convert(contract: dict, text: str, *, include_nonfind: bool = False) -> tuple[dict | None, list[str]]:
    reasons: list[str] = []
    if contract.get("schema_version") != "1.1":
        reasons.append("source_version")
    goals = contract.get("goals", [])
    if len(goals) != 1:
        reasons.append("multi_goal_requires_review")
    if contract.get("clarification", {}).get("required"):
        reasons.append("clarification_requires_review")
    if not goals:
        return None, reasons or ["no_goal"]
    goal = goals[0]
    if goal.get("action") != "find" and not include_nonfind:
        reasons.append("side_effect_or_nonfind_action_requires_review")
    if goal.get("gate") is not None:
        reasons.append("gate_requires_review")
    if not isinstance(goal.get("target", {}).get("type"), str) or not goal["target"]["type"]:
        reasons.append("target_type_unknown")
    facts = []
    for bucket, strength in (("attributes", "hard"), ("constraints", "hard"), ("preferences", "soft")):
        for predicate in goal.get(bucket, []):
            if predicate.get("reference") is not None:
                reasons.append("free_text_reference_requires_review")
            if predicate.get("value") is None:
                reasons.append("null_fact_requires_review")
            fact = {"field": predicate["property"], "op": predicate["operator"],
                    "value": predicate["value"], "strength": strength}
            for key in ("value_to", "unit", "currency"):
                if predicate.get(key) is not None:
                    fact[key] = predicate[key]
            facts.append(fact)
    if reasons:
        return None, sorted(set(reasons))
    # A source requiring repair cannot be silently converted as a gold label.
    source_result = check_contract(contract, text)
    if source_result.status != ACCEPT:
        return None, ["source_guard_" + source_result.status.lower()]
    converted = {"schema_version": "1.2", "kind": "task", "goals": [
        {"id": "g1", "action": goal["action"], "target": {"name": goal["target"]["name"],
                                               "type": goal["target"]["type"]},
         "facts": facts}]}
    issues = validate_contract(converted)
    if issues:
        return None, ["v12_" + issue.code for issue in issues]
    return converted, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/v20")
    parser.add_argument("--include-nonfind", action="store_true")
    args = parser.parse_args()
    global DEST
    DEST = (ROOT / args.output).resolve()
    if ROOT not in DEST.parents:
        parser.error("output must be within repository")
    DEST.mkdir(parents=True, exist_ok=True)
    source_hashes = {split: sha(SOURCE / f"{split}.jsonl") for split in ("train", "valid")}
    candidates = []
    review = []
    rejected = []
    reasons = Counter()
    source_count = Counter()
    for split in ("train", "valid"):
        for index, row in read(SOURCE / f"{split}.jsonl"):
            source_count[split] += 1
            text = source_text(row)
            record_id = hashlib.sha256(f"{split}:{index}:{text}".encode()).hexdigest()[:20]
            try:
                legacy = source_contract(row)
                converted, why = convert(legacy, text, include_nonfind=args.include_nonfind)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                rejected.append({"id": record_id, "split": split, "reason": type(exc).__name__})
                reasons["parse_or_conversion_error"] += 1
                continue
            if why:
                review.append({"id": record_id, "split": split, "text": text, "reasons": why})
                reasons.update(why)
            else:
                candidates.append({"id": record_id, "split": split, "text": text,
                                   "contract": converted, "source_line": index})

    # The old development and release panels are exclusion sets only. No
    # per-case label/prediction informs conversion rules.
    excluded = set()
    exclusion_paths = (ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl")
    for path in exclusion_paths:
        if path.exists():
            excluded.update(normalize_text(row["text"]) for _, row in read(path))
    by_key = defaultdict(list)
    for item in candidates:
        by_key[normalize_text(item["text"])].append(item)
    kept = {"train": [], "valid": []}
    for key, group in sorted(by_key.items()):
        labels = {json.dumps(item["contract"], sort_keys=True, ensure_ascii=False) for item in group}
        splits = {item["split"] for item in group}
        if key in excluded or len(labels) > 1 or len(splits) > 1:
            reason = ("benchmark_overlap" if key in excluded else
                      "conflicting_normalized_prompt" if len(labels) > 1 else "train_valid_overlap")
            for item in group:
                review.append({"id": item["id"], "split": item["split"],
                               "text": item["text"], "reasons": [reason]})
                reasons[reason] += 1
            continue
        first = min(group, key=lambda item: item["source_line"])
        kept[first["split"]].append(first)
        for item in group:
            if item is first:
                continue
            review.append({"id": item["id"], "split": item["split"],
                           "text": item["text"], "reasons": ["duplicate_normalized_prompt"]})
            reasons["duplicate_normalized_prompt"] += 1

    # V19 has many split variants with identical token sets but changed word
    # order. Keep the train side and quarantine suspicious validation prompts.
    postings = defaultdict(set)
    train_tokens = [token_set(item["text"]) for item in kept["train"]]
    for index, tokens in enumerate(train_tokens):
        for token in tokens:
            postings[token].add(index)
    clean_valid = []
    for item in kept["valid"]:
        tokens = token_set(item["text"])
        candidate_ids = set()
        for token in sorted(tokens, key=lambda t: len(postings[t]))[:3]:
            candidate_ids.update(postings[token])
        near = False
        for index in candidate_ids:
            other = train_tokens[index]
            if not tokens or not other or min(len(tokens), len(other)) / max(len(tokens), len(other)) < .85:
                continue
            if len(tokens & other) / len(tokens | other) >= .85:
                near = True
                break
        if near:
            review.append({"id": item["id"], "split": "valid", "text": item["text"],
                           "reasons": ["near_train_overlap"]})
            reasons["near_train_overlap"] += 1
        else:
            clean_valid.append(item)
    kept["valid"] = clean_valid

    for split in ("train", "valid"):
        path = DEST / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in sorted(kept[split], key=lambda r: r["source_line"]):
                answer = json.dumps(item["contract"], ensure_ascii=False, separators=(",", ":"))
                row = {"messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": item["text"]},
                                    {"role": "assistant", "content": answer}],
                       "source_id": item["id"], "source_split": split}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    for name, rows in (("review", review), ("rejected", rejected)):
        with (DEST / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    output_files = {name: sha(DEST / f"{name}.jsonl") for name in ("train", "valid", "review", "rejected")}
    rule = ("one goal, any supported action, no gate/clarification/reference, source Guard ACCEPT, V1.2 valid"
            if args.include_nonfind else
            "one find goal, no gate/clarification/reference, source Guard ACCEPT, V1.2 valid")
    manifest = {"schema_version": "1.2", "source": "data/v19_clean", "source_sha256": source_hashes,
                "migration_rule": rule,
                "near_overlap_threshold": 0.85,
                "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in (
                    Path(__file__).resolve(), ROOT / "jase/goal_contract_v1_2.py",
                    ROOT / "schema/goal_contract_v1_2.schema.json", ROOT / "V20_SYSTEM_PROMPT.txt",
                    ROOT / "jase/guard.py", ROOT / "jase/canon.py", ROOT / "jase/leakage.py")},
                "exclusion_sha256": {str(path.relative_to(ROOT)): sha(path) for path in exclusion_paths if path.exists()},
                "source_rows": dict(source_count),
                "automatically_converted": {split: len(kept[split]) for split in ("train", "valid")},
                "requires_review": len(review), "rejected": len(rejected),
                "reason_counts": dict(sorted(reasons.items())), "output_sha256": output_files,
                "newly_generated_train": 0, "semantic_review_completed": 0,
                "training_readiness": "NOT_READY"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Check that original bytes did not change and no normalized overlap remains.
    assert source_hashes == {split: sha(SOURCE / f"{split}.jsonl") for split in source_hashes}
    assert not ({normalize_text(x["text"]) for x in kept["train"]} &
                {normalize_text(x["text"]) for x in kept["valid"]})
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
