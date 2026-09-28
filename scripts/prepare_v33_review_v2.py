"""Prepare a richer, immutable-input V33 adjudication package.

This script does not decide the gold. All proposed replacements remain pending
independent human review; existing V33_RAW and prior review artifacts are read-only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/v33_review/review_queue.jsonl"
PLANNER = ROOT / "evaluation/replays/v33_qwen_v26_planner_1_3.jsonl"
RAW = ROOT / "benchmarks/v33/blind.jsonl"
OUTPUT = ROOT / "benchmarks/v33_review/review_pending_v2.jsonl"
EXPECTED_RAW = "4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37"

TEMPORAL = {"availability_field", "same_time_reference", "approximate_time", "message_paraphrase", "stops_operator"}
POLICY = {"policy_scope", "english_vehicle_buy_policy", "find_or_book_ambiguous", "send_me_result_not_external_contact"}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    if hashlib.sha256(RAW.read_bytes()).hexdigest() != EXPECTED_RAW:
        raise RuntimeError("V33_RAW hash mismatch")
    planner = {item["id"]: item for item in rows(PLANNER)}
    source = rows(SOURCE)
    if len(source) != 19 or len(planner) != 100:
        raise RuntimeError("Expected 19 flagged gold and 100 frozen Planner rows")
    if OUTPUT.exists():
        raise FileExistsError("Review package is already versioned; do not overwrite")
    enriched = []
    for item in source:
        flags = set(item["review_flags"])
        snapshot = planner[item["case_id"]]
        enriched.append({
            **item,
            "review_artifact_version": "V33_REVIEW_PENDING_2",
            "planner_result": snapshot["planner"],
            "planner_snapshot_label": "v33_qwen_v26_planner_1_3_frozen_replay",
            "temporal_ambiguity": bool(flags & TEMPORAL),
            "policy_ambiguity": bool(flags & POLICY),
            "proposed_reviewed_interpretation": None,
            "exact_fields_to_change": None,
            "canonical_exact_scoring_change": "undetermined_pending_human_adjudication",
            "planner_behavior_change": "undetermined_pending_human_adjudication",
            "candidate_interpretations_for_review": [
                {"source": "V33_RAW", "contract": item["raw_gold"]},
                {"source": "Qwen_V26_RAW", "raw": item["qwen_raw_output"]},
                {"source": "Granite_V26_RAW", "raw": item["granite_raw_output"]},
            ],
            "adjudication_status": "PENDING_INDEPENDENT_HUMAN_REVIEW",
        })
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in enriched), encoding="utf-8")
    print(json.dumps({"cases": len(enriched), "output": str(OUTPUT.relative_to(ROOT)),
                      "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
