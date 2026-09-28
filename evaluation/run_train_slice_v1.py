"""Replay frozen Qwen V26 contracts through the read-only train vertical slice."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.canonical_semantics import canonicalize  # noqa: E402
from jase.train_provider_v1 import GtfsSnapshotTrainProvider  # noqa: E402
from jase.train_vertical_slice_v1 import run_train_search  # noqa: E402

PANEL = ROOT / "benchmarks/train_slice_v1/blind.jsonl"
PREDICTIONS = ROOT / "benchmarks/outputs/qwen-v26-train-slice-v1/cases.jsonl"
FEED = ROOT / "data/provider_cache/trenord_gtfs_2026-09-26.zip"
OUT = ROOT / "evaluation/train_slice_v1"
EXPECTED_PANEL_SHA = "339aad6389e9076e4a2f08436dbe44fcc9f920c3fa58f8f79e204df83ae96783"
REQUEST_AT = datetime(2026, 9, 28, 12, tzinfo=ZoneInfo("Europe/Rome"))


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def presence_normalized_exact(gold: dict, candidate: dict | None) -> bool:
    if not isinstance(candidate, dict):
        return False
    def normalize(value):
        if isinstance(value, dict):
            return {key: normalize(item) for key, item in value.items()
                    if item not in ([], {})}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    try:
        return normalize(canonicalize(gold)) == normalize(canonicalize(candidate))
    except (ValueError, KeyError):
        return False


def compact(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "planner"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if hashlib.sha256(PANEL.read_bytes()).hexdigest() != EXPECTED_PANEL_SHA:
        raise RuntimeError("Frozen train panel hash changed")
    gold = rows(PANEL)
    predicted = {row["id"]: row for row in rows(PREDICTIONS)}
    if len(gold) != 61 or len(predicted) != 61:
        raise RuntimeError("Incomplete panel or Qwen V26 inference")
    OUT.mkdir(exist_ok=True)
    output = OUT / "replay.jsonl"
    summary_path = OUT / "summary.json"
    if (output.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError("Replay already exists; do not silently overwrite")
    provider = GtfsSnapshotTrainProvider(FEED)
    counts = Counter()
    data = []
    for row in gold:
        qwen = predicted[row["id"]]
        try:
            parsed = json.loads(qwen["raw"])
        except (TypeError, ValueError):
            parsed = None
        candidate = run_train_search(qwen["raw"], row["text"], request_at=REQUEST_AT, provider=provider)
        oracle = run_train_search(row["contract"], row["text"], request_at=REQUEST_AT, provider=provider)
        called = candidate["provider_called"]
        gold_call = oracle["provider_called"]
        explicit_no_call = row["expected_system"] in {"NO_PROVIDER_CALL", "NO_WRITE", "HOLD_MISSING_INFORMATION",
                                                     "HOLD_UNRESOLVED_REFERENCE", "HOLD_TEMPORAL_CONFLICT",
                                                     "HOLD_MULTI_GOAL_UNSUPPORTED", "HOLD_IF_UNSUPPORTED",
                                                     "HOLD_IF_PRICE_UNAVAILABLE"}
        provisional_material = []
        if called and explicit_no_call:
            provisional_material.append("provider_called_for_expected_hold_or_no_action")
        if called and gold_call:
            actual_request = candidate["provider_request"]
            gold_request = oracle["provider_request"]
            for key in ("origin", "destination", "travel_date", "time_constraints", "direct_only",
                        "train_category", "max_duration_minutes", "limit", "sort"):
                if actual_request[key] != gold_request[key]:
                    provisional_material.append("provider_request_diff:" + key)
        if called and not gold_call and not provisional_material:
            provisional_material.append("gold_holds_candidate_reads_needs_case_review")
        record = {"id": row["id"], "family": row["family"], "request": row["text"],
                  "expected_system": row["expected_system"], "review_flags": row["review_flags"],
                  "interpreter": {"json_valid": qwen["json_valid"],
                                  "schema_valid": not qwen["metric"].get("candidate_projection_hold"),
                                  "legacy_canonical_exact": qwen["metric"]["canonical_equal"],
                                  "empty_optional_fields_equivalent": presence_normalized_exact(row["contract"], parsed),
                                  "metric": qwen["metric"], "raw": qwen["raw"]},
                  "system": compact(candidate), "gold_system_diagnostic": compact(oracle),
                  "provisional_material_risk_flags": provisional_material,
                  "risk_adjudication": "PENDING_CASE_REVIEW" if provisional_material else "NO_PROVISIONAL_FLAG"}
        data.append(record)
        counts["provider_called"] += called
        counts["gold_provider_called"] += gold_call
        counts["material_flagged_for_review"] += bool(provisional_material)
        counts["false_hold_against_agent_gold"] += gold_call and not called
        counts["called_when_explicit_no_call_expected"] += called and explicit_no_call
        counts["empty_optional_fields_equivalent"] += record["interpreter"]["empty_optional_fields_equivalent"]
        counts["json_valid"] += qwen["json_valid"]
        counts["schema_valid"] += record["interpreter"]["schema_valid"]
        counts["canonical_exact_legacy"] += qwen["metric"]["canonical_equal"]
        counts["consequential_permission_true"] += candidate["execution_permitted"] or candidate["external_side_effect_permitted"]
        counts["status_" + candidate["status"]] += 1
    output.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in data), encoding="utf-8")
    summary = {"panel_sha256": EXPECTED_PANEL_SHA,
               "qwen_cases_sha256": hashlib.sha256(PREDICTIONS.read_bytes()).hexdigest(),
               "feed_sha256": provider.source_sha256,
               "feed_coverage": [provider.coverage_start.isoformat(), provider.coverage_end.isoformat()],
               "provider_source_kind": provider.source_kind,
               "request_at": REQUEST_AT.isoformat(), "cases": len(data),
               "counts": dict(sorted(counts.items())),
               "replay_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
               "limitations": ["Agent-authored gold pending independent human review",
                               "Provisional material flags require case review",
                               "GTFS static scheduled direct trains only; no fare or live availability",
                               "Legacy exact penalizes empty optional arrays in frozen gold"]}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
