"""Offline V1.2 benchmark scorer. It never loads a model or executes actions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.canon import ExtractionError  # noqa: E402
from jase.goal_contract_v1_2 import parse_contract  # noqa: E402
from jase.metrics_v1_2 import aggregate_v12, compare_v12  # noqa: E402


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="benchmarks/v20/hard_dev.jsonl")
    ap.add_argument("--predictions", required=True, help="JSONL with id and contract or raw output")
    ap.add_argument("--output", default="benchmarks/outputs/v20-static-eval.json")
    args = ap.parse_args()
    gold = ROOT / args.gold
    predicted = ROOT / args.predictions
    expected = {row["id"]: row for row in rows(gold)}
    inputs = rows(predicted)
    if len(inputs) != len(expected) or {row.get("id") for row in inputs} != set(expected):
        ap.error("predictions must cover each gold ID exactly once")
    results = []
    for row in inputs:
        item = expected[row["id"]]
        try:
            raw = row.get("contract", row.get("output"))
            if isinstance(raw, dict):
                json_valid = True
            elif isinstance(raw, str):
                json_valid = isinstance(json.loads(raw), dict)
            else:
                json_valid = False
            candidate = parse_contract(raw)
            result = compare_v12(item["contract"], candidate)
            result["json_valid"] = json_valid
            result["complete_success"] = result["complete_success"] and json_valid
        except (ValueError, TypeError, KeyError, json.JSONDecodeError, ExtractionError) as exc:
            result = {"json_valid": False, "schema_valid": False, "complete_success": False,
                      "issues": [type(exc).__name__]}
        result["id"] = row["id"]
        results.append(result)
    summary = aggregate_v12(results)
    summary["json_valid_rate"] = sum(row["json_valid"] for row in results) / len(results)
    summary["metric_scope"] = "all-case success; conditional fact rates only on schema-valid outputs"
    report = {"gold": args.gold, "predictions": args.predictions, "summary": summary}
    if "holdout" not in gold.name.lower():
        report["cases"] = results
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
