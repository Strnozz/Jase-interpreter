"""Offline V1.3 invariant/canonicalization audit of frozen V1.2 generations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.canonical_semantics import compare as canonical_compare  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402
from jase.semantic_guard_v1_3 import ACCEPT, check_contract  # noqa: E402

TEMPORAL = {"date", "time", "arrival_time", "departure_time", "time_of_day"}
CARDINALITY = {"quantity"}


def _slots(contract: dict, fields: set[str]) -> list[str]:
    return sorted(json.dumps({"goal": g["id"], **f}, ensure_ascii=False, sort_keys=True)
                  for g in contract.get("goals", []) for f in g.get("facts", [])
                  if f.get("field") in fields)


def _context(contract: dict) -> list[str]:
    return sorted(json.dumps(f, ensure_ascii=False, sort_keys=True)
                  for f in contract.get("context", []))


def _policy(contract: dict) -> list[str]:
    return sorted(json.dumps({"goal": g["id"], "action": g["action"], "policy": g.get("policy", {})},
                             ensure_ascii=False, sort_keys=True) for g in contract.get("goals", []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True)
    ap.add_argument("--prediction-file", required=True)
    ap.add_argument("--review")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    panel, pred, output = ((ROOT / value).resolve() for value in
                           (args.panel, args.prediction_file, args.output))
    review = (ROOT / args.review).resolve() if args.review else None
    if any(ROOT not in x.parents for x in (panel, pred, output)) or (review and ROOT not in review.parents):
        ap.error("paths must stay in repository")
    gold = {r["id"]: r for r in read_jsonl(panel)}
    rows = read_jsonl(pred)
    if len(rows) != len(gold) or {r["id"] for r in rows} != set(gold):
        raise RuntimeError("Panel and prediction IDs disagree")
    labels = json.loads(review.read_text(encoding="utf-8")) if review else None
    if labels and (labels["panel_sha256"] != sha256(panel) or
                   labels["prediction_sha256"] != sha256(pred) or set(labels["labels"]) != set(gold)):
        raise RuntimeError("Provisional review does not match frozen artifacts")
    counts = Counter()
    codes = Counter()
    records = []
    for row in rows:
        expected = gold[row["id"]]["contract"]
        try:
            got = json.loads(row["raw"])
        except (TypeError, ValueError):
            got = None
        equivalent = canonical_compare(expected, got)
        issues = validate_contract(got, user_text=row["text"])
        decision = check_contract(row["raw"], row["text"])
        exact = bool(row["metric"]["complete_success"])
        label = labels["labels"][row["id"]] if labels else None
        counts["cases"] += 1
        counts["literal_equal"] += equivalent["literal_equal"]
        counts["legacy_exact"] += exact
        counts["canonical_exact"] += equivalent["canonical_equal"]
        counts["semantic_invariant_valid"] += not issues
        counts["guard_accept"] += decision.status == ACCEPT
        counts["accepted_legacy_wrong"] += decision.status == ACCEPT and not exact
        if labels:
            counts["accepted_material_or_routing_risk"] += decision.status == ACCEPT and label == "material_or_routing_risk"
            counts["held_exact_correct"] += decision.status != ACCEPT and label == "exact_correct"
        if isinstance(got, dict) and "goals" in got:
            counts["temporal_slots_correct"] += _slots(expected, TEMPORAL) == _slots(got, TEMPORAL)
            counts["context_correct"] += _context(expected) == _context(got)
            counts["cardinality_correct"] += _slots(expected, CARDINALITY) == _slots(got, CARDINALITY)
            counts["policy_action_correct"] += _policy(expected) == _policy(got)
        for issue in issues:
            codes[issue.code] += 1
        records.append({"id": row["id"], "literal_equal": equivalent["literal_equal"],
                        "legacy_exact": exact, "canonical_exact": equivalent["canonical_equal"],
                        "semantic_issues": [i.as_dict() for i in issues],
                        "guard": {"status": decision.status, "codes": list(decision.codes)},
                        **({"provisional_label": label} if label else {})})
    summary = {"panel": str(panel.relative_to(ROOT)), "panel_sha256": sha256(panel),
               "prediction_file": str(pred.relative_to(ROOT)), "prediction_sha256": sha256(pred),
               "review": str(review.relative_to(ROOT)) if review else None,
               "counts": dict(counts), "issue_codes": dict(codes),
               "limitations": "AI review provisional; canonical equality uses only documented safe aliases; ACCEPT is Planner handoff only",
               "records": records}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"counts": summary["counts"], "issue_codes": summary["issue_codes"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
