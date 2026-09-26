"""Paired offline audit of base and experimental routing Guards on saved raw output."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.guard_v1_2 import ACCEPT, check_contract as base_check  # noqa: E402
from jase.guard_v1_2_routing import check_contract as routing_check  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-file", required=True)
    ap.add_argument("--labels", help="Provisional operational review of these exact predictions")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = (ROOT / args.prediction_file).resolve()
    out = (ROOT / args.output).resolve()
    label_path = (ROOT / args.labels).resolve() if args.labels else None
    if ROOT not in source.parents or ROOT not in out.parents or (label_path and ROOT not in label_path.parents):
        ap.error("All files must be in the repository")
    rows = read_jsonl(source)
    labels = json.loads(label_path.read_text(encoding="utf-8")) if label_path else None
    if labels and set(labels["labels"]) != {row["id"] for row in rows}:
        raise RuntimeError("Operational review IDs do not match prediction IDs")
    if labels and labels["panel_sha256"] != sha256(ROOT / "benchmarks/v24/safety.jsonl"):
        raise RuntimeError("Panel changed after operational review")
    if labels and labels["prediction_sha256"] != sha256(source):
        raise RuntimeError("Predictions changed after operational review")

    report = {"prediction_file": str(source.relative_to(ROOT)), "cases": len(rows),
              "raw_sha256": sha256(source), "labels": str(label_path.relative_to(ROOT)) if label_path else None,
              "guards": {}, "paired_changes": []}
    results = {}
    for name, check in (("base_v1_2", base_check), ("routing_experimental", routing_check)):
        counts = Counter()
        statuses = {}
        for row in rows:
            result = check(row["raw"], row["text"])
            status = result.status
            statuses[row["id"]] = {"status": status, "codes": list(result.codes)}
            counts[status] += 1
            exact = bool(row["metric"]["complete_success"])
            counts["accepted_exact_wrong"] += status == ACCEPT and not exact
            counts["exact_correct_held"] += status != ACCEPT and exact
            if labels:
                label = labels["labels"][row["id"]]
                counts["accepted_material_or_routing_risk"] += status == ACCEPT and label == "material_or_routing_risk"
                counts["held_material_or_routing_risk"] += status != ACCEPT and label == "material_or_routing_risk"
                counts["accepted_uncertain"] += status == ACCEPT and label.startswith("uncertain_")
                counts["held_exact_correct"] += status != ACCEPT and label == "exact_correct"
        report["guards"][name] = {"accept": counts[ACCEPT], "review": counts["REVIEW"],
            "reject": counts["REJECT"], "coverage": counts[ACCEPT] / len(rows),
            "accepted_exact_wrong": counts["accepted_exact_wrong"],
            "exact_correct_held": counts["exact_correct_held"],
            **({key: counts[key] for key in ("accepted_material_or_routing_risk", "held_material_or_routing_risk",
                                         "accepted_uncertain", "held_exact_correct")} if labels else {})}
        results[name] = statuses
    for row in rows:
        id = row["id"]
        before, after = results["base_v1_2"][id], results["routing_experimental"][id]
        if before != after:
            report["paired_changes"].append({"id": id, "before": before, "after": after,
                "exact_correct": row["metric"]["complete_success"],
                **({"operational_label": labels["labels"][id]} if labels else {})})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["guards"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
