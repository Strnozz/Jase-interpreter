"""Paired Planner dry-run replay on the frozen V35 Qwen outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.capability_registry_v1 import REGISTRY_V2_PATH  # noqa: E402
from jase.planner_v1_3 import plan_contract  # noqa: E402

PANEL = ROOT / "benchmarks/v35/blind.jsonl"
PREDICTIONS = ROOT / "benchmarks/outputs/qwen-v26-v35-blind/cases.jsonl"
EXPECTED_SHA = "c0e4d08208406cc4d1198ee84b432d6ee8fc5eaf678ad641dbcbf3074fb41ac7"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("before", "after", "after_v2"))
    args = parser.parse_args()
    if sha(PANEL) != EXPECTED_SHA:
        raise RuntimeError("Frozen V35 panel changed")
    gold = {row["id"]: row for row in read(PANEL)}
    candidates = {row["id"]: row for row in read(PREDICTIONS)}
    if set(gold) != set(candidates) or len(gold) != 60:
        raise RuntimeError("V35 requires all 60 model outputs")
    directory = ROOT / "evaluation/replays"
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / f"v35_qwen_v26_planner_{args.variant}.jsonl"
    summary_file = directory / f"v35_qwen_v26_planner_{args.variant}_summary.json"
    if output.exists() or summary_file.exists():
        raise FileExistsError("Replay variant already exists")
    outcomes, statuses, reasons = [], Counter(), Counter()
    for ident in sorted(gold):
        request = gold[ident]["text"]
        candidate = candidates[ident]
        plan = plan_contract(candidate["raw"], request)
        gold_plan = plan_contract(gold[ident]["contract"], request)
        if plan["execution_permitted"] or gold_plan["execution_permitted"]:
            raise RuntimeError("Dry-run permission boundary failed")
        statuses[plan["planner_status"]] += 1
        reasons.update(plan["reason_codes"])
        outcomes.append({"id": ident, "family": gold[ident]["family"], "request": request,
                         "review_flags": gold[ident]["review_flags"], "gold": gold[ident]["contract"],
                         "raw": candidate["raw"], "canonical_exact_raw": candidate["metric"]["canonical_equal"],
                         "guard_accept_raw": candidate["metric"]["guard_accept"],
                         "planner": plan, "gold_planner": gold_plan})
    summary = {"variant": args.variant, "cases": len(outcomes), "panel_sha256": sha(PANEL),
               "prediction_sha256": sha(PREDICTIONS),
               "planner_sha256": sha(ROOT / "jase/planner_v1_3.py"),
               "registry_sha256": sha(REGISTRY_V2_PATH),
               "guard_sha256": sha(ROOT / "jase/semantic_guard_v1_3_routing_v5.py"),
               "status_counts": dict(statuses), "reason_counts": dict(reasons),
               "canonical_exact_raw": sum(x["canonical_exact_raw"] for x in outcomes),
               "guard_accept_raw": sum(x["guard_accept_raw"] for x in outcomes),
               "exact_held": sum(x["canonical_exact_raw"] and x["planner"]["planner_status"] not in
                                 {"READY_FOR_DRY_RUN", "NO_ACTION"} for x in outcomes),
               "execution_permitted": False,
               "limits": ["Gold agent-authored, not independently human-reviewed",
                          "Provider execution is disabled"]}
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outcomes), encoding="utf-8")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
