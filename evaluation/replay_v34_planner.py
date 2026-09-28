"""Replay frozen V34 model outputs through an isolated Planner snapshot.

The two variants share predictions and gold. This script never invokes a provider.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.planner_v1_3 import plan_contract  # noqa: E402

PANEL = ROOT / "benchmarks/v34/blind.jsonl"
PREDICTIONS = ROOT / "benchmarks/outputs/qwen-v26-v34-blind/cases.jsonl"
SHA = "b4a86432b691eb64220b64a24be29d29ee8dbd9b7b59d2d0fd75e2e900ff7734"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("before", "after"))
    args = parser.parse_args()
    if sha(PANEL) != SHA:
        raise RuntimeError("Frozen V34 panel changed")
    gold = {row["id"]: row for row in rows(PANEL)}
    predicted = {row["id"]: row for row in rows(PREDICTIONS)}
    if set(gold) != set(predicted) or len(gold) != 40:
        raise RuntimeError("V34 replay requires all 40 model outputs")
    out_dir = ROOT / "evaluation/replays"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"v34_qwen_v26_planner_{args.variant}.jsonl"
    summary_file = out_dir / f"v34_qwen_v26_planner_{args.variant}_summary.json"
    if output.exists() or summary_file.exists():
        raise FileExistsError("Replay variant already frozen")

    outcomes = []
    statuses: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    for ident in sorted(gold):
        request = gold[ident]["text"]
        candidate = predicted[ident]
        plan = plan_contract(candidate["raw"], request)
        gold_plan = plan_contract(gold[ident]["contract"], request)
        if plan["execution_permitted"] or gold_plan["execution_permitted"]:
            raise RuntimeError("Execution permission escaped dry-run boundary")
        statuses[plan["planner_status"]] += 1
        reasons.update(plan["reason_codes"])
        outcomes.append({"id": ident, "family": gold[ident]["family"], "request": request,
                         "gold": gold[ident]["contract"],
                         "raw": candidate["raw"], "canonical_exact_raw": candidate["metric"]["canonical_equal"],
                         "guard_accept_raw": candidate["metric"]["guard_accept"],
                         "planner": plan, "gold_planner": gold_plan})
    summary = {"variant": args.variant, "cases": len(outcomes), "panel_sha256": sha(PANEL),
               "prediction_sha256": sha(PREDICTIONS),
               "planner_sha256": sha(ROOT / "jase/planner_v1_3.py"),
               "registry_sha256": sha(ROOT / "configs/capabilities/v1.json"),
               "guard_sha256": sha(ROOT / "jase/semantic_guard_v1_3_routing_v5.py"),
               "status_counts": dict(statuses), "reason_counts": dict(reasons),
               "canonical_exact_raw": sum(x["canonical_exact_raw"] for x in outcomes),
               "guard_accept_raw": sum(x["guard_accept_raw"] for x in outcomes),
               "exact_held": sum(x["canonical_exact_raw"] and x["planner"]["planner_status"] not in
                                 {"READY_FOR_DRY_RUN", "NO_ACTION"} for x in outcomes),
               "execution_permitted": False,
               "limits": ["Gold agent-authored; no independent human review",
                          "READY_FOR_DRY_RUN is not authorization to call a provider"]}
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in outcomes), encoding="utf-8")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
