"""Paired dry-run Planner replay using the frozen V36 model outputs."""
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

PANEL = ROOT / "benchmarks/v36/blind.jsonl"
PREDICTIONS = ROOT / "benchmarks/outputs/qwen-v26-v36-blind/cases.jsonl"
PANEL_SHA = "af40145327eb755a2496344df858f5ec26264a881a72b099ab73ee7640f5078d"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=("before", "after"))
    args = parser.parse_args()
    if sha(PANEL) != PANEL_SHA:
        raise RuntimeError("Frozen V36 panel changed")
    gold = {row["id"]: row for row in load(PANEL)}
    candidate = {row["id"]: row for row in load(PREDICTIONS)}
    if set(gold) != set(candidate) or len(gold) != 42:
        raise RuntimeError("V36 replay requires all 42 model outputs")
    dest = ROOT / "evaluation/replays"
    dest.mkdir(parents=True, exist_ok=True)
    cases = dest / f"v36_qwen_v26_planner_{args.variant}.jsonl"
    summary_file = dest / f"v36_qwen_v26_planner_{args.variant}_summary.json"
    if cases.exists() or summary_file.exists():
        raise FileExistsError("Replay variant is already recorded")
    output, statuses, reasons = [], Counter(), Counter()
    for ident in sorted(gold):
        text = gold[ident]["text"]
        predicted = plan_contract(candidate[ident]["raw"], text)
        expected = plan_contract(gold[ident]["contract"], text)
        if predicted["execution_permitted"] or expected["execution_permitted"]:
            raise RuntimeError("Execution permission escaped dry-run boundary")
        statuses[predicted["planner_status"]] += 1
        reasons.update(predicted["reason_codes"])
        output.append({"id": ident, "family": gold[ident]["family"], "request": text,
                       "review_flags": gold[ident]["review_flags"], "gold": gold[ident]["contract"],
                       "raw": candidate[ident]["raw"],
                       "canonical_exact_raw": candidate[ident]["metric"]["canonical_equal"],
                       "guard_accept_raw": candidate[ident]["metric"]["guard_accept"],
                       "planner": predicted, "gold_planner": expected})
    summary = {"variant": args.variant, "cases": len(output), "panel_sha256": sha(PANEL),
               "prediction_sha256": sha(PREDICTIONS), "planner_sha256": sha(ROOT / "jase/planner_v1_3.py"),
               "registry_sha256": sha(REGISTRY_V2_PATH),
               "guard_sha256": sha(ROOT / "jase/semantic_guard_v1_3_routing_v5.py"),
               "status_counts": dict(statuses), "reason_counts": dict(reasons),
               "canonical_exact_raw": sum(row["canonical_exact_raw"] for row in output),
               "guard_accept_raw": sum(row["guard_accept_raw"] for row in output),
               "exact_held": sum(row["canonical_exact_raw"] and row["planner"]["planner_status"] not in
                                 {"READY_FOR_DRY_RUN", "NO_ACTION"} for row in output),
               "execution_permitted": False,
               "limits": ["Agent-authored gold, no independent human review",
                          "Provider execution is disabled"]}
    cases.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output), encoding="utf-8")
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
