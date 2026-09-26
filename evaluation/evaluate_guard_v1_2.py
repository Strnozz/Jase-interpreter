"""Offline Guard V1.2 coverage and wrong-accepted audit on saved generations."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.guard_v1_2 import ACCEPT, REJECT, REVIEW, check_contract  # noqa: E402
from jase.multimodel import read_jsonl  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prediction-dir", required=True)
    ap.add_argument("--extra-dir")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    source = (ROOT / args.prediction_dir).resolve()
    extra = (ROOT / args.extra_dir).resolve() if args.extra_dir else None
    out = (ROOT / args.output).resolve()
    if ROOT not in source.parents or ROOT not in out.parents or (extra and ROOT not in extra.parents):
        ap.error("All paths must be inside the repository")
    out.mkdir(parents=True, exist_ok=True)
    report = {"prediction_dir": str(source.relative_to(ROOT)), "panels": {},
              "meaning": "ACCEPT permits Planner handoff only; wrong_accepted is exact-gold disagreement, not proven unsafe execution"}
    for panel in ("hard_dev", "holdout", "v19_common", "transfer"):
        path = source / f"{panel}.jsonl"
        if not path.exists() and extra:
            path = extra / f"{panel}.jsonl"
        if not path.exists():
            continue
        records = []
        counts = Counter()
        codes = Counter()
        for row in read_jsonl(path):
            result = check_contract(row["raw"], row["text"])
            correct = bool(row["metric"]["complete_success"])
            counts[result.status] += 1
            counts["complete_gold"] += correct
            counts["wrong_accepted"] += result.status == ACCEPT and not correct
            counts["correct_accepted"] += result.status == ACCEPT and correct
            counts["correct_held"] += result.status != ACCEPT and correct
            for code in result.codes:
                codes[code] += 1
            records.append({"id": row["id"], "status": result.status,
                            "codes": list(result.codes), "complete_gold": correct})
        total = len(records)
        if not total:
            raise RuntimeError(f"Empty panel: {panel}")
        summary = {"cases": total, "accept": counts[ACCEPT], "review": counts[REVIEW],
                   "reject": counts[REJECT], "coverage": counts[ACCEPT] / total,
                   "complete_gold": counts["complete_gold"],
                   "correct_accepted": counts["correct_accepted"],
                   "wrong_accepted": counts["wrong_accepted"],
                   "wrong_accepted_rate_among_accepted": (counts["wrong_accepted"] / counts[ACCEPT]
                                                         if counts[ACCEPT] else None),
                   "correct_held": counts["correct_held"], "codes": dict(codes)}
        report["panels"][panel] = summary
        (out / f"{panel}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
