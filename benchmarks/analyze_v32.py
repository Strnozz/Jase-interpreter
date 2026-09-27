"""Paired V32 diagnostic; exact mismatch is a triage signal, not material risk."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from jase.semantic_guard_v1_3_routing_v4 import check_contract as guard_v4
from jase.semantic_guard_v1_3_routing_v5 import check_contract as guard_v5

ROOT = Path(__file__).resolve().parents[1]
PANEL = {j["id"]: j for j in map(json.loads,
         (ROOT / "benchmarks/v32/blind.jsonl").read_text(encoding="utf-8").splitlines())}
VERSIONS = ("v24", "v25", "v26")
CASES = {v: {j["id"]: j for j in map(json.loads,
         (ROOT / f"benchmarks/outputs/{v}-9b-v32-blind/cases.jsonl")
         .read_text(encoding="utf-8").splitlines())} for v in VERSIONS}


def main():
    for v in VERSIONS:
        rows = CASES[v]
        exact = {i for i, r in rows.items() if r["metric"].get("canonical_equal")}
        print(v, "exact", len(exact), "ids", ",".join(sorted(exact)))
        for name, guard in (("V4", guard_v4), ("V5", guard_v5)):
            statuses = Counter()
            accepted_wrong = []
            held_exact = []
            for i, row in rows.items():
                decision = guard(row["raw"], row["text"])
                statuses[decision.status] += 1
                if decision.status == "ACCEPT" and i not in exact:
                    accepted_wrong.append(i)
                if decision.status != "ACCEPT" and i in exact:
                    held_exact.append(i)
            print(" ", name, dict(statuses), "accepted_nonexact", accepted_wrong,
                  "held_exact", held_exact)
    for old in ("v24", "v25"):
        gains = [i for i in PANEL if CASES["v26"][i]["metric"].get("canonical_equal") and
                 not CASES[old][i]["metric"].get("canonical_equal")]
        losses = [i for i in PANEL if CASES[old][i]["metric"].get("canonical_equal") and
                  not CASES["v26"][i]["metric"].get("canonical_equal")]
        print("v26_vs_" + old, "gains", gains, "losses", losses)
    for i, row in CASES["v26"].items():
        d = guard_v5(row["raw"], row["text"])
        if d.status == "ACCEPT" and not row["metric"].get("canonical_equal"):
            print("REVIEW", i, PANEL[i]["family"], row["text"])
            print("  GOLD", json.dumps(PANEL[i]["contract"], ensure_ascii=False))
            print("  PRED", row["raw"].strip())


if __name__ == "__main__":
    main()
