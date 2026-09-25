#!/usr/bin/env python3
"""Verdetto di release, in un posto solo.

I gate non sono "le metriche sono salite". Sono una lista di condizioni, e
finché una sola non è verde la versione non è pronta. Questo script le
stampa tutte insieme, con il numero che manca, così il verdetto non dipende
da chi legge il report.

    python scripts/release_check.py reports/042-v12-iter4500.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: (nome, percorso nel report, confronto, soglia)
GATES = [
    ("validità JSON", "release_gates.json_validity", "eq", 1.0),
    ("casi mandatory corretti", "release_gates.mandatory_rate", "eq", 1.0),
    ("contratti sbagliati accettati", "release_gates.accepted_but_wrong", "eq", 0),
    ("fatti duplicati", "guarded.fidelity.duplicate_fact_rate", "eq", 0.0),
    ("valori non ancorati ACCETTATI", "_derived.ungrounded_accepted", "eq", 0),
    ("output degenerati", "guard.extraction_errors.degenerate", "absent", 0),
]


def dig(d, path):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("report", nargs="?", default=None)
    ap.add_argument("--production", action="store_true",
                    help="richiede esplicitamente un report marcato release_holdout")
    args = ap.parse_args()

    if args.report is None:
        best = json.loads((ROOT / "reports" / "BEST.json").read_text(encoding="utf-8"))
        path = ROOT / "reports" / best["report"]
    else:
        path = Path(args.report)
    rep = json.loads(path.read_text(encoding="utf-8"))

    # `guarded.fidelity.ungrounded_value_rate` conta anche i contratti che il
    # Guard ha FERMATO, e quelli al Planner non arrivano: come gate sarebbe
    # una misura sbagliata, che boccia una versione per un errore che il
    # sistema ha già intercettato. Qui si contano solo quelli accettati.
    rep.setdefault("_derived", {})["ungrounded_accepted"] = sum(
        1 for f in rep.get("failures") or []
        if f.get("guard") in ("ACCEPT", "REPAIRED")
        and "ungrounded_value" in (f.get("errors") or []))

    role = rep.get("benchmark_role", "dev_regression")
    print(f"{rep.get('label')}   ({path.name})")
    print(f"benchmark_role={role}\n")
    failed = 0

    if args.production and role != "release_holdout":
        failed += 1
        print("  NO   holdout indipendente                 "
              f"{role}   (atteso release_holdout)")

    for name, key, how, want in GATES:
        got = dig(rep, key)
        if how == "absent":
            ok = not got
            shown = got or 0
        elif got is None:
            ok, shown = None, "-"
        else:
            ok, shown = (got == want), got
        mark = "OK  " if ok else ("?   " if ok is None else "NO  ")
        if ok is False:
            failed += 1
        print(f"  {mark} {name:<34} {shown}   (atteso {want})")

    print()
    if failed:
        label = "PRODUZIONE NON PRONTA" if args.production else "NON PRONTA"
        print(f"{label}: {failed} gate non superati.")
        print("Un contratto fermato si recupera con una domanda; uno sbagliato\n"
              "che passa fa agire Jase nel mondo. Inoltre un benchmark DEV\n"
              "non può certificare la generalizzazione di produzione.")
    else:
        if args.production:
            print("PRONTA: tutti i gate superati su release holdout indipendente.")
        else:
            print("DEV GREEN: tutti i gate superati sul benchmark di sviluppo.\n"
                  "Per una release reale rieseguire su un holdout indipendente e usare --production.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
