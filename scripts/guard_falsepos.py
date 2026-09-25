#!/usr/bin/env python3
"""Falsi positivi del Guard, misurati sul benchmark gold.

Ogni contratto gold è corretto per costruzione. Se il Guard lo ferma o lo
ripara, il difetto è nel Guard (o nel gold, che è la stessa urgenza). È il
test che ogni nuovo controllo deve superare PRIMA di essere promosso a
`error`: senza, un controllo che "trova" errori sta solo bloccando richieste
legittime.

    python scripts/guard_falsepos.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.guard import ACCEPT, check_contract  # noqa: E402

GOLD = ROOT / "benchmarks" / "gold_v1.jsonl"


def main() -> int:
    bad = 0
    cases = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    for case in cases:
        res = check_contract(json.loads(json.dumps(case["contract"])),
                             case["text"], ground=True)
        errs = [i for i in res.issues if i.severity == "error"]
        if res.status != ACCEPT or res.repairs or errs:
            bad += 1
            print(f"--- {case['id']}  [{res.status}]")
            print(f"    T: {case['text']}")
            for r in res.repairs:
                print(f"    RIPARAZIONE: {r}")
            for i in errs:
                print(f"    ERRORE {i.code}: {i.message}")
    print(f"\n{len(cases)} casi gold, {bad} con falsi positivi del Guard")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
