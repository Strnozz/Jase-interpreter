#!/usr/bin/env python3
"""Valida e sigilla il release holdout.

Stessi controlli del gold — schema, Guard senza riparazioni, niente
duplicati — più due cose che il gold non ha:

  • un SIGILLO SHA-256 del file serializzato, perché un holdout che cambia
    senza spiegazione rende i numeri prima e dopo non confrontabili;
  • il divieto di leggere i casi falliti: il report si genera con
    `--no-failures`, e questo script ricorda il protocollo a chi lo lancia.

Leggere benchmarks/HOLDOUT_PROTOCOL.md.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "benchmarks"))

from jase.guard import ACCEPT, check_contract  # noqa: E402
from jase.lexicon import SEMANTICS_VERSION  # noqa: E402

from release_holdout_v1_source import CASES  # noqa: E402


def main() -> int:
    out = ROOT / "benchmarks" / "release_holdout_v1.jsonl"
    seal = ROOT / "benchmarks" / "release_holdout_v1.sha256"
    problems: list[str] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()

    for c in CASES:
        cid = c["id"]
        if cid in seen_ids:
            problems.append(f"{cid}: id duplicato")
        seen_ids.add(cid)
        key = c["text"].strip().lower()
        if key in seen_texts:
            problems.append(f"{cid}: testo duplicato")
        seen_texts.add(key)

        # Il Guard deve accettare il contratto atteso SENZA riparazioni.
        # L'ancoraggio resta spento come nel gold: qui si valida la coerenza
        # del contratto con le regole, non con il testo.
        res = check_contract(json.loads(json.dumps(c["contract"])),
                             c["text"], ground=False)
        if res.status != ACCEPT:
            problems.append(f"{cid}: Guard {res.status} "
                            f"{[i.code for i in res.issues if i.severity == 'error']}")
        if res.repairs:
            problems.append(f"{cid}: il Guard lo RIPARA → {res.repairs}")

    if problems:
        print(f"{len(problems)} problemi:")
        for p in problems:
            print("  " + p)
        return 1

    lines = []
    for c in CASES:
        row = dict(c)
        row["semantics_version"] = SEMANTICS_VERSION
        lines.append(json.dumps(row, ensure_ascii=False, sort_keys=True))
    payload = "\n".join(lines) + "\n"
    out.write_text(payload, encoding="utf-8")

    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    previous = seal.read_text(encoding="utf-8").strip() if seal.exists() else ""
    seal.write_text(digest + "\n", encoding="utf-8")

    mand = sum(1 for c in CASES if c.get("mandatory"))
    tags: dict[str, int] = {}
    for c in CASES:
        for t in c["tags"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"OK {len(CASES)} casi holdout → {out.relative_to(ROOT)}")
    print(f"   mandatory: {mand}")
    print(f"   sigillo:   {digest[:16]}…")
    if previous and previous != digest:
        print("   ATTENZIONE: il sigillo è CAMBIATO rispetto a prima.")
        print("   Se non hai aggiunto una riga in HOLDOUT_RUNS.md che lo")
        print("   spieghi, i numeri prima e dopo non sono confrontabili.")
    print("   tag: " + ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    print("\n   Ricorda: non si allena su questi casi, non si corregge il")
    print("   sistema guardandoli, si esegue di rado. HOLDOUT_PROTOCOL.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
