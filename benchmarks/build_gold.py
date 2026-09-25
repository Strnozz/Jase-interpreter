#!/usr/bin/env python3
"""Valida e serializza il benchmark GOLD.

Tre controlli, tutti bloccanti:
  1. ogni contratto atteso è valido contro lo schema;
  2. il Semantic Guard lo accetta SENZA riparazioni — se il Guard ripara il
     gold, o il gold viola la semantica o il Guard è troppo aggressivo: in
     entrambi i casi c'è un bug da discutere, non da ignorare;
  3. nessun testo duplicato, nessun id duplicato.

Emette benchmarks/gold_v1.jsonl (una riga per caso).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.guard import ACCEPT, check_contract  # noqa: E402
from jase.lexicon import SEMANTICS_VERSION  # noqa: E402

from gold_v1_source import CASES  # noqa: E402


def main() -> int:
    out_path = ROOT / "benchmarks" / "gold_v1.jsonl"
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

        # il grounding non si applica al gold: il gold È la verità, e alcune
        # astrazioni legittime (target 'gatto' da 'uno Sphynx') non sono
        # letteralmente nel testo. Si verificano struttura e coerenza.
        res = check_contract(c["contract"], c["text"], ground=False)
        if res.status != ACCEPT:
            problems.append(
                f"{cid}: guard={res.status} "
                f"issues={[i.code for i in res.issues if i.severity == 'error']} "
                f"repairs={res.repairs}")
        else:
            warns = [i.code for i in res.issues if i.severity == "warn"]
            if warns:
                print(f"  warn {cid}: {warns}")

    if problems:
        print("GOLD NON VALIDO:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
        return 1

    with out_path.open("w", encoding="utf-8") as f:
        for c in CASES:
            row = dict(c)
            row["semantics_version"] = SEMANTICS_VERSION
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    mandatory = sum(1 for c in CASES if c["mandatory"])
    tags: dict[str, int] = {}
    for c in CASES:
        for t in c["tags"]:
            tags[t] = tags.get(t, 0) + 1
    print(f"OK {len(CASES)} casi gold → {out_path.relative_to(ROOT)}")
    print(f"   di cui mandatory (release gate): {mandatory}")
    print("   tag:", ", ".join(f"{k}={v}" for k, v in sorted(tags.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
