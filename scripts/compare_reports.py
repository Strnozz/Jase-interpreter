#!/usr/bin/env python3
"""Confronto fra report di eval e scelta del BEST checkpoint.

Un checkpoint nuovo non è automaticamente migliore. Qui si confronta sulle
metriche che contano davvero e si evidenziano le REGRESSIONI: un tag che
peggiora è più informativo di una media che sale.

    python scripts/compare_reports.py reports/*.json
    python scripts/compare_reports.py --set-best reports/002-v2-iter1500.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: ordine di priorità per dichiarare un checkpoint migliore.
#
# ATTENZIONE: `accepted_but_wrong` NON può stare in cima. Un modello che non
# produce nulla di valido ha accepted_but_wrong=0 perché il Guard blocca
# tutto, e vincerebbe il confronto pur essendo inutile. Il conteggio grezzo
# va normalizzato sul numero di contratti effettivamente accettati
# (`accepted_precision`, calcolato in `row()`): fra i contratti che
# RAGGIUNGONO il Planner, quanti sono giusti.
KEYS = [
    ("release_gates", "mandatory_rate", "max"),
    ("guarded", "contract.semantic_exact_rate", "max"),
    ("_derived", "accepted_precision", "max"),
    ("guarded", "roles.bucket_accuracy", "max"),
    ("guarded", "roles.operator_accuracy", "max"),
    ("guarded", "facts.recall", "max"),
    ("guarded", "facts.precision", "max"),
    ("guarded", "fidelity.hallucinated_fact_rate", "min"),
    ("guarded", "fidelity.missing_fact_rate", "min"),
    ("release_gates", "json_validity", "max"),
]


def dig(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def accepted_precision(report: dict) -> float:
    """Fra i contratti che superano il Guard, quale frazione è corretta.

    È la metrica onesta dell'invariante fail-closed: il conteggio grezzo di
    `accepted_but_wrong` premia chi non produce niente."""
    g = report.get("guard", {}).get("status", {}) or {}
    accepted = g.get("ACCEPT", 0) + g.get("REPAIRED", 0)
    if accepted == 0:
        return 0.0
    wrong = report.get("release_gates", {}).get("accepted_but_wrong", 0) or 0
    return round(1 - wrong / accepted, 4)


def score(report: dict) -> tuple:
    """Chiave di ordinamento lessicografica secondo KEYS."""
    report = dict(report)
    report["_derived"] = {"accepted_precision": accepted_precision(report)}
    out = []
    for section, path, direction in KEYS:
        v = dig(report.get(section, {}), path)
        v = 0 if v is None else (1 if v is True else (0 if v is False else v))
        out.append(-v if direction == "min" else v)
    return tuple(out)


def row(report: dict) -> dict:
    return {
        "label": report.get("label"),
        "n": report.get("examples"),
        "mandatory": dig(report, "release_gates.mandatory_rate"),
        "acc_wrong": dig(report, "release_gates.accepted_but_wrong"),
        "acc_prec": accepted_precision(report),
        "json_ok": dig(report, "release_gates.json_validity"),
        "exact": dig(report, "guarded.contract.semantic_exact_rate"),
        "bucket": dig(report, "guarded.roles.bucket_accuracy"),
        "operator": dig(report, "guarded.roles.operator_accuracy"),
        "recall": dig(report, "guarded.facts.recall"),
        "precision": dig(report, "guarded.facts.precision"),
        "halluc": dig(report, "guarded.fidelity.hallucinated_fact_rate"),
        "blocked": dig(report, "guard.blocked_rate"),
    }


def fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reports", nargs="*", default=[])
    ap.add_argument("--set-best", default=None,
                    help="scrive reports/BEST.json puntando a questo report")
    args = ap.parse_args()

    paths = [Path(p) for p in args.reports] or sorted((ROOT / "reports").glob("*.json"))
    paths = [p for p in paths if p.name != "BEST.json"]
    reports = []
    for p in paths:
        try:
            reports.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except Exception as exc:  # noqa: BLE001
            print(f"  salto {p.name}: {exc}")
    if not reports:
        print("nessun report trovato in reports/")
        return 1

    cols = ["label", "n", "mandatory", "acc_wrong", "acc_prec", "json_ok", "exact",
            "bucket", "operator", "recall", "precision", "halluc", "blocked"]
    rows = [row(r) for _p, r in reports]
    widths = {c: max(len(c), *(len(fmt(x[c])) for x in rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    print("  ".join("-" * widths[c] for c in cols))
    for x in rows:
        print("  ".join(fmt(x[c]).ljust(widths[c]) for c in cols))

    best_path, best_report = max(reports, key=lambda pr: score(pr[1]))
    print(f"\nMIGLIORE per criterio semantico: {best_report.get('label')} "
          f"({best_path.name})")

    # regressioni per tag rispetto al report immediatamente precedente
    if len(reports) >= 2:
        (_pa, prev), (_pb, last) = reports[-2], reports[-1]
        regressions = []
        for tag, cur in (last.get("by_tag") or {}).items():
            old = (prev.get("by_tag") or {}).get(tag)
            if not old:
                continue
            delta = cur["semantic_exact_rate"] - old["semantic_exact_rate"]
            if delta < -0.01:
                regressions.append((tag, old["semantic_exact_rate"],
                                    cur["semantic_exact_rate"], cur["n"]))
        if regressions:
            print(f"\nREGRESSIONI {prev.get('label')} → {last.get('label')}:")
            for tag, o, c, n in sorted(regressions, key=lambda r: r[1] - r[2],
                                       reverse=True):
                print(f"  {tag:<28} {o:.3f} → {c:.3f}  (n={n})")
        else:
            print(f"\nNessuna regressione per tag "
                  f"{prev.get('label')} → {last.get('label')}")

    if args.set_best:
        src = Path(args.set_best)
        payload = json.loads(src.read_text(encoding="utf-8"))
        (ROOT / "reports" / "BEST.json").write_text(json.dumps({
            "report": src.name,
            "label": payload.get("label"),
            "adapter": payload.get("adapter"),
            "release_gates": payload.get("release_gates"),
            "guarded": payload.get("guarded"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nBEST aggiornato → {payload.get('label')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
