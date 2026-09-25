#!/usr/bin/env python3
"""Copertura del benchmark da parte del dataset di training.

PERCHÉ ESISTE
    Per cinque iterazioni di fila un fallimento che sembrava un limite del
    modello si è rivelato un buco del dataset: nessuna richiesta senza verbo,
    nessun operatore di lista, nessun inglese, nessuna condizione che non
    parlasse di prezzo. Ogni volta l'ho scoperto DOPO un'ora di training,
    leggendo i fallimenti uno per uno.

    Questo script fa la stessa scoperta in due secondi e prima di allenare:
    estrae le caratteristiche STRUTTURALI di ogni esempio — non il dominio,
    non il lessico, ma la forma semantica — e confronta le due distribuzioni.
    Una caratteristica che il benchmark chiede e il dataset non contiene è un
    buco garantito, non un rischio.

    python scripts/coverage_check.py data/v9/train.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.canon import norm_text  # noqa: E402
from jase.lexicon import (  # noqa: E402
    LIST_OPERATORS, MONETARY_PROPERTIES, NEGATIVE_OPERATORS,
    TEMPORAL_PROPERTIES,
)

_EN = re.compile(r"\b(find|book|buy|need|looking|want|under|least|with|the|"
                 r"please|near|order|rent|compare)\b", re.I)
_NUM_CLASSES = {
    **{p: "time" for p in TEMPORAL_PROPERTIES},
    **{p: "money" for p in MONETARY_PROPERTIES},
    "distance": "distance", "guests": "quantity", "quantity": "quantity",
    "stars": "rating", "age": "age", "size": "size",
}


def features(text: str, contract: dict) -> set[str]:
    """Le caratteristiche strutturali di un esempio.

    Deliberatamente NON guardano il dominio: un panino e un trapano sono la
    stessa cosa, qui. Guardano la forma della richiesta, che è ciò che il
    modello deve generalizzare."""
    f: set[str] = set()
    goals = contract.get("goals") or []
    if len(goals) > 1:
        f.add("multi-goal")
    if (contract.get("clarification") or {}).get("required"):
        f.add("clarification")
    if _EN.search(text) and not re.search(r"\b(un|una|il|la|di|che|per)\b", text):
        f.add("english")
    if len(text) > 80:
        f.add("long-text")
    if len(text.split()) <= 6:
        f.add("telegraphic")
    if "?" in text:
        f.add("question")

    classes_seen: set[str] = set()
    for g in goals:
        if g.get("gate"):
            f.add("gate")
            if (g["gate"] or {}).get("otherwise"):
                f.add("gate-two-branch")
            wprop = norm_text(((g.get("gate") or {}).get("when") or {})
                              .get("property"))
            if wprop and wprop not in MONETARY_PROPERTIES:
                f.add("gate-non-price")
        if norm_text(g.get("raw_action")) == norm_text("(implicito)"):
            f.add("implicit-action")
        if norm_text((g.get("target") or {}).get("name")) == "unknown":
            f.add("unknown-target")
        if g.get("preferences"):
            f.add("preference")
        for bucket in ("attributes", "constraints", "preferences"):
            for p in g.get(bucket) or []:
                op = norm_text(p.get("operator"))
                prop = norm_text(p.get("property"))
                if op in LIST_OPERATORS:
                    f.add("list-operator")
                if op in NEGATIVE_OPERATORS:
                    f.add("negation")
                if op == "between":
                    f.add("range")
                if op in ("lt", "gt"):
                    f.add("strict-threshold")
                if op in ("lte", "gte"):
                    f.add("inclusive-threshold")
                if p.get("unit"):
                    f.add("unit")
                if p.get("currency"):
                    f.add("currency")
                if p.get("reference"):
                    f.add("spatial-reference")
                if prop in TEMPORAL_PROPERTIES:
                    f.add("temporal")
                if isinstance(p.get("value"), (int, float)) and \
                        not isinstance(p.get("value"), bool):
                    cls = _NUM_CLASSES.get(prop)
                    if cls:
                        classes_seen.add(cls)
        if not (g.get("attributes") or g.get("constraints")
                or g.get("preferences") or g.get("gate")):
            f.add("no-facts")
    if len(classes_seen) >= 2:
        f.add("two-number-roles")
    return f


def load(path: Path) -> list[tuple[str, dict]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            user = next(m["content"] for m in row["messages"]
                        if m["role"] == "user")
            out.append((user, json.loads(row["messages"][-1]["content"])))
        else:
            out.append((row["text"], row["contract"]))
    return out


def main() -> int:
    train_path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/v9/train.jsonl")
    gold_path = ROOT / "benchmarks" / "gold_v1.jsonl"
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    train = load(train_path)
    gold = load(gold_path)

    def tally(rows):
        counts: dict[str, int] = {}
        for text, contract in rows:
            for feat in features(text, contract):
                counts[feat] = counts.get(feat, 0) + 1
        return counts

    def tally_props(rows):
        counts: dict[str, int] = {}
        for _text, contract in rows:
            for g in contract.get("goals") or []:
                preds = [p for b in ("attributes", "constraints", "preferences")
                         for p in (g.get(b) or [])]
                gate = g.get("gate")
                if isinstance(gate, dict) and isinstance(gate.get("when"), dict):
                    preds.append(gate["when"])
                for p in preds:
                    name = norm_text(p.get("property"))
                    if name:
                        counts[name] = counts.get(name, 0) + 1
                atype = norm_text((g.get("target") or {}).get("type"))
                if atype:
                    counts["type:" + atype] = counts.get("type:" + atype, 0) + 1
                act = norm_text(g.get("action"))
                if act:
                    counts["action:" + act] = counts.get("action:" + act, 0) + 1
        return counts

    tc, gc = tally(train), tally(gold)
    nt, ng = len(train), len(gold)
    feats = sorted(set(tc) | set(gc))

    print(f"train = {train_path.name} ({nt} esempi)   gold = {ng} casi\n")
    print(f"{'caratteristica':<22} {'gold':>10} {'train':>12}   giudizio")
    print("-" * 68)
    holes = []
    for f in feats:
        g_n, t_n = gc.get(f, 0), tc.get(f, 0)
        g_share, t_share = g_n / ng, t_n / nt
        if g_n == 0:
            verdict = "non richiesta dal benchmark"
        elif t_n == 0:
            verdict = "*** BUCO: il gold la chiede, il dataset non ce l'ha"
            holes.append(f)
        elif t_share < g_share / 12:
            verdict = f"!!! sotto-rappresentata ({g_share / t_share:.0f}× meno)"
            holes.append(f)
        else:
            verdict = "ok"
        print(f"{f:<22} {g_n:>4} ({g_share:5.1%}) {t_n:>6} ({t_share:5.1%})   {verdict}")

    # --- copertura del LESSICO: proprietà, tipi di target, azioni.
    # `coverage_check` guardava solo la forma della richiesta e non avrebbe
    # mai visto che il gold chiede `arrival_time` e il dataset non lo genera.
    tp, gp = tally_props(train), tally_props(gold)
    lex_holes = [k for k in sorted(gp) if tp.get(k, 0) == 0]
    lex_thin = [k for k in sorted(gp)
                if 0 < tp.get(k, 0) and tp[k] / nt < (gp[k] / ng) / 20]
    print("\n--- lessico richiesto dal benchmark ---")
    if lex_holes:
        print("MAI GENERATE:  " + ", ".join(lex_holes))
    if lex_thin:
        print("rare:          " + ", ".join(lex_thin))
    if not lex_holes and not lex_thin:
        print("ogni proprietà, tipo e azione del gold compare nel dataset")
    holes += lex_holes

    print()
    if holes:
        print("DA COLMARE PRIMA DI ALLENARE:", ", ".join(holes))
    else:
        print("Nessun buco strutturale: ogni forma richiesta dal benchmark "
              "ha una controparte nel dataset.")
    return 1 if holes else 0


if __name__ == "__main__":
    raise SystemExit(main())
