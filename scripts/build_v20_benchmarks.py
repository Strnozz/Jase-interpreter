"""Author and seal independent V1.2 static benchmark panels. Never train on these."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.leakage import normalize_text, token_jaccard  # noqa: E402

OUT = ROOT / "benchmarks" / "v20"


def f(field, op, value, strength="hard", **kwargs):
    return {"field": field, "op": op, "value": value, "strength": strength, **kwargs}


def g(id, action, name, type=None, facts=(), **kwargs):
    target = {"name": name}
    if type:
        target["type"] = type
    return {"id": id, "action": action, "target": target, "facts": list(facts), **kwargs}


def c(*goals, kind="task", context=()):
    out = {"schema_version": "1.2", "kind": kind, "goals": list(goals)}
    if context:
        out["context"] = list(context)
    return out


def case(id, text, contract, *tags):
    return {"id": id, "text": text, "contract": contract, "tags": sorted(set(tags))}


HARD = [
    case("d01", "Cerca una trattoria siciliana a Genova, massimo 30 euro, meglio con tavoli fuori; non prenotare.",
         c(g("g1", "find", "trattoria", "place", [f("cuisine", "eq", "siciliana"),
           f("location", "near", "Genova"), f("price", "lte", 30, currency="EUR"),
           f("seating", "eq", "tavoli fuori", "soft")], policy={"forbid": ["book"]})),
         "hard_soft", "negation", "policy"),
    case("d02", "Cerca un treno per Venezia domani e un albergo vicino alla stazione di arrivo.",
         c(g("g1", "find", "treno", "transport", [f("destination", "eq", "Venezia"),
           f("date", "eq", "domani")]),
           g("g2", "find", "albergo", "accommodation", [
             {"field": "location", "op": "near", "value_ref": {"goal": "g1", "field": "arrival_station"},
              "strength": "hard"}], depends_on=["g1"])),
         "multi_goal", "dependency", "reference"),
    case("d03", "Se trovi un volo per Madrid sotto i 250 euro, cerca anche una stanza.",
         c(g("g1", "find", "volo", "transport", [f("destination", "eq", "Madrid"),
           f("price", "lt", 250, currency="EUR")]),
           g("g2", "find", "stanza", "accommodation", [], depends_on=["g1"],
             condition={"goal": "g1", "test": {"field": "result_count", "op": "gt", "value": 0}})),
         "multi_goal", "conditional", "dependency"),
    case("d04", "Invia il messaggio a Giulia soltanto dopo che te lo confermo.",
         c(g("g1", "contact", "Giulia", "person", [],
             missing=[{"field": "message", "blocks": "action", "reason": "unspecified"}],
             policy={"confirm_before": ["contact"]})),
         "confirmation", "missing", "side_effect"),
    case("d05", "Mostrami quattro pizzerie e prenota la terza domani.",
         c(g("g1", "find", "pizzeria", "place", [f("quantity", "eq", 4)]),
           g("g2", "book", "pizzeria", "place", [
             {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "result", "ordinal": 3},
              "strength": "hard"}, f("date", "eq", "domani")], depends_on=["g1"])),
         "multi_goal", "ordinal", "side_effect"),
    case("d06", "Cercami ramen; se non ne trovi, va bene anche un ristorante coreano.",
         c(g("g1", "find", "ramen", "food", []),
           g("g2", "find", "ristorante", "place", [f("cuisine", "eq", "coreana")],
             depends_on=["g1"], condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})),
         "fallback", "conditional"),
    case("d07", "Cerco un hotel entro 90 euro, possibilmente con piscina.",
         c(g("g1", "find", "hotel", "accommodation", [
           f("price", "lte", 90, currency="EUR"), f("amenity", "eq", "piscina", "soft")])),
         "hard_soft", "numeric"),
    case("d08", "Ho già il biglietto; trovami un hotel vicino al teatro.",
         c(g("g1", "find", "hotel", "accommodation", [f("location", "near", "teatro")]),
           context=[{"field": "ticket_owned", "op": "eq", "value": True}]),
         "context", "location"),
    case("d09", "Prenotami un tavolo.",
         c(g("g1", "book", "tavolo", "place", [], missing=[
           {"field": "date", "blocks": "action", "reason": "unspecified"},
           {"field": "time", "blocks": "action", "reason": "unspecified"},
           {"field": "guests", "blocks": "action", "reason": "unspecified"}])),
         "missing", "progressive_execution"),
    case("d10", "Voglio arrivare a Torino prima delle 13 domani, spendendo il meno possibile.",
         c(g("g1", "find", "trasporto", "transport", [
           f("destination", "eq", "Torino"), f("arrival_time", "before", "13:00"),
           f("date", "eq", "domani")], ranking={"field": "price", "direction": "asc"})),
         "temporal", "ranking"),
    case("d11", "La frase 'prenota un volo' è solo un esempio di comando, non una richiesta.",
         c(kind="non_actionable"), "negative", "quoted_action"),
    case("d12", "In teoria, quali dati servono per comprare un biglietto aereo?",
         c(kind="non_actionable"), "negative", "theoretical"),
    case("d13", "Trova una farmacia aperta oggi, ma non comprare nulla.",
         c(g("g1", "find", "farmacia", "place", [f("availability", "contains", "oggi")],
             policy={"forbid": ["buy"]})), "negation", "policy"),
    case("d14", "Cerca un fotografo con almeno cinque anni di esperienza sotto i 400 euro.",
         c(g("g1", "find", "fotografo", "professional_service", [
           f("experience", "gte", 5, unit="years"), f("price", "lt", 400, currency="EUR")])),
         "numeric", "units"),
    case("d15", "Ricordamelo domani.",
         c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "domani")],
             missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])),
         "anaphora", "missing"),
    case("d16", "Siamo in quattro e non abbiamo la macchina. Cerca un ristorante in centro.",
         c(g("g1", "find", "ristorante", "place", [f("location", "near", "centro")]),
           context=[{"field": "party_size", "op": "eq", "value": 4},
                    {"field": "has_car", "op": "eq", "value": False}]),
         "context", "numbers"),
]

HOLDOUT = [
    case("s01", "Mi trovi un bed and breakfast a Parma entro 75 euro? Se c'è il parcheggio, meglio.",
         c(g("g1", "find", "bed and breakfast", "accommodation", [
           f("location", "near", "Parma"), f("price", "lte", 75, currency="EUR"),
           f("amenity", "eq", "parcheggio", "soft")])), "hard_soft"),
    case("s02", "Fammi vedere tre elettricisti disponibili mercoledì, poi contatta il primo solo se confermo.",
         c(g("g1", "find", "elettricista", "professional_service", [
           f("quantity", "eq", 3), f("availability", "contains", "mercoledì")]),
           g("g2", "contact", "elettricista", "professional_service", [
             {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "result", "ordinal": 1},
              "strength": "hard"}], depends_on=["g1"], policy={"confirm_before": ["contact"]})),
         "multi_goal", "confirmation", "reference"),
    case("s03", "Non devi acquistare niente: sto soltanto citando la frase 'compra il biglietto'.",
         c(kind="non_actionable"), "negative", "quoted_action"),
    case("s04", "Prenotami un volo per Oslo.",
         c(g("g1", "book", "volo", "transport", [f("destination", "eq", "Oslo")],
             missing=[{"field": "date", "blocks": "action", "reason": "unspecified"},
                      {"field": "origin", "blocks": "action", "reason": "unspecified"}])),
         "missing", "side_effect"),
    case("s05", "Voglio sushi; se non ne trovi, proponimi qualcosa di thailandese.",
         c(g("g1", "find", "sushi", "food", []),
           g("g2", "find", "ristorante", "place", [f("cuisine", "eq", "thailandese")],
             depends_on=["g1"], condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})),
         "fallback", "conditional"),
    case("s06", "Cerca una casa a Bologna con due camere sotto 900 euro; meglio vicina al centro.",
         c(g("g1", "find", "casa", "real_estate", [f("location", "near", "Bologna"),
           f("rooms", "eq", 2), f("price", "lt", 900, currency="EUR"),
           f("distance_to_center", "near", "centro", "soft")])),
         "hard_soft", "numbers"),
    case("s07", "Trova un museo aperto sabato, senza comprare biglietti.",
         c(g("g1", "find", "museo", "place", [f("availability", "contains", "sabato")],
             policy={"forbid": ["buy"]})), "negation", "policy"),
    case("s08", "Cerco un tablet con almeno 128 GB di memoria, a meno di 250 euro.",
         c(g("g1", "find", "tablet", "product", [f("storage", "gte", 128, unit="GB"),
           f("price", "lt", 250, currency="EUR")])), "numbers", "units"),
    case("s09", "Compra il secondo dei libri che mi hai mostrato prima.",
         c(g("g1", "buy", "libro", "product", [], missing=[
           {"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])),
         "anaphora", "missing"),
    case("s10", "Spiegami come funzionano in generale le prenotazioni alberghiere.",
         c(kind="non_actionable"), "negative", "theoretical"),
    case("s11", "Trova un autobus per Siena prima delle 18 e ordina le opzioni per costo crescente.",
         c(g("g1", "find", "autobus", "transport", [f("destination", "eq", "Siena"),
           f("arrival_time", "before", "18:00")], ranking={"field": "price", "direction": "asc"})),
         "temporal", "ranking"),
    case("s12", "Ho già un alloggio. Cercami un taxi per l'aeroporto domani mattina.",
         c(g("g1", "find", "taxi", "transport", [f("destination", "eq", "aeroporto"),
           f("date", "eq", "domani"), f("time_of_day", "eq", "mattina")]),
           context=[{"field": "lodging_owned", "op": "eq", "value": True}]),
         "context", "temporal"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    panels = {"hard_dev": HARD, "holdout": HOLDOUT}
    known = set()
    for source in (ROOT / "data/v20/train.jsonl", ROOT / "data/v20/valid.jsonl",
                   ROOT / "benchmarks/release_holdout_v1.jsonl",
                   ROOT / "data/v20_quality/dev.jsonl"):
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                text = row.get("text") or next(m["content"] for m in row["messages"] if m["role"] == "user")
                known.add(normalize_text(text))
    panel_texts = []
    for panel, rows in panels.items():
        for row in rows:
            errors = validate_contract(row["contract"])
            if errors:
                raise ValueError(f"{panel}/{row['id']}: {[x.as_dict() for x in errors]}")
            key = normalize_text(row["text"])
            if key in known or key in panel_texts:
                raise ValueError(f"normalized prompt overlap: {row['id']}")
            panel_texts.append(key)
        with (OUT / f"{panel}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"schema_version": "1.2", "panels": {},
                "schema_sha256": hashlib.sha256((ROOT / "schema/goal_contract_v1_2.schema.json").read_bytes()).hexdigest(),
                "builder_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
                "protocol": "hard_dev may guide development; holdout sealed after construction, no training/tuning or per-case inference feedback",
                "normalized_exact_overlap_with_v20_train_valid_and_v19_release": 0}
    for panel, rows in panels.items():
        path = OUT / f"{panel}.jsonl"
        manifest["panels"][panel] = {"rows": len(rows), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                      "tags": sorted({tag for row in rows for tag in row["tags"]})}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
