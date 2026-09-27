"""Freeze a realistic GoalContract 1.3 routing panel before Guard/data changes.

Five independently selected contexts cross 20 capabilities. Every request and
gold is reviewed as a pair in this source; the benchmark remains AI-authored.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

DEST = ROOT / "benchmarks/v27"
PRIOR = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
         ROOT / "benchmarks/v23/transfer.jsonl", ROOT / "benchmarks/v24/safety.jsonl",
         ROOT / "benchmarks/v25/blind.jsonl", ROOT / "benchmarks/v26/blind.jsonl",
         ROOT / "data/v25_9b/train.jsonl", ROOT / "data/v25_9b/valid.jsonl"]


def f(field, value, op="eq", **extra):
    return {"field": field, "op": op, "value": value, "strength": "hard", **extra}


def t(role, value, op="eq"):
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def g(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def c(*goals, context=None):
    out = {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
           "goals": list(goals)}
    if context:
        out["context"] = context
    return out


def missing(field, blocks="action", reason="unspecified"):
    return {"field": field, "blocks": blocks, "reason": reason}


def ctx(field, value):
    return {"scope": "user", "field": field, "op": "eq", "value": value}


def cases():
    records = []
    contexts = [
        {"city": "Bologna", "day": "martedì", "person": "Elisa", "appointment": "visita cardiologica",
         "vehicle": "bicicletta", "vehicle_article": "una", "food": "pizza vegetariana", "food_article": "una",
         "product": "lampada", "origin": "Roma", "flight": "Madrid", "task": "ritirare un documento"},
        {"city": "Padova", "day": "venerdì", "person": "Luca", "appointment": "controllo oculistico",
         "vehicle": "scooter", "vehicle_article": "uno", "food": "panino integrale", "food_article": "un",
         "product": "zaino", "origin": "Milano", "flight": "Berlino", "task": "pagare la bolletta"},
        {"city": "Perugia", "day": "sabato", "person": "Nadia", "appointment": "seduta di fisioterapia",
         "vehicle": "auto", "vehicle_article": "un'", "food": "piadina al formaggio", "food_article": "una",
         "product": "tablet", "origin": "Napoli", "flight": "Parigi", "task": "chiamare l'ufficio"},
        {"city": "Trento", "day": "il 12 giugno", "person": "Pietro", "appointment": "visita dal dentista",
         "vehicle": "furgone", "vehicle_article": "un", "food": "insalata greca", "food_article": "un'",
         "product": "frullatore", "origin": "Venezia", "flight": "Londra", "task": "consegnare le chiavi"},
        {"city": "Modena", "day": "domani", "person": "Anna", "appointment": "consulenza nutrizionale",
         "vehicle": "camper", "vehicle_article": "un", "food": "ramen vegetariano", "food_article": "un",
         "product": "stampante", "origin": "Bologna", "flight": "Amsterdam", "task": "inviare il modulo"},
    ]

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [x.as_dict() for x in issues]))
        records.append({"id": f"v27-{len(records)+1:03d}", "family": family,
                        "text": text, "contract": gold})

    for x in contexts:
        city, day, person, appointment, vehicle = (x[k] for k in
            ("city", "day", "person", "appointment", "vehicle"))
        food, product, origin, flight = (x[k] for k in
            ("food", "product", "origin", "flight"))
        food_with_article = x["food_article"] + ("" if x["food_article"].endswith("'") else " ") + food
        vehicle_with_article = x["vehicle_article"] + ("" if x["vehicle_article"].endswith("'") else " ") + vehicle
        appointment_phrase = ("un " if appointment.startswith("controllo") else "una ") + appointment
        reschedule_day = "al 12 giugno" if day == "il 12 giugno" else "a " + day
        recipient_phrase = "ad Anna" if person == "Anna" else "a " + person
        flight_phrase = "ad Amsterdam" if flight == "Amsterdam" else "a " + flight

        add("quoted_no_action",
            f"Sto scrivendo una scena a {city}: un personaggio dice 'prenota una visita'. Non si tratta di una richiesta per te.",
            c())
        add("actionable_quoted_message",
            f"Invia {recipient_phrase} questo messaggio: 'Arrivo a {city} alle 18'. Prima chiedimi conferma.",
            c(g("contact", person, "person", [f("message", f"Arrivo a {city} alle 18")],
                policy={"confirm_before": ["contact"]})))
        add("find_then_book",
            f"Individua un albergo a {city} per {day}; prenota la prima opzione dopo la mia approvazione.",
            c(g("find", "albergo", "accommodation", [f("location", city, "near")],
                temporal=[t("search_date", day)], idx=1),
              g("book", "albergo", "accommodation", [{"field": "selection", "op": "eq",
                "value_ref": {"goal": "g1", "field": "result", "ordinal": 1}, "strength": "hard"}],
                idx=2, depends_on=["g1"], policy={"confirm_before": ["book"]})))
        add("flight_direct",
            f"Cerca un volo senza scali da {origin} {flight_phrase} per {day}; non acquistare il biglietto.",
            c(g("find", "volo", "transport", [f("origin", origin), f("destination", flight),
                f("stops", 0)], temporal=[t("search_date", day)], policy={"forbid": ["buy", "book"]})))
        add("flight_missing_origin",
            f"Prenota un volo per {flight} {day}; non ti ho ancora detto da quale aeroporto parto.",
            c(g("book", "volo", "transport", [f("destination", flight)],
                temporal=[t("action_date", day)], missing=[missing("origin")])))
        add("explicit_no_booking",
            f"Elencami quattro hotel a {city} in ordine di prezzo, ma non fare nessuna prenotazione.",
            c(g("find", "hotel", "accommodation", [f("location", city, "near")],
                modifiers={"sort": {"field": "price", "direction": "asc"}, "limit": 4},
                policy={"forbid": ["book"]})))
        add("transport_sort",
            f"Per raggiungere {city}, disponi i treni in ordine crescente di prezzo entro le 14.",
            c(g("find", "treno", "transport", [f("destination", city)],
                temporal=[t("arrival_time", "14:00", "before")],
                modifiers={"sort": {"field": "price", "direction": "asc"}})))
        add("food_order",
            f"Fai arrivare a {city} {food_with_article}; sottoponimi l'ordine prima di inviarlo.",
            c(g("order", food, "food", [f("delivery_location", city)],
                policy={"confirm_before": ["order"]})))
        add("compare_cardinality",
            f"Tra le offerte di {product} per {city}, confrontane esattamente tre.",
            c(g("compare", product, "product", [f("location", city, "near")],
                modifiers={"compare_count": 3})))
        add("result_limit",
            f"Mostra soltanto due farmacie aperte vicino al centro di {city}.",
            c(g("find", "farmacia", "place", [f("location", f"centro di {city}", "near"),
                f("open_now", True)], modifiers={"limit": 2})))
        add("conditional_fallback",
            f"Cerca un autobus per {city}. Soltanto se non trovi risultati, passa al treno.",
            c(g("find", "autobus", "transport", [f("destination", city)], idx=1),
              g("find", "treno", "transport", [f("destination", city)], idx=2,
                depends_on=["g1"], condition={"goal": "g1", "test": {
                    "field": "results_count", "op": "eq", "value": 0}})))
        add("appointment_reschedule",
            f"Il mio appuntamento per {appointment_phrase} risulta fissato per il 3 maggio: spostalo {reschedule_day}, previa conferma.",
            c(g("reschedule", appointment, "appointment", [f("original_date", "3 maggio")],
                temporal=[t("action_date", day)], policy={"confirm_before": ["reschedule"]})))
        add("appointment_cancel",
            f"Annulla l'appuntamento per {appointment_phrase} del 3 maggio a {city}.",
            c(g("cancel", appointment, "appointment", [f("original_date", "3 maggio"),
                f("location", city, "near")])))
        add("reminder_content",
            f"{day.capitalize()} alle 09 ricordami di {x['task']}.",
            c(g("notify", "promemoria", "reminder", [f("message", x["task"])],
                temporal=[t("notification_date", day), t("notification_time", "09:00")])))
        add("group_context",
            f"Siamo in tre e viaggiamo senza automobile: cerca un ristorante a {city} raggiungibile a piedi.",
            c(g("find", "ristorante", "place", [f("location", city, "near"),
                f("access", "a piedi")]), context=[ctx("party_size", 3), ctx("has_car", False)]))
        add("search_then_later_action_date",
            f"Cerca oggi un albergo a {city} per il 20 dicembre; prenota la prima opzione tra un'ora solo dopo la mia conferma.",
            c(g("find", "albergo", "accommodation", [f("location", city, "near")],
                temporal=[t("search_date", "20 dicembre")], idx=1),
              g("book", "albergo", "accommodation", [{"field": "selection", "op": "eq",
                "value_ref": {"goal": "g1", "field": "result", "ordinal": 1}, "strength": "hard"}],
                idx=2, depends_on=["g1"], temporal=[t("action_time", "tra un'ora")],
                policy={"confirm_before": ["book"]})))
        add("arrival_role",
            f"Da Firenze trovami un treno che arrivi a {city} prima delle 11:45.",
            c(g("find", "treno", "transport", [f("origin", "Firenze"),
                f("destination", city)], temporal=[t("arrival_time", "11:45", "before")])))
        add("vehicle_rent",
            f"A {city} noleggia {vehicle_with_article} per {day}; aspetta la mia autorizzazione.",
            c(g("rent", vehicle, "vehicle", [f("location", city, "near")],
                temporal=[t("action_date", day)], policy={"confirm_before": ["rent"]})))
        add("vehicle_find_only",
            f"Vorrei vedere le offerte di noleggio per {vehicle_with_article} a {city} {day}, senza impegnarmi.",
            c(g("find", vehicle, "vehicle", [f("location", city, "near")],
                temporal=[t("search_date", day)], policy={"forbid": ["rent"]})))
        add("unresolved_reminder",
            f"Quando sar\u00f2 a {city} {day}, ricordami quella cosa di cui parlavamo ieri.",
            c(g("notify", "promemoria", "reminder", [f("location", city, "near")],
                temporal=[t("notification_date", day)],
                missing=[missing("referent", "search", "unresolved_reference")])))
    return records


def main() -> int:
    if DEST.exists():
        raise FileExistsError("V27 already frozen; do not regenerate")
    rows = cases()
    if len(rows) != 100 or len({normalize_text(r["text"]) for r in rows}) != 100:
        raise RuntimeError("Expected 100 unique requests")
    old = []
    for path in PRIOR:
        for row in read_jsonl(path):
            old.append(row.get("text") or row["messages"][1]["content"])
    keys = {normalize_text(x) for x in old}
    tokens = [token_set(x) for x in old]
    maximum = 0.0
    for row in rows:
        key, tok = normalize_text(row["text"]), token_set(row["text"])
        if key in keys:
            raise RuntimeError(f"Exact prior overlap: {row['id']}")
        maximum = max(maximum, *(len(tok & x) / len(tok | x) for x in tokens if x))
    if maximum >= .85:
        raise RuntimeError(f"Near overlap with prior data: {maximum:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "fresh.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V27", "schema_version": "1.3", "cases": len(rows),
                "families": dict(Counter(r["family"] for r in rows)),
                "panel_sha256": sha256(panel), "generator_sha256": sha256(Path(__file__)),
                "max_prior_jaccard": round(maximum, 5),
                "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR},
                "limitations": ["AI-authored targeted routing panel, not production prevalence",
                                "Five related contexts per family", "Independent human gold review pending"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"cases": len(rows), "sha256": manifest["panel_sha256"],
                      "max_prior_jaccard": maximum}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
