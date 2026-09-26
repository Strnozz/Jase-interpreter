"""Freeze an independently authored V1.3 semantic panel before V25 training data.

The output is a targeted capability panel, not a production-distribution sample.
Do not import this module into a corpus builder. Never rewrite a frozen output.
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

DEST = ROOT / "benchmarks/v26"
EXISTING = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
            ROOT / "benchmarks/v23/transfer.jsonl", ROOT / "benchmarks/v24/safety.jsonl",
            ROOT / "benchmarks/v25/blind.jsonl", ROOT / "data/v24_9b/train.jsonl",
            ROOT / "data/v24_9b/valid.jsonl"]


def fact(field, value, op="eq", strength="hard", **extra):
    return {"field": field, "op": op, "value": value, "strength": strength, **extra}


def temporal(role, value, op="eq", strength="hard"):
    return {"role": role, "op": op, "value": value, "strength": strength}


def goal(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def missing(field, blocks="action", reason="unspecified"):
    return {"field": field, "blocks": blocks, "reason": reason}


def contract(*goals, context=None):
    result = {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
              "goals": list(goals)}
    if context:
        result["context"] = context
    return result


def user_context(field, value):
    return {"scope": "user", "field": field, "op": "eq", "value": value}


def cases():
    out = []
    towns = ["Tarquinia", "Sondrio", "Viterbo", "Matera", "Pordenone", "Lecce"]
    days = ["lunedì", "giovedì", "sabato", "il 18 novembre", "fra tre giorni", "domenica"]
    products = ["lampada", "frullatore", "scrivania", "macchina fotografica", "stampante", "valigia"]
    foods = ["pizza margherita", "sushi vegetariano", "panino al tonno", "insalata greca",
             "ramen miso", "torta al cioccolato"]
    people = ["Elisa", "Marco", "Giulia", "Lorenzo", "Nadia", "Pietro"]
    vehicles = ["scooter", "camper", "furgone", "bicicletta", "moto", "minivan"]
    appointments = ["visita cardiologica", "controllo oculistico", "seduta di fisioterapia",
                    "visita dal dentista", "colloquio di orientamento", "controllo veterinario"]

    def add(family, i, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, i, text, [x.as_dict() for x in issues]))
        out.append({"id": f"v26-{len(out)+1:03d}", "family": family, "text": text,
                    "contract": gold})

    for i in range(6):
        town, day, product, food, person, vehicle, appointment = (
            towns[i], days[i], products[i], foods[i], people[i], vehicles[i], appointments[i])
        add("find_place_time", i,
            f"Mi trovi una libreria aperta a {town} {day} nel pomeriggio?",
            contract(goal("find", "libreria", "place", [fact("location", town, "near")],
                          temporal=[temporal("search_date", day),
                                    temporal("search_time", "pomeriggio")])))
        add("appointment_book", i,
            f"Fissa per {day} alle 09:40 una {appointment} a {town}, previa mia conferma.",
            contract(goal("book", appointment, "appointment", [fact("location", town, "near")],
                          temporal=[temporal("action_date", day), temporal("action_time", "09:40")],
                          policy={"confirm_before": ["book"]})))
        add("appointment_reschedule", i,
            f"Sposta la mia {appointment} del 6 ottobre a {day}; chiedimi prima di modificarla.",
            contract(goal("reschedule", appointment, "appointment", [fact("original_date", "6 ottobre")],
                          temporal=[temporal("action_date", day)],
                          policy={"confirm_before": ["reschedule"]})))
        add("appointment_cancel", i,
            f"Disdici la {appointment} fissata per il 9 dicembre a {town}.",
            contract(goal("cancel", appointment, "appointment", [fact("original_date", "9 dicembre"),
                          fact("location", town, "near")])))
        add("vehicle_rent", i,
            f"Vorrei noleggiare un {vehicle} a {town} per {day}; procedi dopo conferma.",
            contract(goal("rent", vehicle, "vehicle", [fact("location", town, "near")],
                          temporal=[temporal("action_date", day)],
                          policy={"confirm_before": ["rent"]})))
        add("vehicle_find", i,
            f"Cerca a {town} un {vehicle} disponibile a noleggio {day}, senza riservarlo.",
            contract(goal("find", vehicle, "vehicle", [fact("location", town, "near")],
                          temporal=[temporal("search_date", day)],
                          policy={"forbid": ["rent", "book"]})))
        add("transport_sort", i,
            f"Per andare a {town}, ordina le alternative di viaggio dal costo minore al maggiore.",
            contract(goal("find", "alternative di viaggio", "transport", [fact("destination", town)],
                          modifiers={"sort": {"field": "price", "direction": "asc"}})))
        add("food_order", i,
            f"Ordina una {food} da consegnare a {town} e aspetta il mio via libera.",
            contract(goal("order", food, "food", [fact("delivery_location", town)],
                          policy={"confirm_before": ["order"]})))
        add("compare_count", i,
            f"Metti a confronto quattro {product} vendute a {town}, mantenendo quattro opzioni.",
            contract(goal("compare", product, "product", [fact("location", town, "near")],
                          modifiers={"compare_count": 4})))
        add("result_limit", i,
            f"Mostrami al massimo tre alberghi economici per {town}; non prenotarne nessuno.",
            contract(goal("find", "albergo", "accommodation", [fact("location", town, "near")],
                          modifiers={"sort": {"field": "price", "direction": "asc"}, "limit": 3},
                          policy={"forbid": ["book"]})))
        add("contact_message", i,
            f"Scrivi a {person}: arriverò a {town} verso le 19. Prima di inviare mostrami il testo.",
            contract(goal("contact", person, "person", [fact("message", f"Arriverò a {town} verso le 19")],
                          policy={"confirm_before": ["contact"]})))
        add("reminder_content", i,
            f"Ricordami {day} alle 08:15 di telefonare a {person} per la pratica di {town}.",
            contract(goal("notify", "promemoria", "reminder",
                          [fact("message", f"telefonare a {person} per la pratica di {town}")],
                          temporal=[temporal("notification_date", day),
                                    temporal("notification_time", "08:15")])))
        add("find_then_book", i,
            f"Trova una camera a {town} per {day}, poi prenota la prima solo dopo che avrò approvato.",
            contract(goal("find", "camera", "accommodation", [fact("location", town, "near")],
                          temporal=[temporal("search_date", day)], idx=1),
                     goal("book", "camera", "accommodation", [
                         {"field": "selection", "op": "eq", "value_ref": {
                             "goal": "g1", "field": "result", "ordinal": 1}, "strength": "hard"}], idx=2,
                          depends_on=["g1"], policy={"confirm_before": ["book"]})))
        add("two_temporal_roles", i,
            f"Cerca un volo per {town} {day}, ma effettua la prenotazione soltanto il 21 novembre e dopo mia conferma.",
            contract(goal("find", "volo", "transport", [fact("destination", town)],
                          temporal=[temporal("search_date", day)], idx=1),
                     goal("book", "volo", "transport", [fact("destination", town),
                         {"field": "selection", "op": "eq", "value_ref": {
                             "goal": "g1", "field": "result"}, "strength": "hard"}], idx=2,
                          depends_on=["g1"], temporal=[temporal("action_date", "21 novembre")],
                          missing=[missing("origin")], policy={"confirm_before": ["book"]})))
        add("group_context", i,
            f"Siamo in quattro e non abbiamo l'auto: trova un ristorante accessibile a piedi a {town}.",
            contract(goal("find", "ristorante", "place", [fact("location", town, "near"),
                          fact("access", "a piedi")]),
                     context=[user_context("party_size", 4), user_context("has_car", False)]))
        add("conditional_fallback", i,
            f"Cerca un treno per {town}; se non ce n'è uno, cerca un pullman.",
            contract(goal("find", "treno", "transport", [fact("destination", town)], idx=1),
                     goal("find", "pullman", "transport", [fact("destination", town)], idx=2,
                          depends_on=["g1"], condition={"goal": "g1", "test": {
                              "field": "results_count", "op": "eq", "value": 0}})))
        add("non_actionable", i,
            f"Per un racconto ambientato a {town}, immagina qualcuno che dica 'prenotami una {appointment}'.",
            contract())
        add("table_missing", i,
            f"Vorrei prenotare un tavolo in un ristorante a {town}; dimmi cosa manca.",
            contract(goal("book", "tavolo", "place", [fact("location", town, "near")],
                          missing=[missing("date"), missing("time"), missing("guests")])))
        add("flight_missing", i,
            f"Prenota un volo diretto a {town} per {day}, ma non sai da dove parto.",
            contract(goal("book", "volo", "transport", [fact("destination", town),
                          fact("stops", 0)], temporal=[temporal("action_date", day)],
                          missing=[missing("origin")])) )
        add("monitor_price", i,
            f"Tieni sotto controllo il prezzo della {product} modello Selene {i+2}; avvisami sotto 160 euro.",
            contract(goal("monitor", f"{product} modello Selene {i+2}", "product",
                          [fact("price", 160, "lt", currency="EUR")])))
    return out


def main() -> int:
    if DEST.exists():
        raise FileExistsError("V26 panel already frozen")
    rows = cases()
    if len(rows) != 120 or len({normalize_text(r["text"]) for r in rows}) != 120:
        raise RuntimeError("V26 must contain 120 unique requests")
    old_text = []
    for path in EXISTING:
        for row in read_jsonl(path):
            old_text.append(row.get("text") or row["messages"][1]["content"])
    old_keys = {normalize_text(text) for text in old_text}
    old_tokens = [token_set(text) for text in old_text]
    nearest = 0.0
    for row in rows:
        key, tokens = normalize_text(row["text"]), token_set(row["text"])
        if key in old_keys:
            raise RuntimeError(f"Exact prior overlap: {row['id']}")
        nearest = max(nearest, *(len(tokens & prior) / len(tokens | prior)
                                 for prior in old_tokens if prior))
    if nearest >= .85:
        raise RuntimeError(f"V26 near overlap with prior data: {nearest:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V26", "schema_version": "1.3", "frozen_before_v25_data": True,
                "cases": len(rows), "families": dict(Counter(r["family"] for r in rows)),
                "panel_sha256": sha256(panel),
                "prior_corpus_and_panels": {str(path.relative_to(ROOT)): sha256(path) for path in EXISTING},
                "max_prior_jaccard": round(nearest, 5),
                "generator_sha256": sha256(Path(__file__)),
                "limitations": ["AI-authored targeted panel", "Six lexical variants per family",
                                "Not a production prevalence estimate", "Gold needs independent human audit"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"cases": len(rows), "sha256": manifest["panel_sha256"],
                      "max_prior_jaccard": nearest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
