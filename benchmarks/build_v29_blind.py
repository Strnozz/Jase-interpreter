"""Freeze a varied Italian GoalContract 1.3 panel before any V26 data work.

The pairs are individually authored and checked, but not human adjudicated.
Benchmark input/output must never enter the training corpus.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.build_v28_blind import c, ctx, f, fallback, g, missing, selection, t  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

DEST = ROOT / "benchmarks/v29"
PRIOR = [ROOT / name for name in (
    "benchmarks/v20/hard_dev.jsonl", "benchmarks/v20/holdout.jsonl",
    "benchmarks/v23/transfer.jsonl", "benchmarks/v24/safety.jsonl",
    "benchmarks/v25/blind.jsonl", "benchmarks/v26/blind.jsonl",
    "benchmarks/v27/fresh.jsonl", "benchmarks/v28/blind.jsonl",
    "data/v25_9b/train.jsonl", "data/v25_9b/valid.jsonl",
)]


def cases():
    rows = []

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [x.as_dict() for x in issues]))
        rows.append({"id": f"v29-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold})

    add("quoted_instruction", "Nella guida voglio spiegare cosa significa 'prenota un tavolo per due'. Nessuna prenotazione reale.", c())
    add("quoted_instruction", "Traduci in inglese l'espressione 'comprami il biglietto del treno', senza fare acquisti.", c())
    add("quoted_instruction", "Questa frase compare in un romanzo: 'annulla la visita di venerd\u00ec'. Correggine solo la punteggiatura.", c())
    add("quoted_message", "Invia a Sara 'Il medico mi aspetta alle 15'; mostrami il messaggio prima di spedirlo.",
        c(g("contact", "Sara", "person", [f("message", "Il medico mi aspetta alle 15")],
            policy={"confirm_before": ["contact"]})))
    add("quoted_message", "Alle 18 scrivi a Giorgio 'Arriver\u00f2 alle 20', ma aspetta il mio via libera.",
        c(g("contact", "Giorgio", "person", [f("message", "Arriver\u00f2 alle 20")],
            temporal=[t("action_time", "18:00")], policy={"confirm_before": ["contact"]})))
    add("quoted_message", "Domani alle 09 ricordami: 'ritirare il pacco alle 18'.",
        c(g("notify", "promemoria", "reminder", [f("message", "ritirare il pacco alle 18")],
            temporal=[t("notification_date", "domani"), t("notification_time", "09:00")])))

    add("route_arrival", "Mi occorre un treno da Piacenza a Milano Porta Garibaldi con arrivo entro le 08:45.",
        c(g("find", "treno", "transport", [f("origin", "Piacenza"), f("destination", "Milano Porta Garibaldi")],
            temporal=[t("arrival_time", "08:45", "before")])))
    add("route_arrival", "Partendo da Verona, trova un autobus per Trento che sia l\u00ec prima delle 16:00.",
        c(g("find", "autobus", "transport", [f("origin", "Verona"), f("destination", "Trento")],
            temporal=[t("arrival_time", "16:00", "before")])))
    add("route_arrival", "Devo essere a Padova entro le 17:20; parto da Vicenza e preferisco il treno.",
        c(g("find", "treno", "transport", [f("origin", "Vicenza"), f("destination", "Padova")],
            temporal=[t("arrival_time", "17:20", "before")])))
    add("route_arrival", "Per gioved\u00ec mostrami i treni Bologna-Rimini che arrivano prima delle 11 e ordinali per costo crescente.",
        c(g("find", "treno", "transport", [f("origin", "Bologna"), f("destination", "Rimini")],
            temporal=[t("search_date", "gioved\u00ec"), t("arrival_time", "11:00", "before")],
            modifiers={"sort": {"field": "price", "direction": "asc"}})))

    add("flight_direct", "Voglio soltanto vedere voli diretti da Venezia a Barcellona per il 14 ottobre: non prenotare.",
        c(g("find", "volo", "transport", [f("origin", "Venezia"), f("destination", "Barcellona"), f("stops", 0)],
            temporal=[t("search_date", "14 ottobre")], policy={"forbid": ["book"]})))
    add("flight_direct", "Per il 20 novembre prenota un volo senza scalo da Roma a Lisbona, dopo la mia conferma.",
        c(g("book", "volo", "transport", [f("origin", "Roma"), f("destination", "Lisbona"), f("stops", 0)],
            temporal=[t("action_date", "20 novembre")], policy={"confirm_before": ["book"]})))
    add("flight_direct", "Da Milano a Copenaghen cerca un volo senza fermate per luned\u00ec; niente acquisti.",
        c(g("find", "volo", "transport", [f("origin", "Milano"), f("destination", "Copenaghen"), f("stops", 0)],
            temporal=[t("search_date", "luned\u00ec")], policy={"forbid": ["buy"]})))

    add("no_booking", "Elenca quattro hotel con parcheggio a Lecce per il 6 ottobre; non fare alcuna prenotazione.",
        c(g("find", "hotel", "accommodation", [f("location", "Lecce", "near"), f("amenity", "parcheggio")],
            temporal=[t("search_date", "6 ottobre")], modifiers={"limit": 4},
            policy={"forbid": ["book"]})))
    add("no_booking", "Cerco un B&B a Pavia per sabato. Mostrami le opzioni, ma non bloccare nessuna camera.",
        c(g("find", "B&B", "accommodation", [f("location", "Pavia", "near")],
            temporal=[t("search_date", "sabato")], policy={"forbid": ["book"]})))
    add("no_booking", "Confronta tre alberghi a Cremona in ordine di prezzo; la decisione di prenotare spetta a me.",
        c(g("compare", "albergo", "accommodation", [f("location", "Cremona", "near")],
            modifiers={"compare_count": 3, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["book"]})))

    add("dependent_action", "Trovami un hotel a Bologna per venerd\u00ec e prenota la seconda proposta soltanto dopo il mio ok.",
        c(g("find", "hotel", "accommodation", [f("location", "Bologna", "near")],
            temporal=[t("search_date", "venerd\u00ec")]),
          g("book", "hotel", "accommodation", [selection(ordinal=2)], idx=2,
            depends_on=["g1"], temporal=[t("action_date", "venerd\u00ec")],
            policy={"confirm_before": ["book"]})))
    add("dependent_action", "Prima cerca un volo da Firenze a Parigi per il 12 novembre; in seguito prenota il primo se lo approvo.",
        c(g("find", "volo", "transport", [f("origin", "Firenze"), f("destination", "Parigi")],
            temporal=[t("search_date", "12 novembre")]),
          g("book", "volo", "transport", [f("origin", "Firenze"), f("destination", "Parigi"), selection()],
            idx=2, depends_on=["g1"], temporal=[t("action_date", "12 novembre")],
            policy={"confirm_before": ["book"]})))
    add("dependent_action", "Seleziona tre ristoranti a Udine, poi prenota il terzo solo dopo la mia autorizzazione; ancora non so per quando.",
        c(g("find", "ristorante", "place", [f("location", "Udine", "near")], modifiers={"limit": 3}),
          g("book", "ristorante", "place", [selection(ordinal=3)], idx=2, depends_on=["g1"],
            missing=[missing("date"), missing("time"), missing("guests")],
            policy={"confirm_before": ["book"]})))

    add("fallback", "Cerca i treni per la stazione Venezia Santa Lucia; se non esiste alcuna corsa, passa agli autobus.",
        c(g("find", "treno", "transport", [f("destination", "stazione Venezia Santa Lucia")]),
          g("find", "autobus", "transport", [f("destination", "stazione Venezia Santa Lucia")],
            idx=2, depends_on=["g1"], condition=fallback())))
    add("fallback", "Verifica se ci sono appartamenti a Parma per il 9 ottobre. In caso contrario, cerca hotel.",
        c(g("find", "appartamento", "accommodation", [f("location", "Parma", "near")],
            temporal=[t("search_date", "9 ottobre")]),
          g("find", "hotel", "accommodation", [f("location", "Parma", "near")],
            idx=2, depends_on=["g1"], temporal=[t("search_date", "9 ottobre")], condition=fallback())))
    add("independent_goals", "Cerca una farmacia a Saronno e ricordami domani alle 08 di chiamare mia madre.",
        c(g("find", "farmacia", "place", [f("location", "Saronno", "near")]),
          g("notify", "promemoria", "reminder", [f("message", "chiamare mia madre")], idx=2,
            temporal=[t("notification_date", "domani"), t("notification_time", "08:00")])) )

    add("place_constraints", "Adesso a Treviso mi servono due farmacie aperte vicino al municipio.",
        c(g("find", "farmacia", "place", [f("location", "municipio di Treviso", "near"),
            f("open_now", True)], modifiers={"limit": 2})))
    add("place_constraints", "Non ho la macchina: cerca una trattoria a Pesaro raggiungibile a piedi dalla stazione.",
        c(g("find", "trattoria", "place", [f("location", "stazione di Pesaro", "near"),
            f("access", "a piedi")]), context=[ctx("has_car", False)]))
    add("place_constraints", "Siamo quattro: trova un ristorante vegetariano aperto ora a Modena.",
        c(g("find", "ristorante vegetariano", "place", [f("location", "Modena", "near"),
            f("open_now", True)]), context=[ctx("party_size", 4)]))

    add("reminder", "Ricordami marted\u00ec alle 07:15 di prendere il modulo per il Comune.",
        c(g("notify", "promemoria", "reminder", [f("message", "prendere il modulo per il Comune")],
            temporal=[t("notification_date", "marted\u00ec"), t("notification_time", "07:15")])))
    add("reminder", "Fra due giorni fammi ricordare quella cosa di cui abbiamo parlato ieri: non ricordo il contenuto.",
        c(g("notify", "promemoria", "reminder", temporal=[t("notification_date", "fra due giorni")],
            missing=[missing("referent", "search", "unresolved_reference")])))
    add("appointment", "Fissa un controllo dentistico a Forl\u00ec per il 18 novembre alle 14:30, ma prima chiedimi conferma.",
        c(g("book", "controllo dentistico", "appointment", [f("location", "Forl\u00ec", "near")],
            temporal=[t("action_date", "18 novembre"), t("action_time", "14:30")],
            policy={"confirm_before": ["book"]})))
    add("appointment", "La visita ortopedica del 13 ottobre va spostata al 21 ottobre; conferma con me prima.",
        c(g("reschedule", "visita ortopedica", "appointment", [f("original_date", "13 ottobre")],
            temporal=[t("action_date", "21 ottobre")], policy={"confirm_before": ["reschedule"]})))
    add("appointment", "Cancella la seduta logopedica prenotata per il 27 novembre a Cuneo.",
        c(g("cancel", "seduta logopedica", "appointment", [f("original_date", "27 novembre"),
            f("location", "Cuneo", "near")])) )

    add("food_action", "Fammi ordinare una pizza margherita con consegna a Prato, ma invia l'ordine soltanto dopo il mio ok.",
        c(g("order", "pizza margherita", "food", [f("delivery_location", "Prato")],
            policy={"confirm_before": ["order"]})))
    add("food_action", "Metti le offerte di sushi a Bergamo in ordine di prezzo senza comprare niente.",
        c(g("find", "sushi", "food", [f("location", "Bergamo", "near")],
            modifiers={"sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["buy", "order"]})))
    add("service_action", "Trova un elettricista disponibile oggi a Monza; per ora non assumerlo.",
        c(g("find", "elettricista", "professional_service", [f("location", "Monza", "near")],
            temporal=[t("search_date", "oggi")], policy={"forbid": ["hire"]})))
    add("service_action", "Ingaggia un idraulico a Varese, ma sottoponimi la scelta prima di confermare.",
        c(g("hire", "idraulico", "professional_service", [f("location", "Varese", "near")],
            missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("vehicle", "Fammi vedere il noleggio di camper a Rieti per sabato; non affittare ancora nulla.",
        c(g("find", "camper", "vehicle", [f("location", "Rieti", "near")],
            temporal=[t("search_date", "sabato")], policy={"forbid": ["rent"]})))
    add("vehicle", "Per il 15 ottobre noleggia una bicicletta a Ferrara, previa mia autorizzazione.",
        c(g("rent", "bicicletta", "vehicle", [f("location", "Ferrara", "near")],
            temporal=[t("action_date", "15 ottobre")], policy={"confirm_before": ["rent"]})))

    add("count_and_price", "Confronta quattro frigoriferi che costino meno di 600 euro.",
        c(g("compare", "frigorifero", "product", [f("price", 600, "lt", currency="EUR")],
            modifiers={"compare_count": 4})))
    add("count_and_price", "Mostrami non pi\u00f9 di tre librerie a Gorizia, ordinate per distanza.",
        c(g("find", "libreria", "place", [f("location", "Gorizia", "near")],
            modifiers={"limit": 3, "sort": {"field": "distance", "direction": "asc"}})))
    add("missing_information", "Prenota un volo per Bruxelles il 17 novembre; non ti ho detto l'aeroporto di partenza.",
        c(g("book", "volo", "transport", [f("destination", "Bruxelles")],
            temporal=[t("action_date", "17 novembre")], missing=[missing("origin")])))
    add("missing_information", "Scrivigli che sar\u00f2 a casa alle 19, ma non ricordo chi fosse il destinatario.",
        c(g("contact", "destinatario", "person", [f("message", "sar\u00f2 a casa alle 19")],
            missing=[missing("recipient", "search", "unresolved_reference")])))
    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V29 already frozen; do not regenerate")
    rows = cases()
    if len(rows) != 40 or len({normalize_text(r["text"]) for r in rows}) != 40:
        raise RuntimeError(f"Expected 40 unique requests; got {len(rows)}")
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
        maximum = max(maximum, *(len(tok & prev) / len(tok | prev) for prev in tokens if prev))
    if maximum >= .8:
        raise RuntimeError(f"Near prior overlap: {maximum:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V29", "schema_version": "1.3", "cases": len(rows),
                "families": dict(Counter(x["family"] for x in rows)),
                "panel_sha256": sha256(panel), "generator_sha256": sha256(Path(__file__)),
                "helper_sha256": sha256(ROOT / "benchmarks/build_v28_blind.py"),
                "max_prior_jaccard": round(maximum, 5),
                "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR},
                "limitations": ["Agent-authored targeted panel; not a production sample",
                                "Independent human gold review pending"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"cases": len(rows), "panel_sha256": manifest["panel_sha256"],
                      "max_prior_jaccard": maximum}, ensure_ascii=False))


if __name__ == "__main__":
    main()
