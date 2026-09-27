"""Freeze V31 before any further Guard, ontology, data or Planner changes.

The labels were authored independently of model outputs. Human adjudication
is still required before interpreting scores as production accuracy.
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

DEST = ROOT / "benchmarks/v31"
PRIOR = [ROOT / p for p in (
    "benchmarks/v20/hard_dev.jsonl", "benchmarks/v20/holdout.jsonl",
    "data/v20_quality/dev.jsonl", "benchmarks/release_holdout_v1.jsonl",
    "benchmarks/v23/transfer.jsonl", "benchmarks/v24/safety.jsonl",
    "benchmarks/v25/blind.jsonl", "benchmarks/v26/blind.jsonl",
    "benchmarks/v27/fresh.jsonl", "benchmarks/v28/blind.jsonl",
    "benchmarks/v29/blind.jsonl", "benchmarks/v30/blind.jsonl",
    "data/v24_9b/train.jsonl", "data/v24_9b/valid.jsonl",
    "data/v25_9b/train.jsonl", "data/v25_9b/valid.jsonl",
    "data/v26_9b/train.jsonl", "data/v26_9b/valid.jsonl",
)]


def cases():
    rows = []

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [x.as_dict() for x in issues]))
        rows.append({"id": f"v31-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold})

    add("narrative", "Nel mio diario compare la frase 'assumi subito un avvocato'. È una citazione del passato, non un incarico per te.", c())
    add("narrative", "Sto preparando un esempio per un corso: 'annulla la prenotazione'. Limitati ad analizzarne la grammatica.", c())
    add("narrative", "La scritta sul cartello è 'compra i biglietti online'; dimmi solo se è leggibile, senza comprare nulla.", c())

    add("communication", "Prima di mandare a Roberta 'Il pacco arriverà alle 14', fammi rileggere il messaggio e aspetta il mio sì.",
        c(g("contact", "Roberta", "person", [f("message", "Il pacco arriverà alle 14")],
            policy={"confirm_before": ["contact"]})))
    add("communication", "Alle 07:30 invia a Karim 'Ci vediamo alle 19'; non spedirlo finché non approvo.",
        c(g("contact", "Karim", "person", [f("message", "Ci vediamo alle 19")],
            temporal=[t("action_time", "07:30")], policy={"confirm_before": ["contact"]})))
    add("communication", "Per venerdì alle 08:10 avvisami: 'chiudere il gas prima delle 13'.",
        c(g("notify", "promemoria", "reminder", [f("message", "chiudere il gas prima delle 13")],
            temporal=[t("notification_date", "venerdì"), t("notification_time", "08:10")])))

    add("rail_route", "Trovami un regionale che parta da L'Aquila per Sulmona dopo le 06:40 e arrivi entro le 08:00.",
        c(g("find", "treno regionale", "transport", [f("origin", "L'Aquila"), f("destination", "Sulmona")],
            temporal=[t("departure_time", "06:40", "after"), t("arrival_time", "08:00", "before")])))
    add("rail_route", "Mercoledì devo essere a Rimini prima delle 09:15. Parto da Cesena: controlla soltanto i treni.",
        c(g("find", "treno", "transport", [f("origin", "Cesena"), f("destination", "Rimini")],
            temporal=[t("search_date", "mercoledì"), t("arrival_time", "09:15", "before")])))
    add("rail_route", "Verifica i bus da Alba a Bra che lasciano Alba dopo le 17:25, senza prenotare posti.",
        c(g("find", "autobus", "transport", [f("origin", "Alba"), f("destination", "Bra")],
            temporal=[t("departure_time", "17:25", "after")], policy={"forbid": ["book"]})))

    add("flight_constraints", "Quali voli senza scalo da Palermo a Vienna trovi per il 10 marzo? Mostrameli soltanto, niente prenotazioni.",
        c(g("find", "volo", "transport", [f("origin", "Palermo"), f("destination", "Vienna"), f("stops", 0)],
            temporal=[t("search_date", "10 marzo")], policy={"forbid": ["book"]})))
    add("flight_constraints", "Prenota un biglietto aereo da Catania a Bruxelles il 26 febbraio, solo dopo il mio assenso.",
        c(g("book", "volo", "transport", [f("origin", "Catania"), f("destination", "Bruxelles")],
            temporal=[t("action_date", "26 febbraio")], policy={"confirm_before": ["book"]})))
    add("flight_constraints", "Per la prossima settimana cerca un volo per Valencia, ma non ti ho detto da quale aeroporto parto.",
        c(g("find", "volo", "transport", [f("destination", "Valencia")],
            temporal=[t("search_date", "prossima settimana")], missing=[missing("origin", "search")])))

    add("accommodation", "Vorrei vedere al massimo tre alloggi con cucina a Pisa per domenica; la prenotazione la farò io.",
        c(g("find", "alloggio", "accommodation", [f("location", "Pisa", "near"), f("amenity", "cucina")],
            temporal=[t("search_date", "domenica")], modifiers={"limit": 3},
            policy={"forbid": ["book"]})))
    add("accommodation", "Confrontami due campeggi nei dintorni di Ravenna dal meno caro al più caro e non bloccare piazzole.",
        c(g("compare", "campeggio", "accommodation", [f("location", "Ravenna", "near")],
            modifiers={"compare_count": 2, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["book"]})))
    add("accommodation", "Cerca un albergo a Fano, ma fermati alla ricerca: voglio scegliere io se riservarlo.",
        c(g("find", "albergo", "accommodation", [f("location", "Fano", "near")],
            policy={"forbid": ["book"]})))

    add("selected_result", "Mostrami pensioni a Pescia e riserva la seconda solo dopo che ho detto sì.",
        c(g("find", "pensione", "accommodation", [f("location", "Pescia", "near")], idx=1),
          g("book", "pensione", "accommodation", [selection(ordinal=2)], idx=2,
            depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]})))
    add("selected_result", "Individua un portatile economico a Ivrea; se approvo la prima proposta, acquistala.",
        c(g("find", "portatile", "product", [f("location", "Ivrea", "near")], idx=1),
          g("buy", "portatile", "product", [selection(ordinal=1)], idx=2,
            depends_on=["g1"], policy={"confirm_before": ["buy"]})))
    add("selected_result", "Fammi vedere due ristoranti a Sora; quando scelgo il primo prenotalo, ma devo ancora dirti giorno, ora e coperti.",
        c(g("find", "ristorante", "place", [f("location", "Sora", "near")], modifiers={"limit": 2}, idx=1),
          g("book", "ristorante", "place", [selection(ordinal=1)], idx=2, depends_on=["g1"],
            missing=[missing("date"), missing("time"), missing("guests")],
            policy={"confirm_before": ["book"]})))

    add("alternatives", "Se a Chioggia non risultano treni per domani, cerca corriere; non mostrarmi entrambi se il treno c'è.",
        c(g("find", "treno", "transport", [f("destination", "Chioggia")],
            temporal=[t("search_date", "domani")], idx=1),
          g("find", "corriera", "transport", [f("destination", "Chioggia")],
            temporal=[t("search_date", "domani")], idx=2, depends_on=["g1"], condition=fallback())))
    add("alternatives", "Prima cerca un B&B a Vasto per lunedì; soltanto se la lista è vuota prova un hotel.",
        c(g("find", "B&B", "accommodation", [f("location", "Vasto", "near")],
            temporal=[t("search_date", "lunedì")], idx=1),
          g("find", "hotel", "accommodation", [f("location", "Vasto", "near")],
            temporal=[t("search_date", "lunedì")], idx=2, depends_on=["g1"], condition=fallback())))
    add("alternatives", "Cerca sia un parcheggio a Massa sia un cinema a Carrara; sono esigenze indipendenti.",
        c(g("find", "parcheggio", "place", [f("location", "Massa", "near")], idx=1),
          g("find", "cinema", "place", [f("location", "Carrara", "near")], idx=2)))

    add("local_filters", "Non guido: da piazza Duomo a Cremona trovami una pasticceria aperta adesso, raggiungibile camminando.",
        c(g("find", "pasticceria", "place", [f("location", "piazza Duomo a Cremona", "near"),
            f("access", "a piedi"), f("open_now", True)]), context=[ctx("has_car", False)]))
    add("local_filters", "Siamo sei persone: a Biella cerca un ristorante vegano aperto in questo momento.",
        c(g("find", "ristorante vegano", "place", [f("location", "Biella", "near"),
            f("open_now", True)]), context=[ctx("party_size", 6)]))
    add("local_filters", "Da via Roma ad Ascoli Piceno mostrami due farmacie aperte ora nelle vicinanze.",
        c(g("find", "farmacia", "place", [f("location", "via Roma ad Ascoli Piceno", "near"),
            f("open_now", True)], modifiers={"limit": 2})))

    add("medical_appointment", "Fissa una visita dermatologica a Rieti il 5 marzo alle 15:20. Prima di impegnarmi mostrami la proposta.",
        c(g("book", "visita dermatologica", "appointment", [f("location", "Rieti", "near")],
            temporal=[t("action_date", "5 marzo"), t("action_time", "15:20")],
            policy={"confirm_before": ["book"]})))
    add("medical_appointment", "La seduta di logopedia a Latina del 3 aprile va annullata, ma chiedimi un sì esplicito prima di cancellarla.",
        c(g("cancel", "seduta di logopedia", "appointment", [f("location", "Latina", "near"),
            f("original_date", "3 aprile")], policy={"confirm_before": ["cancel"]})))
    add("medical_appointment", "Vorrei spostare la visita neurologica del 6 marzo al 20 marzo: attendi la mia approvazione.",
        c(g("reschedule", "visita neurologica", "appointment", [f("original_date", "6 marzo")],
            temporal=[t("action_date", "20 marzo")], policy={"confirm_before": ["reschedule"]})))

    add("food_and_products", "Mandami quattro offerte per una lavatrice a Padova, ordinate dalla più economica; non comprarla.",
        c(g("find", "lavatrice", "product", [f("location", "Padova", "near")],
            modifiers={"limit": 4, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["buy"]})))
    add("food_and_products", "Confronta tre modelli di tablet sotto i 250 euro; il confronto non è un ordine d'acquisto.",
        c(g("compare", "tablet", "product", [f("price", 250, "lt", currency="EUR")],
            modifiers={"compare_count": 3}, policy={"forbid": ["buy"]})))
    add("food_and_products", "Fammi ordinare due porzioni di lasagne per consegna a Fidenza, ma aspetta il mio via libera per inviare l'ordine.",
        c(g("order", "lasagne", "food", [f("quantity", 2), f("delivery_location", "Fidenza")],
            policy={"confirm_before": ["order"]})))

    add("professional", "Trova un serramentista disponibile oggi a Gubbio, senza assumerlo ancora.",
        c(g("find", "serramentista", "professional_service", [f("location", "Gubbio", "near")],
            temporal=[t("search_date", "oggi")], policy={"forbid": ["hire"]})))
    add("professional", "Assumi un idraulico a Monfalcone solo quando avrai identificato chi è e dopo il mio consenso.",
        c(g("hire", "idraulico", "professional_service", [f("location", "Monfalcone", "near")],
            missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("professional", "Cerca un imbianchino a Nola, poi chiedi al secondo candidato per messaggio se può iniziare martedì.",
        c(g("find", "imbianchino", "professional_service", [f("location", "Nola", "near")], idx=1),
          g("contact", "imbianchino", "professional_service", [selection(ordinal=2),
            f("message", "Puoi iniziare martedì?")], idx=2, depends_on=["g1"])))

    add("mobility", "Per domenica elenca gli scooter a noleggio a Taranto, senza affittarne uno adesso.",
        c(g("find", "scooter", "vehicle", [f("location", "Taranto", "near")],
            temporal=[t("search_date", "domenica")], policy={"forbid": ["rent"]})))
    add("mobility", "Noleggia una canoa al lago di Bracciano il 19 aprile dopo che ti avrò detto di procedere.",
        c(g("rent", "canoa", "vehicle", [f("location", "lago di Bracciano", "near")],
            temporal=[t("action_date", "19 aprile")], policy={"confirm_before": ["rent"]})))
    add("mobility", "Metti a confronto quattro furgoni a noleggio a Rho per sabato, però non prenderne ancora nessuno.",
        c(g("compare", "furgone", "vehicle", [f("location", "Rho", "near")],
            temporal=[t("search_date", "sabato")], modifiers={"compare_count": 4},
            policy={"forbid": ["rent"]})))
    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V31 already frozen")
    rows = cases()
    if len(rows) != 36 or len({normalize_text(r["text"]) for r in rows}) != len(rows):
        raise RuntimeError(f"Expected 36 unique cases, got {len(rows)}")
    previous = [r.get("text") or r["messages"][1]["content"]
                for path in PRIOR for r in read_jsonl(path)]
    exact = {normalize_text(x) for x in previous}
    tokens = [token_set(x) for x in previous]
    maximum = 0.0
    for row in rows:
        key, tok = normalize_text(row["text"]), token_set(row["text"])
        if key in exact:
            raise RuntimeError(f"Exact prior overlap: {row['id']}")
        maximum = max(maximum, *(len(tok & other) / len(tok | other)
                                 for other in tokens if other))
    if maximum >= .8:
        raise RuntimeError(f"Near prior overlap: {maximum:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V31", "schema_version": "1.3", "cases": len(rows),
                "families": dict(Counter(r["family"] for r in rows)),
                "panel_sha256": sha256(panel), "generator_sha256": sha256(Path(__file__)),
                "helper_sha256": sha256(ROOT / "benchmarks/build_v28_blind.py"),
                "max_prior_jaccard": round(maximum, 5),
                "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR},
                "limitations": ["Agent-authored panel; independent human review pending",
                                "Not a random production sample"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"cases": len(rows), "max_prior_jaccard": maximum,
                      "panel_sha256": manifest["panel_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
