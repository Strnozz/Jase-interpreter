"""Freeze a conversational V32 panel before Planner handoff and Guard changes.

Agent-authored gold is checked against GoalContract 1.3 and prior corpora, but
is not a substitute for independent human adjudication.
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

DEST = ROOT / "benchmarks/v32"
PRIOR = [ROOT / p for p in (
    "benchmarks/v20/hard_dev.jsonl", "benchmarks/v20/holdout.jsonl",
    "data/v20_quality/dev.jsonl", "benchmarks/release_holdout_v1.jsonl",
    "benchmarks/v23/transfer.jsonl", "benchmarks/v24/safety.jsonl",
    "benchmarks/v25/blind.jsonl", "benchmarks/v26/blind.jsonl",
    "benchmarks/v27/fresh.jsonl", "benchmarks/v28/blind.jsonl",
    "benchmarks/v29/blind.jsonl", "benchmarks/v30/blind.jsonl",
    "benchmarks/v31/blind.jsonl", "data/v24_9b/train.jsonl",
    "data/v24_9b/valid.jsonl", "data/v25_9b/train.jsonl",
    "data/v25_9b/valid.jsonl", "data/v26_9b/train.jsonl",
    "data/v26_9b/valid.jsonl",
)]


def cases():
    rows = []

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [i.as_dict() for i in issues]))
        rows.append({"id": f"v32-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold})

    add("narrative", "Sto leggendo un dialogo: 'prenota il ristorante ora', dice un personaggio. Non è una richiesta rivolta a te.", c())
    add("narrative", "La frase 'noleggia una macchina' è nel compito di italiano. Correggi la frase e basta, senza noleggi.", c())
    add("narrative", "Ho copiato da un cartello 'assumi un tecnico'; voglio solo capire che cosa vuol dire.", c())

    add("messages", "Scrivi a Ivo 'Il sopralluogo è alle 16:30'; prima di mandarlo voglio leggere la bozza.",
        c(g("contact", "Ivo", "person", [f("message", "Il sopralluogo è alle 16:30")],
            policy={"confirm_before": ["contact"]})))
    add("messages", "Manda a Daria alle 10:15 il testo 'Passo da te alle 18'; aspetta che ti dica di procedere.",
        c(g("contact", "Daria", "person", [f("message", "Passo da te alle 18")],
            temporal=[t("action_time", "10:15")], policy={"confirm_before": ["contact"]})))
    add("messages", "Alle 07:50 di domani ricordami: 'consegnare le chiavi entro le 12'.",
        c(g("notify", "promemoria", "reminder", [f("message", "consegnare le chiavi entro le 12")],
            temporal=[t("notification_date", "domani"), t("notification_time", "07:50")])))

    add("transport", "Ho un appuntamento alle 11 a Vigevano: da Mortara cercami un treno che arrivi prima delle 10:30.",
        c(g("find", "treno", "transport", [f("origin", "Mortara"), f("destination", "Vigevano")],
            temporal=[t("arrival_time", "10:30", "before")])))
    add("transport", "Trova un autobus da Empoli a Pontedera con partenza successiva alle 13:05 e arrivo entro le 14:20.",
        c(g("find", "autobus", "transport", [f("origin", "Empoli"), f("destination", "Pontedera")],
            temporal=[t("departure_time", "13:05", "after"), t("arrival_time", "14:20", "before")])))
    add("transport", "Martedì controlla i treni regionali fra Pescara e Chieti, ma mostrami solo le partenze dopo le 17:40.",
        c(g("find", "treno regionale", "transport", [f("origin", "Pescara"), f("destination", "Chieti")],
            temporal=[t("search_date", "martedì"), t("departure_time", "17:40", "after")])))

    add("flights", "Il 28 aprile voglio vedere voli diretti da Bologna a Praga. Nessun acquisto né prenotazione.",
        c(g("find", "volo", "transport", [f("origin", "Bologna"), f("destination", "Praga"), f("stops", 0)],
            temporal=[t("search_date", "28 aprile")], policy={"forbid": ["buy", "book"]})))
    add("flights", "Prenota il volo da Bari per Oslo il 18 maggio, ma solo quando avrò detto sì.",
        c(g("book", "volo", "transport", [f("origin", "Bari"), f("destination", "Oslo")],
            temporal=[t("action_date", "18 maggio")], policy={"confirm_before": ["book"]})))
    add("flights", "Per il 2 giugno prenota un volo diretto verso Marsiglia; devi prima chiedermi da dove voglio partire.",
        c(g("book", "volo", "transport", [f("destination", "Marsiglia"), f("stops", 0)],
            temporal=[t("action_date", "2 giugno")], missing=[missing("origin")])))

    add("lodging", "Mi bastano tre opzioni di ostello a Pavia con bagno privato, ordinate per prezzo; la camera non va bloccata.",
        c(g("find", "ostello", "accommodation", [f("location", "Pavia", "near"),
            f("amenity", "bagno privato")], modifiers={"limit": 3,
            "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["book"]})))
    add("lodging", "Confronta due case vacanza a Sanremo dal prezzo più basso, senza effettuare alcuna prenotazione.",
        c(g("compare", "casa vacanza", "accommodation", [f("location", "Sanremo", "near")],
            modifiers={"compare_count": 2, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["book"]})))
    add("lodging", "A Viareggio cerca un albergo disponibile sabato. Fermati prima di riservare: decido io dopo aver visto le opzioni.",
        c(g("find", "albergo", "accommodation", [f("location", "Viareggio", "near")],
            temporal=[t("search_date", "sabato")], policy={"forbid": ["book"]})))

    add("multi_step", "Trovami un hotel a Piombino; se il secondo mi piace lo prenoti, ma chiedimi l'ok e la data prima.",
        c(g("find", "hotel", "accommodation", [f("location", "Piombino", "near")], idx=1),
          g("book", "hotel", "accommodation", [selection(ordinal=2)], idx=2,
            depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]})))
    add("multi_step", "Mostra quattro ristoranti a Massa Marittima, poi prenota il primo solo se te lo autorizzo; mancano giorno, ora e coperti.",
        c(g("find", "ristorante", "place", [f("location", "Massa Marittima", "near")],
            modifiers={"limit": 4}, idx=1),
          g("book", "ristorante", "place", [selection(ordinal=1)], idx=2, depends_on=["g1"],
            missing=[missing("date"), missing("time"), missing("guests")],
            policy={"confirm_before": ["book"]})))
    add("multi_step", "Cerca un meccanico a Voghera e manda al secondo la domanda 'Puoi venire venerdì?'.",
        c(g("find", "meccanico", "professional_service", [f("location", "Voghera", "near")], idx=1),
          g("contact", "meccanico", "professional_service", [selection(ordinal=2),
            f("message", "Puoi venire venerdì?")], idx=2, depends_on=["g1"])))

    add("alternatives", "Prova a cercare un treno per Aosta domani; se non ne trovi, consulta gli autobus per la stessa destinazione.",
        c(g("find", "treno", "transport", [f("destination", "Aosta")],
            temporal=[t("search_date", "domani")], idx=1),
          g("find", "autobus", "transport", [f("destination", "Aosta")],
            temporal=[t("search_date", "domani")], idx=2, depends_on=["g1"], condition=fallback())))
    add("alternatives", "Se non c'è una pensione a Jesi per domenica, passa agli hotel; non mostrare entrambi se la pensione c'è.",
        c(g("find", "pensione", "accommodation", [f("location", "Jesi", "near")],
            temporal=[t("search_date", "domenica")], idx=1),
          g("find", "hotel", "accommodation", [f("location", "Jesi", "near")],
            temporal=[t("search_date", "domenica")], idx=2, depends_on=["g1"], condition=fallback())))
    add("alternatives", "Per oggi cerca sia un'edicola a Salò sia un bancomat a Desenzano: sono due ricerche indipendenti.",
        c(g("find", "edicola", "place", [f("location", "Salò", "near")],
            temporal=[t("search_date", "oggi")], idx=1),
          g("find", "bancomat", "place", [f("location", "Desenzano", "near")],
            temporal=[t("search_date", "oggi")], idx=2)))

    add("local", "Non possiedo un'auto: trova a Vercelli due farmacie aperte ora che si raggiungano a piedi dalla stazione.",
        c(g("find", "farmacia", "place", [f("location", "stazione di Vercelli", "near"),
            f("open_now", True), f("access", "a piedi")], modifiers={"limit": 2}),
          context=[ctx("has_car", False)]))
    add("local", "Per il nostro gruppo di quattro cerca una trattoria senza glutine aperta adesso a Casale Monferrato.",
        c(g("find", "trattoria", "place", [f("location", "Casale Monferrato", "near"),
            f("diet", "senza glutine"), f("open_now", True)]), context=[ctx("party_size", 4)]))
    add("local", "Dalla biblioteca a Foggia trovami un bar raggiungibile camminando, ma non chiedere se ha tavoli liberi.",
        c(g("find", "bar", "place", [f("location", "biblioteca di Foggia", "near"),
            f("access", "a piedi")])))

    add("actions", "Fissa una visita cardiologica a Sondrio il 4 giugno alle 10:20; mostrala prima di confermare la prenotazione.",
        c(g("book", "visita cardiologica", "appointment", [f("location", "Sondrio", "near")],
            temporal=[t("action_date", "4 giugno"), t("action_time", "10:20")],
            policy={"confirm_before": ["book"]})))
    add("actions", "Annulla la visita oculistica del 15 giugno a Rieti, ma soltanto dopo il mio consenso.",
        c(g("cancel", "visita oculistica", "appointment", [f("original_date", "15 giugno"),
            f("location", "Rieti", "near")], policy={"confirm_before": ["cancel"]})))
    add("actions", "Per mercoledì affitta una bici a Ragusa dopo che ti avrò dato il via libera.",
        c(g("rent", "bicicletta", "vehicle", [f("location", "Ragusa", "near")],
            temporal=[t("action_date", "mercoledì")], policy={"confirm_before": ["rent"]})))

    add("shopping_service", "Confronta quattro stampanti sotto i 180 euro; mettile in ordine di prezzo, ma non acquistarne.",
        c(g("compare", "stampante", "product", [f("price", 180, "lt", currency="EUR")],
            modifiers={"compare_count": 4, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["buy"]})))
    add("shopping_service", "Ingaggia un falegname a Volterra, previo mio assenso; il professionista va ancora scelto.",
        c(g("hire", "falegname", "professional_service", [f("location", "Volterra", "near")],
            missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("shopping_service", "Per lunedì mostrami i furgoni a noleggio a Varese; non prenderne ancora uno.",
        c(g("find", "furgone", "vehicle", [f("location", "Varese", "near")],
            temporal=[t("search_date", "lunedì")], policy={"forbid": ["rent"]})))
    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V32 already frozen")
    rows = cases()
    if len(rows) != 30 or len({normalize_text(r["text"]) for r in rows}) != len(rows):
        raise RuntimeError(f"Expected 30 unique cases, got {len(rows)}")
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
    manifest = {"version": "V32", "schema_version": "1.3", "cases": len(rows),
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
