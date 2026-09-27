"""Freeze independent V30 tasks before changing ontology, Guard or training data.

Requests and gold are individually authored here; no prior benchmark output is
used as a label. This remains an agent-authored panel pending human review.
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

DEST = ROOT / "benchmarks/v30"
PRIOR = [ROOT / p for p in (
    "benchmarks/v20/hard_dev.jsonl", "benchmarks/v20/holdout.jsonl",
    "data/v20_quality/dev.jsonl", "benchmarks/release_holdout_v1.jsonl",
    "benchmarks/v23/transfer.jsonl", "benchmarks/v24/safety.jsonl",
    "benchmarks/v25/blind.jsonl", "benchmarks/v26/blind.jsonl",
    "benchmarks/v27/fresh.jsonl", "benchmarks/v28/blind.jsonl",
    "benchmarks/v29/blind.jsonl", "data/v24_9b/train.jsonl",
    "data/v24_9b/valid.jsonl", "data/v25_9b/train.jsonl",
    "data/v25_9b/valid.jsonl", "data/v26_9b/train.jsonl",
    "data/v26_9b/valid.jsonl",
)]


def cases():
    rows = []

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [x.as_dict() for x in issues]))
        rows.append({"id": f"v30-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold})

    add("reported_command", "Sto scrivendo un verbale: il cliente ha pronunciato 'comprate il biglietto'. Riporta soltanto la frase, non eseguire ordini.", c())
    add("reported_command", "Nel romanzo il protagonista grida 'cancella tutte le visite'. Ti sto raccontando la scena, non dandoti un incarico.", c())
    add("reported_command", "La frase 'prenota un taxi per me' è un esempio grammaticale. Voglio solo discuterne il significato.", c())
    add("reported_command", "La mail ricevuta dice 'manda il contratto'. Archiviala come informazione, non spedire documenti.", c())

    add("message_time", "Devo avvisare Federica che 'la riunione inizia alle 17'. Prepara l'invio, ma mostrami prima il testo.",
        c(g("contact", "Federica", "person", [f("message", "la riunione inizia alle 17")],
            policy={"confirm_before": ["contact"]})))
    add("message_time", "Spedisci a Mauro alle 19 il messaggio 'Il corriere passerà alle 21'; attendi il mio consenso prima di inviarlo.",
        c(g("contact", "Mauro", "person", [f("message", "Il corriere passerà alle 21")],
            temporal=[t("action_time", "19:00")], policy={"confirm_before": ["contact"]})))
    add("message_time", "Per dopodomani alle 08 imposta un promemoria con testo 'pagare la quota prima delle 16'.",
        c(g("notify", "promemoria", "reminder", [f("message", "pagare la quota prima delle 16")],
            temporal=[t("notification_date", "dopodomani"), t("notification_time", "08:00")])))
    add("message_time", "Scrivi a Paolo 'L'aereo atterra alle 22', ma non ti ho indicato quando mandare il messaggio.",
        c(g("contact", "Paolo", "person", [f("message", "L'aereo atterra alle 22")])))

    add("travel_roles", "Vorrei arrivare in treno a Bolzano entro le 10:30 partendo da Bressanone; guarda le corse di martedì.",
        c(g("find", "treno", "transport", [f("origin", "Bressanone"), f("destination", "Bolzano")],
            temporal=[t("search_date", "martedì"), t("arrival_time", "10:30", "before")])))
    add("travel_roles", "Dal porto di Livorno verso Pisa, cerca un autobus che parta non prima delle 18:15.",
        c(g("find", "autobus", "transport", [f("origin", "porto di Livorno"), f("destination", "Pisa")],
            temporal=[t("departure_time", "18:15", "after")])))
    add("travel_roles", "Per sabato mi serve un treno da Como a Lecco che arrivi prima delle 12 e costi meno di 20 euro.",
        c(g("find", "treno", "transport", [f("origin", "Como"), f("destination", "Lecco"),
            f("price", 20, "lt", currency="EUR")], temporal=[t("search_date", "sabato"),
            t("arrival_time", "12:00", "before")])))
    add("travel_roles", "Controlla i collegamenti ferroviari da Cagliari a Oristano con partenza dopo le 07:45 e arrivo entro le 09:30.",
        c(g("find", "treno", "transport", [f("origin", "Cagliari"), f("destination", "Oristano")],
            temporal=[t("departure_time", "07:45", "after"), t("arrival_time", "09:30", "before")])))

    add("air_travel", "Per il 3 dicembre mostrami soltanto voli senza scalo fra Napoli e Madrid, senza fare prenotazioni.",
        c(g("find", "volo", "transport", [f("origin", "Napoli"), f("destination", "Madrid"), f("stops", 0)],
            temporal=[t("search_date", "3 dicembre")], policy={"forbid": ["book"]})))
    add("air_travel", "Prenota un volo diretto da Torino a Berlino il 9 gennaio solo dopo che l'avrò approvato.",
        c(g("book", "volo", "transport", [f("origin", "Torino"), f("destination", "Berlino"), f("stops", 0)],
            temporal=[t("action_date", "9 gennaio")], policy={"confirm_before": ["book"]})))
    add("air_travel", "Vorrei prenotare il volo per Amsterdam il 2 febbraio, ma non ho ancora deciso da quale città partire.",
        c(g("book", "volo", "transport", [f("destination", "Amsterdam")],
            temporal=[t("action_date", "2 febbraio")], missing=[missing("origin")])))

    add("lodging_policy", "Fai vedere due pensioni con colazione inclusa nei dintorni di Arezzo per venerdì; non riservare camere.",
        c(g("find", "pensione", "accommodation", [f("location", "Arezzo", "near"), f("amenity", "colazione inclusa")],
            temporal=[t("search_date", "venerdì")], modifiers={"limit": 2}, policy={"forbid": ["book"]})))
    add("lodging_policy", "Metti a confronto quattro ostelli a Lucca dal più economico al più caro, senza bloccarne nessuno.",
        c(g("compare", "ostello", "accommodation", [f("location", "Lucca", "near")],
            modifiers={"compare_count": 4, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["book"]})))
    add("lodging_policy", "Trovami una camera a Faenza. Se ce n'è una adatta, chiedimi prima se vuoi prenotarla; per ora cerca soltanto.",
        c(g("find", "camera", "accommodation", [f("location", "Faenza", "near")],
            policy={"forbid": ["book"]})))

    add("dependent", "Cerca un agriturismo a Grosseto per sabato, poi riserva la terza soluzione solo quando ti darò il permesso.",
        c(g("find", "agriturismo", "accommodation", [f("location", "Grosseto", "near")],
            temporal=[t("search_date", "sabato")], idx=1),
          g("book", "agriturismo", "accommodation", [selection(ordinal=3)], idx=2,
            temporal=[t("action_date", "sabato")], depends_on=["g1"], policy={"confirm_before": ["book"]})))
    add("dependent", "Trova una stampante laser a Ravenna e, dopo avermi fatto scegliere la prima proposta, comprala con il mio consenso.",
        c(g("find", "stampante laser", "product", [f("location", "Ravenna", "near")], idx=1),
          g("buy", "stampante laser", "product", [selection(ordinal=1)], idx=2,
            depends_on=["g1"], policy={"confirm_before": ["buy"]})))
    add("dependent", "Seleziona due pizzerie a Empoli e prenota la seconda dopo il mio sì; ancora non ho stabilito giorno, ora e numero di persone.",
        c(g("find", "pizzeria", "place", [f("location", "Empoli", "near")], modifiers={"limit": 2}, idx=1),
          g("book", "pizzeria", "place", [selection(ordinal=2)], idx=2, depends_on=["g1"],
            missing=[missing("date"), missing("time"), missing("guests")],
            policy={"confirm_before": ["book"]})))

    add("fallback", "Guarda se c'è un pullman per Vercelli. Se la ricerca non restituisce corse, prova con i treni.",
        c(g("find", "pullman", "transport", [f("destination", "Vercelli")], idx=1),
          g("find", "treno", "transport", [f("destination", "Vercelli")], idx=2,
            depends_on=["g1"], condition=fallback())))
    add("fallback", "Verifica la disponibilità di un ostello a Imola per giovedì; se non ne compare nessuno, cerca un B&B.",
        c(g("find", "ostello", "accommodation", [f("location", "Imola", "near")],
            temporal=[t("search_date", "giovedì")], idx=1),
          g("find", "B&B", "accommodation", [f("location", "Imola", "near")],
            temporal=[t("search_date", "giovedì")], idx=2, depends_on=["g1"], condition=fallback())))
    add("independent", "Mi servono sia un benzinaio a Frosinone sia una farmacia a Ceccano; sono due ricerche separate.",
        c(g("find", "benzinaio", "place", [f("location", "Frosinone", "near")], idx=1),
          g("find", "farmacia", "place", [f("location", "Ceccano", "near")], idx=2)))

    add("local_context", "Non posso usare l'auto: cerca un bar raggiungibile a piedi dal museo a Urbino, aperto adesso.",
        c(g("find", "bar", "place", [f("location", "museo di Urbino", "near"), f("access", "a piedi"),
            f("open_now", True)]), context=[ctx("has_car", False)]))
    add("local_context", "Siamo in cinque e ci troviamo a Macerata: cerca una trattoria vegetariana aperta ora.",
        c(g("find", "trattoria vegetariana", "place", [f("location", "Macerata", "near"),
            f("open_now", True)]), context=[ctx("party_size", 5)]))
    add("local_context", "Dal teatro di Pistoia cercami tre caffè che si possano raggiungere camminando.",
        c(g("find", "caffè", "place", [f("location", "teatro di Pistoia", "near"),
            f("access", "a piedi")], modifiers={"limit": 3})))

    add("reminder", "Imposta per lunedì alle 06:30 un avviso per portare i documenti all'ufficio postale.",
        c(g("notify", "promemoria", "reminder", [f("message", "portare i documenti all'ufficio postale")],
            temporal=[t("notification_date", "lunedì"), t("notification_time", "06:30")])))
    add("reminder", "Fra tre giorni ricordami quell'impegno di cui parlavamo prima; adesso non ricordo quale fosse.",
        c(g("notify", "promemoria", "reminder", [], temporal=[t("notification_date", "fra tre giorni")],
            missing=[missing("referent", "search", "unresolved_reference")])))
    add("reminder", "Domattina ricordamelo: devo chiamare il commercialista, ma non so ancora a che ora.",
        c(g("notify", "promemoria", "reminder", [f("message", "chiamare il commercialista")],
            temporal=[t("notification_time", "domattina")])))

    add("appointment", "Prenota una visita oculistica a Teramo il 22 dicembre alle 11:45 e chiedimi il via libera prima di confermare.",
        c(g("book", "visita oculistica", "appointment", [f("location", "Teramo", "near")],
            temporal=[t("action_date", "22 dicembre"), t("action_time", "11:45")],
            policy={"confirm_before": ["book"]})))
    add("appointment", "Annulla il colloquio di fisioterapia fissato per il 7 gennaio a Novara, ma mostrami prima quale appuntamento hai trovato.",
        c(g("cancel", "colloquio di fisioterapia", "appointment", [f("original_date", "7 gennaio"),
            f("location", "Novara", "near")], policy={"confirm_before": ["cancel"]})))
    add("appointment", "Sposta il controllo dal dentista del 4 febbraio al 12 febbraio. Non modificare l'agenda finché non confermo.",
        c(g("reschedule", "controllo dal dentista", "appointment", [f("original_date", "4 febbraio")],
            temporal=[t("action_date", "12 febbraio")], policy={"confirm_before": ["reschedule"]})))

    add("commerce", "Ordina un panino vegetariano con consegna in via Verdi a La Spezia; prima di pagare chiedimi conferma.",
        c(g("order", "panino vegetariano", "food", [f("delivery_location", "via Verdi a La Spezia")],
            policy={"confirm_before": ["order"]})))
    add("commerce", "Fammi vedere i prezzi delle focacce a Reggio Emilia dal più basso al più alto, senza effettuare acquisti.",
        c(g("find", "focaccia", "food", [f("location", "Reggio Emilia", "near")],
            modifiers={"sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["buy", "order"]})))
    add("commerce", "Confronta cinque monitor sotto i 300 euro e ordina i risultati per costo, senza comprare nulla.",
        c(g("compare", "monitor", "product", [f("price", 300, "lt", currency="EUR")],
            modifiers={"compare_count": 5, "sort": {"field": "price", "direction": "asc"}},
            policy={"forbid": ["buy"]})))

    add("service", "Cerca un fabbro a Pordenone disponibile oggi, ma non ingaggiare nessuno per ora.",
        c(g("find", "fabbro", "professional_service", [f("location", "Pordenone", "near")],
            temporal=[t("search_date", "oggi")], policy={"forbid": ["hire"]})))
    add("service", "Vorrei assumere un elettricista per un lavoro a Lodi; mostrami il professionista e chiedimi l'ok prima di impegnarmi.",
        c(g("hire", "elettricista", "professional_service", [f("location", "Lodi", "near")],
            missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("service", "Trova un tecnico della caldaia a Belluno e chiedi al primo se può venire venerdì: questo è il testo da inviargli.",
        c(g("find", "tecnico della caldaia", "professional_service", [f("location", "Belluno", "near")], idx=1),
          g("contact", "tecnico della caldaia", "professional_service", [selection(),
            f("message", "Puoi venire venerdì?")], idx=2, depends_on=["g1"])))

    add("vehicle", "Cercami un'auto elettrica da noleggiare a Mantova domenica, senza riservarla.",
        c(g("find", "auto elettrica", "vehicle", [f("location", "Mantova", "near")],
            temporal=[t("search_date", "domenica")], policy={"forbid": ["rent"]})))
    add("vehicle", "Prendi a noleggio uno scooter a Chieti per il 16 gennaio, ma aspetta la mia autorizzazione.",
        c(g("rent", "scooter", "vehicle", [f("location", "Chieti", "near")],
            temporal=[t("action_date", "16 gennaio")], policy={"confirm_before": ["rent"]})))
    add("vehicle", "Metti a confronto tre camper disponibili a Grosseto per sabato senza prenderne ancora nessuno.",
        c(g("compare", "camper", "vehicle", [f("location", "Grosseto", "near")],
            temporal=[t("search_date", "sabato")], modifiers={"compare_count": 3},
            policy={"forbid": ["rent"]})))
    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V30 already frozen; do not regenerate")
    rows = cases()
    if len(rows) != 42 or len({normalize_text(r["text"]) for r in rows}) != len(rows):
        raise RuntimeError(f"Expected 42 unique cases, got {len(rows)}")
    previous = [r.get("text") or r["messages"][1]["content"]
                for path in PRIOR for r in read_jsonl(path)]
    exact = {normalize_text(x) for x in previous}
    tokens = [token_set(x) for x in previous]
    maximum = 0.0
    for row in rows:
        key, tok = normalize_text(row["text"]), token_set(row["text"])
        if key in exact:
            raise RuntimeError(f"Exact overlap: {row['id']}")
        maximum = max(maximum, *(len(tok & other) / len(tok | other)
                                 for other in tokens if other))
    if maximum >= .8:
        raise RuntimeError(f"Near overlap with prior data: {maximum:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V30", "schema_version": "1.3", "cases": len(rows),
                "families": dict(Counter(r["family"] for r in rows)),
                "panel_sha256": sha256(panel), "generator_sha256": sha256(Path(__file__)),
                "helper_sha256": sha256(ROOT / "benchmarks/build_v28_blind.py"),
                "max_prior_jaccard": round(maximum, 5),
                "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR},
                "limitations": ["Agent-authored targeted cases; independent human review pending",
                                "Not a random production sample"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"cases": len(rows), "max_prior_jaccard": maximum,
                      "panel_sha256": manifest["panel_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
