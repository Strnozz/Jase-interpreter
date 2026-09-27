"""Freeze diverse, realistic GoalContract 1.3 requests before new fixes.

These manually curated pairs are agent-authored; independent human review is
still needed. Do not regenerate the frozen output or train on these requests.
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

DEST = ROOT / "benchmarks/v28"
PRIOR = [ROOT / name for name in (
    "benchmarks/v20/hard_dev.jsonl", "benchmarks/v20/holdout.jsonl",
    "benchmarks/v23/transfer.jsonl", "benchmarks/v24/safety.jsonl",
    "benchmarks/v25/blind.jsonl", "benchmarks/v26/blind.jsonl",
    "benchmarks/v27/fresh.jsonl", "data/v25_9b/train.jsonl",
    "data/v25_9b/valid.jsonl",
)]


def f(field, value, op="eq", **extra):
    return {"field": field, "op": op, "value": value, "strength": "hard", **extra}


def t(role, value, op="eq"):
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def g(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action,
            "target": {"name": name, "type": typ}, "facts": facts or [], **extra}


def c(*goals, context=None):
    out = {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
           "goals": list(goals)}
    if context:
        out["context"] = context
    return out


def ctx(field, value):
    return {"scope": "user", "field": field, "op": "eq", "value": value}


def missing(field, blocks="action", reason="unspecified"):
    return {"field": field, "blocks": blocks, "reason": reason}


def selection(goal="g1", ordinal=1):
    return {"field": "selection", "op": "eq", "strength": "hard",
            "value_ref": {"goal": goal, "field": "result", "ordinal": ordinal}}


def fallback(prior="g1"):
    return {"goal": prior, "test": {"field": "results_count", "op": "eq", "value": 0}}


def cases():
    rows = []

    def add(family, text, gold):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [x.as_dict() for x in issues]))
        rows.append({"id": f"v28-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold})

    # Quoted commands in a discussion are distinct from quoted message content.
    add("quoted_non_action", "Nel copione compare la battuta 'prenotami un taxi per l'aeroporto'. Mi dici se suona naturale?", c())
    add("quoted_non_action", "Sto analizzando un esempio di comando: 'compra due biglietti per il concerto'. Non eseguire nulla.", c())
    add("quoted_non_action", "Per la lezione di italiano, spiega la frase 'ordina una pizza stasera'.", c())
    add("quoted_non_action", "La nota che ho copiato dice 'annulla la visita'. Voglio solo archiviarla, non agire.", c())
    add("quoted_contact", "Scrivi a Chiara: 'Sono davanti alla stazione di Parma'. Chiedimi conferma prima di inviare.",
        c(g("contact", "Chiara", "person", [f("message", "Sono davanti alla stazione di Parma")],
            policy={"confirm_before": ["contact"]})))
    add("quoted_contact", "Manda a Marco il messaggio 'La riunione inizia alle 16', ma mostramelo prima.",
        c(g("contact", "Marco", "person", [f("message", "La riunione inizia alle 16")],
            policy={"confirm_before": ["contact"]})))

    # Route endpoints and time roles affect actual searches.
    add("arrival_route", "Parto da Parma: trovami un treno per Milano Centrale che arrivi entro le 10:15.",
        c(g("find", "treno", "transport", [f("origin", "Parma"), f("destination", "Milano Centrale")],
            temporal=[t("arrival_time", "10:15", "before")])))
    add("arrival_route", "Da Pisa devo essere a Roma Termini prima delle 13:00; cerca i treni adatti.",
        c(g("find", "treno", "transport", [f("origin", "Pisa"), f("destination", "Roma Termini")],
            temporal=[t("arrival_time", "13:00", "before")])))
    add("arrival_route", "Per domani cerca un autobus da Bologna a Ferrara con arrivo non oltre le 18.",
        c(g("find", "autobus", "transport", [f("origin", "Bologna"), f("destination", "Ferrara")],
            temporal=[t("search_date", "domani"), t("arrival_time", "18:00", "before")])))
    add("arrival_route", "Vorrei partire da Torino e arrivare a Genova in treno entro le 09:30 di venerd\u00ec.",
        c(g("find", "treno", "transport", [f("origin", "Torino"), f("destination", "Genova")],
            temporal=[t("search_date", "venerd\u00ec"), t("arrival_time", "09:30", "before")])))
    add("direct_flight", "Mostrami voli senza scalo da Roma a Madrid il 5 novembre; non comprare nulla.",
        c(g("find", "volo", "transport", [f("origin", "Roma"), f("destination", "Madrid"), f("stops", 0)],
            temporal=[t("search_date", "5 novembre")], policy={"forbid": ["buy"]})))
    add("direct_flight", "Per sabato cerca un volo diretto Venezia-Londra, ma lascia a me la prenotazione.",
        c(g("find", "volo", "transport", [f("origin", "Venezia"), f("destination", "Londra"), f("stops", 0)],
            temporal=[t("search_date", "sabato")], policy={"forbid": ["book"]})))
    add("direct_flight", "Prenota, dopo il mio via libera, un volo senza scali da Milano a Berlino per il 3 ottobre.",
        c(g("book", "volo", "transport", [f("origin", "Milano"), f("destination", "Berlino"), f("stops", 0)],
            temporal=[t("action_date", "3 ottobre")], policy={"confirm_before": ["book"]})))
    add("direct_flight", "Da Napoli a Parigi cerco un volo diretto per mercoled\u00ec; mostra le opzioni, non prenotare.",
        c(g("find", "volo", "transport", [f("origin", "Napoli"), f("destination", "Parigi"), f("stops", 0)],
            temporal=[t("search_date", "mercoled\u00ec")], policy={"forbid": ["book"]})))

    # Explicit prohibitions and result cardinality must survive interpretation.
    add("no_booking", "Cerca tre alberghi a Firenze vicino alla stazione, ma non riservare camere.",
        c(g("find", "albergo", "accommodation", [f("location", "stazione di Firenze", "near")],
            modifiers={"limit": 3}, policy={"forbid": ["book"]})))
    add("no_booking", "Fammi vedere due B&B a Verona per sabato senza effettuare prenotazioni.",
        c(g("find", "B&B", "accommodation", [f("location", "Verona", "near")],
            temporal=[t("search_date", "sabato")], modifiers={"limit": 2}, policy={"forbid": ["book"]})))
    add("no_booking", "Per il 4 ottobre confronta esattamente due camere a Padova; niente prenotazioni.",
        c(g("compare", "camera", "accommodation", [f("location", "Padova", "near")],
            temporal=[t("search_date", "4 ottobre")], modifiers={"compare_count": 2},
            policy={"forbid": ["book"]})))
    add("no_booking", "Vorrei soltanto le offerte di hotel a Siena: non bloccare ancora nessuna stanza.",
        c(g("find", "hotel", "accommodation", [f("location", "Siena", "near")],
            policy={"forbid": ["book"]})))

    # Later action on search results must be a separate dependent goal.
    add("find_then_book", "Trova un albergo a Napoli per il 7 ottobre; poi prenota la prima opzione solo se confermo.",
        c(g("find", "albergo", "accommodation", [f("location", "Napoli", "near")],
            temporal=[t("search_date", "7 ottobre")]),
          g("book", "albergo", "accommodation", [selection()], idx=2, depends_on=["g1"],
            policy={"confirm_before": ["book"]})))
    add("find_then_book", "Cerca tre hotel a Ravenna ordinati dal meno caro; prenota il primo solo dopo la mia conferma.",
        c(g("find", "hotel", "accommodation", [f("location", "Ravenna", "near")],
            modifiers={"limit": 3, "sort": {"field": "price", "direction": "asc"}}),
          g("book", "hotel", "accommodation", [selection()], idx=2, depends_on=["g1"],
            policy={"confirm_before": ["book"]})))
    add("find_then_book", "Individua un volo da Bari a Parigi per il 10 novembre, quindi prenota il primo previa approvazione.",
        c(g("find", "volo", "transport", [f("origin", "Bari"), f("destination", "Parigi")],
            temporal=[t("search_date", "10 novembre")]),
          g("book", "volo", "transport", [f("origin", "Bari"), f("destination", "Parigi"), selection()],
            idx=2, depends_on=["g1"], temporal=[t("action_date", "10 novembre")],
            policy={"confirm_before": ["book"]})))
    add("find_then_book", "Prima mostrami i ristoranti giapponesi a Bologna; prenota il secondo solo dopo che ti dico di procedere.",
        c(g("find", "ristorante giapponese", "place", [f("location", "Bologna", "near")]),
          g("book", "ristorante giapponese", "place", [selection(ordinal=2)], idx=2,
            depends_on=["g1"], missing=[missing("date"), missing("time"), missing("guests")],
            policy={"confirm_before": ["book"]})))

    # Search constraints versus context; no unsupported spatial block.
    add("open_and_access", "Mi serve una farmacia aperta adesso nel centro di Piacenza.",
        c(g("find", "farmacia", "place", [f("location", "centro di Piacenza", "near"),
            f("open_now", True)])))
    add("open_and_access", "Siamo senza auto: trova a Lucca una pizzeria raggiungibile a piedi.",
        c(g("find", "pizzeria", "place", [f("location", "Lucca", "near"), f("access", "a piedi")]),
          context=[ctx("has_car", False)]))
    add("open_and_access", "Per me e mia figlia cerca una libreria aperta ora vicino a Piazza Maggiore a Bologna.",
        c(g("find", "libreria", "place", [f("location", "Piazza Maggiore a Bologna", "near"),
            f("open_now", True)]), context=[ctx("party_size", 2)]))
    add("open_and_access", "Non posso usare l'auto: mostrami un ristorante vicino alla stazione di Trieste dove arrivare a piedi.",
        c(g("find", "ristorante", "place", [f("location", "stazione di Trieste", "near"),
            f("access", "a piedi")]), context=[ctx("has_car", False)]))

    # Outcome-dependent fallback and unresolved reminders.
    add("conditional_fallback", "Prima cerca un treno da Modena a Bologna. Solo se non ce ne sono, cerca un autobus sulla stessa tratta.",
        c(g("find", "treno", "transport", [f("origin", "Modena"), f("destination", "Bologna")]),
          g("find", "autobus", "transport", [f("origin", "Modena"), f("destination", "Bologna")],
            idx=2, depends_on=["g1"], condition=fallback())))
    add("conditional_fallback", "Trova un hotel a Bergamo; se la ricerca restituisce zero risultati, prova con un B&B.",
        c(g("find", "hotel", "accommodation", [f("location", "Bergamo", "near")]),
          g("find", "B&B", "accommodation", [f("location", "Bergamo", "near")],
            idx=2, depends_on=["g1"], condition=fallback())))
    add("conditional_fallback", "Cerca un idraulico disponibile oggi a Como, e solo in assenza di risultati cerca un pronto intervento idraulico.",
        c(g("find", "idraulico", "professional_service", [f("location", "Como", "near")],
            temporal=[t("search_date", "oggi")]),
          g("find", "pronto intervento idraulico", "professional_service", [f("location", "Como", "near")],
            idx=2, depends_on=["g1"], temporal=[t("search_date", "oggi")], condition=fallback())))
    add("conditional_fallback", "Verifica gli autobus per l'aeroporto di Bologna; se non trovi corse, mostrami i treni.",
        c(g("find", "autobus", "transport", [f("destination", "aeroporto di Bologna")]),
          g("find", "treno", "transport", [f("destination", "aeroporto di Bologna")],
            idx=2, depends_on=["g1"], condition=fallback())))
    add("reminder", "Domani alle 08:30 ricordami di portare il referto in ospedale.",
        c(g("notify", "promemoria", "reminder", [f("message", "portare il referto in ospedale")],
            temporal=[t("notification_date", "domani"), t("notification_time", "08:30")])))
    add("reminder", "Venerd\u00ec alle 17 fammi un promemoria: ritirare le chiavi da Laura.",
        c(g("notify", "promemoria", "reminder", [f("message", "ritirare le chiavi da Laura")],
            temporal=[t("notification_date", "venerd\u00ec"), t("notification_time", "17:00")])))
    add("reminder", "Domani ricordami quella cosa di cui parlavamo ieri: non ricordo quale.",
        c(g("notify", "promemoria", "reminder", temporal=[t("notification_date", "domani")],
            missing=[missing("referent", "search", "unresolved_reference")])))

    # Appointment and rental operations remain distinct from searches.
    add("appointment", "Sposta la visita dermatologica fissata il 2 ottobre al 9 ottobre, ma chiedimi conferma.",
        c(g("reschedule", "visita dermatologica", "appointment", [f("original_date", "2 ottobre")],
            temporal=[t("action_date", "9 ottobre")], policy={"confirm_before": ["reschedule"]})))
    add("appointment", "Annulla la seduta di fisioterapia del 6 novembre a Reggio Emilia.",
        c(g("cancel", "seduta di fisioterapia", "appointment", [f("original_date", "6 novembre"),
            f("location", "Reggio Emilia", "near")])) )
    add("appointment", "Fissa una visita oculistica a Udine per il 12 novembre alle 11, previa mia conferma.",
        c(g("book", "visita oculistica", "appointment", [f("location", "Udine", "near")],
            temporal=[t("action_date", "12 novembre"), t("action_time", "11:00")],
            policy={"confirm_before": ["book"]})))
    add("vehicle", "Fammi vedere le biciclette a noleggio a Mantova per sabato; per ora non affittare nulla.",
        c(g("find", "bicicletta", "vehicle", [f("location", "Mantova", "near")],
            temporal=[t("search_date", "sabato")], policy={"forbid": ["rent"]})))
    add("vehicle", "Noleggia uno scooter a Pisa per luned\u00ec, ma sottoponimi la scelta prima di confermare.",
        c(g("rent", "scooter", "vehicle", [f("location", "Pisa", "near")],
            temporal=[t("action_date", "luned\u00ec")], policy={"confirm_before": ["rent"]})))

    # Ranking and counts are modifiers, not provider actions or item quantities.
    add("ranking_count", "Confronta esattamente tre computer portatili sotto 900 euro.",
        c(g("compare", "computer portatile", "product", [f("price", 900, "lt", currency="EUR")],
            modifiers={"compare_count": 3})))
    add("ranking_count", "Mostrami al massimo due ristoranti thailandesi a Ravenna.",
        c(g("find", "ristorante thailandese", "place", [f("location", "Ravenna", "near")],
            modifiers={"limit": 2})))
    add("ranking_count", "Metti i treni da Firenze a Bologna in ordine di prezzo, dal pi\u00f9 economico.",
        c(g("find", "treno", "transport", [f("origin", "Firenze"), f("destination", "Bologna")],
            modifiers={"sort": {"field": "price", "direction": "asc"}})))
    add("ranking_count", "Per arrivare a Milano entro mezzogiorno, cerca i treni da Brescia e ordina i risultati per prezzo crescente.",
        c(g("find", "treno", "transport", [f("origin", "Brescia"), f("destination", "Milano")],
            temporal=[t("arrival_time", "12:00", "before")],
            modifiers={"sort": {"field": "price", "direction": "asc"}})))
    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V28 already frozen; do not regenerate")
    rows = cases()
    if len(rows) != 42 or len({normalize_text(r["text"]) for r in rows}) != 42:
        raise RuntimeError("Expected 42 distinct requests")
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
            raise RuntimeError(f"Exact overlap: {row['id']}")
        maximum = max(maximum, *(len(tok & prior) / len(tok | prior)
                                 for prior in tokens if prior))
    if maximum >= .8:
        raise RuntimeError(f"Near overlap: {maximum:.3f}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V28", "schema_version": "1.3", "cases": len(rows),
                "families": dict(Counter(x["family"] for x in rows)),
                "panel_sha256": sha256(panel), "generator_sha256": sha256(Path(__file__)),
                "max_prior_jaccard": round(maximum, 5),
                "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR},
                "limitations": ["Agent-authored targeted panel, not a production sample",
                                "Independent human gold review pending"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"cases": len(rows), "panel_sha256": manifest["panel_sha256"],
                      "max_prior_jaccard": maximum}, ensure_ascii=False))


if __name__ == "__main__":
    main()
