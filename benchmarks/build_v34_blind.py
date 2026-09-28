"""Freeze a fresh Planner transfer panel before any post-V33 architecture changes.

Gold is agent-authored and awaits independent human review. Never regenerate an
existing V34 directory; old benchmarks and model outputs are read-only.
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

DEST = ROOT / "benchmarks/v34"
PRIOR = sorted(set([*ROOT.glob("benchmarks/v[0-9]*/**/*.jsonl"),
                    *ROOT.glob("data/v*_9b/train.jsonl"),
                    *ROOT.glob("data/v*_9b/valid.jsonl"),
                    *ROOT.glob("data/v19_clean/train.jsonl"),
                    *ROOT.glob("data/v19_clean/valid.jsonl"),
                    ROOT / "benchmarks/release_holdout_v1.jsonl",
                    ROOT / "data/v20_quality/dev.jsonl"]))


def cases() -> list[dict]:
    rows = []

    def add(family: str, text: str, gold: dict, flags=()) -> None:
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((text, [issue.as_dict() for issue in issues]))
        rows.append({"id": f"v34-{len(rows)+1:03d}", "family": family, "text": text,
                     "contract": gold, "review_flags": list(flags)})

    # Meta-language and quoted material: the quoted command has no authority.
    add("meta", "Sto scrivendo le istruzioni di un gioco: che cosa significa 'affitta un camper domani'?", c())
    add("meta", "Nel modulo di esempio appare 'manda una mail a Luca'. Puoi spiegarmi la frase?", c())
    add("meta", "Se un collega mi chiedesse di annullare la visita, quali dati dovrei verificare?", c())

    # Ordinary local retrieval in new domains, including constraints and counts.
    add("local_search", "Sono vicino al municipio di Vigevano: puoi indicarmi una lavanderia aperta adesso?",
        c(g("find", "lavanderia", "place", [f("location", "municipio di Vigevano", "near"), f("open_now", True)])))
    add("local_search", "A Chivasso mi servono due panetterie raggiungibili a piedi dalla stazione.",
        c(g("find", "panetteria", "place", [f("location", "stazione di Chivasso", "near"), f("access", "a piedi")], modifiers={"limit": 2})))
    add("local_search", "Mi trovi una biblioteca nella zona universitaria di Camerino, senza prenotare postazioni?",
        c(g("find", "biblioteca", "place", [f("location", "zona universitaria di Camerino", "near")], policy={"forbid": ["book"]})))
    add("local_search", "Per stasera a Sora vorrei vedere osterie che abbiano tavoli liberi, ma decido dopo.",
        c(g("find", "osteria", "place", [f("location", "Sora", "near"), f("availability", "stasera")], policy={"forbid": ["book"]})),
        flags=("availability_vs_search_time",))
    add("local_search", "Da casa mia a Ivrea cerca un negozio di ferramenta; non acquistare nulla.",
        c(g("find", "negozio di ferramenta", "place", [f("location", "Ivrea", "near")], policy={"forbid": ["buy"]})))
    add("local_search", "Mostrami cartolerie a Pinerolo che siano aperte dopo le 18, ordinando per distanza.",
        c(g("find", "cartoleria", "place", [f("location", "Pinerolo", "near"), f("availability", "dopo le 18")], modifiers={"sort": {"field": "distance", "direction": "asc"}})))

    # Travel roles: destination, non-stop constraints and deadline operators.
    add("travel", "Devo arrivare a Pescara prima delle 11:20: quali treni partono da Termoli?",
        c(g("find", "treno", "transport", [f("origin", "Termoli"), f("destination", "Pescara")], temporal=[t("arrival_time", "11:20", "before")])))
    add("travel", "Per mercoledì mostrami voli da Catania a Praga senza scalo, dal più economico.",
        c(g("find", "volo", "transport", [f("origin", "Catania"), f("destination", "Praga"), f("stops", 0)], temporal=[t("search_date", "mercoledì")], modifiers={"sort": {"field": "price", "direction": "asc"}})))
    add("travel", "Cerco pullman da Salerno verso Matera che lascino Salerno dopo le 17:45.",
        c(g("find", "pullman", "transport", [f("origin", "Salerno"), f("destination", "Matera")], temporal=[t("departure_time", "17:45", "after")])))
    add("travel", "Quali sono le prime tre corse ferroviarie da Grosseto a Roma per venerdì mattina?",
        c(g("find", "treno", "transport", [f("origin", "Grosseto"), f("destination", "Roma")], temporal=[t("search_date", "venerdì"), t("search_time", "mattina")], modifiers={"limit": 3})))
    add("travel", "Non ho ancora scelto da quale città partire: cerca voli per Bruxelles per il 21 giugno.",
        c(g("find", "volo", "transport", [f("destination", "Bruxelles")], temporal=[t("search_date", "21 giugno")], missing=[missing("origin", "search")])))

    # Delayed actions, missing identity, and explicit prohibitions.
    add("policy", "Esplora gli hotel di Oristano per agosto. Nessuna prenotazione, per favore.",
        c(g("find", "hotel", "accommodation", [f("location", "Oristano", "near")], temporal=[t("search_date", "agosto")], policy={"forbid": ["book"]})))
    add("policy", "Per lunedì cerca una bicicletta a noleggio a Senigallia, però non affittarla ancora.",
        c(g("find", "bicicletta", "vehicle", [f("location", "Senigallia", "near")], temporal=[t("search_date", "lunedì")], policy={"forbid": ["rent"]})))
    add("policy", "Prenota un volo da Treviso a Valencia per il 3 dicembre, dopo che ti avrò dato l'ok.",
        c(g("book", "volo", "transport", [f("origin", "Treviso"), f("destination", "Valencia")], temporal=[t("action_date", "3 dicembre")], policy={"confirm_before": ["book"]})))
    add("policy", "Ingaggia un elettricista a Benevento sabato, ma prima fammi scegliere chi chiamare.",
        c(g("hire", "elettricista", "professional_service", [f("location", "Benevento", "near")], temporal=[t("action_date", "sabato")], missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("policy", "Puoi bloccare un tavolo a Faenza per quattro persone domenica alle 13? Vorrei approvare il locale.",
        c(g("book", "ristorante", "place", [f("location", "Faenza", "near"), f("guests", 4)], temporal=[t("action_date", "domenica"), t("action_time", "13:00")], missing=[missing("provider")], policy={"confirm_before": ["book"]})))
    add("policy", "Controlla se a Fabriano ci sono tecnici per la caldaia disponibili oggi; non dare incarichi.",
        c(g("find", "tecnico caldaia", "professional_service", [f("location", "Fabriano", "near"), f("availability", "oggi")], policy={"forbid": ["hire"]})))

    # Message content, notification time and provider identity must remain distinct.
    add("message", "Domani alle 07:40 ricordami di dire a Paolo che il treno parte alle 10.",
        c(g("notify", "promemoria", "reminder", [f("message", "dire a Paolo che il treno parte alle 10")], temporal=[t("notification_date", "domani"), t("notification_time", "07:40")])))
    add("message", "Scrivi a Silvia 'Il preventivo scade venerdì', ma lascia l'invio in attesa del mio sì.",
        c(g("contact", "Silvia", "person", [f("message", "Il preventivo scade venerdì")], policy={"confirm_before": ["contact"]})))
    add("message", "Alle 08:30 manda a Renzo il testo 'Ci sentiamo alle 18'; chiedimi conferma prima di spedirlo.",
        c(g("contact", "Renzo", "person", [f("message", "Ci sentiamo alle 18")], temporal=[t("action_time", "08:30")], policy={"confirm_before": ["contact"]})))
    add("message", "Ricordami mercoledì di rinnovare l'abbonamento, non so ancora a che ora.",
        c(g("notify", "promemoria", "reminder", [f("message", "rinnovare l'abbonamento")], temporal=[t("notification_date", "mercoledì")], missing=[missing("notification_time")])))

    # Comparison is a read action; sort and cardinality are not purchase intent.
    add("comparison", "Metti a confronto tre frigoriferi sotto 700 euro, iniziando dal meno caro. Non ordinarli.",
        c(g("compare", "frigorifero", "product", [f("price", 700, "lt", currency="EUR")], modifiers={"compare_count": 3, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})))
    add("comparison", "Quali due compagnie internet a Nuoro hanno il canone mensile più basso? Solo confronto.",
        c(g("compare", "internet", "service", [f("location", "Nuoro", "near")], modifiers={"compare_count": 2, "sort": {"field": "price", "direction": "asc"}})), flags=("comparison_target_ontology",))
    add("comparison", "Confronta quattro pensioni a Foligno per valutazione, senza riservare camere.",
        c(g("compare", "pensione", "accommodation", [f("location", "Foligno", "near")], modifiers={"compare_count": 4, "sort": {"field": "rating", "direction": "desc"}}, policy={"forbid": ["book"]})))

    # Two goals, result references and fallback conditions.
    add("sequence", "Prima mostrami i falegnami di Fidenza; se scelgo il secondo, contattalo solo dopo aver letto la bozza.",
        c(g("find", "falegname", "professional_service", [f("location", "Fidenza", "near")], idx=1), g("contact", "falegname", "professional_service", [selection(ordinal=2)], depends_on=["g1"], missing=[missing("message")], policy={"confirm_before": ["contact"]}, idx=2)))
    add("sequence", "Cerca alloggi a Castelfranco Veneto e prenota quello che scelgo soltanto dopo averti dato la data e il permesso.",
        c(g("find", "alloggio", "accommodation", [f("location", "Castelfranco Veneto", "near")], idx=1), g("book", "alloggio", "accommodation", [selection()], depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]}, idx=2)), flags=("selection_unresolved",))
    add("sequence", "Guarda se ci sono treni per Parma; in mancanza di risultati cerca autobus per la stessa destinazione.",
        c(g("find", "treno", "transport", [f("destination", "Parma")], idx=1), g("find", "autobus", "transport", [f("destination", "Parma")], depends_on=["g1"], condition=fallback(), idx=2)))
    add("sequence", "Trova due alberghi a Montepulciano, poi chiedimi quale preferisco prima di prenotare.",
        c(g("find", "albergo", "accommodation", [f("location", "Montepulciano", "near")], modifiers={"limit": 2}, idx=1), g("book", "albergo", "accommodation", [selection()], depends_on=["g1"], missing=[missing("provider")], policy={"confirm_before": ["book"]}, idx=2)), flags=("selection_unresolved",))
    add("sequence", "Da Cesano Maderno mostrami treni per Como e, separatamente, hotel a Erba per sabato.",
        c(g("find", "treno", "transport", [f("origin", "Cesano Maderno"), f("destination", "Como")], idx=1), g("find", "hotel", "accommodation", [f("location", "Erba", "near")], temporal=[t("search_date", "sabato")], idx=2)))

    # User-owned objects, appointments, and incomplete required information.
    add("appointment", "Sposta la mia visita dal fisiatra del 10 febbraio alle 15:30 al 12 febbraio alle 09:00, ma attendi la conferma.",
        c(g("reschedule", "visita dal fisiatra", "appointment", [f("original_date", "10 febbraio"), f("original_time", "15:30")], temporal=[t("action_date", "12 febbraio"), t("action_time", "09:00")], policy={"confirm_before": ["reschedule"]})))
    add("appointment", "Annulla l'appuntamento dal dentista che ho il 4 marzo a Udine; prima mostrami quale record cancelleresti.",
        c(g("cancel", "appuntamento dal dentista", "appointment", [f("original_date", "4 marzo"), f("location", "Udine", "near")], policy={"confirm_before": ["cancel"]})))
    add("appointment", "Fissa un controllo allergologico a Cosenza, ma non conosco ancora il giorno né lo studio.",
        c(g("book", "controllo allergologico", "appointment", [f("location", "Cosenza", "near")], missing=[missing("date"), missing("provider")])))
    add("appointment", "Vorrei noleggiare un'auto a Viterbo domenica, dopo aver controllato il prezzo e scelto l'agenzia.",
        c(g("rent", "auto", "vehicle", [f("location", "Viterbo", "near")], temporal=[t("action_date", "domenica")], missing=[missing("provider")], policy={"confirm_before": ["rent"]})))
    add("appointment", "Mi serve un idraulico a Foggia, però non ho deciso quale professionista assumere.",
        c(g("hire", "idraulico", "professional_service", [f("location", "Foggia", "near")], missing=[missing("provider")])))

    # Context and unsupported assumptions.
    add("context", "Non ho la macchina: cerca un supermercato raggiungibile a piedi dal centro di Alba.",
        c(g("find", "supermercato", "place", [f("location", "centro di Alba", "near"), f("access", "a piedi")]), context=[ctx("has_car", False)]))
    add("context", "Siamo in quattro e vorremmo vedere ristoranti aperti stasera a Campobasso, senza prenotare.",
        c(g("find", "ristorante", "place", [f("location", "Campobasso", "near"), f("availability", "stasera")], policy={"forbid": ["book"]}), context=[ctx("party_size", 4)]))
    add("context", "Per la riunione a Rovigo, cerca una sala con accesso per sedia a rotelle; mostrami solo le opzioni.",
        c(g("find", "sala riunioni", "place", [f("location", "Rovigo", "near"), f("access", "sedia a rotelle")], policy={"forbid": ["book"]})))

    return rows


def main() -> None:
    if DEST.exists():
        raise FileExistsError("V34 already frozen; never regenerate")
    rows = cases()
    if len(rows) < 40 or len({normalize_text(row["text"]) for row in rows}) != len(rows):
        raise RuntimeError("V34 must contain at least 40 distinct requests")
    previous = []
    for path in PRIOR:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            user_text = row.get("text") or row.get("request")
            if user_text is None:
                user_text = next((msg["content"] for msg in row.get("messages", [])
                                  if msg.get("role") == "user"), None)
            if user_text is not None:
                previous.append((path, user_text))
    prior_exact = {normalize_text(text) for _, text in previous}
    prior_tokens = [(path, text, token_set(text)) for path, text in previous]
    maximum, nearest = 0.0, None
    for row in rows:
        if normalize_text(row["text"]) in prior_exact:
            raise RuntimeError(f"Exact prior overlap: {row['id']}")
        current = token_set(row["text"])
        for path, text, old in prior_tokens:
            score = len(current & old) / len(current | old) if current | old else 0.0
            if score > maximum:
                maximum = score
                nearest = {"case": row["id"], "source": str(path.relative_to(ROOT)), "prior_text": text}
    within = 0.0
    for i, row in enumerate(rows):
        a = token_set(row["text"])
        for other in rows[:i]:
            b = token_set(other["text"])
            within = max(within, len(a & b) / len(a | b) if a | b else 0.0)
    if maximum >= 0.8 or within >= 0.8:
        raise RuntimeError(f"Near duplicate: prior={maximum:.3f}, within={within:.3f}, nearest={nearest}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    review = DEST / "human_review_queue.jsonl"
    with review.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps({"id": row["id"], "request": row["text"], "family": row["family"],
                                     "raw_gold": row["contract"], "review_flags": row["review_flags"],
                                     "human_review_status": "pending", "human_reviewer": None,
                                     "human_notes": None}, ensure_ascii=False) + "\n")
    manifest = {"version": "V34", "schema_version": "1.3", "cases": len(rows),
                "families": dict(sorted(Counter(row["family"] for row in rows).items())),
                "panel_sha256": sha256(panel), "review_queue_sha256": sha256(review),
                "generator_sha256": sha256(Path(__file__)),
                "max_prior_token_jaccard": round(maximum, 6),
                "max_within_panel_token_jaccard": round(within, 6),
                "nearest_prior_source": nearest,
                "prior_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in PRIOR if path.is_file()},
                "gold_review": "agent-authored; pending independent human review",
                "purpose": "independent Planner 1.3 transfer test, frozen before post-V33 fixes"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("cases", "panel_sha256", "max_prior_token_jaccard",
                                                    "max_within_panel_token_jaccard")}, indent=2))


if __name__ == "__main__":
    main()
