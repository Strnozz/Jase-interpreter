"""Freeze a fresh, varied Planner panel before further architecture changes.

Gold and review notes are agent-authored. This file must never regenerate V35.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.build_v28_blind import c, ctx, f, fallback, g, missing, selection, t  # noqa: E402
from benchmarks.build_v34_blind import PRIOR as OLDER  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

DEST = ROOT / "benchmarks/v35"


def cases() -> list[dict]:
    result = []

    def add(family: str, request: str, gold: dict, flags=()) -> None:
        issues = validate_contract(gold, user_text=request)
        if issues:
            raise ValueError((request, [issue.as_dict() for issue in issues]))
        result.append({"id": f"v35-{len(result)+1:03d}", "family": family, "text": request,
                       "contract": gold, "review_flags": list(flags)})

    # Ordinary information requests, with different location and availability scopes.
    add("local", "Ho finito il detersivo. Qual è la lavanderia self service aperta ora vicino a piazza Garibaldi a Cremona?",
        c(g("find", "lavanderia", "place", [f("location", "piazza Garibaldi a Cremona", "near"), f("open_now", True)])), ("target_variant",))
    add("local", "Da via Po a Torino, mostrami farmacie raggiungibili a piedi; mi basta la più vicina.",
        c(g("find", "farmacia", "place", [f("location", "via Po a Torino", "near"), f("access", "a piedi")], modifiers={"limit": 1, "sort": {"field": "distance", "direction": "asc"}})))
    add("local", "Sono al porto di Ancona. Mi trovi un bar aperto adesso che sia lì vicino?",
        c(g("find", "bar", "place", [f("location", "porto di Ancona", "near"), f("open_now", True)])))
    add("local", "Vorrei vedere biblioteche in centro a Pavia, senza riservare un posto per ora.",
        c(g("find", "biblioteca", "place", [f("location", "centro di Pavia", "near")], policy={"forbid": ["book"]})))
    add("local", "C'è una ferramenta nei pressi della stazione di Biella? Non comprare ancora niente.",
        c(g("find", "ferramenta", "place", [f("location", "stazione di Biella", "near")], policy={"forbid": ["buy"]})))
    add("local", "Per la conferenza, cerca sale riunioni accessibili in carrozzina nel centro di Pesaro.",
        c(g("find", "sala riunioni", "place", [f("location", "centro di Pesaro", "near"), f("access", "carrozzina")])))
    add("local", "Mi servono due supermercati aperti questa sera nei dintorni di Porta Romana a Milano.",
        c(g("find", "supermercato", "place", [f("location", "Porta Romana a Milano", "near"), f("availability", "questa sera")], modifiers={"limit": 2})))
    add("local", "A Recanati cerca un museo visitabile domenica mattina, solo per vedere le opzioni.",
        c(g("find", "museo", "place", [f("location", "Recanati", "near"), f("availability", "domenica mattina")])))
    add("local", "La gomma perde aria: cerca un gommista disponibile lunedì mattina a Carpi, senza incaricarlo.",
        c(g("find", "gommista", "professional_service", [f("location", "Carpi", "near"), f("availability", "lunedì mattina")], policy={"forbid": ["hire"]})))

    # Route direction, deadlines, stop count and travel-day semantics.
    add("travel", "Per andare da Asti a Verona venerdì, quali treni arrivano entro le 16?",
        c(g("find", "treno", "transport", [f("origin", "Asti"), f("destination", "Verona")], temporal=[t("search_date", "venerdì"), t("arrival_time", "16:00", "before")])))
    add("travel", "Partendo da Bari dopo le 6:30, cerca pullman diretti a Taranto per domani.",
        c(g("find", "pullman", "transport", [f("origin", "Bari"), f("destination", "Taranto")], temporal=[t("departure_time", "06:30", "after"), t("search_date", "domani")])))
    add("travel", "Voglio il primo volo senza scali da Venezia a Lisbona il 9 maggio; mostrami gli orari, non prenotare.",
        c(g("find", "volo", "transport", [f("origin", "Venezia"), f("destination", "Lisbona"), f("stops", 0)], temporal=[t("search_date", "9 maggio")], modifiers={"limit": 1}, policy={"forbid": ["book"]})))
    add("travel", "Da Modena per Rimini, mostrami autobus che arrivino prima delle 10:15 di sabato.",
        c(g("find", "autobus", "transport", [f("origin", "Modena"), f("destination", "Rimini")], temporal=[t("search_date", "sabato"), t("arrival_time", "10:15", "before")])))
    add("travel", "Quanto ci metto ad arrivare a Ravenna? Non ti ho ancora detto da dove parto: cerca i treni quando lo saprai.",
        c(g("find", "treno", "transport", [f("destination", "Ravenna")], missing=[missing("origin", "search")])), flags=("actionability_ambiguous",))
    add("travel", "Mi interessano voli Bologna-Edimburgo per giovedì pomeriggio, ordinati per durata crescente.",
        c(g("find", "volo", "transport", [f("origin", "Bologna"), f("destination", "Edimburgo")], temporal=[t("search_date", "giovedì"), t("search_time", "pomeriggio")], modifiers={"sort": {"field": "duration", "direction": "asc"}})))
    add("travel", "Cerca treni da Lucca per La Spezia dopo le 19, e non acquistare biglietti.",
        c(g("find", "treno", "transport", [f("origin", "Lucca"), f("destination", "La Spezia")], temporal=[t("departure_time", "19:00", "after")], policy={"forbid": ["buy"]})))
    add("travel", "Per lunedì sera, voglio le tre corse più economiche da Como a Trento in treno.",
        c(g("find", "treno", "transport", [f("origin", "Como"), f("destination", "Trento")], temporal=[t("search_date", "lunedì"), t("search_time", "sera")], modifiers={"limit": 3, "sort": {"field": "price", "direction": "asc"}})))

    # Hospitality and availability are read operations until a reservation is explicit.
    add("hospitality", "A Matera cerca una stanza per due persone per il prossimo fine settimana, ma fammi vedere i prezzi prima.",
        c(g("find", "hotel", "accommodation", [f("location", "Matera", "near"), f("availability", "prossimo fine settimana")], policy={"forbid": ["book"]}), context=[ctx("party_size", 2)]), flags=("target_generalization",))
    add("hospitality", "Mostrami trattorie vegetariane a Spoleto aperte sabato a pranzo; non riservare nulla.",
        c(g("find", "trattoria", "place", [f("location", "Spoleto", "near"), f("diet", "vegetariano"), f("availability", "sabato a pranzo")], policy={"forbid": ["book"]})))
    add("hospitality", "Mi trovi un ostello a Trieste vicino alla stazione, per stanotte? Decido io quale scegliere.",
        c(g("find", "ostello", "accommodation", [f("location", "stazione di Trieste", "near"), f("availability", "stanotte")], policy={"forbid": ["book"]})))
    add("hospitality", "A Trapani voglio verificare se un ristorante ha un tavolo per cinque alle 20 di domani; non prenotare.",
        c(g("find", "ristorante", "place", [f("location", "Trapani", "near"), f("availability", "domani alle 20")], policy={"forbid": ["book"]}), context=[ctx("party_size", 5)]))
    add("hospitality", "Fammi vedere alberghi di Lecce disponibili dal 14 al 16 giugno, senza bloccare camere.",
        c(g("find", "albergo", "accommodation", [f("location", "Lecce", "near"), f("availability", "dal 14 al 16 giugno")], policy={"forbid": ["book"]})))

    # Ranking and selection boundaries.
    add("comparison", "Confronta tre lavatrici sotto 500 euro per prezzo, poi mostrami quella meno cara; non comprare.",
        c(g("compare", "lavatrice", "product", [f("price", 500, "lt", currency="EUR")], modifiers={"compare_count": 3, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})))
    add("comparison", "Vorrei mettere a confronto due alberghi a Gubbio in base alla valutazione, senza prenotarli.",
        c(g("compare", "albergo", "accommodation", [f("location", "Gubbio", "near")], modifiers={"compare_count": 2, "sort": {"field": "rating", "direction": "desc"}}, policy={"forbid": ["book"]})))
    add("comparison", "Quali sono i quattro abbonamenti fibra a Savona dal canone più basso al più alto? Solo confronto.",
        c(g("compare", "fibra", "service", [f("location", "Savona", "near")], modifiers={"compare_count": 4, "sort": {"field": "price", "direction": "asc"}})))
    add("comparison", "Metti a confronto cinque biciclette usate a Lecco, partendo dalle più vicine.",
        c(g("compare", "bicicletta usata", "vehicle", [f("location", "Lecco", "near")], modifiers={"compare_count": 5, "sort": {"field": "distance", "direction": "asc"}})))
    add("comparison", "Tra tre stampanti laser entro 300 euro, mostrami prima quella meglio valutata; niente acquisti.",
        c(g("compare", "stampante laser", "product", [f("price", 300, "lte", currency="EUR")], modifiers={"compare_count": 3, "sort": {"field": "rating", "direction": "desc"}}, policy={"forbid": ["buy"]})))
    add("comparison", "Sto valutando un frigo nuovo: confronta due modelli sotto 800 euro, ma non ordinare nulla.",
        c(g("compare", "frigorifero", "product", [f("price", 800, "lt", currency="EUR")], modifiers={"compare_count": 2}, policy={"forbid": ["buy"]})))

    # Time in message content is not the trigger or sending time.
    add("communication", "Alle 7:10 di domani ricordami di avvisare Marta che la visita è fissata alle 11.",
        c(g("notify", "promemoria", "reminder", [f("message", "avvisare Marta che la visita è fissata alle 11")], temporal=[t("notification_date", "domani"), t("notification_time", "07:10")])))
    add("communication", "Ricordami martedì alle 18 di portare i documenti del mutuo.",
        c(g("notify", "promemoria", "reminder", [f("message", "portare i documenti del mutuo")], temporal=[t("notification_date", "martedì"), t("notification_time", "18:00")])))
    add("communication", "Domani manda a Giulia il messaggio 'Ti richiamo alle 21', ma mostrami la bozza prima dell'invio.",
        c(g("contact", "Giulia", "person", [f("message", "Ti richiamo alle 21")], temporal=[t("action_date", "domani")], policy={"confirm_before": ["contact"]})))
    add("communication", "Puoi scrivere a mio fratello che il pacco è arrivato? Ti confermerò io quando inviarlo.",
        c(g("contact", "mio fratello", "person", [f("message", "il pacco è arrivato")], policy={"confirm_before": ["contact"]})), flags=("recipient_identity",))
    add("communication", "Alle 09:20 scrivi a Nina 'Sono sotto casa alle 17', e aspetta il mio sì per inviare.",
        c(g("contact", "Nina", "person", [f("message", "Sono sotto casa alle 17")], temporal=[t("action_time", "09:20")], policy={"confirm_before": ["contact"]})))
    add("communication", "Ricordami sabato di rinnovare la tessera della biblioteca; l'ora possiamo sceglierla dopo.",
        c(g("notify", "promemoria", "reminder", [f("message", "rinnovare la tessera della biblioteca")], temporal=[t("notification_date", "sabato")], missing=[missing("notification_time")])))

    # Consequential actions require trusted identity, full slots and confirmation.
    add("consent", "Vorrei prenotare un hotel a Noto per due notti a ottobre; prima mostrami quale camera hai scelto.",
        c(g("book", "hotel", "accommodation", [f("location", "Noto", "near")], temporal=[t("action_date", "ottobre")], missing=[missing("provider"), missing("check_in"), missing("check_out")], policy={"confirm_before": ["book"]})), flags=("booking_dates",))
    add("consent", "Assumi un giardiniere a Brindisi venerdì, ma soltanto dopo che avrò approvato il professionista.",
        c(g("hire", "giardiniere", "professional_service", [f("location", "Brindisi", "near")], temporal=[t("action_date", "venerdì")], missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("consent", "Annulla la mia visita cardiologica del 22 ottobre a Pistoia, dopo avermi mostrato il record e chiesto conferma.",
        c(g("cancel", "visita cardiologica", "appointment", [f("original_date", "22 ottobre"), f("location", "Pistoia", "near")], policy={"confirm_before": ["cancel"]})))
    add("consent", "Noleggia un furgone a Rimini per sabato. Voglio autorizzare l'agenzia prima di procedere.",
        c(g("rent", "furgone", "vehicle", [f("location", "Rimini", "near")], temporal=[t("action_date", "sabato")], missing=[missing("provider")], policy={"confirm_before": ["rent"]})))
    add("consent", "Fissa la visita dal dentista a Perugia il 7 novembre alle 10:00, ma prima verifica lo studio con me.",
        c(g("book", "dentista", "appointment", [f("location", "Perugia", "near")], temporal=[t("action_date", "7 novembre"), t("action_time", "10:00")], missing=[missing("provider")], policy={"confirm_before": ["book"]})))
    add("consent", "Sposta l'appuntamento del meccanico dal 3 aprile al 5 aprile, se trovi quello giusto; chiedimi prima.",
        c(g("reschedule", "appuntamento", "appointment", [f("original_date", "3 aprile")], temporal=[t("action_date", "5 aprile")], missing=[missing("provider")], policy={"confirm_before": ["reschedule"]})))

    # Multiple goals, fallbacks, and references to unseen search results.
    add("sequence", "Prima cerca meccanici a Sondrio; poi preparami un messaggio per quello che scelgo, senza inviarlo.",
        c(g("find", "meccanico", "professional_service", [f("location", "Sondrio", "near")], idx=1), g("contact", "meccanico", "professional_service", [selection()], depends_on=["g1"], missing=[missing("message")], policy={"confirm_before": ["contact"]}, idx=2)))
    add("sequence", "Cerca treni da Novara a Padova; se non ce ne sono, guarda autobus per Padova.",
        c(g("find", "treno", "transport", [f("origin", "Novara"), f("destination", "Padova")], idx=1), g("find", "autobus", "transport", [f("origin", "Novara"), f("destination", "Padova")], depends_on=["g1"], condition=fallback(), idx=2)))
    add("sequence", "Mostrami due alberghi a Siena. Poi prenota il primo soltanto quando ti avrò dato il mio sì.",
        c(g("find", "albergo", "accommodation", [f("location", "Siena", "near")], modifiers={"limit": 2}, idx=1), g("book", "albergo", "accommodation", [selection(ordinal=1)], depends_on=["g1"], policy={"confirm_before": ["book"]}, idx=2)))
    add("sequence", "Trova il negozio di biciclette più vicino a Lodi e, separatamente, una farmacia aperta ora.",
        c(g("find", "negozio di biciclette", "place", [f("location", "Lodi", "near")], modifiers={"limit": 1, "sort": {"field": "distance", "direction": "asc"}}, idx=1), g("find", "farmacia", "place", [f("location", "Lodi", "near"), f("open_now", True)], idx=2)))
    add("sequence", "Cerca un volo Firenze-Atene e un hotel ad Atene per la settimana prossima, senza prenotare nessuno dei due.",
        c(g("find", "volo", "transport", [f("origin", "Firenze"), f("destination", "Atene")], temporal=[t("search_date", "settimana prossima")], policy={"forbid": ["book"]}, idx=1), g("find", "hotel", "accommodation", [f("location", "Atene", "near")], temporal=[t("search_date", "settimana prossima")], policy={"forbid": ["book"]}, idx=2)))
    add("sequence", "Guarda gli elettricisti a Chieti, poi contatta il secondo soltanto se te lo confermo.",
        c(g("find", "elettricista", "professional_service", [f("location", "Chieti", "near")], idx=1), g("contact", "elettricista", "professional_service", [selection(ordinal=2)], depends_on=["g1"], missing=[missing("message")], policy={"confirm_before": ["contact"]}, idx=2)))

    # Meta requests and mentions have no actionable intent.
    add("meta", "Nel testo della presentazione c'è scritto 'compra il biglietto oggi': me lo traduci in inglese?", c())
    add("meta", "Sto insegnando a un assistente cosa vuol dire 'prenota il tavolo'. Spiegami il significato.", c())
    add("meta", "Se un amico dicesse 'cancella tutto', quali domande gli faresti prima di agire?", c())
    add("meta", "Nel romanzo il personaggio dice 'mandagli una mail'. Non farlo: analizza la scena.", c())
    add("meta", "Come distingueresti una ricerca di hotel da una prenotazione in un manuale utente?", c())

    # Missing information and implicit user context.
    add("incomplete", "Trova un volo per Copenaghen il mese prossimo; la città di partenza te la dico più tardi.",
        c(g("find", "volo", "transport", [f("destination", "Copenaghen")], temporal=[t("search_date", "mese prossimo")], missing=[missing("origin", "search")])), flags=("search_can_be_broad",))
    add("incomplete", "Prenotami un ristorante per stasera; non so ancora in quale città né per quante persone.",
        c(g("book", "ristorante", "place", [], temporal=[t("action_date", "stasera")], missing=[missing("location"), missing("guests"), missing("provider")], policy={"confirm_before": ["book"]})))
    add("incomplete", "Ricordamelo fra una settimana: ti dirò poi cosa va ricordato.",
        c(g("notify", "promemoria", "reminder", [], temporal=[t("notification_date", "fra una settimana")], missing=[missing("message"), missing("notification_time")])), flags=("time_precision",))
    add("incomplete", "Mi serve un elettricista urgente a Mantova, ma non ho ancora scelto a chi affidare il lavoro.",
        c(g("find", "elettricista", "professional_service", [f("location", "Mantova", "near"), f("availability", "urgente")])), flags=("hire_vs_find",))
    add("incomplete", "Devo spostare una visita ma non so quale prenotazione risulta nel calendario. Aiutami a identificarla.",
        c(g("reschedule", "visita", "appointment", [], missing=[missing("provider"), missing("original_date"), missing("date")])), flags=("identification_vs_action",))

    add("context", "Non guido; per arrivare da Mantova a Brescia mostrami i treni del mattino.",
        c(g("find", "treno", "transport", [f("origin", "Mantova"), f("destination", "Brescia")], temporal=[t("search_time", "mattino")]), context=[ctx("has_car", False)]))
    add("context", "Viaggio con un cane: cerca un hotel a Udine che accetti animali per venerdì.",
        c(g("find", "hotel", "accommodation", [f("location", "Udine", "near"), f("availability", "venerdì")]), context=[ctx("pet", "cane")]))
    add("context", "Siamo sei colleghi; mostrami ristoranti vicino al teatro di Prato per domani sera.",
        c(g("find", "ristorante", "place", [f("location", "teatro di Prato", "near"), f("availability", "domani sera")]), context=[ctx("party_size", 6)]))
    add("context", "La mia bici è a Cesena: trova un'officina nei dintorni aperta sabato, senza ingaggiarne nessuna.",
        c(g("find", "officina", "professional_service", [f("location", "Cesena", "near"), f("availability", "sabato")], policy={"forbid": ["hire"]}), context=[ctx("vehicle_location", "Cesena")]))

    return result


def main() -> None:
    if DEST.exists():
        raise FileExistsError("V35 is frozen; never regenerate")
    panel = cases()
    if len(panel) < 60 or len({normalize_text(row["text"]) for row in panel}) != len(panel):
        raise RuntimeError("V35 requires at least 60 distinct requests")
    prior = []
    for path in OLDER:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            request = row.get("text") or row.get("request")
            if request is None:
                request = next((item["content"] for item in row.get("messages", [])
                                if item.get("role") == "user"), None)
            if request is not None:
                prior.append((path, request))
    prior_exact = {normalize_text(text) for _, text in prior}
    prior_tokens = [(path, text, token_set(text)) for path, text in prior]
    maximum, nearest, within = 0.0, None, 0.0
    for i, row in enumerate(panel):
        if normalize_text(row["text"]) in prior_exact:
            raise RuntimeError(f"Exact overlap: {row['id']}")
        current = token_set(row["text"])
        for path, text, old in prior_tokens:
            score = len(current & old) / len(current | old) if current | old else 0.0
            if score > maximum:
                maximum, nearest = score, {"case": row["id"], "source": str(path.relative_to(ROOT)),
                                           "prior_text": text}
        for earlier in panel[:i]:
            other = token_set(earlier["text"])
            within = max(within, len(current & other) / len(current | other) if current | other else 0.0)
    if maximum >= 0.8 or within >= 0.8:
        raise RuntimeError(f"Near overlap: prior={maximum:.3f}, within={within:.3f}, nearest={nearest}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel_path, review_path = DEST / "blind.jsonl", DEST / "human_review_queue.jsonl"
    with panel_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in panel:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8", newline="\n") as output:
        for row in panel:
            output.write(json.dumps({"id": row["id"], "request": row["text"], "family": row["family"],
                                     "raw_gold": row["contract"], "review_flags": row["review_flags"],
                                     "human_review_status": "pending", "human_reviewer": None,
                                     "human_notes": None}, ensure_ascii=False) + "\n")
    manifest = {"version": "V35", "schema_version": "1.3", "cases": len(panel),
                "families": dict(sorted(Counter(row["family"] for row in panel).items())),
                "panel_sha256": sha256(panel_path), "review_queue_sha256": sha256(review_path),
                "generator_sha256": sha256(Path(__file__)),
                "max_prior_token_jaccard": round(maximum, 6),
                "max_within_panel_token_jaccard": round(within, 6),
                "nearest_prior_source": nearest,
                "prior_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in OLDER if path.is_file()},
                "gold_review": "agent-authored; pending independent human review",
                "purpose": "independent Planner 1.3 transfer test, frozen before post-V34 fixes"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("cases", "panel_sha256", "max_prior_token_jaccard",
                                                    "max_within_panel_token_jaccard")}, indent=2))


if __name__ == "__main__":
    main()
