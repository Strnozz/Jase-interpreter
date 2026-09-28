"""Freeze an independent, natural-language Planner handoff panel before fixes.

The gold is agent-authored and pending independent human adjudication.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.build_v28_blind import c, ctx, f, fallback, g, missing, selection, t  # noqa: E402
from benchmarks.build_v35_blind import OLDER  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

DEST = ROOT / "benchmarks/v36"
PRIOR = sorted(set([*OLDER, *ROOT.glob("benchmarks/v35/**/*.jsonl")]))


def cases() -> list[dict]:
    result = []

    def add(family: str, request: str, gold: dict, flags=()) -> None:
        issues = validate_contract(gold, user_text=request)
        if issues:
            raise ValueError((request, [issue.as_dict() for issue in issues]))
        result.append({"id": f"v36-{len(result)+1:03d}", "family": family, "text": request,
                       "contract": gold, "review_flags": list(flags)})

    add("shopping", "Il mio portatile si è rotto: confronta tre PC sotto i 900 euro, dal più economico. Per ora non comprare.",
        c(g("compare", "pc portatile", "product", [f("price", 900, "lt", currency="EUR")], modifiers={"compare_count": 3, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})))
    add("shopping", "Mi indichi una libreria nei dintorni di piazza Dante a Napoli? Voglio solo sapere dove si trova.",
        c(g("find", "libreria", "place", [f("location", "piazza Dante a Napoli", "near")])))
    add("shopping", "Sto pensando a una bici a noleggio per domenica a Ravenna. Mostrami le opzioni, non affittarne una.",
        c(g("find", "bicicletta", "vehicle", [f("location", "Ravenna", "near")], temporal=[t("search_date", "domenica")], policy={"forbid": ["rent"]})))
    add("shopping", "Quali due alberghi a Padova hanno la valutazione più alta? Non fare prenotazioni.",
        c(g("compare", "albergo", "accommodation", [f("location", "Padova", "near")], modifiers={"compare_count": 2, "sort": {"field": "rating", "direction": "desc"}}, policy={"forbid": ["book"]})))
    add("shopping", "Cerco ristoranti che propongano cucina greca a Novara per sabato sera, siamo in tre.",
        c(g("find", "ristorante", "place", [f("location", "Novara", "near"), f("cuisine", "greca"), f("availability", "sabato sera")]), context=[ctx("party_size", 3)]))
    add("shopping", "La lavastoviglie perde acqua: cerca un tecnico disponibile oggi ad Arezzo, ma non incaricarlo.",
        c(g("find", "tecnico lavastoviglie", "professional_service", [f("location", "Arezzo", "near"), f("availability", "oggi")], policy={"forbid": ["hire"]})), flags=("capability_ontology",))

    add("transport", "Per una visita a Pesaro devo arrivare prima delle 9:30. Ci sono treni da Fano?",
        c(g("find", "treno", "transport", [f("origin", "Fano"), f("destination", "Pesaro")], temporal=[t("arrival_time", "09:30", "before")])))
    add("transport", "Dammi voli diretti da Napoli a Stoccolma il 18 agosto; non acquistare né prenotare.",
        c(g("find", "volo", "transport", [f("origin", "Napoli"), f("destination", "Stoccolma"), f("stops", 0)], temporal=[t("search_date", "18 agosto")], policy={"forbid": ["buy", "book"]})))
    add("transport", "Domani sera devo lasciare Piacenza dopo le 20: cerca autobus per Parma.",
        c(g("find", "autobus", "transport", [f("origin", "Piacenza"), f("destination", "Parma")], temporal=[t("search_date", "domani"), t("departure_time", "20:00", "after")])))
    add("transport", "Mostrami voli per Porto per il 12 settembre; scelgo la città di partenza più tardi.",
        c(g("find", "volo", "transport", [f("destination", "Porto")], temporal=[t("search_date", "12 settembre")], missing=[missing("origin", "search")])), flags=("broad_route_search",))
    add("transport", "Sono a Trento e domenica vorrei vedere sia i treni per Bolzano sia le camere disponibili lì.",
        c(g("find", "treno", "transport", [f("origin", "Trento"), f("destination", "Bolzano")], temporal=[t("search_date", "domenica")], idx=1),
          g("find", "hotel", "accommodation", [f("location", "Bolzano", "near"), f("availability", "domenica")], idx=2)))
    add("transport", "Da Pordenone a Udine venerdì mattina, elenca i primi due treni per prezzo crescente.",
        c(g("find", "treno", "transport", [f("origin", "Pordenone"), f("destination", "Udine")], temporal=[t("search_date", "venerdì"), t("search_time", "mattina")], modifiers={"limit": 2, "sort": {"field": "price", "direction": "asc"}})))

    add("communication", "Ricordami alle 7 di consegnare il modulo: l'ufficio chiude alle 12.",
        c(g("notify", "promemoria", "reminder", [f("message", "consegnare il modulo: l'ufficio chiude alle 12")], temporal=[t("notification_time", "07:00")], missing=[missing("notification_date")])), flags=("punctuated_message",))
    add("communication", "Mercoledì alle 18:30 mandami un promemoria per ritirare gli occhiali.",
        c(g("notify", "promemoria", "reminder", [f("message", "ritirare gli occhiali")], temporal=[t("notification_date", "mercoledì"), t("notification_time", "18:30")])))
    add("communication", "Prepara per Elisa il messaggio 'Arriverò mezz'ora dopo'; non inviarlo finché non dico sì.",
        c(g("contact", "Elisa", "person", [f("message", "Arriverò mezz'ora dopo")], policy={"confirm_before": ["contact"]})))
    add("communication", "Domani alle 08:00 scrivi a Riccardo che il pacco arriverà alle 15. Chiedimi conferma per l'invio.",
        c(g("contact", "Riccardo", "person", [f("message", "il pacco arriverà alle 15")], temporal=[t("action_date", "domani"), t("action_time", "08:00")], policy={"confirm_before": ["contact"]})))
    add("communication", "Vorrei avvisare il medico del ritardo, ma non ti ho ancora dato il suo recapito né il testo.",
        c(g("contact", "medico", "person", [], missing=[missing("message"), missing("recipient")], policy={"confirm_before": ["contact"]})), flags=("recipient_binding",))
    add("communication", "Mi ricordi lunedì di comprare le medicine? L'orario ancora non lo so.",
        c(g("notify", "promemoria", "reminder", [f("message", "comprare le medicine")], temporal=[t("notification_date", "lunedì")], missing=[missing("notification_time")])))

    add("consent", "Prenota un tavolo a Parma per due persone venerdì alle 21, ma fammi approvare il ristorante.",
        c(g("book", "ristorante", "place", [f("location", "Parma", "near"), f("guests", 2)], temporal=[t("action_date", "venerdì"), t("action_time", "21:00")], missing=[missing("provider")], policy={"confirm_before": ["book"]})))
    add("consent", "Mi serve un veterinario a Massa: puoi cercarne uno senza fissare subito la visita?",
        c(g("find", "veterinario", "professional_service", [f("location", "Massa", "near")], policy={"forbid": ["book"]})), flags=("target_type",))
    add("consent", "Cancella il controllo del 6 dicembre a Ferrara, dopo che mi avrai mostrato quale appuntamento hai trovato.",
        c(g("cancel", "appuntamento", "appointment", [f("original_date", "6 dicembre"), f("location", "Ferrara", "near")], policy={"confirm_before": ["cancel"]})))
    add("consent", "Sposta la visita del 10 maggio alle 14 al 12 maggio alle 16; prima assicurati di aver trovato la prenotazione.",
        c(g("reschedule", "visita", "appointment", [f("original_date", "10 maggio"), f("original_time", "14:00")], temporal=[t("action_date", "12 maggio"), t("action_time", "16:00")], policy={"confirm_before": ["reschedule"]})))
    add("consent", "Affitta un'auto a Cuneo per sabato solo dopo che avrò scelto l'agenzia e confermato il prezzo.",
        c(g("rent", "auto", "vehicle", [f("location", "Cuneo", "near")], temporal=[t("action_date", "sabato")], missing=[missing("provider")], policy={"confirm_before": ["rent"]})))
    add("consent", "Ingaggia un idraulico a Livorno oggi, ma non contattarne nessuno prima del mio via libera.",
        c(g("hire", "idraulico", "professional_service", [f("location", "Livorno", "near")], temporal=[t("action_date", "oggi")], missing=[missing("provider")], policy={"confirm_before": ["hire"]})))

    add("dependency", "Trova tre alberghi a Mantova; quando ne avrò scelto uno, chiedimi prima di riservarlo.",
        c(g("find", "albergo", "accommodation", [f("location", "Mantova", "near")], modifiers={"limit": 3}, idx=1),
          g("book", "albergo", "accommodation", [selection()], depends_on=["g1"], missing=[missing("provider")], policy={"confirm_before": ["book"]}, idx=2)))
    add("dependency", "Se non ci sono voli per Roma da Verona sabato, cerca i treni per la stessa meta.",
        c(g("find", "volo", "transport", [f("origin", "Verona"), f("destination", "Roma")], temporal=[t("search_date", "sabato")], idx=1),
          g("find", "treno", "transport", [f("origin", "Verona"), f("destination", "Roma")], temporal=[t("search_date", "sabato")], depends_on=["g1"], condition=fallback(), idx=2)))
    add("dependency", "Prima elenca le officine vicino a Crema. Poi prepara una mail alla seconda chiedendo disponibilità, senza inviarla.",
        c(g("find", "officina", "professional_service", [f("location", "Crema", "near")], idx=1),
          g("contact", "officina", "professional_service", [f("message", "chiedendo disponibilità"), selection(ordinal=2)], depends_on=["g1"], policy={"confirm_before": ["contact"]}, idx=2)), flags=("message_paraphrase",))
    add("dependency", "Vorrei vedere un hotel a Frosinone e, separatamente, una farmacia aperta ora nei dintorni.",
        c(g("find", "hotel", "accommodation", [f("location", "Frosinone", "near")], idx=1),
          g("find", "farmacia", "place", [f("location", "Frosinone", "near"), f("open_now", True)], idx=2)))
    add("dependency", "Cerca un treno da Pisa per Arezzo e ricordami domani alle 7 di controllare il biglietto.",
        c(g("find", "treno", "transport", [f("origin", "Pisa"), f("destination", "Arezzo")], idx=1),
          g("notify", "promemoria", "reminder", [f("message", "controllare il biglietto")], temporal=[t("notification_date", "domani"), t("notification_time", "07:00")], idx=2)))
    add("dependency", "Indicami i due voli meno cari Milano-Bruxelles e prenota solo quello che confermerò più tardi.",
        c(g("find", "volo", "transport", [f("origin", "Milano"), f("destination", "Bruxelles")], modifiers={"limit": 2, "sort": {"field": "price", "direction": "asc"}}, idx=1),
          g("book", "volo", "transport", [selection(), f("origin", "Milano"), f("destination", "Bruxelles")], depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]}, idx=2)))

    add("non_actionable", "Per un glossario, spiegami che cosa vuol dire 'noleggia un furgone'.", c())
    add("non_actionable", "La frase 'manda subito il messaggio' compare in una sceneggiatura. Analizzane il tono, non eseguirla.", c())
    add("non_actionable", "Se un cliente ti chiedesse di prenotare un volo senza dirti quando, quali dati gli chiederesti?", c())
    add("non_actionable", "Sto preparando un test: traduci in francese 'annulla la visita di domani'.", c())
    add("non_actionable", "Secondo te perché un assistente dovrebbe chiedere conferma prima di pagare?", c())
    add("non_actionable", "Nel manuale di un'app compare 'trova un elettricista': descrivi il significato del comando.", c())

    add("incomplete", "Trova un ristorante per cena, ma non so ancora in quale città saremo.",
        c(g("find", "ristorante", "place", [f("availability", "cena")], missing=[missing("location", "search")])), flags=("broad_location",))
    add("incomplete", "Vorrei prenotare un treno per Bologna, ma non conosco né il giorno né la stazione di partenza.",
        c(g("book", "treno", "transport", [f("destination", "Bologna")], missing=[missing("date"), missing("origin"), missing("selection")], policy={"confirm_before": ["book"]})))
    add("incomplete", "Mandagli una mail per favore. Non ti ho ancora detto chi sia 'lui' o cosa scrivere.",
        c(g("contact", "lui", "person", [], missing=[missing("recipient"), missing("message")], policy={"confirm_before": ["contact"]})), flags=("pronoun_resolution",))
    add("incomplete", "Domani ricordami qualcosa, ma ancora non so che cosa né a che ora.",
        c(g("notify", "promemoria", "reminder", [], temporal=[t("notification_date", "domani")], missing=[missing("message"), missing("notification_time")])))
    add("incomplete", "Mi serve una camera a Rieti per un fine settimana; per ora voglio solo vedere possibilità, non prenotare.",
        c(g("find", "hotel", "accommodation", [f("location", "Rieti", "near"), f("availability", "un fine settimana")], policy={"forbid": ["book"]})), flags=("target_generalization",))
    add("incomplete", "Non ho scelto il professionista: cerca un imbianchino a Vercelli e fermati prima di assegnargli il lavoro.",
        c(g("find", "imbianchino", "professional_service", [f("location", "Vercelli", "near")], policy={"forbid": ["hire"]})))

    return result


def main() -> None:
    if DEST.exists():
        raise FileExistsError("V36 is already frozen")
    panel = cases()
    if len(panel) < 42 or len({normalize_text(x["text"]) for x in panel}) != len(panel):
        raise RuntimeError("Need at least 42 distinct V36 requests")
    older = []
    for path in PRIOR:
        if not path.is_file():
            continue
        for row in read_jsonl(path):
            text = row.get("text") or row.get("request")
            if text is None:
                text = next((m["content"] for m in row.get("messages", [])
                             if m.get("role") == "user"), None)
            if text is not None:
                older.append((path, text))
    previous = [(path, text, token_set(text)) for path, text in older]
    exact = {normalize_text(text) for _, text in older}
    maximum, nearest, within = 0.0, None, 0.0
    for i, row in enumerate(panel):
        if normalize_text(row["text"]) in exact:
            raise RuntimeError(f"Exact prior request: {row['id']}")
        current = token_set(row["text"])
        for path, text, old in previous:
            score = len(current & old) / len(current | old) if current | old else 0.0
            if score > maximum:
                maximum, nearest = score, {"case": row["id"], "source": str(path.relative_to(ROOT)),
                                           "prior_text": text}
        for other in panel[:i]:
            old = token_set(other["text"])
            within = max(within, len(current & old) / len(current | old) if current | old else 0.0)
    if maximum >= 0.8 or within >= 0.8:
        raise RuntimeError(f"Near duplicate: prior={maximum:.3f}, within={within:.3f}, nearest={nearest}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel_path, review_path = DEST / "blind.jsonl", DEST / "human_review_queue.jsonl"
    with panel_path.open("w", encoding="utf-8", newline="\n") as out:
        for row in panel:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    with review_path.open("w", encoding="utf-8", newline="\n") as out:
        for row in panel:
            out.write(json.dumps({"id": row["id"], "request": row["text"], "family": row["family"],
                                  "raw_gold": row["contract"], "review_flags": row["review_flags"],
                                  "human_review_status": "pending", "human_reviewer": None,
                                  "human_notes": None}, ensure_ascii=False) + "\n")
    manifest = {"version": "V36", "schema_version": "1.3", "cases": len(panel),
                "families": dict(sorted(Counter(x["family"] for x in panel).items())),
                "panel_sha256": sha256(panel_path), "review_queue_sha256": sha256(review_path),
                "generator_sha256": sha256(Path(__file__)),
                "max_prior_token_jaccard": round(maximum, 6),
                "max_within_panel_token_jaccard": round(within, 6), "nearest_prior_source": nearest,
                "prior_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in PRIOR if path.is_file()},
                "gold_review": "agent-authored; pending independent human review",
                "purpose": "independent typed Planner handoff transfer, frozen before post-V35 fixes"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("cases", "panel_sha256", "max_prior_token_jaccard",
                                                  "max_within_panel_token_jaccard")}, indent=2))


if __name__ == "__main__":
    main()
