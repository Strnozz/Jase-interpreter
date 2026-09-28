"""Freeze agent-authored train-search E2E cases before provider implementation.

Not training data. The gold and system expectations require independent review.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.build_v28_blind import c, f, g, missing, t  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

OUT = ROOT / "benchmarks/train_slice_v1"
PRIOR = sorted(set(ROOT.glob("benchmarks/**/*.jsonl")) | set(ROOT.glob("data/*/train.jsonl")) |
             set(ROOT.glob("data/*/valid.jsonl")))


def cases() -> list[dict]:
    rows = []

    def add(family, text, gold, expected, flags=()):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((text, [x.as_dict() for x in issues]))
        rows.append({"id": f"train-v1-{len(rows)+1:03d}", "family": family, "text": text,
                     "contract": gold, "expected_system": expected, "review_flags": list(flags)})

    def train(origin=None, destination=None, date=None, *, times=(), extra=(), modifiers=None,
              policy=None, action="find", name="treno", absent=()):
        facts = ([f("origin", origin)] if origin else []) + ([f("destination", destination)] if destination else []) + list(extra)
        temporal = ([t("search_date", date)] if date else []) + list(times)
        missing_fields = [missing(field, "search") for field in absent]
        kwargs = {"temporal": temporal, "missing": missing_fields}
        if modifiers:
            kwargs["modifiers"] = modifiers
        if policy:
            kwargs["policy"] = policy
        return c(g(action, name, "transport", facts, **kwargs))

    # Ordinary requests and relative dates.
    add("basic", "Cerco il treno da Lecco a Monza per il 14 ottobre; mostrami gli orari.", train("Lecco", "Monza", "14 ottobre"), "READ_IF_COVERED")
    add("basic", "Per sabato vorrei andare in ferrovia da Brescia a Bergamo. Quali corse ci sono?", train("Brescia", "Bergamo", "sabato"), "READ_IF_COVERED")
    add("basic", "Mi elenchi i collegamenti ferroviari fra Como San Giovanni e Milano Centrale il 3 novembre?", train("Como San Giovanni", "Milano Centrale", "3 novembre"), "READ_IF_COVERED")
    add("basic", "Dovrei partire da Pavia e arrivare a Lodi martedì: cerca i treni disponibili.", train("Pavia", "Lodi", "martedì"), "READ_IF_COVERED")
    add("relative_date", "Sono a Varese. Domani devo raggiungere Saronno in treno.", train("Varese", "Saronno", "domani"), "READ_IF_COVERED")
    add("relative_date", "Cerca treni da Milano Porta Garibaldi verso Novara dopodomani.", train("Milano Porta Garibaldi", "Novara", "dopodomani"), "READ_IF_COVERED")
    add("relative_date", "Per venerdì prossimo mi servono le corse ferroviarie Mantova-Cremona.", train("Mantova", "Cremona", "venerdì prossimo"), "READ_IF_COVERED")
    add("relative_date", "Vorrei sapere come andare in treno da Desenzano a Verona domenica.", train("Desenzano", "Verona", "domenica"), "READ_IF_COVERED")

    # Operators must survive the Interpreter and binding unchanged.
    add("departure_after", "Giovedì esco dall'ufficio alle 15: da Milano Centrale a Como voglio partire dopo quell'ora.", train("Milano Centrale", "Como", "giovedì", times=[t("departure_time", "15:00", "after")]), "READ_IF_COVERED")
    add("departure_after", "Da Bergamo per Treviglio il 9 dicembre, nessuna partenza prima delle 18:20.", train("Bergamo", "Treviglio", "9 dicembre", times=[t("departure_time", "18:20", "after")]), "READ_IF_COVERED")
    add("departure_after", "Quale treno prendo da Gallarate a Busto Arsizio lunedì se posso salire solo dopo le 11?", train("Gallarate", "Busto Arsizio", "lunedì", times=[t("departure_time", "11:00", "after")]), "READ_IF_COVERED")
    add("arrival_before", "Mercoledì ho una visita a Milano alle dieci: trovami un treno da Como che arrivi entro le 09:30.", train("Como", "Milano", "mercoledì", times=[t("arrival_time", "09:30", "before")]), "READ_IF_COVERED")
    add("arrival_before", "Da Cremona a Brescia venerdì, devo essere sul posto prima delle 08:45.", train("Cremona", "Brescia", "venerdì", times=[t("arrival_time", "08:45", "before")]), "READ_IF_COVERED")
    add("arrival_before", "Il 22 novembre devo arrivare a Monza non oltre le 17:00 partendo da Lecco.", train("Lecco", "Monza", "22 novembre", times=[t("arrival_time", "17:00", "before")]), "READ_IF_COVERED")
    add("arrival_after", "Sabato da Lodi a Pavia, cerca una corsa che arrivi dopo le 19.", train("Lodi", "Pavia", "sabato", times=[t("arrival_time", "19:00", "after")]), "READ_IF_COVERED")
    add("ranges", "Tra le 8 e le 10 di domenica vorrei partire da Varese per Milano Cadorna in treno.", train("Varese", "Milano Cadorna", "domenica", times=[{**t("departure_time", "08:00", "between"), "value_to": "10:00"}]), "READ_IF_COVERED")
    add("ranges", "Per il 5 dicembre mostra solo partenze da Novara a Milano fra le 13:30 e le 15:30.", train("Novara", "Milano", "5 dicembre", times=[{**t("departure_time", "13:30", "between"), "value_to": "15:30"}]), "READ_IF_COVERED")
    add("daypart", "Domani mattina devo andare da Monza a Milano in treno.", train("Monza", "Milano", "domani", times=[t("search_time", "mattina")]), "READ_IF_COVERED")
    add("daypart", "Martedì sera vedrei volentieri i treni Brescia-Verona.", train("Brescia", "Verona", "martedì", times=[t("search_time", "sera")]), "READ_IF_COVERED")
    add("daypart", "Cerco una corsa ferroviaria pomeridiana da Parma a Piacenza per lunedì.", train("Parma", "Piacenza", "lunedì", times=[t("search_time", "pomeriggio")]), "READ_IF_COVERED")

    # Filters: some must HOLD if data or schema cannot support them.
    add("direct", "Da Milano a Varese sabato, mostrami soltanto treni senza cambi.", train("Milano", "Varese", "sabato", extra=[f("stops", 0)]), "READ_IF_COVERED")
    add("direct", "Vorrei un collegamento diretto fra Pavia e Milano il 12 ottobre, niente coincidenze.", train("Pavia", "Milano", "12 ottobre", extra=[f("stops", 0)]), "READ_IF_COVERED")
    add("regional", "Per domani cerca solo regionali da Brescia a Iseo.", train("Brescia", "Iseo", "domani", extra=[f("train_category", "regional")], name="treno regionale"), "READ_IF_SUPPORTED")
    add("regional", "Mi servono gli orari dei treni regionali fra Lecco e Colico per domenica.", train("Lecco", "Colico", "domenica", extra=[f("train_category", "regional")], name="treno regionale"), "READ_IF_SUPPORTED")
    add("high_speed", "Cerca un'alta velocità da Milano a Bologna per venerdì mattina.", train("Milano", "Bologna", "venerdì", extra=[f("train_category", "high_speed")], times=[t("search_time", "mattina")]), "HOLD_IF_UNSUPPORTED")
    add("high_speed", "Il 18 novembre voglio solo Frecciarossa da Torino a Roma.", train("Torino", "Roma", "18 novembre", extra=[f("train_service", "Frecciarossa")]), "HOLD_IF_UNSUPPORTED")
    add("category", "Per sabato c'è un Intercity da Milano Centrale a Genova Brignole?", train("Milano Centrale", "Genova Brignole", "sabato", extra=[f("train_service", "Intercity")]), "HOLD_IF_UNSUPPORTED")
    add("class", "Cerca da Como a Milano domani soltanto soluzioni in prima classe.", train("Como", "Milano", "domani", extra=[f("travel_class", "first")]), "HOLD_IF_UNSUPPORTED")
    add("max_price", "Da Milano a Bergamo martedì spendendo al massimo 12 euro, quali treni trovo?", train("Milano", "Bergamo", "martedì", extra=[f("price", 12, "lte", currency="EUR")]), "HOLD_IF_PRICE_UNAVAILABLE")
    add("max_price", "Per il 20 ottobre cerca corse da Como a Monza che costino meno di 8 euro.", train("Como", "Monza", "20 ottobre", extra=[f("price", 8, "lt", currency="EUR")]), "HOLD_IF_PRICE_UNAVAILABLE")
    add("max_duration", "Domani da Saronno a Milano, non voglio viaggi oltre 45 minuti.", train("Saronno", "Milano", "domani", extra=[f("duration", 45, "lte", unit="minute")]), "READ_IF_COVERED")
    add("max_duration", "Il 21 ottobre da Lecco a Como: massimo un'ora di percorso.", train("Lecco", "Como", "21 ottobre", extra=[f("duration", 60, "lte", unit="minute")]), "READ_IF_COVERED")
    add("limits", "Elencami i primi tre treni da Pavia a Milano per venerdì, senza prenotare.", train("Pavia", "Milano", "venerdì", modifiers={"limit": 3}, policy={"forbid": ["book"]}), "READ_IF_COVERED")
    add("limits", "Da Monza a Bergamo il 2 novembre fammi vedere solo due opzioni.", train("Monza", "Bergamo", "2 novembre", modifiers={"limit": 2}), "READ_IF_COVERED")
    add("sort", "Per domani ordina i treni Milano-Novara dall'orario di partenza più vicino.", train("Milano", "Novara", "domani", modifiers={"sort": {"field": "departure_time", "direction": "asc"}}), "READ_IF_COVERED")
    add("sort", "Da Brescia a Verona sabato, metti per primo quello che arriva prima.", train("Brescia", "Verona", "sabato", modifiers={"sort": {"field": "arrival_time", "direction": "asc"}}), "READ_IF_COVERED")
    add("sort", "Lunedì da Como a Milano: proponi le corse dalla durata minore.", train("Como", "Milano", "lunedì", modifiers={"sort": {"field": "duration", "direction": "asc"}}), "READ_IF_COVERED")
    add("sort", "Venerdì da Varese a Saronno, ordinami i biglietti dal meno caro.", train("Varese", "Saronno", "venerdì", modifiers={"sort": {"field": "price", "direction": "asc"}}), "HOLD_IF_PRICE_UNAVAILABLE")

    # Missing, conflicting and contextual information.
    add("missing", "Per martedì mostrami i treni diretti a Cremona; la partenza te la dico dopo.", train(destination="Cremona", date="martedì", absent=("origin",)), "HOLD_MISSING_INFORMATION")
    add("missing", "Partirei da Bergamo venerdì, ma non ho ancora scelto dove andare in treno.", train(origin="Bergamo", date="venerdì", absent=("destination",)), "HOLD_MISSING_INFORMATION")
    add("missing", "Cerca collegamenti ferroviari da Monza a Varese; decidiamo il giorno più avanti.", train("Monza", "Varese", absent=("date",)), "HOLD_MISSING_INFORMATION")
    add("missing", "Vorrei un treno per Roma domani mattina; non so da quale stazione partirò.", train(destination="Roma", date="domani", times=[t("search_time", "mattina")], absent=("origin",)), "HOLD_MISSING_INFORMATION")
    add("context", "Cerca il treno da qui a Milano per lunedì: non hai ancora la mia posizione.", train(destination="Milano", date="lunedì", absent=("origin",)), "HOLD_UNRESOLVED_REFERENCE")
    add("context", "Dal posto in cui lavoro a Torino mercoledì, mostrami i treni; non sai dov'è il mio ufficio.", train(destination="Torino", date="mercoledì", absent=("origin",)), "HOLD_UNRESOLVED_REFERENCE")
    add("conflict", "Cerca un treno da Milano a Como martedì che parta dopo le 19 ma arrivi prima delle 18.", train("Milano", "Como", "martedì", times=[t("departure_time", "19:00", "after"), t("arrival_time", "18:00", "before")]), "HOLD_TEMPORAL_CONFLICT")
    add("conflict", "Da Brescia a Bergamo giovedì, parto dopo le 16 e devo arrivare entro le 15 dello stesso giorno.", train("Brescia", "Bergamo", "giovedì", times=[t("departure_time", "16:00", "after"), t("arrival_time", "15:00", "before")]), "HOLD_TEMPORAL_CONFLICT")

    # Meta-language and quoted requests must never trigger a provider read.
    add("non_action", "Come tradurresti in inglese «trovami un treno per Roma domani»?", c(), "NO_PROVIDER_CALL")
    add("non_action", "Scrivimi un esempio di domanda che potrei fare per cercare un treno a Napoli.", c(), "NO_PROVIDER_CALL")
    add("non_action", "Se dicessi 'cerca un treno per Milano', quale informazione ti mancherebbe?", c(), "NO_PROVIDER_CALL")
    add("non_action", "Sto preparando un articolo sui regionali che passano da Pavia; non ho chiesto una ricerca di orari.", c(), "NO_PROVIDER_CALL")
    add("non_action", "Nel manuale dell'app compare il comando 'trova treni da Lecco a Milano'. Spiegamelo.", c(), "NO_PROVIDER_CALL")
    add("non_action", "Puoi correggere la grammatica della frase «prenotami un Frecciarossa per domani»?", c(), "NO_PROVIDER_CALL")

    # Multi-goal and write requests. Search may proceed only if the plan preserves all goals atomically.
    add("policy", "Cerca da Milano a Como per sabato ma non comprare né riservare alcun posto.", train("Milano", "Como", "sabato", policy={"forbid": ["buy", "book"]}), "READ_IF_COVERED")
    add("policy", "Mostrami le corse da Cremona a Pavia domani; scelgo io se acquistare.", train("Cremona", "Pavia", "domani", policy={"forbid": ["buy"]}), "READ_IF_COVERED")
    add("booking", "Prenota un posto sul treno da Milano a Torino per il 17 ottobre, dopo avermi mostrato le opzioni.", c(g("book", "treno", "transport", [f("origin", "Milano"), f("destination", "Torino")], temporal=[t("action_date", "17 ottobre")], policy={"confirm_before": ["book"]}, missing=[missing("selection")])), "NO_WRITE")
    add("booking", "Compra subito un biglietto ferroviario per domani da Verona a Brescia.", c(g("buy", "biglietto ferroviario", "ticket", [f("origin", "Verona"), f("destination", "Brescia")], temporal=[t("action_date", "domani")])), "NO_WRITE")
    add("multi_goal", "Cerca un treno Lecco-Milano per domani e ricordami alle sette di prendere la valigia.", c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Milano")], temporal=[t("search_date", "domani")], idx=1), g("notify", "promemoria", "reminder", [f("message", "prendere la valigia")], temporal=[t("notification_time", "07:00")], missing=[missing("date")], idx=2)), "HOLD_MULTI_GOAL_UNSUPPORTED")
    add("multi_goal", "Per lunedì trova i treni da Monza a Como e, separatamente, un albergo vicino alla stazione.", c(g("find", "treno", "transport", [f("origin", "Monza"), f("destination", "Como")], temporal=[t("search_date", "lunedì")], idx=1), g("find", "albergo", "accommodation", [f("location", "stazione di Como", "near")], idx=2)), "HOLD_MULTI_GOAL_UNSUPPORTED")
    add("references", "Tra i treni trovati ieri per Milano, mostrami il secondo: non hai conservato quei risultati.", train(destination="Milano", absent=("origin", "date")), "HOLD_UNRESOLVED_REFERENCE")
    add("references", "Quello più economico dei treni per Bergamo: non abbiamo ancora fatto una ricerca.", train(destination="Bergamo", absent=("origin", "date"), modifiers={"sort": {"field": "price", "direction": "asc"}}), "HOLD_UNRESOLVED_REFERENCE")
    add("references", "Confronta il primo e il terzo collegamento per Pavia, ma non ci sono risultati precedenti.", c(g("compare", "treno", "transport", [f("destination", "Pavia")], missing=[missing("origin", "search"), missing("date", "search")], modifiers={"compare_count": 2})), "HOLD_UNRESOLVED_REFERENCE")
    return rows


def main() -> None:
    if OUT.exists():
        raise FileExistsError("Frozen train panel already exists")
    rows = cases()
    if len(rows) < 50 or len({normalize_text(x["text"]) for x in rows}) != len(rows):
        raise RuntimeError("Need >=50 unique requests")
    older = []
    for path in PRIOR:
        if path.is_file():
            for row in read_jsonl(path):
                text = row.get("text") or row.get("request") or next(
                    (m["content"] for m in row.get("messages", []) if m.get("role") == "user"), None)
                if text:
                    older.append((path, text, token_set(text)))
    exact = {normalize_text(text) for _, text, _ in older}
    maximum, nearest, within = 0.0, None, 0.0
    for index, row in enumerate(rows):
        if normalize_text(row["text"]) in exact:
            raise RuntimeError(f"Exact overlap: {row['id']}")
        current = token_set(row["text"])
        for path, text, tokens in older:
            score = len(current & tokens) / len(current | tokens) if current | tokens else 0
            if score > maximum:
                maximum, nearest = score, {"id": row["id"], "source": str(path.relative_to(ROOT)), "text": text}
        for prior in rows[:index]:
            tokens = token_set(prior["text"])
            within = max(within, len(current & tokens) / len(current | tokens) if current | tokens else 0)
    if maximum >= 0.8 or within >= 0.8:
        raise RuntimeError(f"Near-overlap prior={maximum:.3f} within={within:.3f} nearest={nearest}")
    OUT.mkdir()
    panel = OUT / "blind.jsonl"
    panel.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    review = OUT / "human_review_queue.jsonl"
    review.write_text("".join(json.dumps({"id": row["id"], "request": row["text"],
                                  "raw_gold": row["contract"], "expected_system": row["expected_system"],
                                  "status": "PENDING_INDEPENDENT_HUMAN_REVIEW"}, ensure_ascii=False) + "\n"
                              for row in rows), encoding="utf-8")
    manifest = {"version": "TRAIN_SLICE_V1", "schema_version": "1.3", "cases": len(rows),
                "families": dict(sorted(Counter(row["family"] for row in rows).items())),
                "panel_sha256": sha256(panel), "review_queue_sha256": sha256(review),
                "generator_sha256": sha256(Path(__file__)),
                "max_prior_token_jaccard": round(maximum, 6),
                "max_within_panel_token_jaccard": round(within, 6), "nearest_prior": nearest,
                "gold_review": "agent-authored; pending independent human review",
                "training_use": "FORBIDDEN", "frozen_before_provider_implementation": True}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("cases", "panel_sha256", "max_prior_token_jaccard", "max_within_panel_token_jaccard")}, indent=2))


if __name__ == "__main__":
    main()
