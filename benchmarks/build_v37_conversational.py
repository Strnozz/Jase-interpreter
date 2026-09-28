"""Freeze conversational transfer cases before V27 data or runtime remediation.

Agent-authored gold is a diagnostic proposal, not independent human adjudication.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "v37_conversational"


def norm(value: str) -> str:
    import re
    return " ".join(re.findall(r"\w+", value.casefold(), re.UNICODE))


def tokens(value: str) -> set[str]:
    return set(norm(value).split())


def raw_text(row: dict) -> str:
    if "messages" in row:
        return " ".join(m["content"] for m in row["messages"] if m["role"] == "user")
    if "turns" in row:
        return " ".join(t["user"] for t in row["turns"])
    return row.get("text", row.get("request", ""))


def make() -> list[dict]:
    panel: list[dict] = []

    def add(family: str, turns: list[tuple[str, str, str, str, str, dict, str]]) -> None:
        built = []
        for utterance, mode, operation, action, target, slots, next_step in turns:
            built.append({"user": utterance, "gold": {"mode": mode, "operation": operation,
                          "action": action or None, "target": target or None,
                          "slot_updates": slots, "expected_next": next_step}})
        panel.append({"id": f"v37-{len(panel)+1:03d}", "family": family,
                      "turns": built, "human_review_status": "pending"})

    nonaction = {
        "CHAT": [
            "Che bella luce c'è oggi sul lago.", "Sono contento che domani sia festa.",
            "Il viaggio di ieri è stato davvero piacevole.", "Mi piace ascoltare la pioggia dalla finestra.",
            "Oggi ho finalmente finito il mio libro.", "La città di sera sembra diversa dal mattino.",
            "Non vedo l'ora di raccontarti della vacanza.", "Il profumo del pane mi ricorda casa.",
            "Ho incontrato un vecchio amico in stazione.", "Mi sento più tranquillo dopo una passeggiata."],
        "QUESTION": [
            "Qual è la differenza generale tra un treno regionale e uno veloce?",
            "Come si calcola di solito la durata di un viaggio con cambio?",
            "Che cosa significa una tariffa ferroviaria flessibile?",
            "Perché alcuni alberghi chiedono una carta a garanzia?",
            "Come funziona in generale una lista d'attesa al ristorante?",
            "Che differenza c'è tra un promemoria e un appuntamento?",
            "Quando conviene confrontare più preventivi per un idraulico?",
            "Che cosa indica la classe di un biglietto ferroviario?",
            "Come vengono ordinati i risultati di ricerca per prezzo?",
            "Cosa cambia tra noleggiare una bici e comprarla?"],
        "META": [
            "Se ti chiedessi di cercare un treno, quali dati domanderesti?",
            "Fammi un esempio di frase per richiedere un albergo senza prenotarlo.",
            "Spiegami come interpreteresti l'ordine 'manda un messaggio a Lea'.",
            "Sto scrivendo un manuale: descrivi un dialogo di prenotazione.",
            "In una demo, che cosa significherebbe 'trova un veterinario'?",
            "Mostrami la struttura di una richiesta per fissare una visita medica.",
            "Per un articolo, illustra quando chiedere conferma prima di comprare.",
            "Nella documentazione, come rappresenteresti una ricerca di voli?",
            "Spiega a un collega cosa faresti davanti a 'cerca un ristorante'.",
            "Vorrei capire quali passaggi seguirebbe Jase per noleggiare un'auto."],
        "TRANSLATION": [
            "Come traduco in inglese 'trovami un treno per Pisa'?",
            "Traduci in francese la frase 'prenota un tavolo per sei'.",
            "In tedesco, come si dice 'non comprare ancora il biglietto'?",
            "Dammi la traduzione spagnola di 'cerca un hotel vicino al porto'.",
            "Qual è la resa inglese di 'ricordami di ritirare il pacco'?",
            "Puoi tradurre 'manda una mail a Giulia' in portoghese?",
            "Come si esprime in inglese 'il secondo risultato è più economico'?",
            "Traduci 'trova un medico disponibile oggi' in francese.",
            "Per un esercizio, traduci 'prenotami un volo diretto'.",
            "In inglese britannico come diresti 'cerco un treno regionale'?"],
        "HYPOTHETICAL": [
            "Se un giorno dovessi partire da Lecce, potrei cercare un treno qui?",
            "Immagina che volessi cambiare hotel: quali vincoli sarebbero rilevanti?",
            "Se ipoteticamente ordinassi una pizza, come funzionerebbe la conferma?",
            "Supponiamo che qualcuno debba arrivare a Parma presto: cosa servirebbe sapere?",
            "Se in futuro cercassi un'officina, potrei indicare il quartiere?",
            "In uno scenario fittizio, una persona chiede un volo per Oslo: cosa succede?",
            "Se avessi bisogno di un idraulico, potresti spiegarmi il processo?",
            "Immagina di ricevere 'prenota quello': come sapresti a cosa si riferisce?",
            "Se domani volessi visitare Ravenna, come valuterei i collegamenti?",
            "In teoria, come chiederesti un promemoria per una visita?"],
    }
    for mode, utterances in nonaction.items():
        for utterance in utterances:
            add(mode.casefold(), [(utterance, mode, "NO_GOAL", "", "", {}, "NO_ACTION")])

    complete = [
        ("Trovami treni da Pavia a Genova mercoledì mattina, senza cambi.", "find", "treno", {"origin": "Pavia", "destination": "Genova", "search_date": "mercoledì", "direct_only": True}),
        ("Cerca un albergo nel centro di Trieste per sabato notte.", "find", "hotel", {"location": "centro di Trieste", "search_date": "sabato"}),
        ("Mostrami tre ristoranti vegani a Perugia aperti stasera.", "find", "ristorante", {"location": "Perugia", "cuisine": "vegana", "limit": 3, "search_date": "stasera"}),
        ("Trova un elettricista disponibile oggi a Imola.", "find", "elettricista", {"location": "Imola", "availability": "oggi"}),
        ("Cerca voli da Venezia a Lisbona venerdì, solo diretti.", "find", "volo", {"origin": "Venezia", "destination": "Lisbona", "search_date": "venerdì", "direct_only": True}),
        ("Ricordami domani alle sette di portare la cartella.", "notify", "promemoria", {"action_date": "domani", "action_time": "07:00", "message": "portare la cartella"}),
        ("Confronta quattro biciclette pieghevoli sotto i 600 euro, dalla meno cara.", "compare", "bicicletta", {"compare_count": 4, "price_max": 600, "sort": "price"}),
        ("Fammi vedere auto a noleggio a Modena per lunedì, senza prenotarne una.", "find", "auto", {"location": "Modena", "search_date": "lunedì", "forbid": "rent"}),
        ("Cerca una farmacia aperta ora vicino a piazza Duomo a Lecce.", "find", "farmacia", {"location": "piazza Duomo a Lecce", "open_now": True}),
        ("Mandami un promemoria martedì alle 16: ritirare le chiavi.", "notify", "promemoria", {"action_date": "martedì", "action_time": "16:00", "message": "ritirare le chiavi"}),
        ("Trova treni da Savona a Torino con arrivo prima delle 11 di domani.", "find", "treno", {"origin": "Savona", "destination": "Torino", "search_date": "domani", "arrival_before": "11:00"}),
        ("Cerca un ristorante giapponese a Fano per venerdì sera.", "find", "ristorante", {"location": "Fano", "cuisine": "giapponese", "search_date": "venerdì"}),
        ("Mostrami alberghi a Orvieto ordinati dal più economico, senza prenotare.", "find", "hotel", {"location": "Orvieto", "sort": "price", "forbid": "book"}),
        ("Trova un treno da Cuneo a Milano dopo le 17 di giovedì.", "find", "treno", {"origin": "Cuneo", "destination": "Milano", "search_date": "giovedì", "departure_after": "17:00"}),
        ("Vorrei tre opzioni di volo Bologna-Atene per domenica mattina.", "find", "volo", {"origin": "Bologna", "destination": "Atene", "search_date": "domenica", "limit": 3}),
        ("Cerca un veterinario aperto oggi a Mantova.", "find", "veterinario", {"location": "Mantova", "open_now": True}),
        ("Mostrami ristoranti a Como accessibili a piedi dalla stazione.", "find", "ristorante", {"location": "stazione di Como", "access": "walk"}),
        ("Trova treni regionali fra Vicenza e Verona per lunedì pomeriggio.", "find", "treno", {"origin": "Vicenza", "destination": "Verona", "search_date": "lunedì", "category": "regionale"}),
        ("Cerca camere a Spoleto disponibili sabato prossimo.", "find", "camera", {"location": "Spoleto", "search_date": "sabato prossimo", "availability": True}),
        ("Ricordami venerdì mattina di chiamare l'assicurazione.", "notify", "promemoria", {"action_date": "venerdì", "daypart": "mattina", "message": "chiamare l'assicurazione"}),
    ]
    for utterance, action, target, slot in complete:
        add("complete", [(utterance, "ACTIONABLE_GOAL", "OPEN_DRAFT", action, target, slot,
                          "CHECK_REQUIREMENTS")])

    incomplete = [
        ("Trovami un treno per Ancona dopodomani.", "find", "treno", {"destination": "Ancona", "search_date": "dopodomani"}, "origin"),
        ("Vorrei un treno da Brindisi per sabato.", "find", "treno", {"origin": "Brindisi", "search_date": "sabato"}, "destination"),
        ("Cercami un regionale da Firenze a Pisa.", "find", "treno", {"origin": "Firenze", "destination": "Pisa", "category": "regionale"}, "search_date"),
        ("Prenota un tavolo domani sera.", "book", "ristorante", {"search_date": "domani", "daypart": "sera"}, "location,party_size"),
        ("Mi serve una cena per sei sabato; trova un locale e poi prenota.", "book", "ristorante", {"party_size": 6, "search_date": "sabato"}, "location"),
        ("Prenotami un albergo a Urbino per il weekend.", "book", "hotel", {"location": "Urbino", "stay_period": "weekend"}, "DISCOVER_RESULT"),
        ("Cerca un hotel vicino al teatro.", "find", "hotel", {"location": "vicino al teatro"}, "DISAMBIGUATE_LOCATION"),
        ("Mi trovi un meccanico disponibile oggi?", "find", "meccanico", {"availability": "oggi"}, "location"),
        ("Ricordami di passare in banca.", "notify", "promemoria", {"message": "passare in banca"}, "action_time"),
        ("Fammi una ricerca di voli per Praga venerdì.", "find", "volo", {"destination": "Praga", "search_date": "venerdì"}, "origin"),
        ("Vorrei noleggiare una macchina per domani.", "rent", "auto", {"search_date": "domani"}, "location"),
        ("Fissa una visita dal dentista a maggio.", "book", "dentista", {"month": "maggio"}, "date,time"),
        ("Cerca una farmacia nelle vicinanze.", "find", "farmacia", {}, "location"),
        ("Prenotami due posti per un volo a Berlino.", "book", "volo", {"destination": "Berlino", "party_size": 2}, "origin,date,DISCOVER_RESULT"),
        ("Trova un idraulico per una perdita nel bagno.", "find", "idraulico", {"specialization": "perdita nel bagno"}, "location"),
        ("Mandagli un messaggio dicendo che arriverò tardi.", "contact", "persona", {"message": "arriverò tardi"}, "recipient"),
        ("Cerca un posto per mangiare oggi.", "find", "ristorante", {"search_date": "oggi"}, "location"),
        ("Mi piacerebbe confrontare tre offerte di furgoni.", "compare", "furgone", {"compare_count": 3}, "location"),
        ("Prenota una camera per due la prossima settimana.", "book", "camera", {"party_size": 2, "stay_period": "prossima settimana"}, "location,DISCOVER_RESULT"),
        ("Trova treni che arrivino prima delle otto.", "find", "treno", {"arrival_before": "08:00"}, "origin,destination,date"),
    ]
    for utterance, action, target, slot, missing in incomplete:
        next_step = "DISCOVER_READ_ONLY" if missing == "DISCOVER_RESULT" else "ASK_USER"
        add("incomplete", [(utterance, "ACTIONABLE_GOAL", "OPEN_DRAFT", action, target,
                            slot, next_step)])

    sequences = [
        ("train_origin", "Cercami un treno per Rimini domani dopo le 14.", "Da Ferrara.", "origin", "Ferrara"),
        ("train_origin", "Mi serve un regionale per Udine mercoledì mattina.", "Parto da Gorizia.", "origin", "Gorizia"),
        ("train_destination", "Trovami collegamenti ferroviari da Arezzo sabato.", "Verso Bologna.", "destination", "Bologna"),
        ("train_date", "Cerca treni da Bergamo a Brescia.", "Per lunedì prossimo.", "search_date", "lunedì prossimo"),
        ("train_date", "Voglio andare in treno da Asti a Torino.", "Domani mattina.", "search_date", "domani"),
        ("train_correction", "Cerca treni da Verona a Padova venerdì.", "Anzi, parto da Vicenza.", "origin", "Vicenza"),
        ("train_correction", "Treni da Siena a Roma giovedì, per favore.", "Facciamo sabato invece.", "search_date", "sabato"),
        ("train_constraint", "Mostrami treni da Pisa a Livorno domenica.", "Solo senza cambi.", "direct_only", True),
        ("train_constraint", "Cerca treni da Como a Milano domani.", "Devo arrivare entro le nove.", "arrival_before", "09:00"),
        ("restaurant_party", "Vorrei prenotare un ristorante a Lucca sabato alle 20.", "Siamo cinque.", "party_size", 5),
        ("restaurant_location", "Prenotami una cena per due martedì sera.", "A Treviso, possibilmente in centro.", "location", "centro di Treviso"),
        ("restaurant_correction", "Cerca ristoranti a Salerno per quattro.", "No, siamo in tre.", "party_size", 3),
        ("hotel_location", "Mi trovi un hotel per venerdì notte?", "A Viterbo.", "location", "Viterbo"),
        ("hotel_policy", "Cerca camere ad Ascoli per domenica.", "Però non prenotare niente.", "forbid", "book"),
        ("service_location", "Mi serve un elettricista oggi.", "Nel quartiere San Paolo a Bari.", "location", "San Paolo a Bari"),
        ("service_cancellation", "Trova un idraulico a Rovigo per lunedì.", "Lascia perdere, ho risolto.", "", ""),
        ("reminder_time", "Ricordami di chiamare mia sorella.", "Alle diciotto di venerdì.", "action_time", "18:00"),
        ("reminder_correction", "Ricordami alle 8 di comprare il pane.", "Non alle otto: alle nove.", "action_time", "09:00"),
        ("flight_origin", "Mostrami voli per Cracovia domenica.", "Da Napoli.", "origin", "Napoli"),
        ("flight_policy", "Cerca voli da Catania a Madrid il 7 ottobre.", "Non acquistare ancora.", "forbid", "buy"),
        ("vehicle_location", "Vorrei cercare una bici a noleggio domani.", "A La Spezia.", "location", "La Spezia"),
        ("appointment_cancel", "Fissa una visita dal medico per martedì.", "No, annulla la richiesta.", "", ""),
        ("result_selection", "Mostrami tre alberghi a Pesaro per il weekend.", "Il secondo mi interessa.", "result_ordinal", 2),
        ("result_selection", "Cerca quattro treni da Faenza a Ravenna lunedì.", "Mostrami il più economico.", "result_rank", "cheapest"),
        ("confirmation", "Prepara un messaggio a Carla: arrivo alle 19.", "Sì, invialo.", "", ""),
        ("confirmation_denial", "Vorrei una prenotazione al ristorante vicino al museo.", "No, non prenotare.", "forbid", "book"),
        ("quote_followup", "Spiegami cosa significherebbe 'cerca un treno per Roma'.", "E se lo dicessi in spagnolo?", "", ""),
        ("multi_goal", "Cerca un hotel a Trento e un treno da Bolzano per sabato.", "L'hotel vicino alla stazione, il treno verso Trento.", "destination", "Trento"),
        ("reference_unresolved", "Vorrei prenotare il secondo ristorante.", "Quello visto ieri, credo.", "result_ordinal", 2),
        ("shopping_correction", "Confronta telefoni sotto 400 euro.", "In realtà il limite è 350.", "price_max", 350),
    ]
    for family, opening, followup, field, value in sequences:
        if family == "quote_followup":
            add(family, [(opening, "META", "NO_GOAL", "", "", {}, "NO_ACTION"),
                         (followup, "META", "NO_GOAL", "", "", {}, "NO_ACTION")])
            continue
        action = "find"
        target = ("treno" if family.startswith("train") or family in {"result_selection"} and "treni" in opening
                  else "ristorante" if family.startswith("restaurant") or "ristorante" in opening
                  else "hotel" if family.startswith("hotel") or "alberghi" in opening
                  else "elettricista" if family.startswith("service") and "elettricista" in opening
                  else "idraulico" if family.startswith("service") else "promemoria" if family.startswith("reminder")
                  else "volo" if family.startswith("flight") else "bicicletta" if family.startswith("vehicle")
                  else "visita" if family.startswith("appointment") else "messaggio" if family.startswith("confirmation")
                  else "telefono" if family.startswith("shopping") else "treno")
        if family.startswith("reminder"):
            action = "notify"
        elif family in {"appointment_cancel", "restaurant_party", "restaurant_location", "confirmation_denial"}:
            action = "book"
        elif family == "confirmation":
            action = "contact"
        elif family == "shopping_correction":
            action = "compare"
        mode = "ANSWER_TO_CLARIFICATION" if family in {"train_origin", "train_destination", "train_date", "restaurant_party", "restaurant_location", "hotel_location", "service_location", "reminder_time", "flight_origin", "vehicle_location"} else "CONTINUATION"
        op = "UPDATE_DRAFT"
        next_step = "CHECK_REQUIREMENTS"
        if family.endswith("cancel") or family.endswith("cancellation"):
            mode, op, next_step = "CANCELLATION", "CANCEL_DRAFT", "NO_ACTION"
        elif family == "confirmation":
            mode, op, next_step = "CONFIRMATION", "CONFIRM_DRAFT", "CHECK_CONFIRMATION_BINDING"
        elif family == "confirmation_denial":
            mode, op, next_step = "REJECTION", "REJECT_DRAFT", "NO_ACTION"
        elif family in {"result_selection", "reference_unresolved"}:
            mode, next_step = "CONTINUATION", "RESOLVE_RESULT_REFERENCE"
        elif family == "multi_goal":
            next_step = "CHECK_GOAL_SEPARATION"
        add(family, [(opening, "ACTIONABLE_GOAL", "OPEN_DRAFT", action, target, {}, "CHECK_REQUIREMENTS"),
                     (followup, mode, op, action, target, {field: value} if field else {}, next_step)])
    assert len(panel) == 120, len(panel)
    return panel


def prior_rows():
    paths = sorted(set([*ROOT.glob("data/*/train.jsonl"), *ROOT.glob("data/*/valid.jsonl"),
                        *ROOT.glob("benchmarks/v*/blind.jsonl"),
                        ROOT / "benchmarks/train_slice_v1/blind.jsonl"]))
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                yield path, raw_text(json.loads(line))


def audit(panel: list[dict]) -> dict:
    prior = [(path, tokens(text), norm(text)) for path, text in prior_rows() if text]
    exact = {normalized for _, _, normalized in prior}
    inverted: dict[str, set[int]] = defaultdict(set)
    for i, (_, wordset, _) in enumerate(prior):
        for word in wordset:
            inverted[word].add(i)
    maximum = 0.0
    nearest = None
    within = 0.0
    panel_tokens = []
    for row in panel:
        text = raw_text(row)
        if norm(text) in exact:
            raise ValueError(f"Exact prior overlap: {row['id']}")
        wordset = tokens(text)
        candidates = set().union(*(inverted.get(word, set()) for word in wordset))
        for idx in candidates:
            score = len(wordset & prior[idx][1]) / len(wordset | prior[idx][1])
            if score > maximum:
                maximum, nearest = score, str(prior[idx][0].relative_to(ROOT))
        for old in panel_tokens:
            within = max(within, len(wordset & old) / len(wordset | old))
        panel_tokens.append(wordset)
    if maximum >= 0.85:
        raise ValueError(f"Near prior overlap {maximum:.3f} at {nearest}")
    return {"prior_rows": len(prior), "exact_prior_overlap": 0,
            "max_prior_token_jaccard": round(maximum, 6),
            "max_within_panel_token_jaccard": round(within, 6),
            "nearest_prior_source": nearest}


def main() -> None:
    panel = make()
    contamination = audit(panel)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "blind.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                            for row in panel), encoding="utf-8", newline="\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"version": "V37_CONVERSATIONAL", "cases": len(panel),
                "turns": sum(len(row["turns"]) for row in panel),
                "families": dict(sorted(Counter(row["family"] for row in panel).items())),
                "panel_sha256": digest,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "gold_review": "agent-authored; pending independent human review",
                "frozen_before": "V27 targeted training data and runtime remediation",
                "contamination": contamination}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "review_queue.jsonl").write_text("".join(json.dumps({"id": row["id"], "family": row["family"],
        "turns": row["turns"], "human_review_status": "pending", "reviewer": None,
        "review_notes": None}, ensure_ascii=False) + "\n" for row in panel), encoding="utf-8", newline="\n")
    print(json.dumps({"sha256": digest, "cases": len(panel), "turns": manifest["turns"],
                      **contamination}, indent=2))


if __name__ == "__main__":
    main()
