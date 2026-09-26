"""V23 continuation data with structural contrast and all-action replay."""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from scripts.build_v21_9b_data import contract as c, fact as f, goal as g  # noqa: E402

DEST = ROOT / "data/v23_9b"
SYSTEM = (ROOT / "V23_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
SEED = 2029
BENCHMARKS = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
              ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl",
              ROOT / "benchmarks/v23/transfer.jsonl"]


def row(text: str, gold: dict, category: str, split: str) -> dict:
    issues = validate_contract(gold)
    if issues:
        raise ValueError((text, [x.as_dict() for x in issues]))
    return {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))}],
            "category": category, "source_split": split,
            "source_id": hashlib.sha256(f"v23:{split}:{text}".encode()).hexdigest()[:20]}


def miss(*names: str) -> list[dict]:
    return [{"field": n, "blocks": "action", "reason": "unspecified"} for n in names]


def generate(split: str) -> list[dict]:
    train = split == "train"
    cities = (["Matera", "Pordenone", "Monza", "Brindisi", "Avellino", "Pesaro", "Savona", "Chieti"]
              if train else ["Imperia", "Prato", "Sondrio", "Agrigento"])
    out = []
    add = lambda text, gold, cat: out.append(row(text, gold, cat, split))
    bare = (["Puoi riservare un tavolo?", "Avrei bisogno che prenotassi un tavolo.",
             "Organizza una prenotazione per un tavolo, i dettagli arriveranno.",
             "Mi occorre un tavolo prenotato, poi ti darò le informazioni.",
             "Trova un modo per riservarmi un tavolo, ma ancora non so quando.",
             "Vorrei bloccare un tavolo senza avere ancora i particolari.",
             "Occupati di prenotare un tavolo, giorno e ospiti da concordare.",
             "Fai una prenotazione al ristorante per un tavolo, ora e data da definire."]
            if train else
            ["Mi aiuti a riservare un tavolo? I dati te li comunicherò dopo.",
             "Per un tavolo da prenotare non ho ancora deciso giorno, orario e ospiti.",
             "Serve una prenotazione per un tavolo, ma non posso darti i particolari ora."])
    for text in bare:
        add(text, c(g("g1", "book", "tavolo", "place", [], missing=miss("date", "time", "guests"))),
            "missing_table_bare")
    for n in (2, 3, 4, 5) if train else (6, 7):
        for form in range(4 if train else 2):
            text = ([f"Prenotami un tavolo per {n} persone; ti dirò dopo giorno e orario.",
                     f"Per {n} ospiti mi serve un tavolo prenotato, senza data né ora fissate.",
                     f"Riserva un tavolo per il gruppo di {n}; quando ci andiamo lo scelgo più avanti.",
                     f"Puoi bloccare un tavolo per {n} commensali? Non abbiamo ancora scelto quando."]
                    if train else
                    [f"Vorremmo un tavolo per {n}, ma giorno e ora sono ancora da decidere.",
                     f"Prenotazione per {n} persone: data e orario li comunicherò successivamente."])[form]
            add(text, c(g("g1", "book", "tavolo", "place", [f("guests", "eq", n)],
                        missing=miss("date", "time"))), "missing_table_partial")
    days = (["domani", "lunedì", "venerdì", "sabato", "tra tre giorni", "la prossima settimana",
             "martedì", "giovedì", "stasera", "domenica"] if train else
            ["mercoledì", "tra quattro giorni", "il mese prossimo", "lunedì mattina"])
    for day in days:
        for form in range(3 if train else 2):
            text = ([f"{day.capitalize()} ricordami quella cosa di cui parlavamo.",
                     f"Per {day} imposta un promemoria su ciò che ho detto prima.",
                     f"Avvisami {day} riguardo a quella faccenda, anche se qui non è specificata."]
                    if train else
                    [f"Vorrei essere avvisato {day} su quella questione precedente.",
                     f"{day.capitalize()} fammi un promemoria per ciò di cui discutevamo."])[form]
            add(text, c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", day)],
                missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])),
                "missing_reminder_reference")
    for city in cities:
        for time in ("10:00", "12:00", "16:00", "19:00"):
            forms = ([f"Cercami il mezzo meno caro per raggiungere {city} entro le {time} sabato.",
                      f"Voglio arrivare a {city} sabato prima delle {time} spendendo il minimo possibile.",
                      f"Per {city} sabato, come arrivo entro le {time} al costo più basso? Cerca le opzioni."]
                     if train else
                     [f"Mostrami il trasporto più economico per {city} sabato prima delle {time}.",
                      f"Entro le {time} di sabato devo essere a {city}; trova la soluzione meno costosa."])
            for text in forms:
                add(text, c(g("g1", "find", "trasporto", "transport", [f("destination", "eq", city),
                    f("date", "eq", "sabato"), f("arrival_time", "before", time)],
                    ranking={"field": "price", "direction": "asc"})), "ranking_no_fabrication")
    foods = ([('pho', 'vietnamita'), ('kebab', 'turca'), ('paella', 'spagnola'),
              ('dumpling', 'cinese'), ('cous cous', 'marocchina'), ('tacos', 'messicana'),
              ('pizza', 'napoletana'), ('falafel', 'libanese'), ('gnocchi', 'italiana'),
              ('samosa', 'indiana')] if train else
             [('udon', 'giapponese'), ('arepa', 'venezuelana'), ('pierogi', 'polacca')])
    for food, cuisine in foods:
        forms = ([f"Prova a cercare {food}; se non ce ne sono, trova un ristorante {cuisine}.",
                  f"Mi interessano {food}, ma in mancanza di risultati cerca cucina {cuisine}.",
                  f"Vorrei {food}; qualora non li trovi, proponi un ristorante di cucina {cuisine}.",
                  f"Prima cerca {food}; solo se la ricerca è vuota, mostrami un ristorante {cuisine}.",
                  f"Cerca {food}, altrimenti passa a un ristorante {cuisine}."]
                 if train else
                 [f"Trova {food}; se non sono disponibili, cerca un ristorante {cuisine}.",
                  f"Voglio {food}, oppure cucina {cuisine} soltanto in assenza di risultati."])
        for text in forms:
            add(text, c(g("g1", "find", food, "food", []),
                g("g2", "find", "ristorante", "place", [f("cuisine", "eq", cuisine)],
                  depends_on=["g1"], condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})),
                "fallback_without_city")
    targets = ([('pizzeria', 'pizzerie', 'place', 'book', 'prenota'),
                ('ristorante', 'ristoranti', 'place', 'book', 'prenota'),
                ('libreria', 'librerie', 'place', 'contact', 'contatta'),
                ('idraulico', 'idraulici', 'professional_service', 'contact', 'contatta'),
                ('tablet', 'tablet', 'product', 'buy', 'compra'),
                ('lampada', 'lampade', 'product', 'buy', 'compra')]
               if train else
               [('fioraio', 'fiorai', 'place', 'contact', 'contatta'),
                ('fotografo', 'fotografi', 'professional_service', 'contact', 'contatta'),
                ('orologio', 'orologi', 'product', 'buy', 'compra')])
    for name, plural, typ, action, verb in targets:
        for city in cities:
            n = 4 if action == "book" else 3
            ordinal = 2 if action == "buy" else 1
            forms = ([f"Mostrami {n} {plural} a {city} e poi {verb} il risultato numero {ordinal}.",
                      f"Cerca {n} {plural} a {city}; dopo la ricerca {verb} l'opzione numero {ordinal}.",
                      f"Prima trova {n} {plural} a {city}, quindi {verb} la scelta numero {ordinal}."]
                     if train else
                     [f"Elenca {n} {plural} a {city}; in seguito {verb} il risultato numero {ordinal}.",
                      f"Vorrei vedere {n} {plural} a {city} e poi {verb} la scelta numero {ordinal}."])
            # Contact has no message; booking has no date/time/guests. Explicitly
            # mark the blocked action instead of matching inconsistent old gold.
            extras = (miss("date", "time", "guests") if action == "book" else
                      miss("message") if action == "contact" else [])
            first = g("g1", "find", name, typ, [f("quantity", "eq", n), f("location", "near", city)])
            second = g("g2", action, name, typ, [{"field": "selection", "op": "eq",
                "value_ref": {"goal": "g1", "field": "result", "ordinal": ordinal}, "strength": "hard"}],
                depends_on=["g1"], **({"missing": extras} if extras else {}))
            for text in forms:
                add(text, c(first, second), "ordinal_cross_domain")
    modes = ([('treno', 'arrival_station'), ('autobus', 'arrival_stop'), ('volo', 'arrival_airport')]
             if train else [('traghetto', 'arrival_port'), ('pullman', 'arrival_stop')])
    for mode, field in modes:
        for city in cities:
            forms = ([f"Cerca un {mode} per {city} e un hotel vicino al suo punto di arrivo.",
                      f"Trovami un {mode} verso {city}; poi un albergo nei pressi del punto di arrivo.",
                      f"Prima individua un {mode} per {city}, quindi cerca una stanza accanto all'arrivo."]
                     if train else
                     [f"Mi serve un {mode} diretto a {city} e, vicino alla sua destinazione finale, un hotel.",
                      f"Mostrami un {mode} per {city}; in seguito cerca alloggio presso il punto d'arrivo."])
            name = "hotel" if (train or city != cities[-1]) else "alloggio"
            gold = c(g("g1", "find", mode, "transport", [f("destination", "eq", city)]),
                g("g2", "find", name, "accommodation", [{"field": "location", "op": "near",
                    "value_ref": {"goal": "g1", "field": field}, "strength": "hard"}], depends_on=["g1"]))
            for text in forms:
                add(text, gold, "dependent_location_reference")
    for city in cities:
        for n in (2, 3, 4, 5):
            text = (f"Noi {n} siamo senza macchina; cercaci un ristorante nel centro di {city}." if train else
                    f"Siamo un gruppo di {n}, senza auto. Trova un ristorante nel centro di {city}.")
            add(text, c(g("g1", "find", "ristorante", "place", [f("location", "near", f"centro di {city}")]),
                context=[{"field": "party_size", "op": "eq", "value": n},
                         {"field": "has_car", "op": "eq", "value": False}]), "context_not_filter")
        for amenity in ("piscina", "parcheggio", "giardino", "colazione"):
            text = (f"Un hotel a {city} entro 100 euro, possibilmente con {amenity}." if train else
                    f"Cerca un hotel a {city} da massimo 100 euro; {amenity} sarebbe un plus.")
            add(text, c(g("g1", "find", "hotel", "accommodation", [f("location", "near", city),
                f("price", "lte", 100, currency="EUR"), f("amenity", "eq", amenity, strength="soft")])),
                "soft_amenity")
        for name, typ, prohibited in (('museo', 'place', 'buy'), ('hotel', 'accommodation', 'book'),
                                      ('tablet', 'product', 'buy')):
            text = (f"Mostrami un {name} a {city}, ma non voglio {('acquistare nulla' if prohibited == 'buy' else 'prenotare nulla')}."
                    if train else
                    f"Trova un {name} a {city}; fermati prima di qualsiasi {('acquisto' if prohibited == 'buy' else 'prenotazione')}.")
            add(text, c(g("g1", "find", name, typ, [f("location", "near", city)],
                policy={"forbid": [prohibited]})), "policy_goal_scope")
    return out


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    source = ROOT / "data/v22_9b"
    data = {split: read_jsonl(source / f"{split}.jsonl") for split in ("train", "valid")}
    for split in data:
        for item in data[split]:
            item["messages"][0]["content"] = SYSTEM
        data[split].extend(generate(split))
    excluded = {normalize_text(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)}
    benchmark_tokens = [token_set(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)]
    seen = {}
    dropped = Counter()
    for split in ("train", "valid"):
        unique = []
        for item in data[split]:
            text = item["messages"][1]["content"]
            key, tokens = normalize_text(text), token_set(text)
            if key in excluded or any(tokens and other and len(tokens & other) / len(tokens | other) >= .85
                                      for other in benchmark_tokens):
                dropped["benchmark_overlap"] += 1
                continue
            if key in seen:
                if seen[key] != item["messages"][-1]["content"]:
                    dropped["label_conflict"] += 1
                else:
                    dropped["duplicate"] += 1
                continue
            seen[key] = item["messages"][-1]["content"]
            unique.append(item)
        data[split] = unique
    if dropped["label_conflict"]:
        raise RuntimeError(f"Conflicting labels: {dropped['label_conflict']}")
    postings = defaultdict(set)
    train_tokens = [token_set(item["messages"][1]["content"]) for item in data["train"]]
    for index, tokens in enumerate(train_tokens):
        for token in tokens:
            postings[token].add(index)
    kept = []
    for item in data["valid"]:
        tokens = token_set(item["messages"][1]["content"])
        candidates = set()
        for token in sorted(tokens, key=lambda t: len(postings[t]))[:3]:
            candidates.update(postings[token])
        if any(tokens and train_tokens[j] and len(tokens & train_tokens[j]) / len(tokens | train_tokens[j]) >= .85
               for j in candidates):
            dropped["near_train_validation_overlap"] += 1
        else:
            kept.append(item)
    data["valid"] = kept
    for split in data:
        rng.shuffle(data[split])
        with (DEST / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in data[split]:
                gold = json.loads(item["messages"][-1]["content"])
                if validate_contract(gold):
                    raise RuntimeError(f"Invalid {split} gold: {item['source_id']}")
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"version": "V23_9B", "schema_version": "1.2", "seed": SEED,
        "source": "V22 continuation plus no-city, cross-domain and negative-contrast semantic repairs",
        "source_sha256": {s: sha256(source / f"{s}.jsonl") for s in data},
        "excluded_benchmark_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in BENCHMARKS},
        "generator_sha256": sha256(Path(__file__)),
        "system_prompt_sha256": sha256(ROOT / "V23_SYSTEM_PROMPT.txt"),
        "rows": {s: len(data[s]) for s in data}, "dropped": dict(dropped),
        "categories": {s: dict(Counter(x["category"] for x in data[s])) for s in data},
        "output_sha256": {s: sha256(DEST / f"{s}.jsonl") for s in data},
        "limitations": ["Synthetic examples need independent human semantic review",
                        "The new transfer panel is feature-targeted rather than a random production sample"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "dropped": manifest["dropped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
