"""Build independent V1.3 contrastive corpus after V26 was frozen.

Source templates and entity pools are hand authored here; benchmark contents are
used solely for exact/near overlap exclusion. V24 supplies stratified replay.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.migrate_contract_v1_3 import MigrationHold, upgrade_contract  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

DEST = ROOT / "data/v25_9b"
SOURCE = ROOT / "data/v24_9b"
PROMPT = (ROOT / "V25_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
BENCHMARKS = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
              ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl",
              ROOT / "benchmarks/v23/transfer.jsonl", ROOT / "benchmarks/v24/safety.jsonl",
              ROOT / "benchmarks/v25/blind.jsonl", ROOT / "benchmarks/v26/blind.jsonl"]
SEED = 2031


def f(field, value, op="eq", **extra):
    return {"field": field, "op": op, "value": value, "strength": "hard", **extra}


def t(role, value, op="eq"):
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def g(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def m(field, blocks="action", reason="unspecified"):
    return {"field": field, "blocks": blocks, "reason": reason}


def c(*goals, context=None):
    result = {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
              "goals": list(goals)}
    if context:
        result["context"] = context
    return result


def ctx(field, value):
    return {"scope": "user", "field": field, "op": "eq", "value": value}


def row(text, gold, category, split, source="contrastive"):
    issues = validate_contract(gold, user_text=text)
    if issues:
        raise ValueError((category, text, [issue.as_dict() for issue in issues]))
    return {"messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False,
                                                                     separators=(",", ":"))}],
            "category": category, "source_split": split, "source": source,
            "source_id": hashlib.sha256(f"v25:{split}:{text}".encode()).hexdigest()[:20]}


def generate(split):
    train = split == "train"
    cities = (["Asti", "Udine", "Forlì", "Ragusa", "Pescara", "Latina", "Pavia", "Novara",
               "Crotone", "Ferrara"] if train else ["Agrigento", "Savona", "Imperia", "Caserta"])
    food = (["focaccia", "piadina", "lasagna", "gelato", "crostata", "risotto", "zuppa"]
            if train else ["gnocchi", "falafel", "quiche", "tiramisù", "couscous", "polenta", "omelette"])
    vehicle = (["auto", "monopattino", "barca", "camper", "bicicletta", "furgone", "scooter"]
               if train else ["moto", "canoa", "pulmino", "quad", "minibus", "e-bike", "caravan"])
    product = (["microfono", "tablet", "tostapane", "zaino", "monitor", "stampante", "materasso"]
               if train else ["lavatrice", "valigia", "router", "frigorifero", "scrivania", "cuffie", "forno"])
    service = (["visita allergologica", "visita ortopedica", "controllo dentistico",
                "seduta logopedica", "visita neurologica", "controllo dermatologico", "visita pediatrica"]
               if train else ["visita reumatologica", "seduta osteopatica", "controllo uditivo",
                              "visita oculistica", "colloquio psicologico", "visita nefrologica",
                              "controllo cardiologico"])
    names = (["Sara", "Davide", "Nora", "Enrico", "Irene", "Matteo", "Chiara"]
             if train else ["Lucia", "Omar", "Beatrice", "Paolo", "Alma", "Stefano", "Elena"])
    day = "mercoledì" if train else "venerdì"
    examples = []

    def add(text, gold, category):
        examples.append(row(text, gold, category, split))

    for ci, city in enumerate(cities):
        for j in range(7):
            variants = range(3) if train else [(ci * 7 + j) % 3]
            for v in variants:
                x, food_name, vehicle_name, product_name = service[j], food[j], vehicle[j], product[j]
                name = names[j]
                n = 2 + (j % 3)

                # Small action/modifier difference, with target type held fixed.
                if v == 1:
                    add(f"Ordina {food_name} a {city}; chiedimi il permesso prima dell'acquisto.",
                        c(g("order", food_name, "food", [f("delivery_location", city)],
                            policy={"confirm_before": ["order"]})), "order_vs_sort")
                else:
                    lead = "Ordina" if v == 0 else "Metti in ordine"
                    add(f"{lead} le offerte di {food_name} a {city} per costo crescente, senza acquistare.",
                        c(g("find", food_name, "food", [f("location", city, "near")],
                            modifiers={"sort": {"field": "price", "direction": "asc"}},
                            policy={"forbid": ["order"]})), "order_vs_sort")

                if v == 1:
                    add(f"A {city} mostrami un {vehicle_name} che si possa prendere a noleggio {day}; non riservarlo.",
                        c(g("find", vehicle_name, "vehicle", [f("location", city, "near")],
                            temporal=[t("search_date", day)], policy={"forbid": ["rent"]})),
                        "rent_vs_search")
                else:
                    phrase = "Prendi in affitto" if v == 0 else "Noleggia"
                    add(f"{phrase} un {vehicle_name} per {day} a {city}; fammi confermare prima.",
                        c(g("rent", vehicle_name, "vehicle", [f("location", city, "near")],
                            temporal=[t("action_date", day)], policy={"confirm_before": ["rent"]})),
                        "rent_vs_search")

                if v == 1:
                    add(f"Trovami {n} risultati per {product_name} a {city}, non un confronto.",
                        c(g("find", product_name, "product", [f("location", city, "near")],
                            modifiers={"limit": n})), "count_role")
                else:
                    lead = "Confronta" if v == 0 else "Metti a confronto"
                    add(f"{lead} {n} alternative di {product_name} disponibili a {city}.",
                        c(g("compare", product_name, "product", [f("location", city, "near")],
                            modifiers={"compare_count": n})), "count_role")

                mode = "treno" if j % 2 else "autobus"
                if v == 0:
                    add(f"Per {city} mostrami un {mode} che arrivi entro le 16:20.",
                        c(g("find", mode, "transport", [f("destination", city)],
                            temporal=[t("arrival_time", "16:20", "before")])), "temporal_role")
                elif v == 1:
                    add(f"Da {city} cercami un {mode} che parta dopo le 16:20.",
                        c(g("find", mode, "transport", [f("origin", city)],
                            temporal=[t("departure_time", "16:20", "after")])), "temporal_role")
                else:
                    add(f"Cerca un {mode} per {city} disponibile {day} in serata.",
                        c(g("find", mode, "transport", [f("destination", city)],
                            temporal=[t("search_date", day), t("search_time", "serata")])),
                        "temporal_role")

                if v == 0:
                    add(f"Fissami una {x} a {city} {day} alle 11:10.",
                        c(g("book", x, "appointment", [f("location", city, "near")],
                            temporal=[t("action_date", day), t("action_time", "11:10")])),
                        "appointment_lifecycle")
                elif v == 1:
                    add(f"Annulla la mia {x} del 14 maggio a {city}.",
                        c(g("cancel", x, "appointment", [f("original_date", "14 maggio"),
                            f("location", city, "near")])), "appointment_lifecycle")
                else:
                    add(f"Sposta la {x} del 14 maggio a {day} alle 11:10 a {city}.",
                        c(g("reschedule", x, "appointment", [f("original_date", "14 maggio"),
                            f("location", city, "near")], temporal=[t("action_date", day),
                            t("action_time", "11:10")])), "appointment_lifecycle")

                if v == 0:
                    add(f"Ricordami {day} alle 07:30 di chiamare {name} per la riunione a {city}.",
                        c(g("notify", "promemoria", "reminder",
                            [f("message", f"chiamare {name} per la riunione a {city}")],
                            temporal=[t("notification_date", day), t("notification_time", "07:30")])),
                        "message_scope")
                elif v == 1:
                    add(f"Avvisa {name} che sarò a {city} alle 07:30; chiedimi conferma prima di inviare.",
                        c(g("contact", name, "person", [f("message", f"Sarò a {city} alle 07:30")],
                            policy={"confirm_before": ["contact"]})), "message_scope")
                else:
                    add(f"Ricordamelo {day} alle 07:30 quando sarò a {city}; non ti ho detto cosa.",
                        c(g("notify", "promemoria", "reminder", [f("location", city, "near")],
                            temporal=[t("notification_date", day), t("notification_time", "07:30")],
                            missing=[m("referent", "search", "unresolved_reference")])),
                        "message_scope")

                ref = {"goal": "g1", "field": "result", "ordinal": 1}
                if v == 0:
                    add(f"Cerca un albergo a {city} e prenota il primo solo dopo che te lo confermo.",
                        c(g("find", "albergo", "accommodation", [f("location", city, "near")], idx=1),
                          g("book", "albergo", "accommodation", [{"field": "selection", "op": "eq",
                              "value_ref": ref, "strength": "hard"}], idx=2, depends_on=["g1"],
                            policy={"confirm_before": ["book"]})), "dependent_policy")
                elif v == 1:
                    add(f"Trova un negozio di {product_name} a {city}, poi compra la prima opzione previo mio consenso.",
                        c(g("find", product_name, "product", [f("location", city, "near")], idx=1),
                          g("buy", product_name, "product", [{"field": "selection", "op": "eq",
                              "value_ref": ref, "strength": "hard"}], idx=2, depends_on=["g1"],
                            policy={"confirm_before": ["buy"]})), "dependent_policy")
                else:
                    add(f"Cerca un idraulico a {city} e poi chiedigli per messaggio se può venire {day}.",
                        c(g("find", "idraulico", "professional_service", [f("location", city, "near")], idx=1),
                          g("contact", "idraulico", "professional_service", [
                              {"field": "selection", "op": "eq", "value_ref": ref, "strength": "hard"},
                              f("message", f"Puoi venire {day}?")], idx=2, depends_on=["g1"])),
                        "dependent_policy")

                if v == 0:
                    add(f"Siamo {n} persone e non abbiamo un'auto: cerca un locale raggiungibile a piedi a {city}.",
                        c(g("find", "locale", "place", [f("location", city, "near"),
                            f("access", "a piedi")]), context=[ctx("party_size", n),
                                                                  ctx("has_car", False)]),
                        "user_context")
                elif v == 1:
                    add(f"Abbiamo già un albergo a {city}; trovami un taxi dall'aeroporto.",
                        c(g("find", "taxi", "transport", [f("origin", "aeroporto"),
                            f("destination", city)]), context=[ctx("lodging_owned", True)]),
                        "user_context")
                else:
                    add(f"Io e {name} siamo in {n}: cerca un tavolo a {city}, senza prenotare.",
                        c(g("find", "tavolo", "place", [f("location", city, "near")],
                            policy={"forbid": ["book"]}), context=[ctx("party_size", n)]),
                        "user_context")

                if v == 0:
                    add(f"Cerca un treno per {city}; se non trovi posti, prova un pullman.",
                        c(g("find", "treno", "transport", [f("destination", city)], idx=1),
                          g("find", "pullman", "transport", [f("destination", city)], idx=2,
                            depends_on=["g1"], condition={"goal": "g1", "test": {
                                "field": "results_count", "op": "eq", "value": 0}})),
                        "fallback_vs_independent")
                elif v == 1:
                    add(f"Cerca un treno per {city} e anche un hotel vicino alla stazione.",
                        c(g("find", "treno", "transport", [f("destination", city)], idx=1),
                          g("find", "hotel", "accommodation", [f("location", "stazione", "near")],
                            idx=2)), "fallback_vs_independent")
                else:
                    add(f"In un dialogo di fantasia a {city} qualcuno pronuncia 'prenota un treno' senza chiedertelo.",
                        c(), "fallback_vs_independent")

                if v == 0:
                    add(f"Prenota un tavolo a {city}; non ho specificato né il giorno, né l'ora, né i coperti.",
                        c(g("book", "tavolo", "place", [f("location", city, "near")],
                            missing=[m("date"), m("time"), m("guests")])), "missing_information")
                elif v == 1:
                    add(f"Prenota un volo per {city} {day}; manca la città di partenza.",
                        c(g("book", "volo", "transport", [f("destination", city)],
                            temporal=[t("action_date", day)], missing=[m("origin")])),
                        "missing_information")
                else:
                    add(f"Contatta {name} a {city}, ma prima chiedimi cosa devo dirgli.",
                        c(g("contact", name, "person", [f("location", city, "near")],
                            missing=[m("message")])), "missing_information")
    return examples


def replay(split, count, rng):
    by_category = defaultdict(list)
    held = Counter()
    for old in read_jsonl(SOURCE / f"{split}.jsonl"):
        text = old["messages"][1]["content"]
        try:
            gold = upgrade_contract(json.loads(old["messages"][-1]["content"]))
        except MigrationHold as exc:
            held[str(exc)] += 1
            continue
        by_category[old["category"]].append(row(text, gold, old["category"], split, "v24_replay"))
    for category in by_category:
        rng.shuffle(by_category[category])
    chosen = []
    categories = sorted(by_category)
    while len(chosen) < count and categories:
        next_round = []
        for category in categories:
            if len(chosen) == count:
                break
            if by_category[category]:
                chosen.append(by_category[category].pop())
            if by_category[category]:
                next_round.append(category)
        categories = next_round
    if len(chosen) != count:
        raise RuntimeError(f"Insufficient replay rows: {len(chosen)}/{count}")
    return chosen, dict(held)


def main() -> int:
    if DEST.exists():
        raise FileExistsError("V25 corpus already exists; refuse regeneration")
    if not (ROOT / "benchmarks/v26/manifest.json").is_file():
        raise RuntimeError("Freeze V26 before generating corpus")
    rng = random.Random(SEED)
    data = {}
    held = {}
    for split, amount in (("train", 500), ("valid", 70)):
        generated = generate(split)
        selected, held[split] = replay(split, amount, rng)
        data[split] = generated + selected
    excluded = {normalize_text(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)}
    benchmark_tokens = [token_set(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)]
    seen = {}
    drops = Counter()
    for split in ("train", "valid"):
        filtered = []
        for item in data[split]:
            text = item["messages"][1]["content"]
            key, tokens = normalize_text(text), token_set(text)
            if key in excluded or any(tokens and other and len(tokens & other) / len(tokens | other) >= .85
                                      for other in benchmark_tokens):
                drops[f"{split}_benchmark_overlap"] += 1
                continue
            if key in seen:
                if seen[key] != item["messages"][-1]["content"]:
                    raise RuntimeError(f"Conflicting gold for the same request: {key}")
                drops[f"{split}_duplicate"] += 1
                continue
            seen[key] = item["messages"][-1]["content"]
            filtered.append(item)
        data[split] = filtered
    # Remove near-duplicates from validation, preserving its genuine lexical holdout.
    train_tokens = [token_set(r["messages"][1]["content"]) for r in data["train"]]
    filtered = []
    for item in data["valid"]:
        toks = token_set(item["messages"][1]["content"])
        if any(toks and train_tokens[j] and len(toks & train_tokens[j]) /
               len(toks | train_tokens[j]) >= .85 for j in range(len(train_tokens))):
            drops["valid_near_train"] += 1
        else:
            filtered.append(item)
    data["valid"] = filtered
    for split in data:
        rng.shuffle(data[split])
    DEST.mkdir(parents=True, exist_ok=False)
    for split, rows in data.items():
        with (DEST / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in rows:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"version": "V25_9B", "schema_version": "1.3", "seed": SEED,
                "panel_frozen_before_data": {"path": "benchmarks/v26/blind.jsonl",
                                             "sha256": sha256(ROOT / "benchmarks/v26/blind.jsonl")},
                "source_sha256": {s: sha256(SOURCE / f"{s}.jsonl") for s in data},
                "excluded_benchmark_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in BENCHMARKS},
                "generator_sha256": sha256(Path(__file__)),
                "system_prompt_sha256": sha256(ROOT / "V25_SYSTEM_PROMPT.txt"),
                "rows": {s: len(rows) for s, rows in data.items()},
                "replay_rows": {s: sum(x["source"] == "v24_replay" for x in data[s]) for s in data},
                "migration_held": held, "dropped": dict(drops),
                "categories": {s: dict(Counter(x["category"] for x in data[s])) for s in data},
                "output_sha256": {s: sha256(DEST / f"{s}.jsonl") for s in data},
                "limitations": ["Template-derived synthetic data", "AI-authored gold needs human audit",
                                "V26 is a targeted capability panel, not a production sample"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "replay": manifest["replay_rows"],
                      "drops": manifest["dropped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
