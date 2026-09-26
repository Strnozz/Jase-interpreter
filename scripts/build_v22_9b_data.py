"""Build V22 continuation: all-action V20 replay plus V21 features and repairs.

The fixed benchmark panels are exclusions, never training inputs. V21's prior
synthetic train/valid split is retained; V20 replay is newly stratified by action.
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

from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from scripts.build_v21_9b_data import contract, fact, goal  # noqa: E402

DEST = ROOT / "data/v22_9b"
PROMPT = (ROOT / "V22_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
SEED = 2028
BENCHMARKS = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
              ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl"]


def format_row(text: str, gold: dict, category: str, split: str) -> dict:
    issues = validate_contract(gold)
    if issues:
        raise ValueError((text, [i.as_dict() for i in issues]))
    return {"messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))}],
            "category": category, "source_split": split,
            "source_id": hashlib.sha256(f"v22:{split}:{text}".encode()).hexdigest()[:20]}


def repair_rows(split: str) -> list[dict]:
    # Keep forms and cities disjoint across the splits. The examples target
    # structural failure modes, not the benchmark's exact prompts or entities.
    cities = (["Bologna", "Verona", "Lecce", "Ravenna", "Pisa", "Trieste", "Perugia", "Bari"]
              if split == "train" else ["Cremona", "Viterbo", "Rimini", "Asti"])
    results = []
    add = lambda text, gold, category: results.append(format_row(text, gold, category, split))
    for i, city in enumerate(cities):
        for name in ("osteria", "trattoria", "ristorante", "pizzeria"):
            text = (f"Mi prenoti un tavolo in una {name} a {city}? L'orario e il giorno li scelgo più avanti."
                    if split == "train" else
                    f"Vorrei riservare un tavolo da una {name} a {city}; non so ancora quando né per quanti.")
            gold = contract(goal("g1", "book", "tavolo", "place", [fact("location", "near", city)],
                missing=[{"field": f, "blocks": "action", "reason": "unspecified"}
                         for f in ("date", "time", "guests")]))
            add(text, gold, "repair_missing_table")
        for profession in ("fabbro", "tecnico caldaie", "giardiniere", "elettricista"):
            n = 3 + i % 2
            ordinal = 2 if i % 2 else 1
            word = "secondo" if ordinal == 2 else "primo"
            text = (f"Elencami {n} {profession} liberi venerdì a {city}; in seguito contatta il {word} soltanto con il mio via libera."
                    if split == "train" else
                    f"Mostrami {n} {profession} disponibili venerdì a {city}, quindi senti il {word} dopo che confermo.")
            first = goal("g1", "find", profession, "professional_service", [fact("quantity", "eq", n),
                fact("availability", "contains", "venerdì"), fact("location", "near", city)])
            second = goal("g2", "contact", profession, "professional_service", [
                {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "result", "ordinal": ordinal},
                 "strength": "hard"}], depends_on=["g1"], policy={"confirm_before": ["contact"]},
                missing=[{"field": "message", "blocks": "action", "reason": "unspecified"}])
            add(text, contract(first, second), "repair_chain_contact")
        for destination in ("stazione", "centro storico", "aeroporto"):
            text = (f"Per arrivare a {destination} a {city} entro le 15 di sabato, cercami l'opzione di trasporto meno cara."
                    if split == "train" else
                    f"Cerco il modo più economico per raggiungere {destination} a {city} sabato prima delle 15.")
            gold = contract(goal("g1", "find", "trasporto", "transport", [
                fact("destination", "eq", destination), fact("location", "near", city),
                fact("date", "eq", "sabato"), fact("arrival_time", "before", "15:00")],
                ranking={"field": "price", "direction": "asc"}))
            add(text, gold, "repair_ranking_mode")
        for food, cuisine in (("tacos", "messicana"), ("ramen", "coreana"), ("falafel", "libanese")):
            text = (f"Cerca {food} a {city}; nel caso non ci sia, mostrami invece un ristorante di cucina {cuisine}."
                    if split == "train" else
                    f"Vorrei {food} a {city}; se la ricerca è vuota, trova un ristorante {cuisine}.")
            first = goal("g1", "find", food, "food", [fact("location", "near", city)])
            second = goal("g2", "find", "ristorante", "place", [fact("cuisine", "eq", cuisine),
                fact("location", "near", city)], depends_on=["g1"],
                condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})
            add(text, contract(first, second), "repair_fallback")
        for item, owned, context_field in (("taxi", "alloggio", "lodging_owned"),
                                           ("albergo", "biglietto", "ticket_owned")):
            target_type = "transport" if item == "taxi" else "accommodation"
            text = (f"Ho già un {owned}; per domenica cerca un {item} a {city}."
                    if split == "train" else
                    f"Il {owned} ce l'ho già. Mi trovi un {item} a {city} domenica?")
            gold = contract(goal("g1", "find", item, target_type, [fact("location", "near", city),
                fact("date", "eq", "domenica")]),
                context=[{"field": context_field, "op": "eq", "value": True}])
            add(text, gold, "repair_context_no_forbid")
        for item, typ, verb in (("museo", "place", "comprare"), ("tablet", "product", "acquistare")):
            text = (f"Trova un {item} a {city} aperto domenica, ma senza {verb} nulla."
                    if split == "train" else
                    f"Cerca un {item} a {city} disponibile domenica; niente acquisti.")
            gold = contract(goal("g1", "find", item, typ, [fact("location", "near", city),
                fact("availability", "contains", "domenica")], policy={"forbid": ["buy"]}))
            add(text, gold, "repair_policy")
    # Crucial no-location forms: the previous V21 table-booking template always
    # supplied a city, encouraging omission of missing on bare requests.
    for text in (["Puoi occuparti della prenotazione di un tavolo?",
                  "Mi serve un tavolo prenotato, ma i dettagli arriveranno dopo.",
                  "Riservami un posto al ristorante: non ho ancora stabilito quando e per quanti."]
                 if split == "train" else
                 ["Vorrei una prenotazione al ristorante, i particolari te li darò più avanti.",
                  "Per favore occupati di riservare un tavolo; data, ora e ospiti da definire."]):
        gold = contract(goal("g1", "book", "tavolo", "place", [],
            missing=[{"field": f, "blocks": "action", "reason": "unspecified"}
                     for f in ("date", "time", "guests")]))
        add(text, gold, "repair_missing_table_bare")
    return results


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    data = {"train": [], "valid": []}
    v21 = ROOT / "data/v21_9b"
    v20 = ROOT / "data/v20_9b"
    for split in data:
        for item in read_jsonl(v21 / f"{split}.jsonl"):
            if item["category"] == "v20_rehearsal":
                continue
            item["messages"][0]["content"] = PROMPT
            data[split].append(item)
        data[split].extend(repair_rows(split))
    quota = {"train": {"find": 500, "book": 350, "buy": 250, "compare": 150,
                       "contact": 150, "rent": 150, "monitor": 130, "order": 100,
                       "hire": 100, "cancel": 100},
             "valid": {"find": 80, "book": 45, "buy": 35, "compare": 20,
                       "contact": 20, "rent": 20, "monitor": 15, "order": 15,
                       "hire": 15, "cancel": 15}}
    for split, limits in quota.items():
        by_action = defaultdict(list)
        for item in read_jsonl(v20 / f"{split}.jsonl"):
            gold = json.loads(item["messages"][-1]["content"])
            by_action[gold["goals"][0]["action"]].append(item)
        for action, n in limits.items():
            selected = rng.sample(by_action[action], min(n, len(by_action[action])))
            for item in selected:
                item["messages"][0]["content"] = PROMPT
                item["category"] = "v20_replay_" + action
                data[split].append(item)
    excluded = {normalize_text(row["text"]) for p in BENCHMARKS for row in read_jsonl(p)}
    benchmark_tokens = [token_set(row["text"]) for p in BENCHMARKS for row in read_jsonl(p)]
    seen = {}
    dropped = Counter()
    for split in ("train", "valid"):
        unique = []
        for item in data[split]:
            text = item["messages"][1]["content"]
            key = normalize_text(text)
            ts = token_set(text)
            if key in excluded or any(ts and b and len(ts & b) / len(ts | b) >= .85
                                      for b in benchmark_tokens):
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
        raise RuntimeError(f"Conflicting gold labels: {dropped['label_conflict']}")
    postings = defaultdict(set)
    train_tokens = [token_set(item["messages"][1]["content"]) for item in data["train"]]
    for index, tokens in enumerate(train_tokens):
        for token in tokens:
            postings[token].add(index)
    clean_valid = []
    for item in data["valid"]:
        tokens = token_set(item["messages"][1]["content"])
        candidates = set()
        for token in sorted(tokens, key=lambda t: len(postings[t]))[:3]:
            candidates.update(postings[token])
        if any(tokens and train_tokens[j] and
               len(tokens & train_tokens[j]) / len(tokens | train_tokens[j]) >= .85
               for j in candidates):
            dropped["near_train_validation_overlap"] += 1
        else:
            clean_valid.append(item)
    data["valid"] = clean_valid
    rng.shuffle(data["train"])
    rng.shuffle(data["valid"])
    for split in data:
        path = DEST / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for item in data[split]:
                gold = json.loads(item["messages"][-1]["content"])
                if validate_contract(gold):
                    raise RuntimeError(f"Invalid gold in {split}: {item['source_id']}")
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"version": "V22_9B", "schema_version": "1.2", "seed": SEED,
        "source": "V21 feature examples + stratified all-action V20 replay + reviewed repair templates",
        "source_sha256": {"v21": {s: sha256(v21 / f"{s}.jsonl") for s in data},
                          "v20": {s: sha256(v20 / f"{s}.jsonl") for s in data}},
        "excluded_benchmark_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in BENCHMARKS},
        "generator_sha256": sha256(Path(__file__)),
        "system_prompt_sha256": sha256(ROOT / "V22_SYSTEM_PROMPT.txt"),
        "rows": {s: len(data[s]) for s in data}, "dropped": dict(dropped),
        "categories": {s: dict(Counter(x["category"] for x in data[s])) for s in data},
        "output_sha256": {s: sha256(DEST / f"{s}.jsonl") for s in data},
        "limitations": ["Repair examples are templated and need human review", "GoalContract remains V1.2 for paired benchmarks"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "dropped": manifest["dropped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
