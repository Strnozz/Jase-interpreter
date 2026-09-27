"""Build V26 training data from prior training corpora and independent contrasts.

All benchmark panels through V29 are exclusion-only. Never use their gold here.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.migrate_contract_v1_3 import MigrationHold, upgrade_contract  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

SEED = 2032
DEST = ROOT / "data/v26_9b"
PROMPT_FILE = ROOT / "V26_SYSTEM_PROMPT.txt"
PROMPT = PROMPT_FILE.read_text(encoding="utf-8").strip()
V24 = ROOT / "data/v24_9b"
V25 = ROOT / "data/v25_9b"
PANELS = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
          ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl",
          *[ROOT / f"benchmarks/v{i}/{name}.jsonl" for i, name in
            [(23, "transfer"), (24, "safety"), (25, "blind"), (26, "blind"),
             (27, "fresh"), (28, "blind"), (29, "blind")]]]


def fact(field, value, op="eq"):
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def temporal(role, value, op="eq"):
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def goal(action, name, typ, facts=None, **more):
    return {"id": "g1", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **more}


def contract(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


def make_row(text, gold, category, split, source):
    issues = validate_contract(gold, user_text=text)
    if issues:
        raise ValueError((category, text, [i.as_dict() for i in issues]))
    return {"messages": [{"role": "system", "content": PROMPT},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False,
                                                                     separators=(",", ":"))}],
            "category": category, "source": source, "source_split": split,
            "source_id": hashlib.sha256(f"v26:{source}:{split}:{text}".encode()).hexdigest()[:20]}


def fresh(split):
    # Entity pools are disjoint across splits; wording and combinations vary by family.
    cities = (["Aosta", "Cuneo", "Siena", "Trento", "Matera", "Modena", "Lecce", "Perugia",
               "Viterbo", "Fermo", "Prato", "Brescia"] if split == "train" else
              ["Nuoro", "Rovigo", "Terni", "Gorizia"])
    origins = (["Bari", "Bologna", "Genova", "Palermo", "Verona", "Pisa", "Catania", "Ancona",
                "Napoli", "Firenze", "Torino", "Trieste"] if split == "train" else
               ["Brindisi", "Sassari", "Parma", "Taranto"])
    people = (["Marta", "Luca", "Elisa", "Gianni", "Nadia", "Pietro", "Ada", "Sergio",
               "Ilaria", "Fabio", "Miriam", "Carlo"] if split == "train" else
              ["Viola", "Dario", "Olga", "Rita"])
    out = []
    def add(text, gold, category):
        out.append(make_row(text, gold, category, split, "v26_fresh"))

    for i, (city, origin, person) in enumerate(zip(cities, origins, people)):
        date = ["lunedì", "martedì", "giovedì", "sabato"][i % 4]
        time = ["13:20", "15:10", "18:30", "20:40"][i % 4]
        for form in range(4 if split == "train" else 2):
            intro = ["Cerca", "Trovami", "Mostrami", "Individua"][form]
            add(f"{intro} un treno da {origin} a {city} che arrivi entro le {time} {date}.",
                contract(goal("find", "treno", "transport", [fact("origin", origin),
                         fact("destination", city)], temporal=[temporal("arrival_time", time, "before"),
                         temporal("search_date", date)])), "route_arrival")
            add(f"{intro} un treno da {origin} a {city} con partenza dopo le {time} {date}.",
                contract(goal("find", "treno", "transport", [fact("origin", origin),
                         fact("destination", city)], temporal=[temporal("departure_time", time, "after"),
                         temporal("search_date", date)])), "route_departure")
            add(f"{intro} un volo diretto da {origin} a {city} per {date}, senza acquistare biglietti.",
                contract(goal("find", "volo", "transport", [fact("origin", origin),
                         fact("destination", city), fact("direct", True)],
                         temporal=[temporal("search_date", date)], policy={"forbid": ["book"]})),
                "direct_no_book")
            msg = f"Arriverò alle {time}"
            add(f"Scrivi a {person}: '{msg}'. Inviagli il messaggio solo dopo la mia conferma.",
                contract(goal("contact", person, "person", [fact("message", msg)],
                         policy={"confirm_before": ["contact"]})), "quoted_message")
            add(f"{intro} un ristorante aperto adesso a {city}, raggiungibile a piedi; non prenotare.",
                contract(goal("find", "ristorante", "place", [fact("location", city, "near"),
                         fact("open_now", True), fact("access", "a piedi")],
                         policy={"forbid": ["book"]})), "place_constraints")
            add(f"Ricordami {date} alle {time} di chiamare {person}.",
                contract(goal("notify", "promemoria", "reminder", [fact("message", f"chiamare {person}")],
                         temporal=[temporal("notification_date", date),
                                   temporal("notification_time", time)])), "reminder_content")
            add(f"Nel romanzo un personaggio dice 'prenota un volo da {origin} a {city}' e poi cambia argomento.",
                contract(), "quoted_non_actionable")
    return out


BAD_GRAMMAR = re.compile(r"\b(?:una|la|mia) controllo\b|\bun (?:auto|barca|bicicletta|moto|canoa|e-bike)\b", re.I)


def source_rows(split, drops):
    result = []
    for old in read_jsonl(V24 / f"{split}.jsonl"):
        text = old["messages"][1]["content"]
        try:
            gold = upgrade_contract(json.loads(old["messages"][-1]["content"]))
        except MigrationHold as exc:
            drops[f"v24_{split}_migration_hold:{exc}"] += 1
            continue
        result.append(make_row(text, gold, old["category"], split, "v24_converted"))
    for old in read_jsonl(V25 / f"{split}.jsonl"):
        text = old["messages"][1]["content"]
        if BAD_GRAMMAR.search(text):
            drops[f"v25_{split}_grammar"] += 1
            continue
        gold = json.loads(old["messages"][-1]["content"])
        result.append(make_row(text, gold, old["category"], split, "v25_selected"))
    return result


def overlap(a, b):
    return bool(a and b) and len(a & b) / len(a | b) >= .85


def main():
    if DEST.exists():
        raise FileExistsError("V26 already exists; never regenerate frozen corpus")
    panel_manifest = json.loads((ROOT / "benchmarks/v29/manifest.json").read_text(encoding="utf-8"))
    if sha256(ROOT / "benchmarks/v29/blind.jsonl") != panel_manifest["panel_sha256"]:
        raise RuntimeError("V29 panel changed before generation")
    rng = random.Random(SEED)
    drops = Counter()
    benchmark_texts = [r["text"] for p in PANELS for r in read_jsonl(p)]
    benchmark_exact = {normalize_text(s) for s in benchmark_texts}
    benchmark_tokens = [token_set(s) for s in benchmark_texts]
    seen = {}
    data = {}
    for split in ("train", "valid"):
        candidates = source_rows(split, drops) + fresh(split)
        rng.shuffle(candidates)
        filtered = []
        for row in candidates:
            text = row["messages"][1]["content"]
            key, tokens = normalize_text(text), token_set(text)
            if key in benchmark_exact or any(overlap(tokens, b) for b in benchmark_tokens):
                drops[f"{split}_benchmark_overlap"] += 1
                continue
            if key in seen:
                if seen[key] != row["messages"][-1]["content"]:
                    drops[f"{split}_conflicting_duplicate"] += 1
                else:
                    drops[f"{split}_duplicate"] += 1
                continue
            seen[key] = row["messages"][-1]["content"]
            filtered.append(row)
        data[split] = filtered
    train_tokens = [token_set(r["messages"][1]["content"]) for r in data["train"]]
    validation = []
    for row in data["valid"]:
        tok = token_set(row["messages"][1]["content"])
        if any(overlap(tok, train) for train in train_tokens):
            drops["valid_near_train"] += 1
        else:
            validation.append(row)
    data["valid"] = validation
    DEST.mkdir(parents=True, exist_ok=False)
    for split, rows in data.items():
        with (DEST / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V26_9B", "schema_version": "1.3", "seed": SEED,
                "panel_frozen_before_data": {"path": "benchmarks/v29/blind.jsonl",
                                             "sha256": panel_manifest["panel_sha256"]},
                "source_sha256": {f"v{v}_{s}": sha256(p / f"{s}.jsonl")
                                  for v, p in ((24, V24), (25, V25)) for s in data},
                "excluded_benchmark_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PANELS},
                "generator_sha256": sha256(Path(__file__)),
                "system_prompt_sha256": sha256(PROMPT_FILE),
                "rows": {s: len(rows) for s, rows in data.items()},
                "sources": {s: dict(Counter(r["source"] for r in rows)) for s, rows in data.items()},
                "categories": {s: dict(Counter(r["category"] for r in rows)) for s, rows in data.items()},
                "dropped": dict(drops), "output_sha256": {s: sha256(DEST / f"{s}.jsonl") for s in data},
                "limitations": ["Synthetic training data; human gold audit pending",
                                "V29 is a small agent-authored targeted panel"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows", "sources", "dropped")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
