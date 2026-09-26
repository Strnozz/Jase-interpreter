"""V24 continuation: fill measured routing gaps with balanced, held-out forms.

The V25 blind panel was frozen before this file. This generator never reads
benchmark gold into examples; benchmarks are used only for leakage exclusion.
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
from scripts.build_v21_9b_data import contract as c, fact as f, goal as g  # noqa: E402

DEST = ROOT / "data/v24_9b"
SOURCE = ROOT / "data/v23_9b"
SYSTEM = (ROOT / "V24_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
SEED = 2030
BENCHMARKS = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
              ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl",
              ROOT / "benchmarks/v23/transfer.jsonl", ROOT / "benchmarks/v24/safety.jsonl",
              ROOT / "benchmarks/v25/blind.jsonl"]


def row(text: str, gold: dict, category: str, split: str) -> dict:
    issues = validate_contract(gold)
    if issues:
        raise ValueError((text, [x.as_dict() for x in issues]))
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": json.dumps(gold, ensure_ascii=False, separators=(",", ":"))}],
            "category": category, "source_split": split,
            "source_id": hashlib.sha256(f"v24:{split}:{text}".encode()).hexdigest()[:20]}


def generate(split: str) -> list[dict]:
    train = split == "train"
    cities = (["Catanzaro", "Alessandria", "Nuoro", "Teramo", "La Spezia", "Enna",
               "Massa", "Rieti", "Trapani", "Lecco"] if train else
              ["Benevento", "Biella", "Belluno", "Oristano"])
    out: list[dict] = []

    def add(text: str, gold: dict, category: str) -> None:
        out.append(row(text, gold, category, split))

    specialties = (["visita neurologica", "visita allergologica", "visita podologica"] if train else
                   ["visita endocrinologica", "visita urologica"])
    for city in cities:
        for name in specialties:
            date = "mercoledì" if train else "giovedì"
            gold = c(g("g1", "cancel", name, "appointment", [f("date", "eq", date),
                f("location", "near", city)]))
            add(f"Disdici la {name} di {date} a {city}.", gold, "appointment_cancel")
            if train:
                add(f"A {city} annulla la mia {name} fissata per {date}.", gold,
                    "appointment_cancel")
            gold = c(g("g1", "book", name, "appointment", [f("date", "eq", date),
                f("time", "eq", "10:30"), f("location", "near", city)]))
            add(f"Prenota una {name} a {city} {date} alle 10:30.", gold,
                "appointment_book")
            if train:
                add(f"Per {date} alle 10:30 fissami a {city} una {name}.", gold,
                    "appointment_book")

    vehicles = (["bicicletta", "scooter", "automobile", "camper", "furgone"] if train else
                ["motocicletta", "monopattino elettrico", "minivan"])
    for city in cities:
        for vehicle in vehicles:
            article = "una" if vehicle in {"bicicletta", "automobile", "motocicletta"} else "un"
            gold = c(g("g1", "rent", vehicle, "vehicle", [f("location", "near", city),
                f("date", "eq", "sabato")]))
            add(f"A {city} noleggia {article} {vehicle} per sabato.", gold, "vehicle_rent")
            if train:
                add(f"Per sabato mi serve in affitto {article} {vehicle} a {city}.", gold,
                    "vehicle_rent")
            gold = c(g("g1", "find", vehicle, "vehicle", [f("location", "near", city),
                f("price", "lt", 75, currency="EUR")]))
            add(f"Trova {article} {vehicle} da noleggiare a {city} sotto 75 euro.", gold,
                "vehicle_find_price")

    reminders = (["pagare la quota del corso", "restituire il libro al vicino",
                  "inviare la ricevuta a Marta", "chiamare il tecnico",
                  "ritirare gli occhiali", "comprare i biglietti del museo",
                  "portare il passaporto", "rinnovare l'abbonamento",
                  "consegnare il modulo", "telefonare alla nonna"] if train else
                 ["ritirare la posta", "stampare il contratto", "comprare il latte",
                  "avvisare l'allenatore"])
    days = ["martedì", "sabato", "tra due giorni"] if train else ["giovedì", "domenica"]
    for message in reminders:
        for day in days:
            gold = c(g("g1", "notify", "promemoria", "reminder",
                [f("date", "eq", day), f("message", "eq", message)]))
            add(f"{day.capitalize()} ricordami di {message}.", gold, "reminder_message")
            if train:
                when = day if day.startswith("tra ") else f"per {day}"
                add(f"Imposta un promemoria {when}: devo {message}.", gold,
                    "reminder_message")

    products = ([("fotocamera", "fotocamere"), ("aspirapolvere", "aspirapolvere"),
                 ("sedia ergonomica", "sedie ergonomiche"), ("zaino", "zaini"),
                 ("lavatrice", "lavatrici"), ("tastiera", "tastiere"),
                 ("forno", "forni"), ("casco", "caschi"), ("router", "router"),
                 ("poltrona", "poltrone")] if train else
                [("ventilatore", "ventilatori"), ("borsone", "borsoni"),
                 ("scanner", "scanner"), ("microfono", "microfoni")])
    quantities = [2, 3, 4, 5] if train else [3, 5]
    number_words = {2: "due", 3: "tre", 4: "quattro", 5: "cinque"}
    for product, plural in products:
        for n in quantities:
            gold = c(g("g1", "compare", product, "product", [f("quantity", "eq", n)]))
            add(f"Confronta {number_words[n]} {plural} tra loro.", gold,
                "compare_quantity")
            if train:
                add(f"Voglio una comparazione fra {n} {plural}.", gold,
                    "compare_quantity")

    models = (["tablet Helios R5", "telefono Amato Q8", "portatile Tera N4",
               "orologio Axis P6", "fotocamera Nivo S2", "stampante Velo X9",
               "router Cavo Z6", "monitor Lume C3", "console Fenix T4",
               "aspirapolvere Aria M7", "cuffie Rivo B2", "proiettore Luma K5"] if train else
              ["tablet Filo D8", "telefono Lento V3", "monitor Prisma J2", "router Onda E7"])
    for name in models:
        gold = c(g("g1", "monitor", name, "product", [f("price", "lt", 280, currency="EUR")]))
        add(f"Monitora il prezzo di {name}: avvisami se va sotto 280 euro.", gold,
            "model_identity")
        if train:
            add(f"Tieni d'occhio {name} finché il prezzo non scende sotto 280 euro.",
                gold, "model_identity")

    modes = ["treno", "autobus", "pullman", "traghetto"] if train else ["treno", "pullman"]
    for city in cities:
        for mode in modes:
            arrival = c(g("g1", "find", mode, "transport", [f("destination", "eq", city),
                f("arrival_time", "before", "13:30")]))
            departure = c(g("g1", "find", mode, "transport", [f("origin", "eq", city),
                f("departure_time", "after", "13:30")]))
            add(f"Cerca un {mode} che arrivi a {city} prima delle 13:30.", arrival,
                "arrival_role")
            add(f"Cerca un {mode} che parta da {city} dopo le 13:30.", departure,
                "departure_role")
            if train:
                add(f"Per {city} vorrei un {mode} con arrivo entro le 13:30.", arrival,
                    "arrival_role")
                add(f"Da {city} vorrei un {mode} in partenza dopo le 13:30.", departure,
                    "departure_role")
    return out


def main() -> int:
    if DEST.exists():
        raise FileExistsError("V24 corpus already exists; do not regenerate silently")
    rng = random.Random(SEED)
    data = {split: read_jsonl(SOURCE / f"{split}.jsonl") for split in ("train", "valid")}
    for split in data:
        for item in data[split]:
            item["messages"][0]["content"] = SYSTEM
        data[split].extend(generate(split))
    excluded = {normalize_text(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)}
    benchmark_tokens = [token_set(r["text"]) for path in BENCHMARKS for r in read_jsonl(path)]
    seen: dict[str, str] = {}
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
    for split in data:
        rng.shuffle(data[split])
    DEST.mkdir(parents=True, exist_ok=False)
    for split in data:
        with (DEST / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in data[split]:
                gold = json.loads(item["messages"][-1]["content"])
                if validate_contract(gold):
                    raise RuntimeError(f"Invalid gold in {split}: {item['source_id']}")
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"version": "V24_9B", "schema_version": "1.2", "seed": SEED,
        "source": "V23 continuation plus audited appointment, vehicle, reminder, quantity, identity and temporal-role coverage",
        "source_sha256": {s: sha256(SOURCE / f"{s}.jsonl") for s in data},
        "excluded_benchmark_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in BENCHMARKS},
        "generator_sha256": sha256(Path(__file__)),
        "system_prompt_sha256": sha256(ROOT / "V24_SYSTEM_PROMPT.txt"),
        "rows": {s: len(data[s]) for s in data}, "dropped": dict(dropped),
        "categories": {s: dict(Counter(x["category"] for x in data[s])) for s in data},
        "output_sha256": {s: sha256(DEST / f"{s}.jsonl") for s in data},
        "limitations": ["Synthetic templates need human review", "V25 blind panel is small and feature-targeted"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": manifest["rows"], "dropped": manifest["dropped"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
