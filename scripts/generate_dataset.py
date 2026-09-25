#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = Path(__file__).resolve().parents[1].joinpath("TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()

# Broad but intentionally finite seed vocabulary. The model is expected to learn the
# STRUCTURE; held-out and invented entities test whether it can copy unseen values.
TARGETS_TRAIN = [
    ("panino", "food"), ("pizza", "food"), ("ristorante", "place"),
    ("elettricista", "professional_service"), ("idraulico", "professional_service"),
    ("fotografo", "professional_service"), ("hotel", "accommodation"),
    ("volo", "transport"), ("treno", "transport"), ("auto usata", "product"),
    ("laptop", "product"), ("scheda grafica", "product"), ("chitarra", "product"),
    ("appartamento", "real_estate"), ("libro", "media"), ("corso", "education"),
    ("gatto", "animal"), ("cane", "animal"), ("biglietto", "ticket"),
    ("grafico", "professional_service"), ("regalo", "product"), ("farmacia", "place"),
]

TARGETS_HELDOUT = [
    ("stampante 3D", "product"), ("violino", "product"), ("veterinario", "professional_service"),
    ("coworking", "place"), ("bicicletta cargo", "product"), ("pappagallo", "animal"),
    ("traduttore", "professional_service"), ("camper", "product"),
]

LOCATIONS = ["Milano", "Torino", "Roma", "Verbania", "Bologna", "Firenze", "Napoli", "Genova", "Palermo", "Bari"]
TIMES = ["stasera", "domani", "sabato", "entro venerdì", "questa settimana", "alle 20:30"]
CURRENCIES = ["EUR"]

ATTRS_BY_TYPE = {
    "food": [("ingredient", ["mortadella", "prosciutto", "mozzarella", "melanzane"]), ("style", ["vegano", "senza glutine", "piccante"])],
    "animal": [("breed", ["Siamese", "Sphynx", "Labrador", "Maine Coon"]), ("sex", ["female", "male"]), ("age_months", [3, 6, 12])],
    "product": [("color", ["rosso", "nero", "bianco", "blu"]), ("condition", ["nuovo", "usato", "ricondizionato"]), ("feature", ["USB-C", "32 GB RAM", "4K", "elettrico"])],
    "professional_service": [("skill", ["urgenze", "impianti", "ritratti", "branding", "riparazioni"]), ("rating", [4.0, 4.5, 4.8])],
    "accommodation": [("stars", [3, 4, 5]), ("feature", ["parcheggio", "colazione", "spa", "wifi"])],
    "transport": [("class", ["economy", "business", "prima classe"]), ("feature", ["bagaglio incluso", "diretto", "wifi"])],
    "real_estate": [("rooms", [1, 2, 3, 4]), ("feature", ["balcone", "garage", "arredato", "ascensore"])],
    "media": [("format", ["cartaceo", "ebook", "audiolibro"]), ("language", ["italiano", "inglese"])],
    "education": [("format", ["online", "in presenza"]), ("level", ["base", "intermedio", "avanzato"])],
    "place": [("rating", [4.0, 4.5, 4.8]), ("feature", ["aperto tardi", "parcheggio", "accessibile"]),],
    "ticket": [("section", ["platea", "tribuna", "curva"]), ("quantity", [1, 2, 4])],
}

ACTION_TEMPLATES = {
    "find": ["trova", "cercami", "mi trovi", "sto cercando"],
    "buy": ["voglio comprare", "comprami", "cercami da comprare", "vorrei acquistare"],
    "book": ["prenotami", "vorrei prenotare", "trova e prenota", "mi serve una prenotazione per"],
    "hire": ["mi serve", "trova", "vorrei contattare", "cercami"],
    "compare": ["confrontami", "metti a confronto", "fammi confrontare", "paragona"],
}

ACTION_BY_TYPE = {
    "food": ["find", "buy"], "place": ["find", "book"], "professional_service": ["find", "hire"],
    "accommodation": ["find", "book"], "transport": ["find", "book"], "product": ["find", "buy", "compare"],
    "real_estate": ["find"], "media": ["find", "buy"], "education": ["find", "book"],
    "animal": ["find"], "ticket": ["find", "buy"],
}


def pred(prop, op, value, unit=None, currency=None, reference=None):
    d = {"property": prop, "operator": op, "value": value}
    if unit is not None: d["unit"] = unit
    if currency is not None: d["currency"] = currency
    if reference is not None: d["reference"] = reference
    return d


def goal(action, raw_action, target_name, target_type, raw_target, attrs=None, constraints=None, prefs=None):
    return {
        "action": action,
        "raw_action": raw_action,
        "target": {"name": target_name, "type": target_type, "raw": raw_target},
        "attributes": attrs or [],
        "constraints": constraints or [],
        "preferences": prefs or [],
    }


def contract(goals, clarification=False, reason=None, missing=None):
    return {
        "schema_version": "1.0",
        "goals": goals,
        "clarification": {"required": clarification, "reason": reason, "missing_fields": missing or []},
    }


def choose_attr(rng, target_type):
    choices = ATTRS_BY_TYPE.get(target_type, [("feature", ["specifico", "compatibile", "recente"])])
    prop, values = rng.choice(choices)
    value = rng.choice(values)
    return prop, value


def render_request(rng, action, target_name, target_type, include_attr=True, include_location=True,
                   include_budget=True, include_time=False, soft_attr=False, invented=False):
    raw_action = rng.choice(ACTION_TEMPLATES[action])
    raw_target = target_name
    parts = []
    attrs, constraints, prefs = [], [], []

    # Main clause
    if raw_action == "sto cercando":
        parts.append(f"Sto cercando {target_name}")
    elif target_type == "professional_service" and raw_action in {"mi serve", "vorrei contattare"}:
        parts.append(f"{raw_action.capitalize()} un {target_name}")
    else:
        article = "un"
        if target_name in {"pizza", "scheda grafica", "farmacia", "bicicletta cargo", "stampante 3D"}:
            article = "una"
        parts.append(f"{raw_action.capitalize()} {article} {target_name}")

    if include_attr:
        prop, value = choose_attr(rng, target_type)
        op = "eq"
        phrase = None
        if prop == "ingredient":
            op = "contains"; phrase = f"con {value}"
        elif prop == "rating":
            op = "gte"; phrase = f"con almeno {value} stelle"
        elif prop == "age_months":
            op = "lte"; phrase = f"di massimo {value} mesi"
        elif prop == "quantity":
            op = "eq"; phrase = f"per {value} persone"
        elif prop == "rooms":
            op = "gte"; phrase = f"con almeno {value} stanze"
        elif prop == "stars":
            op = "gte"; phrase = f"da almeno {value} stelle"
        elif prop == "feature":
            op = "contains"; phrase = f"con {value}"
        elif prop == "skill":
            op = "contains"; phrase = f"esperto in {value}"
        else:
            phrase = f"{prop} {value}"

        bucket = prefs if soft_attr else attrs
        bucket.append(pred(prop, op, value, unit="months" if prop == "age_months" else None))
        if soft_attr:
            parts.append(f"possibilmente {phrase}")
        else:
            parts.append(phrase)

    if include_location:
        location = rng.choice(LOCATIONS)
        constraints.append(pred("location", "near", location))
        parts.append(f"a {location}")

    if include_budget:
        budget = rng.choice([10, 20, 50, 100, 300, 700, 1200, 2000])
        constraints.append(pred("price", "lte", budget, currency="EUR"))
        parts.append(rng.choice([f"sotto i {budget}€", f"massimo {budget} euro", f"senza superare {budget}€"]))

    if include_time:
        t = rng.choice(TIMES)
        constraints.append(pred("time", "before" if t.startswith("entro") else "eq", t))
        parts.append(t)

    # Slightly vary punctuation/noise.
    request = rng.choice([
        ", ".join(parts) + ".",
        " ".join(parts) + " per favore",
        ", ".join(parts),
    ])

    return request, goal(action, raw_action, target_name, target_type, raw_target, attrs, constraints, prefs)


def invented_target(rng):
    # Unseen names teach copy/generalization instead of memorization.
    prefix = rng.choice(["Zorba", "Nexa", "Velun", "Kiro", "Aster", "Movo", "Prax"])
    suffix = rng.choice(["X7", "Mini", "Pro", "One", "Nova", "Q", "Lite"])
    return f"{prefix} {suffix}", "product"


def make_example(rng, split):
    heldout = split == "test" and rng.random() < 0.45
    invented = split == "test" and rng.random() < 0.18
    if invented:
        target_name, target_type = invented_target(rng)
    else:
        target_name, target_type = rng.choice(TARGETS_HELDOUT if heldout else TARGETS_TRAIN)

    action = rng.choice(ACTION_BY_TYPE[target_type])
    include_attr = rng.random() < 0.72
    include_location = rng.random() < 0.72
    include_budget = rng.random() < 0.58
    include_time = rng.random() < 0.35
    soft_attr = include_attr and rng.random() < 0.22

    request, g = render_request(rng, action, target_name, target_type, include_attr,
                                include_location, include_budget, include_time, soft_attr, invented)

    # Negation hard cases.
    if rng.random() < 0.12:
        prop = "color" if target_type == "product" else "feature"
        value = rng.choice(["rosso", "abbonamento", "scalo", "glutine"])
        g["constraints"].append(pred(prop, "neq" if prop == "color" else "excludes", value))
        request += f", ma non {value}"

    # A small share of intentionally ambiguous requests.
    clarification = False
    reason = None
    missing = []
    if rng.random() < 0.05:
        request = rng.choice(["Prenotamelo per domani.", "Trova quello più vicino.", "Compramelo entro stasera."])
        g = goal(action, raw_action=action, target_name="unknown", target_type=None, raw_target="unknown")
        clarification = True
        reason = "target_missing"
        missing = ["target"]

    # Multi-goal requests teach goals[] rather than a single-intent assumption.
    goals = [g]
    if not clarification and rng.random() < 0.08:
        second_name, second_type = rng.choice(TARGETS_TRAIN)
        second_action = rng.choice(ACTION_BY_TYPE[second_type])
        req2, g2 = render_request(rng, second_action, second_name, second_type,
                                  include_attr=rng.random() < .4,
                                  include_location=False,
                                  include_budget=rng.random() < .3,
                                  include_time=False,
                                  soft_attr=False)
        request = request.rstrip(".") + " e poi " + req2[0].lower() + req2[1:]
        goals.append(g2)

    c = contract(goals, clarification, reason, missing)
    return request, c


def as_chat(request, contract_obj):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": request},
            {"role": "assistant", "content": json.dumps(contract_obj, ensure_ascii=False, separators=(",", ":"))},
        ]
    }


def write_split(path, n, rng, split):
    with path.open("w", encoding="utf-8") as f:
        for _ in range(n):
            request, c = make_example(rng, split)
            f.write(json.dumps(as_chat(request, c), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", help="Output directory")
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--valid", type=int, default=300)
    ap.add_argument("--test", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    write_split(out / "train.jsonl", args.train, rng, "train")
    write_split(out / "valid.jsonl", args.valid, rng, "valid")
    write_split(out / "test.jsonl", args.test, rng, "test")
    print(f"Generated {args.train} train, {args.valid} valid, {args.test} test examples in {out}")
    print("The test split intentionally contains held-out and invented entities.")

if __name__ == "__main__":
    main()
