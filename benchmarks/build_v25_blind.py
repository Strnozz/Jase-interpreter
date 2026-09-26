"""Freeze a new blind diagnostic panel before V24 corpus construction.

Author-written Italian requests, not sampled from model output. Never use these
requests or gold contracts as training data. The panel is small and needs human
review before any production-quality claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.multimodel import sha256  # noqa: E402
from scripts.build_v21_9b_data import contract as c, fact as f, goal as g  # noqa: E402

OUT = ROOT / "benchmarks/v25/blind.jsonl"
ROWS: list[dict] = []


def add(category: str, text: str, gold: dict) -> None:
    issues = validate_contract(gold)
    if issues:
        raise ValueError((text, [x.as_dict() for x in issues]))
    ROWS.append({"id": f"b{len(ROWS)+1:02d}", "category": category,
                 "text": text, "contract": gold})


def miss(field: str, reason: str = "unspecified") -> list[dict]:
    return [{"field": field, "blocks": "action", "reason": reason}]


add("appointment_cancel", "Cancella il controllo dermatologico di giovedì alle 15 a Cuneo.",
    c(g("g1", "cancel", "controllo dermatologico", "appointment", [f("date", "eq", "giovedì"),
      f("time", "eq", "15:00"), f("location", "near", "Cuneo")])))
add("appointment_cancel", "Disdici la visita ortopedica fissata per il 12 ottobre.",
    c(g("g1", "cancel", "visita ortopedica", "appointment", [f("date", "eq", "12 ottobre")])))
add("appointment_book", "Prenota una visita dal fisioterapista a Rovigo martedì mattina.",
    c(g("g1", "book", "visita dal fisioterapista", "appointment", [f("location", "near", "Rovigo"),
      f("date", "eq", "martedì"), f("time_of_day", "eq", "mattina")])))
add("appointment_confirm", "Prenota il taglio dal barbiere venerdì alle 11, ma chiedimi conferma prima.",
    c(g("g1", "book", "taglio dal barbiere", "appointment", [f("date", "eq", "venerdì"),
      f("time", "eq", "11:00")], policy={"confirm_before": ["book"]})))
add("vehicle_rent", "Affitta un furgoncino a Bolzano per sabato e domenica.",
    c(g("g1", "rent", "furgoncino", "vehicle", [f("location", "near", "Bolzano"),
      f("date", "eq", "sabato e domenica")])))
add("vehicle_rent", "Mi serve a noleggio una mountain bike a Grosseto per lunedì.",
    c(g("g1", "rent", "mountain bike", "vehicle", [f("location", "near", "Grosseto"),
      f("date", "eq", "lunedì")])))
add("vehicle_rent", "Noleggia un motorino a Varese domani, meglio se elettrico.",
    c(g("g1", "rent", "motorino", "vehicle", [f("location", "near", "Varese"),
      f("date", "eq", "domani"), f("feature", "eq", "elettrico", strength="soft")])))
add("vehicle_rent", "Cerca un camper da affittare a Potenza sotto 80 euro al giorno.",
    c(g("g1", "find", "camper", "vehicle", [f("location", "near", "Potenza"),
      f("price", "lt", 80, currency="EUR")])))
add("reminder_message", "Martedì ricordami di spedire il modulo al commercialista.",
    c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "martedì"),
      f("message", "eq", "spedire il modulo al commercialista")])))
add("reminder_message", "Domani alle 8 avvisami di prendere le medicine.",
    c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "domani"),
      f("time", "eq", "08:00"), f("message", "eq", "prendere le medicine")])))
add("reminder_missing", "Ricordami venerdì quella scadenza di cui parlavamo; qui non è specificata.",
    c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "venerdì")],
      missing=miss("referent", "unresolved_reference"))))
add("contact_message", "Scrivi a Lidia: porterò io le chiavi sabato.",
    c(g("g1", "contact", "Lidia", "person", [f("message", "eq", "porterò io le chiavi sabato")])))
add("compare_count", "Metti a confronto quattro stampanti laser sotto 190 euro.",
    c(g("g1", "compare", "stampante laser", "product", [f("quantity", "eq", 4),
      f("price", "lt", 190, currency="EUR")])))
add("compare_count", "Confronta due valigie da cabina, una rigida e una morbida.",
    c(g("g1", "compare", "valigia da cabina", "product", [f("quantity", "eq", 2),
      f("material", "eq", "rigida e morbida")])))
add("model_identity", "Controlla se il portatile Zenbook Q77 scende sotto 900 euro.",
    c(g("g1", "monitor", "portatile Zenbook Q77", "product", [f("price", "lt", 900, currency="EUR")])))
add("model_identity", "Compra il caricatore VoltPro M2 nero, massimo 35 euro.",
    c(g("g1", "buy", "caricatore VoltPro M2", "product", [f("color", "eq", "nero"),
      f("price", "lte", 35, currency="EUR")])))
add("arrival", "Mostrami treni per Asti che arrivino prima delle 12:15.",
    c(g("g1", "find", "treno", "transport", [f("destination", "eq", "Asti"),
      f("arrival_time", "before", "12:15")])))
add("departure", "Cerca autobus in partenza da Asti dopo le 18:45.",
    c(g("g1", "find", "autobus", "transport", [f("origin", "eq", "Asti"),
      f("departure_time", "after", "18:45")])))
add("ranking", "Devo raggiungere Todi entro le 10 di lunedì: ordina i mezzi per prezzo crescente.",
    c(g("g1", "find", "trasporto", "transport", [f("destination", "eq", "Todi"),
      f("arrival_time", "before", "10:00"), f("date", "eq", "lunedì")],
      ranking={"field": "price", "direction": "asc"})))
add("soft_vs_hard", "Trova un agriturismo a Cremona con parcheggio; la piscina sarebbe un bonus.",
    c(g("g1", "find", "agriturismo", "accommodation", [f("location", "near", "Cremona"),
      f("amenity", "eq", "parcheggio"), f("amenity", "eq", "piscina", strength="soft")])))
add("forbid", "Fammi vedere negozi di biciclette a Viterbo, senza acquistare nulla.",
    c(g("g1", "find", "negozio di biciclette", "place", [f("location", "near", "Viterbo")],
      policy={"forbid": ["buy"]})))
add("non_actionable", "In questa poesia compare 'ordina un taxi', ma sto solo chiedendo un commento sul testo.",
    c(kind="non_actionable"))
add("context", "Siamo in tre e abbiamo già l'hotel. Cerca un taxi per la fiera di Parma.",
    c(g("g1", "find", "taxi", "transport", [f("destination", "eq", "fiera di Parma")]),
      context=[{"field": "party_size", "op": "eq", "value": 3},
               {"field": "lodging_owned", "op": "eq", "value": True}]))
add("two_goals", "Trova una farmacia a Vibo Valentia e, a parte, un cinema a Isernia.",
    c(g("g1", "find", "farmacia", "place", [f("location", "near", "Vibo Valentia")]),
      g("g2", "find", "cinema", "place", [f("location", "near", "Isernia")])))


def main() -> None:
    if OUT.exists():
        raise FileExistsError("V25 blind panel is frozen and must not be rewritten")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ROWS), encoding="utf-8")
    manifest = {"version": "V25_BLIND", "cases": len(ROWS), "sha256": sha256(OUT),
                "method": "Manually authored diagnostic, frozen before V24 dataset; never used for training",
                "limitation": "Small feature-targeted AI-authored panel; human gold review still needed"}
    OUT.with_name("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
