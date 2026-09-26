"""Freeze a schema-validated diagnostic panel before V23 training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.multimodel import sha256  # noqa: E402
from scripts.build_v21_9b_data import contract as c, fact as f, goal as g  # noqa: E402

OUT = ROOT / "benchmarks/v23/transfer.jsonl"
ROWS = []


def add(category: str, text: str, gold: dict) -> None:
    issues = validate_contract(gold)
    if issues:
        raise ValueError((text, [i.as_dict() for i in issues]))
    ROWS.append({"id": f"t{len(ROWS)+1:02d}", "category": category,
                 "text": text, "contract": gold})


def missing(*fields: str) -> list[dict]:
    return [{"field": name, "blocks": "action", "reason": "unspecified"} for name in fields]


add("simple", "Cercami un dentista a Udine.", c(g("g1", "find", "dentista", "professional_service",
    [f("location", "near", "Udine")])))
add("constraints", "Voglio un agriturismo a Mantova sotto 110 euro con parcheggio.",
    c(g("g1", "find", "agriturismo", "accommodation", [f("location", "near", "Mantova"),
        f("price", "lt", 110, currency="EUR"), f("amenity", "eq", "parcheggio")])))
add("soft", "Cerca un hotel a Lecco entro 95 euro; la colazione inclusa sarebbe un vantaggio.",
    c(g("g1", "find", "hotel", "accommodation", [f("location", "near", "Lecco"),
        f("price", "lte", 95, currency="EUR"), f("amenity", "eq", "colazione inclusa", strength="soft")])))
add("forbid", "Mostrami librerie a Padova, ma non comprare alcun libro.",
    c(g("g1", "find", "libreria", "place", [f("location", "near", "Padova")],
        policy={"forbid": ["buy"]})))
add("missing_table", "Mi occupi un tavolo per cena? Non ho ancora scelto giorno, ora o numero di ospiti.",
    c(g("g1", "book", "tavolo", "place", [], missing=missing("date", "time", "guests"))))
add("missing_flight", "Vorrei prenotare un aereo per Reykjavik; da dove e quando parto te lo dico dopo.",
    c(g("g1", "book", "volo", "transport", [f("destination", "eq", "Reykjavik")],
        missing=missing("origin", "date"))))
add("confirm_contact", "Avvisa Nora che arriverò alle 18, ma aspetta la mia conferma prima di contattarla.",
    c(g("g1", "contact", "Nora", "person", [f("message", "eq", "arriverò alle 18")],
        policy={"confirm_before": ["contact"]})))
add("missing_message", "Scrivi a Paolo, il testo del messaggio te lo darò più tardi.",
    c(g("g1", "contact", "Paolo", "person", [], missing=missing("message"))))
add("missing_reference", "Fra due giorni ricordami quella faccenda di cui parlavamo.",
    c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "fra due giorni")],
        missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])))
add("notify", "Mercoledì ricordami di pagare la bolletta della luce.",
    c(g("g1", "notify", "promemoria", "reminder", [f("date", "eq", "mercoledì"),
        f("message", "eq", "pagare la bolletta della luce")])))
add("independent", "Cerca una farmacia a Sassari e, separatamente, un teatro a Como.",
    c(g("g1", "find", "farmacia", "place", [f("location", "near", "Sassari")]),
      g("g2", "find", "teatro", "place", [f("location", "near", "Como")])))
add("dependent_reference", "Trova un autobus per Parma e poi un hotel vicino alla fermata d'arrivo.",
    c(g("g1", "find", "autobus", "transport", [f("destination", "eq", "Parma")]),
      g("g2", "find", "hotel", "accommodation", [{"field": "location", "op": "near",
        "value_ref": {"goal": "g1", "field": "arrival_stop"}, "strength": "hard"}], depends_on=["g1"])))
add("ordinal_chain", "Fammi vedere cinque librerie a Terni e poi contatta la seconda, ma solo se ti do il via libera.",
    c(g("g1", "find", "libreria", "place", [f("quantity", "eq", 5), f("location", "near", "Terni")]),
      g("g2", "contact", "libreria", "place", [{"field": "selection", "op": "eq",
        "value_ref": {"goal": "g1", "field": "result", "ordinal": 2}, "strength": "hard"}],
        depends_on=["g1"], policy={"confirm_before": ["contact"]}, missing=missing("message"))))
add("fallback", "Cerca dei burrito; se non ne trovi, cerca un ristorante peruviano.",
    c(g("g1", "find", "burrito", "food", []),
      g("g2", "find", "ristorante", "place", [f("cuisine", "eq", "peruviana")],
        depends_on=["g1"], condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})))
add("positive_condition", "Se esiste un treno per Pisa sotto 35 euro, cerca anche un bed and breakfast.",
    c(g("g1", "find", "treno", "transport", [f("destination", "eq", "Pisa"),
        f("price", "lt", 35, currency="EUR")]),
      g("g2", "find", "bed and breakfast", "accommodation", [], depends_on=["g1"],
        condition={"goal": "g1", "test": {"field": "result_count", "op": "gt", "value": 0}})))
add("ranking_generic", "Devo essere a Vercelli sabato entro le 11; cerca il trasporto meno costoso.",
    c(g("g1", "find", "trasporto", "transport", [f("destination", "eq", "Vercelli"),
        f("date", "eq", "sabato"), f("arrival_time", "before", "11:00")],
        ranking={"field": "price", "direction": "asc"})))
add("ranking_mode", "Mostrami i pullman per Novara prima delle 17, dal meno caro al più caro.",
    c(g("g1", "find", "pullman", "transport", [f("destination", "eq", "Novara"),
        f("arrival_time", "before", "17:00")], ranking={"field": "price", "direction": "asc"})))
add("context_owned", "L'albergo è già prenotato. Mi trovi un taxi per il porto lunedì mattina?",
    c(g("g1", "find", "taxi", "transport", [f("destination", "eq", "porto"),
        f("date", "eq", "lunedì"), f("time_of_day", "eq", "mattina")]),
        context=[{"field": "lodging_owned", "op": "eq", "value": True}]))
add("context_group", "Siamo in sei, senza automobile. Trova un ristorante in centro a Pescara.",
    c(g("g1", "find", "ristorante", "place", [f("location", "near", "centro a Pescara")]),
        context=[{"field": "party_size", "op": "eq", "value": 6},
                 {"field": "has_car", "op": "eq", "value": False}]))
add("non_actionable_theory", "Quali informazioni servono di norma per noleggiare una barca? È una domanda teorica.",
    c(kind="non_actionable"))
add("non_actionable_quote", "Nel mio documento compare la frase «prenota un taxi» come esempio, non eseguirla.",
    c(kind="non_actionable"))
add("prior_reference", "Acquista il quarto orologio dell'elenco che mi hai mostrato ieri, ma ora non vedo l'elenco.",
    c(g("g1", "buy", "orologio", "product", [],
        missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])))
add("cancel", "Annulla la visita dal dentista di venerdì a Padova.",
    c(g("g1", "cancel", "visita dal dentista", "appointment", [f("date", "eq", "venerdì"),
        f("location", "near", "Padova")])))
add("hire", "Assumi un giardiniere a Lucca per martedì.",
    c(g("g1", "hire", "giardiniere", "professional_service", [f("location", "near", "Lucca"),
        f("date", "eq", "martedì")])))
add("compare", "Confronta due portatili con almeno 16 GB di RAM.",
    c(g("g1", "compare", "portatile", "product", [f("quantity", "eq", 2),
        f("ram", "gte", 16, unit="GB")])))
add("rent", "Noleggia una bicicletta a Ferrara per sabato.",
    c(g("g1", "rent", "bicicletta", "vehicle", [f("location", "near", "Ferrara"),
        f("date", "eq", "sabato")])))
add("monitor", "Monitora il prezzo del tablet X sotto 300 euro.",
    c(g("g1", "monitor", "tablet X", "product", [f("price", "lt", 300, currency="EUR")])))
add("order", "Ordina un mazzo di tulipani per mia sorella a Livorno domani.",
    c(g("g1", "order", "mazzo di tulipani", "product", [f("recipient", "eq", "mia sorella"),
        f("location", "near", "Livorno"), f("date", "eq", "domani")])))
add("buy", "Compra una lampada da scrivania blu entro 45 euro.",
    c(g("g1", "buy", "lampada da scrivania", "product", [f("color", "eq", "blu"),
        f("price", "lte", 45, currency="EUR")])))
add("soft_location", "Una casa a Vicenza con tre camere, meglio se vicino alla stazione.",
    c(g("g1", "find", "casa", "real_estate", [f("location", "near", "Vicenza"),
        f("rooms", "eq", 3), f("distance_to_station", "near", "stazione", strength="soft")])))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in ROWS:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {"version": "V23_TRANSFER", "cases": len(ROWS), "sha256": sha256(OUT),
        "method": "Manually authored diagnostic, frozen before V23 dataset construction; no model outputs copied",
        "limitation": "Feature-targeted small panel, not a population-level independent production sample"}
    OUT.with_name("manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
