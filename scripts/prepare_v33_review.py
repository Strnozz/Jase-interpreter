"""Prepare a human review queue without changing frozen V33 gold or outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "benchmarks/v33/blind.jsonl"
QUEUE = ROOT / "benchmarks/v33/human_review_queue.jsonl"
OUT = ROOT / "benchmarks/v33_review/review_queue.jsonl"
EXPECTED_SHA = "4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37"

REASONS = {
    "translation_of_target": "Target inglese e traduzione italiana richiedono una regola di equivalenza esplicita.",
    "availability_field": "Disponibilità per la sera e data di ricerca hanno ruoli diversi.",
    "selection_unspecified": "La struttura da prenotare non è stata ancora selezionata; il riferimento non va inventato.",
    "context_ontology": "La collocazione tra target, fatto e contesto utente cambia il binding Planner.",
    "same_time_reference": "«Stesso orario» rinvia all'appuntamento precedente e richiede una sorgente verificata.",
    "verb_hire_or_contact": "«Fai venire» può implicare ingaggio o contatto preliminare.",
    "approximate_time": "«Verso le nove» esprime un orario approssimativo, non un istante esatto.",
    "monthly_price_sort": "«Per canone mensile» può indicare il criterio di confronto senza ordinamento crescente.",
    "english_vehicle_buy_policy": "Bici usata: classificazione vehicle/product e ambito del divieto di acquisto.",
    "translated_city_names": "Torino/Turin e Berlino/Berlin richiedono normalizzazione multilingue esplicita.",
    "message_paraphrase": "Il testo del messaggio può essere parafrasato; data citata non è data di invio.",
    "policy_scope": "Il divieto di prenotazione può essere irrilevante per il solo invio oppure globale.",
    "selection_then_third": "«Seleziona un ristorante» e «prenota il terzo» hanno un riferimento potenzialmente incoerente.",
    "stops_operator": "«Con scalo» può significare almeno uno scalo, non esattamente uno.",
    "context_duplicate_location": "Ubicazione nel contesto e fatto location possono essere entrambi necessari o ridondanti.",
    "context_field": "Localizzazione dell'auto dell'utente può essere contesto distinto dalla location della ricerca.",
    "send_me_result_not_external_contact": "«Mandami il nome» può essere un output del sistema, non un contatto esterno.",
    "find_or_book_ambiguous": "«Vorrei un biglietto» può esprimere ricerca o intenzione di acquisto non ancora eseguibile.",
}


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    if hashlib.sha256(PANEL.read_bytes()).hexdigest() != EXPECTED_SHA:
        raise RuntimeError("V33 raw panel hash changed")
    gold = {row["id"]: row for row in rows(PANEL)}
    qwen = {row["id"]: row for row in rows(ROOT / "benchmarks/outputs/qwen-v26-v33-blind/cases.jsonl")}
    granite = {row["id"]: row for row in rows(ROOT / "benchmarks/outputs/granite41-v26-final-v33/cases.jsonl")}
    review = []
    for row in rows(QUEUE):
        flags = row.get("review_flags") or []
        # The frozen queue has one string-valued flag; keep the source untouched.
        if isinstance(flags, str):
            flags = [flags]
        if not flags:
            continue
        ident = row["id"]
        review.append({
            "case_id": ident,
            "request": gold[ident]["text"],
            "family": gold[ident]["family"],
            "raw_gold": gold[ident]["contract"],
            "qwen_raw_output": qwen[ident]["raw"],
            "granite_raw_output": granite[ident]["raw"],
            "review_flags": flags,
            "gold_question": [REASONS.get(flag, flag) for flag in flags],
            "ontology_ambiguity": any(flag in {"translation_of_target", "context_ontology", "english_vehicle_buy_policy", "translated_city_names", "find_or_book_ambiguous"} for flag in flags),
            "normalization_ambiguity": any(flag in {"translation_of_target", "approximate_time", "translated_city_names", "message_paraphrase", "stops_operator"} for flag in flags),
            "proposed_reviewed_gold": None,
            "proposed_change_explanation": None,
            "exact_match_impact": "pending_human_review",
            "planner_semantics_impact": "pending_human_review",
            "human_review_status": "pending",
            "human_reviewer": None,
        })
    if len(review) != 19:
        raise RuntimeError(f"Expected 19 flagged cases, found {len(review)}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in review), encoding="utf-8")
    print(f"Prepared {len(review)} review records: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
