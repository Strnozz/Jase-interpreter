"""Conservative GoalContract 1.2 handoff gate.

ACCEPT permits the Planner to inspect a contract; it never authorizes a side
effect. REVIEW holds a plausible contract for clarification/review. REJECT
holds malformed or structurally contradictory output. No candidate is repaired
or silently normalized here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .canon import strip_accents
from .goal_contract_v1_2 import validate_contract

ACCEPT, REVIEW, REJECT = "ACCEPT", "REVIEW", "REJECT"


@dataclass(frozen=True)
class GuardResult:
    status: str
    codes: tuple[str, ...]
    contract: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "codes": list(self.codes), "contract": self.contract}


def _flat(text: str) -> str:
    return " ".join(strip_accents(text.casefold()).split())


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardResult:
    """Gate a V1.2 candidate without inventing user facts or execution policy."""
    try:
        contract = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        return GuardResult(REJECT, ("invalid_json",), None)
    if not isinstance(contract, dict):
        return GuardResult(REJECT, ("not_object",), None)
    issues = validate_contract(contract)
    if issues:
        return GuardResult(REJECT, tuple(sorted({"schema:" + x.code for x in issues})), None)

    text = _flat(user_text)
    codes: set[str] = set()
    goals = contract["goals"]
    all_facts = [fact for goal in goals for fact in goal["facts"]]
    all_context = contract.get("context", [])

    for goal in goals:
        action = goal["action"]
        target = _flat(goal["target"]["name"])
        fields = {fact["field"] for fact in goal["facts"]}
        missing = {item["field"] for item in goal.get("missing", [])}
        if missing:
            codes.add("declared_missing_input")
        policy = goal.get("policy", {})
        if policy.get("confirm_before"):
            codes.add("explicit_confirmation_pending")
        if action == "book" and any(word in target for word in
                                    ("tavolo", "ristorante", "pizzeria", "trattoria", "osteria")):
            if not {"date", "time", "guests"}.issubset(fields):
                codes.add("table_booking_requires_details")
        if action == "book" and any(word in target for word in ("volo", "aereo", "flight")):
            if not {"origin", "date"}.issubset(fields):
                codes.add("flight_booking_requires_details")
        if action == "contact" and "message" not in fields:
            codes.add("contact_requires_message")
        for fact in goal["facts"]:
            if fact["field"] in ("price", "budget", "cost") and isinstance(fact.get("value"), (int, float)):
                if fact["value"] < 0:
                    codes.add("negative_money_value")
                elif fact["value"] == 0 and fact["op"] in ("lt", "lte"):
                    codes.add("nonpositive_price_ceiling")
        if "condition" in goal and not _has(text, r"\b(se|qualora|nel caso|a condizione|soltanto se)\b"):
            codes.add("ungrounded_condition")

    if contract["kind"] == "task":
        if _has(text, r"\b(se non (?:ne )?(?:trovi|trova|ci sono)|in assenza di risultati)\b") and len(goals) < 2:
            codes.add("possible_missing_fallback_goal")
        if _has(text, r"\bpoi\s+(?:prenota|contatta|compra|acquista|cerca|trova|ordina|noleggia|assumi|invia|avvisa)\b") and len(goals) < 2:
            codes.add("possible_missing_followup_goal")
        if _has(text, r"\b(meno caro|meno costoso|piu economico|prezzo crescente|costo crescente|spendendo il meno possibile)\b"):
            if not any(goal.get("ranking", {}).get("field") == "price" and
                       goal["ranking"]["direction"] == "asc" for goal in goals):
                codes.add("missing_price_ranking")
        if _has(text, r"\b(non prenotare|senza prenotare|non effettuare prenotazioni)\b") and not any(
                "book" in goal.get("policy", {}).get("forbid", []) for goal in goals):
            codes.add("missing_explicit_booking_prohibition")
        if _has(text, r"\b(non comprare|senza comprare|niente acquisti|non acquistare)\b") and not any(
                "buy" in goal.get("policy", {}).get("forbid", []) for goal in goals):
            codes.add("missing_explicit_purchase_prohibition")
        if _has(text, r"\b(solo se conferm\w*|dopo (?:la |una |mia )?conferma|solo dopo (?:la |mia )?conferma|soltanto dopo che (?:te lo )?conferm\w*)\b") and not any(
                goal.get("policy", {}).get("confirm_before") for goal in goals):
            codes.add("missing_explicit_confirmation")
        if _has(text, r"\b(ricordamelo|quella faccenda|quella cosa|cio di cui parlavamo)\b") and any(
                goal["action"] == "notify" for goal in goals):
            if not any(fact["field"] == "message" for fact in all_facts):
                codes.add("unresolved_reminder_reference")
        if _has(text, r"\b(siamo in|senza (?:la )?(?:macchina|auto|automobile))\b") and not all_context:
            codes.add("possible_missing_user_context")

    if "negative_money_value" in codes or "nonpositive_price_ceiling" in codes:
        return GuardResult(REJECT, tuple(sorted(codes)), None)
    if codes:
        return GuardResult(REVIEW, tuple(sorted(codes)), contract)
    return GuardResult(ACCEPT, (), contract)
