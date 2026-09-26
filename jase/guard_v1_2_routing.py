"""Experimental routing checks after the unchanged GoalContract 1.2 Guard.

The check is conservative: it holds contracts with known capability-routing
ambiguities. It never rewrites a target, invents facts, or authorizes execution.
"""
from __future__ import annotations

import re
from typing import Any

from .canon import strip_accents
from .guard_v1_2 import ACCEPT, REVIEW, GuardResult, check_contract as base_check

# Explicit, versioned routing vocabulary. Subtypes only collapse for *routing*;
# the original target.type stays in the contract and exact metrics are untouched.
TARGET_TYPE_FAMILY = {
    "appointment": "appointment", "medical_appointment": "appointment",
    "vehicle": "vehicle", "product": "product", "electronics": "product",
    "flower": "product", "professional_service": "professional_service",
    "place": "place", "accommodation": "accommodation",
    "transport": "transport", "reminder": "reminder", "person": "person",
}

PROFESSIONAL_NAMES = frozenset({
    "farmacista", "dentista", "medico", "idraulico", "elettricista",
    "fotografo", "giardiniere", "avvocato", "commercialista",
})
VEHICLE_NAMES = frozenset({
    "bicicletta", "bici", "scooter", "automobile", "auto", "furgone",
    "moto", "motocicletta", "camper",
})
TRANSPORT_NAMES = frozenset({
    "treno", "pullman", "autobus", "aereo", "volo", "traghetto", "taxi",
})
COUNT_WORDS = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", strip_accents(text.casefold())))


def _flat(text: str) -> str:
    return " ".join(strip_accents(text.casefold()).split())


def _routing_codes(contract: dict[str, Any], user_text: str) -> set[str]:
    codes: set[str] = set()
    text = _flat(user_text)
    for goal in contract["goals"]:
        action = goal["action"]
        name = _flat(goal["target"]["name"])
        words = _words(name)
        family = TARGET_TYPE_FAMILY.get(goal["target"].get("type", ""))
        fields = {fact["field"] for fact in goal["facts"]}
        missing = {item["field"] for item in goal.get("missing", [])}

        if action == "notify":
            if family != "reminder":
                codes.add("notify_target_not_reminder")
            if "message" not in fields and not ({"message", "referent"} & missing):
                codes.add("notify_message_unresolved")
        if action == "cancel" and {"visita", "appuntamento"} & words and family != "appointment":
            codes.add("appointment_routing_mismatch")
        if action == "rent" and words & VEHICLE_NAMES and family != "vehicle":
            codes.add("vehicle_routing_mismatch")
        if action in {"find", "hire"} and words & PROFESSIONAL_NAMES and family != "professional_service":
            codes.add("professional_routing_mismatch")
        if action == "compare":
            requested = re.search(r"\bconfronta(?:mi)?\s+(due|tre|quattro|cinque|sei|[2-6])\b", text)
            if requested:
                count = COUNT_WORDS.get(requested.group(1), None)
                count = count if count is not None else int(requested.group(1))
                if not any(fact["field"] == "quantity" and fact["op"] == "eq" and
                           fact.get("value") == count for fact in goal["facts"]):
                    codes.add("comparison_quantity_missing")
        if action == "find" and words & TRANSPORT_NAMES:
            if any(fact["field"] == "time" and fact["op"] in {"before", "after"}
                   for fact in goal["facts"]):
                codes.add("transport_time_role_ambiguous")
            if re.search(r"\b(arriv\w*|entro le)\b", text) and re.search(r"\b(?:alle|le)\s+\d{1,2}(?::\d{2})?\b", text):
                if "arrival_time" not in fields:
                    codes.add("arrival_time_field_missing")
            if re.search(r"\b(part\w*)\b", text) and re.search(r"\b(?:alle|le)\s+\d{1,2}(?::\d{2})?\b", text):
                if "departure_time" not in fields:
                    codes.add("departure_time_field_missing")
    return codes


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardResult:
    """Run base Guard, then hold known routing ambiguities for Planner review."""
    base = base_check(raw, user_text)
    if base.status != ACCEPT or base.contract is None:
        return base
    codes = _routing_codes(base.contract, user_text)
    if codes:
        return GuardResult(REVIEW, tuple(sorted(codes)), base.contract)
    return base
