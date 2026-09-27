"""Opt-in, conservative policy checks for explicit delayed consent and no booking.

This gate only holds questionable contracts. It never rewrites a field, selects a
provider, or permits execution. It is not wired into the frontend.
"""
from __future__ import annotations

import re

from .semantic_guard_v1_3 import ACCEPT, HOLD_POLICY_MISMATCH, GuardDecision
from .semantic_guard_v1_3_routing_v4 import _plain, check_contract as previous_check

_DELAYED_CONSENT = re.compile(
    r"\b(?:solo )?dopo che (?:ti )?(?:avro|avremo) (?:detto di procedere|dato (?:il )?(?:via libera|permesso|ok))\b"
    r"|\b(?:solo )?dopo (?:il )?mio (?:via libera|permesso|consenso)\b")
_NO_BOOK = re.compile(
    r"\b(?:senza (?:prenotare|fare (?:la |una )?prenotazione|prenotarne|riservarne)"
    r"|non (?:prenotare|prenotarne|riservarne)(?: (?:ancora|adesso|per ora))?"
    r"|nessuna prenotazione(?: (?:per ora|adesso))?)\b")


def check_contract(raw: str | dict, user_text: str) -> GuardDecision:
    previous = previous_check(raw, user_text)
    if previous.contract is None or previous.contract.get("schema_version") != "1.3":
        return previous
    text = _plain(user_text)
    goals = previous.contract["goals"]
    codes = set(previous.codes)
    consequential = [g for g in goals if g["action"] in
                     {"book", "buy", "contact", "rent", "order", "hire", "cancel", "reschedule"}]
    if len(consequential) == 1 and _DELAYED_CONSENT.search(text):
        goal = consequential[0]
        if goal["action"] not in goal.get("policy", {}).get("confirm_before", []):
            codes.add("explicit_delayed_consent_missing_v5")
    if _NO_BOOK.search(text):
        for goal in goals:
            if goal["target"]["type"] not in {"accommodation", "place", "transport", "appointment"}:
                continue
            if goal["action"] == "book" or (goal["action"] in {"find", "compare"} and
                                          "book" not in goal.get("policy", {}).get("forbid", [])):
                codes.add("explicit_no_book_policy_missing_v5")
    if not codes or previous.status != ACCEPT:
        return GuardDecision(previous.status, tuple(sorted(codes)), previous.contract)
    return GuardDecision(HOLD_POLICY_MISMATCH, tuple(sorted(codes)), previous.contract)
