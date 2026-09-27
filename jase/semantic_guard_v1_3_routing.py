"""Experimental conservative routing gate on top of the frozen V1.3 Guard.

The outcome permits Planner inspection only. No automatic field rewrite or
provider call is performed. Historical benchmarks continue to use the old gate.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .field_ontology_v1_3 import field_issue
from .semantic_guard_v1_3 import (ACCEPT, HOLD_MISSING_REQUIRED_INFORMATION,
                                    HOLD_POLICY_MISMATCH, HOLD_SEMANTIC_CONFLICT,
                                    HOLD_UNSUPPORTED_ACTION, GuardDecision,
                                    check_contract as base_check)

_NUMBER = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold())
                    .encode("ascii", "ignore").decode("ascii").split())


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text) is not None


def _issues(contract: dict[str, Any], user_text: str) -> set[str]:
    if contract.get("schema_version") != "1.3":
        return set()
    text = _plain(user_text)
    goals = contract["goals"]
    codes: set[str] = set()
    for item in contract.get("context", []):
        issue = field_issue(item["field"], scope="context")
        if issue:
            codes.add(issue)

    # The quoted imperative belongs to a story/example, not the user request.
    meta_quote = (_has(text, r"\b(?:per un racconto|sto scrivendo una scena)\b") and
                  _has(text, r"\b(?:immagina qualcuno che dica|un personaggio dice|non si tratta di una richiesta per te)\b"))
    if meta_quote and (contract["kind"] != "non_actionable" or goals):
        codes.add("quoted_command_misread_as_action")

    for i, goal in enumerate(goals):
        action = goal["action"]
        target = goal["target"]
        facts = goal["facts"]
        policy = goal.get("policy", {})
        for fact in facts:
            issue = field_issue(fact["field"])
            if issue:
                codes.add(issue)
        condition = goal.get("condition")
        if condition:
            issue = field_issue(condition["test"]["field"], scope="result")
            if issue:
                codes.add(issue)

        if target["type"] == "transport" and _has(target["name"].casefold(), r"\b(?:volo|aereo|flight)\b"):
            if _has(text, r"\b(?:volo diretto|senza scali|senza scalo)\b"):
                if not any(f["field"] == "stops" and f.get("op") == "eq" and
                           f.get("value") == 0 and f.get("strength") == "hard" for f in facts):
                    codes.add("direct_flight_constraint_missing")

        if _has(text, r"\b(?:non prenotare|non prenotarne|senza prenotare|non fare (?:nessuna |alcuna )?prenotazione)\b"):
            if action == "book" or (target["type"] == "accommodation" and action == "find" and
                                     "book" not in policy.get("forbid", [])):
                codes.add("explicit_no_booking_policy_missing")
        if _has(text, r"\b(?:non acquistare|senza acquistare)\b"):
            if action == "buy" or (target["type"] == "transport" and action == "find" and
                                    "buy" not in policy.get("forbid", [])):
                codes.add("explicit_no_purchase_policy_missing")

        if action == "compare":
            match = re.search(r"\bconfronta(?:ne|mi)?\s+(?:esattamente\s+)?(due|tre|quattro|cinque|sei|[2-6])\b", text)
            if match:
                count = _NUMBER.get(match.group(1), None)
                count = count if count is not None else int(match.group(1))
                if goal.get("modifiers", {}).get("compare_count") != count:
                    codes.add("compare_cardinality_missing")
        if action == "find":
            match = re.search(r"\b(?:mostra|elenca)(?:mi)?\s+(?:soltanto|solo|al massimo)\s+(due|tre|quattro|cinque|sei|[2-6])\b", text)
            if match:
                count = _NUMBER.get(match.group(1), None)
                count = count if count is not None else int(match.group(1))
                if goal.get("modifiers", {}).get("limit") != count:
                    codes.add("result_limit_missing")

    # A request to search and then book entails two separate goals, a
    # dependency, and an explicit reference to the selected search result.
    if (_has(text, r"\b(?:cerca|trova|individua|esplora)\b") and
            _has(text, r"\b(?:poi|dopo|quindi)\b") and
            _has(text, r"\bprenota\b")):
        finds = {g["id"] for g in goals if g["action"] == "find"}
        bookings = [g for g in goals if g["action"] == "book"]
        if not finds or not bookings:
            codes.add("requested_booking_goal_missing")
        elif not any(set(b.get("depends_on", [])) & finds and any(
                f.get("value_ref", {}).get("goal") in finds for f in b["facts"])
                     for b in bookings):
            codes.add("booking_result_dependency_missing")
    return codes


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardDecision:
    base = base_check(raw, user_text)
    if base.contract is None:
        return base
    codes = set(base.codes) | _issues(base.contract, user_text)
    if not codes:
        return base
    if base.status != ACCEPT:
        return GuardDecision(base.status, tuple(sorted(codes)), base.contract)
    if "quoted_command_misread_as_action" in codes:
        status = HOLD_SEMANTIC_CONFLICT
    elif any("policy" in c for c in codes):
        status = HOLD_POLICY_MISMATCH
    elif any("field" in c or "alias" in c or "dependency" in c for c in codes):
        status = HOLD_UNSUPPORTED_ACTION
    else:
        status = HOLD_MISSING_REQUIRED_INFORMATION
    return GuardDecision(status, tuple(sorted(codes)), base.contract)
