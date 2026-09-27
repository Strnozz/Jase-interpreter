"""Second experimental routing gate; the V1 Guard and V27 audit stay frozen."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .semantic_guard_v1_3 import (ACCEPT, HOLD_MISSING_REQUIRED_INFORMATION,
                                    HOLD_POLICY_MISMATCH, HOLD_SEMANTIC_CONFLICT,
                                    HOLD_UNSUPPORTED_ACTION, GuardDecision)
from .semantic_guard_v1_3_routing import check_contract as previous_check

_COUNT = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold())
                    .encode("ascii", "ignore").decode("ascii").split())


def _number(word: str) -> int:
    return _COUNT[word] if word in _COUNT else int(word)


def _facts(goal: dict[str, Any], field: str, value: Any = None) -> bool:
    return any(f["field"] == field and f.get("strength") == "hard" and
               (value is None or f.get("value") == value) for f in goal["facts"])


def _issues(contract: dict[str, Any], user_text: str) -> set[str]:
    if contract.get("schema_version") != "1.3":
        return set()
    text = _plain(user_text)
    goals = contract["goals"]
    codes: set[str] = set()
    meta = re.search(r"\b(?:copione|sceneggiatura|lezione di italiano|esempio di comando|nota che ho copiato)\b", text)
    meta_intent = re.search(r"\b(?:mi dici|spiega|non eseguire|voglio solo|voglio soltanto)\b", text)
    if meta and meta_intent and (contract["kind"] != "non_actionable" or goals):
        codes.add("quoted_example_misread_as_action")

    no_book = re.search(r"\b(?:non riservare|senza effettuare prenotazioni|niente prenotazioni|non bloccare|lascia a me la prenotazione)\b", text)
    no_buy = re.search(r"\bnon comprare\b", text)
    arrival = re.search(r"\b(?:arriv\w*|arrivo|essere a|raggiungere)\b.{0,100}\b(?:entro|prima delle|prima di|non oltre)\b", text)
    # Also captures 'entro le 14' after a stated destination/arrival purpose.
    arrival = arrival or re.search(r"\b(?:entro|prima delle|prima di|non oltre)\b.{0,45}\barriv\w*\b", text)
    walking = re.search(r"\b(?:raggiungibil\w*|arrivare|arrivarci)\s+a piedi\b", text)
    opened_now = re.search(r"\bapert[aoie]\s+(?:ora|adesso)\b", text)
    from_route = re.search(r"\b(?:parto da|partire da|treni da|treno da|autobus da)\b", text)
    from_route = from_route or (text.startswith("da ") and re.search(r"\b(?:treno|treni|autobus|volo)\b", text))

    for goal in goals:
        action = goal["action"]
        typ = goal["target"]["type"]
        policy = goal.get("policy", {})
        temporal = goal.get("temporal", [])
        if no_book and (action == "book" or
                        (action in {"find", "compare"} and typ == "accommodation" and
                         "book" not in policy.get("forbid", []))):
            codes.add("explicit_no_booking_policy_missing_v2")
        if no_buy and (action == "buy" or
                       (action == "find" and typ == "transport" and
                        "buy" not in policy.get("forbid", []))):
            codes.add("explicit_no_purchase_policy_missing_v2")
        if action == "find" and typ == "transport":
            if arrival and not any(t["role"] == "arrival_time" for t in temporal):
                codes.add("arrival_deadline_role_missing")
            if from_route and not _facts(goal, "origin"):
                codes.add("explicit_origin_missing")
        if action == "find" and typ in {"place", "accommodation"}:
            if walking and not _facts(goal, "access", "a piedi"):
                codes.add("walking_access_missing")
            if opened_now and not _facts(goal, "open_now", True):
                codes.add("open_now_missing")
        if action == "find":
            count = re.search(r"\b(?:cerca|trova|mostrami|mostra|fammi vedere)\s+(?:al massimo\s+|soltanto\s+|solo\s+)?(due|tre|quattro|cinque|sei|[2-6])\b", text)
            if count and goal.get("modifiers", {}).get("limit") != _number(count.group(1)):
                codes.add("result_limit_missing_v2")
        if action == "book" and typ == "place" and re.search(r"\bristorant\w*\b", _plain(goal["target"]["name"])):
            present = {f["field"] for f in goal["facts"]} | {t["role"] for t in temporal}
            absent = {m["field"] for m in goal.get("missing", [])}
            if any(x not in present and x.removeprefix("action_") not in absent
                   for x in ("action_date", "action_time", "guests")):
                codes.add("restaurant_booking_slots_missing")

    if re.search(r"\b(?:solo se non ce ne sono|se la ricerca restituisce zero risultati|solo in assenza di risultati|se non trovi corse)\b", text):
        if len(goals) < 2 or not any(g.get("condition", {}).get("test", {}).get("field") == "results_count" and
                                      g["condition"]["test"].get("op") == "eq" and
                                      g["condition"]["test"].get("value") == 0 and
                                      g["condition"]["goal"] in g.get("depends_on", [])
                                      for g in goals[1:]):
            codes.add("zero_results_fallback_missing")
    if (re.search(r"\b(?:trova|cerca|individua|mostrami|mostra)\b", text) and
            re.search(r"\bprenota\b", text) and
            re.search(r"\b(?:poi|quindi|solo dopo)\b", text)):
        find_ids = {g["id"] for g in goals if g["action"] == "find"}
        books = [g for g in goals if g["action"] == "book"]
        if not find_ids or not books:
            codes.add("requested_booking_goal_missing_v2")
        elif not any(set(b.get("depends_on", [])) & find_ids and
                     any(f.get("value_ref", {}).get("goal") in find_ids for f in b["facts"])
                     for b in books):
            codes.add("booking_result_dependency_missing_v2")
    return codes


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardDecision:
    previous = previous_check(raw, user_text)
    if previous.contract is None:
        return previous
    codes = set(previous.codes) | _issues(previous.contract, user_text)
    if not codes:
        return previous
    if previous.status != ACCEPT:
        return GuardDecision(previous.status, tuple(sorted(codes)), previous.contract)
    if any(c.startswith("quoted_") for c in codes):
        status = HOLD_SEMANTIC_CONFLICT
    elif any("policy" in c for c in codes):
        status = HOLD_POLICY_MISMATCH
    elif any("dependency" in c or "fallback" in c or "booking_goal" in c for c in codes):
        status = HOLD_UNSUPPORTED_ACTION
    else:
        status = HOLD_MISSING_REQUIRED_INFORMATION
    return GuardDecision(status, tuple(sorted(codes)), previous.contract)
