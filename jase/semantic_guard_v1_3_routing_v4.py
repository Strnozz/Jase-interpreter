"""Opt-in routing safety checks after V3; no normalization or provider calls."""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .semantic_guard_v1_3 import (ACCEPT, HOLD_MISSING_REQUIRED_INFORMATION,
                                    HOLD_POLICY_MISMATCH, GuardDecision)
from .semantic_guard_v1_3_routing_v3 import check_contract as previous_check


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold())
                    .encode("ascii", "ignore").decode("ascii").split())


_APPROVAL = re.compile(
    r"\b(?:dopo (?:il )?mio si|prima di (?:procedere|impegnarmi) .*?(?:mostrami|chiedimi)"
    r"|prima di [^.;]{1,80}?(?:mostrami|fammi vedere|chiedimi)"
    r"|(?:mostrami|fammi vedere) prima (?:quale|quali|la|il)"
    r"|(?:attendi|aspetta) (?:la |il )?mia approvazione"
    r"|(?:solo )?quando (?:avrai|ti avro) (?:il |detto di )?(?:mio )?(?:assenso|si|procedere)"
    r"|(?:chiedimi|richiedi) (?:un )?si esplicito)\b")
_NO_RENT = re.compile(
    r"\b(?:senza (?:prenderne|prenderla|prenderlo|affittarne|noleggiarne)"
    r"|non (?:prenderne|prenderla|prenderlo|affittarne|noleggiarne))\b")
_NO_BUY = re.compile(
    r"\b(?:non (?:comprarla|comprarlo|comprarne|acquistarla|acquistarlo|acquistarne)"
    r"|non e un ordine d acquisto|l acquisto lo faro io)\b")
_NO_HIRE = re.compile(r"\b(?:senza assumerlo|non assumerlo|senza ingaggiarlo|non ingaggiarlo)\b")
_ORDINAL = {"primo": 1, "prima": 1, "secondo": 2, "seconda": 2,
            "terzo": 3, "terza": 3, "quarto": 4, "quarta": 4}
_ACTION_SELECTION = re.compile(
    r"\b(?:prenota|riserva|compra|acquista|contatta|chiedi)\b.{0,50}?\b"
    r"(primo|prima|secondo|seconda|terzo|terza|quarto|quarta)\b")
_COUNT = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}
_COMPARE_COUNT = re.compile(
    r"\b(?:metti a confronto|confrontami|confronta|paragona)\s+"
    r"(?:al massimo |soltanto |solo )?(due|tre|quattro|cinque|sei|[2-6])\b")


def _issues(contract: dict[str, Any], user_text: str) -> set[str]:
    if contract.get("schema_version") != "1.3":
        return set()
    text = _plain(user_text)
    goals = contract["goals"]
    codes: set[str] = set()
    consequential = [g for g in goals if g["action"] in
                     {"book", "buy", "contact", "rent", "order", "hire", "cancel", "reschedule"}]
    if len(consequential) == 1 and _APPROVAL.search(text):
        goal = consequential[0]
        if goal["action"] not in goal.get("policy", {}).get("confirm_before", []):
            codes.add("explicit_approval_missing_v4")

    no_rent, no_buy, no_hire = bool(_NO_RENT.search(text)), bool(_NO_BUY.search(text)), bool(_NO_HIRE.search(text))
    count_match = _COMPARE_COUNT.search(text)
    selection_match = _ACTION_SELECTION.search(text)
    for goal in goals:
        action = goal["action"]
        typ = goal["target"]["type"]
        forbid = set(goal.get("policy", {}).get("forbid", []))
        if no_rent and typ == "vehicle" and (action == "rent" or
                (action in {"find", "compare"} and "rent" not in forbid)):
            codes.add("explicit_no_rent_policy_missing_v4")
        if no_buy and typ in {"product", "food"} and (action in {"buy", "order"} or
                (action in {"find", "compare"} and "buy" not in forbid)):
            codes.add("explicit_no_purchase_policy_missing_v4")
        if no_hire and typ == "professional_service" and (action == "hire" or
                (action == "find" and "hire" not in forbid)):
            codes.add("explicit_no_hire_policy_missing_v4")
        if count_match and action == "compare":
            word = count_match.group(1)
            count = _COUNT.get(word, int(word) if word.isdigit() else -1)
            if goal.get("modifiers", {}).get("compare_count") != count:
                codes.add("compare_count_missing_v4")
        if action == "compare" and re.search(r"\b(?:dal meno caro al piu caro|dal piu economico al piu caro)\b", text):
            if goal.get("modifiers", {}).get("sort") != {"field": "price", "direction": "asc"}:
                codes.add("explicit_price_sort_missing_v4")

        if action == "find" and typ == "transport" and \
                _plain(goal["target"]["name"]) in {"trasporto", "mezzo", "collegamento"} and \
                re.search(r"\b(?:treni?|ferroviari[oa]?|regionale)\b", text):
            codes.add("rail_mode_not_bound_to_target_v4")

    # A requested ordinal selection from a prior search must retain an actual
    # result reference; a literal 'second' cannot bind a provider result.
    if selection_match:
        ordinal = _ORDINAL[selection_match.group(1)]
        find_ids = {g["id"] for g in goals if g["action"] in {"find", "compare"}}
        later = [g for g in goals if g["action"] in {"book", "buy", "contact"}]
        if find_ids and later and not any(
                set(g.get("depends_on", [])) & find_ids and
                any(f.get("value_ref", {}).get("goal") in find_ids and
                    f["value_ref"].get("ordinal") == ordinal for f in g["facts"])
                for g in later):
            codes.add("ordinal_result_reference_missing_v4")
    return codes


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardDecision:
    previous = previous_check(raw, user_text)
    if previous.contract is None:
        return previous
    codes = set(previous.codes) | _issues(previous.contract, user_text)
    if not codes or previous.status != ACCEPT:
        return GuardDecision(previous.status, tuple(sorted(codes)), previous.contract)
    status = (HOLD_POLICY_MISMATCH if any("policy" in x or "approval" in x for x in codes)
              else HOLD_MISSING_REQUIRED_INFORMATION)
    return GuardDecision(status, tuple(sorted(codes)), previous.contract)
