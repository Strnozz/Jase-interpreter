"""Experimental conservative safety checks after the opt-in V2 routing gate.

ACCEPT is only permission for Planner inspection. The checks use explicit user
language; they neither rewrite a contract nor authorize provider execution.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any

from .semantic_guard_v1_3 import (ACCEPT, HOLD_MISSING_REQUIRED_INFORMATION,
                                    HOLD_POLICY_MISMATCH, GuardDecision)
from .semantic_guard_v1_3_routing_v2 import check_contract as previous_check


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold())
                    .encode("ascii", "ignore").decode("ascii").split())


_CONFIRM = re.compile(
    r"\b(?:chiedimi (?:prima )?(?:la )?(?:conferma|autorizzazione|ok|via libera)"
    r"|(?:dopo|con|previa) (?:il |la |mia |mio )*(?:mia |mio )?(?:conferma|autorizzazione|consenso|ok)"
    r"|(?:aspetta|attendi) (?:il |la )?(?:mio |mia )?(?:consenso|via libera|autorizzazione|conferma)"
    r"|(?:mostrami|fammi vedere).*?prima di (?:spedir\w*|inviar\w*|confermar\w*|pagare)"
    r"|(?:solo )?dopo che (?:l )?avr[oai] (?:approvato|confermato)"
    r"|solo quando (?:ti )?(?:dar[o ]+|avrai) (?:il )?(?:permesso|consenso|ok)"
    r"|non (?:spedire|inviare|prenotare|pagare|modificare).*?finche non confermo)\b")
_NO_PURCHASE = re.compile(
    r"\b(?:senza (?:fare |effettuare )?(?:acquisti|comprare|acquistare)"
    r"|senza comprare niente|non (?:comprare|acquistare|fare acquisti)"
    r"|niente acquisti)\b")
_NO_HIRE = re.compile(r"\b(?:non (?:assumere|ingaggiare)|senza (?:assumere|ingaggiare))\b")
_NO_RENT = re.compile(r"\b(?:non (?:affittare|noleggiare)|senza (?:affittare|noleggiare))\b")
_GENERIC_PROVIDER = re.compile(
    r"\b(?:idraulico|elettricista|fabbro|tecnico|muratore|imbianchino|giardiniere|riparatore)\b")
_REMINDER_TIME = re.compile(
    r"\b(?:alle\s+|(?:domani|dopodomani|oggi|lunedi|martedi|mercoledi|giovedi|venerdi|sabato|domenica)\s+alle\s+)"
    r"(\d{1,2})(?::(\d{2}))?\b.{0,35}\b(?:ricordami|promemoria|avviso|imposta)\b")


def _issues(contract: dict[str, Any], user_text: str) -> set[str]:
    if contract.get("schema_version") != "1.3":
        return set()
    text = _plain(user_text)
    goals = contract["goals"]
    codes: set[str] = set()

    # Infer scope only when there is a single consequential action. Search-only
    # requests with a future hypothetical booking must not be converted to book.
    consequential = [g for g in goals if g["action"] in
                     {"book", "buy", "contact", "rent", "order", "hire", "cancel", "reschedule"}]
    if len(consequential) == 1 and _CONFIRM.search(text):
        g = consequential[0]
        if g["action"] not in g.get("policy", {}).get("confirm_before", []):
            codes.add("explicit_confirmation_missing_v3")

    no_purchase = bool(_NO_PURCHASE.search(text))
    no_hire = bool(_NO_HIRE.search(text))
    no_rent = bool(_NO_RENT.search(text))
    for goal in goals:
        action = goal["action"]
        typ = goal["target"]["type"]
        forbid = set(goal.get("policy", {}).get("forbid", []))
        if no_purchase and typ in {"food", "product", "transport"}:
            required = {"buy", "order"} if typ == "food" else {"buy"}
            if action in required or (action in {"find", "compare"} and not required <= forbid):
                codes.add("explicit_no_purchase_policy_missing_v3")
        if no_hire and typ == "professional_service" and (action == "hire" or
                (action == "find" and "hire" not in forbid)):
            codes.add("explicit_no_hire_policy_missing_v3")
        if no_rent and typ == "vehicle" and (action == "rent" or
                (action in {"find", "compare"} and "rent" not in forbid)):
            codes.add("explicit_no_rent_policy_missing_v3")

        if action == "hire" and typ == "professional_service" and \
                _GENERIC_PROVIDER.search(_plain(goal["target"]["name"])) and \
                not any(f["field"] == "selection" and "value_ref" in f for f in goal["facts"]) and \
                not any(m["field"] == "provider" for m in goal.get("missing", [])):
            codes.add("provider_identity_missing_v3")

        if action == "notify" and typ == "reminder":
            # The anchor is outside any quoted message and precedes the reminder
            # verb. Ignore the later hour inside message content.
            prefix = re.split(r"['\"«]", text, maxsplit=1)[0]
            hit = _REMINDER_TIME.search(prefix)
            if hit:
                hour = int(hit.group(1))
                minute = int(hit.group(2) or 0)
                expected = f"{hour:02d}:{minute:02d}"
                if not any(t["role"] == "notification_time" and t.get("value") == expected
                           for t in goal.get("temporal", [])):
                    codes.add("explicit_reminder_time_mismatch_v3")
    return codes


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardDecision:
    previous = previous_check(raw, user_text)
    if previous.contract is None:
        return previous
    codes = set(previous.codes) | _issues(previous.contract, user_text)
    if not codes or previous.status != ACCEPT:
        return GuardDecision(previous.status, tuple(sorted(codes)), previous.contract)
    status = (HOLD_POLICY_MISMATCH if any("policy" in c or "confirmation" in c for c in codes)
              else HOLD_MISSING_REQUIRED_INFORMATION)
    return GuardDecision(status, tuple(sorted(codes)), previous.contract)
