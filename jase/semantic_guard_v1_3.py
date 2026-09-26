"""Opt-in semantic handoff gate for V1.2 and V1.3 contracts.

ACCEPT only permits Planner inspection, never provider execution. Existing
frontend and V1.1/V1.2 Guards remain untouched.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .guard_v1_2_routing import check_contract as routing_v12
from .semantic_contract_v1_3 import CONSEQUENTIAL, SemanticIssue, validate_contract

ACCEPT = "ACCEPT"
HOLD_NEEDS_CLARIFICATION = "HOLD_NEEDS_CLARIFICATION"
HOLD_SEMANTIC_CONFLICT = "HOLD_SEMANTIC_CONFLICT"
HOLD_UNSUPPORTED_ACTION = "HOLD_UNSUPPORTED_ACTION"
HOLD_MISSING_REQUIRED_INFORMATION = "HOLD_MISSING_REQUIRED_INFORMATION"
HOLD_POLICY_MISMATCH = "HOLD_POLICY_MISMATCH"
REJECT_INVALID = "REJECT_INVALID"


@dataclass(frozen=True)
class GuardDecision:
    status: str
    codes: tuple[str, ...]
    contract: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "codes": list(self.codes), "contract": self.contract}


def _grounding_issues(contract: dict[str, Any], text: str) -> list[SemanticIssue]:
    user = text.casefold()
    issues: list[SemanticIssue] = []
    goals = contract["goals"]
    if re.search(r"\b(?:ordina|ordinami)\b.*\b(?:per prezzo|per costo|dal pi[uù] economico|prezzo crescente)\b", user):
        if any(g["action"] == "order" for g in goals):
            issues.append(SemanticIssue("sort_misread_as_order", "$.goals", "sorting request"))
    if re.search(r"\b(?:noleggia|affitta|mi serve a noleggio)\b", user) and not re.search(r"\b(?:cerca|trova)\b", user):
        if any(g["action"] == "find" and g["target"].get("type") == "vehicle" for g in goals):
            issues.append(SemanticIssue("rental_misread_as_find", "$.goals", "direct rental wording"))
    if re.search(r"\b(?:chiedimi conferma|dopo la mia conferma|solo dopo conferma)\b", user):
        consequential = [(i, goal) for i, goal in enumerate(goals) if goal["action"] in CONSEQUENTIAL]
        # Multi-goal natural-language scope is not determined by this regex.
        if len(consequential) == 1:
            i, goal = consequential[0]
            if goal["action"] not in goal.get("policy", {}).get("confirm_before", []):
                issues.append(SemanticIssue("confirmation_not_bound_to_action", f"$.goals[{i}].policy", goal["action"]))
    context_fields = {c["field"] for c in contract.get("context", [])}
    if re.search(r"\b(?:siamo in|siamo un gruppo di)\s+(?:due|tre|quattro|cinque|sei|\d+)\b", user):
        if "party_size" not in context_fields:
            issues.append(SemanticIssue("missing_user_context", "$.context", "party_size"))
    if re.search(r"\b(?:senza (?:la )?(?:macchina|auto)|non abbiamo (?:la )?(?:macchina|auto))\b", user):
        if "has_car" not in context_fields:
            issues.append(SemanticIssue("missing_user_context", "$.context", "has_car"))
    if re.search(r"\b(?:abbiamo gi[aà] (?:l[' ]hotel|un hotel)|hotel gi[aà] prenotato)\b", user):
        if "lodging_owned" not in context_fields:
            issues.append(SemanticIssue("missing_user_context", "$.context", "lodging_owned"))
    return issues


def _status(issues: list[SemanticIssue]) -> str:
    codes = {x.code for x in issues}
    if any(x.startswith("schema") or x in {"unsupported_version", "not_object"} for x in codes):
        return REJECT_INVALID
    if "conflicting_hard_equality" in codes or "policy_conflict" in codes:
        return HOLD_SEMANTIC_CONFLICT
    if any("policy" in x or "confirmation" in x for x in codes):
        return HOLD_POLICY_MISMATCH
    if any("missing_required" in x or x == "cardinality_lost" for x in codes):
        return HOLD_MISSING_REQUIRED_INFORMATION
    if any(x in {"unsupported_action_target", "order_sort_confusion", "sort_misread_as_order",
                 "rental_misread_as_find", "vehicle_target_mismatch", "notify_target_mismatch"}
           for x in codes):
        return HOLD_UNSUPPORTED_ACTION
    return HOLD_NEEDS_CLARIFICATION


def check_contract(raw: str | dict[str, Any], user_text: str) -> GuardDecision:
    try:
        contract = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return GuardDecision(REJECT_INVALID, ("invalid_json",), None)
    issues = validate_contract(contract, user_text=user_text)
    if isinstance(contract, dict) and not any(x.code.startswith("schema") for x in issues):
        issues.extend(_grounding_issues(contract, user_text))
    if issues:
        status = _status(issues)
        return GuardDecision(status, tuple(sorted({i.code for i in issues})),
                             contract if status != REJECT_INVALID else None)
    if contract["schema_version"] == "1.2":
        old = routing_v12(contract, user_text)
        if old.status != ACCEPT:
            return GuardDecision(HOLD_NEEDS_CLARIFICATION if old.status == "REVIEW" else REJECT_INVALID,
                                 old.codes, old.contract)
    if any(goal["action"] == "reschedule" for goal in contract["goals"]):
        return GuardDecision(HOLD_UNSUPPORTED_ACTION, ("reschedule_provider_not_verified",), contract)
    return GuardDecision(ACCEPT, (), contract)
