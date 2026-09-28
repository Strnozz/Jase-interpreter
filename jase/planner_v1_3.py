"""Fail-closed GoalContract 1.3 Planner: inspectable dry run, never execution.

Inputs may include trusted context and mock result sets. No provider, frontend,
or external side effect is reachable from this module.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .capability_registry_v1 import Capability, CapabilityRegistry, REGISTRY_V2_PATH
from .provenance_v1 import (DETERMINISTIC_DERIVATION, MODEL_INFERENCE,
                            TRUSTED_CONVERSATION_CONTEXT, USER_EXPLICIT,
                            ground_value)
from .semantic_guard_v1_3 import ACCEPT
from .semantic_guard_v1_3_routing_v5 import check_contract

VERSION = "1.3-dry-run-2"
_TRUSTED = {USER_EXPLICIT, TRUSTED_CONVERSATION_CONTEXT, DETERMINISTIC_DERIVATION,
            "PROVIDER_RESULT", "CAPABILITY_RESULT"}
_CONTEXT_FIELDS = {"has_car", "party_size", "user_location", "vehicle_location", "pet", "lodging_owned"}
_SORT_FIELDS = {"price", "distance", "duration", "rating"}
_ACTION_WORDS = re.compile(r"\b(?:prenota|prenotami|cerca|cercami|trova|trovami|manda|mandami|assumi|"
                           r"noleggia|affitta|avvisami|ricordami|compra|annulla|sposta|fissa|confronta)\b", re.I)
_META_WORDS = re.compile(r"\b(?:traduzione|traduci|significato|spiegami|grammatica|grammaticale|"
                         r"manuale|dialogo|battuta|citazione|ipotetic[oa]|esempio|hypothetical|translation)\b", re.I)
_HYPOTHETICAL = re.compile(r"\b(?:se qualcuno|supponiamo che|ipotizziamo che|in un esempio|if someone|imagine that)\b", re.I)
_EXPLANATION = re.compile(r"\b(?:cosa significa|che significa|spiegami|spiega|what would|what does|significato)\b", re.I)
_QUOTED = re.compile(r"['\"‘“][^'\"’”]+['\"’”]")
_FORBIDS = {
    "book": re.compile(r"\b(?:senza|non)\s+(?:prenotare|prenotarlo|prenotarla|riservare)\b|\bnessuna prenotazione\b|\b(?:decido|deciderò|scelgo|sceglierò)\s+dopo\b", re.I),
    "buy": re.compile(r"\b(?:senza|non)\s+(?:comprare|acquistare|ordinare|fare acquisti)\b|\bniente acquisti\b|\bnon acquistare nulla\b", re.I),
    "rent": re.compile(r"\b(?:senza|non)\s+(?:affittare|noleggiare|prenderne)\b|\bsolo ricerche\b", re.I),
    "hire": re.compile(r"\b(?:senza|non)\s+(?:assumere|ingaggiare|incaricare)\b|\bnessun incarico\b", re.I),
}
_FORBID_TYPES = {"book": {"accommodation", "place", "transport", "appointment"},
                 "buy": {"product", "food", "transport", "place"},
                 "rent": {"vehicle"}, "hire": {"professional_service"}}


def _shell(status: str, codes: list[str] | None = None, registry_version: str = "2") -> dict[str, Any]:
    return {"planner_version": VERSION, "registry_version": registry_version, "planner_status": status,
            "reason_codes": codes or [], "resolved_capabilities": [], "goal_bindings": [],
            "resolved_references": [], "missing_slots": [], "unresolved_references": [],
            "semantic_conflicts": [], "provenance_issues": [], "policy_state": [],
            "dependencies": [], "execution_plan_steps": [], "blocked_steps": [],
            "execution_permitted": False}


def _meta_only(text: str) -> bool:
    if _HYPOTHETICAL.search(text) and _EXPLANATION.search(text):
        return True
    quotes = list(_QUOTED.finditer(text))
    if not quotes or not _META_WORDS.search(text):
        return False
    outside = list(text)
    for match in quotes:
        outside[match.start():match.end()] = " " * (match.end() - match.start())
    return _ACTION_WORDS.search("".join(outside)) is None


def _reminder_content(text: str) -> str | None:
    """Extract a single explicit reminder clause; ambiguity stays with the Guard."""
    match = re.search(r"\b(?:ricordami|avvisami)\b[^,.;!?]{0,45}?\bdi\s+([^.;!?]+)", text, re.I)
    if not match or "," in match.group(1):
        return None
    return " ".join(match.group(1).strip().casefold().split())


def _reference(ref: dict[str, Any], results: Mapping[str, list[dict[str, Any]]],
               dependencies: list[str], spec: Capability) -> tuple[dict[str, Any] | None, str | None]:
    source, field = ref["goal"], ref["field"]
    if source not in dependencies:
        return None, "reference_without_dependency"
    if field == "result" and "result_ordinal" not in spec.supported_reference_types:
        return None, "unsupported_ordinal_reference"
    if field == "cheapest" and "cheapest_result" not in spec.supported_reference_types:
        return None, "unsupported_cheapest_reference"
    if field not in {"result", "cheapest"}:
        return None, "unknown_reference_type"
    items = results.get(source)
    if not items:
        return None, "result_set_unavailable"
    if field == "result":
        ordinal = ref.get("ordinal")
        if not isinstance(ordinal, int) or ordinal < 1 or ordinal > len(items):
            return None, "ordinal_out_of_range"
        chosen = items[ordinal - 1]
    else:
        priced = [item for item in items if isinstance(item.get("price"), (int, float))]
        if len(priced) != len(items):
            return None, "price_not_available_for_all_results"
        minimum = min(item["price"] for item in priced)
        cheapest = [item for item in priced if item["price"] == minimum]
        if len(cheapest) != 1:
            return None, "cheapest_result_ambiguous"
        chosen = cheapest[0]
    if not isinstance(chosen.get("id"), str) or not chosen.get("trusted") is True:
        return None, "result_not_trusted_or_identified"
    return {"source_goal": source, "reference_field": field, "result_id": chosen["id"],
            "source": "PROVIDER_RESULT"}, None


def plan_contract(raw: str | dict[str, Any], user_text: str, *,
                  registry: CapabilityRegistry | None = None,
                  trusted_context: Mapping[str, Any] | None = None,
                  provider_values: tuple[Mapping[str, Any], ...] = (),
                  capability_values: tuple[Mapping[str, Any], ...] = (),
                  results_by_goal: Mapping[str, list[dict[str, Any]]] | None = None,
                  external_bindings_by_goal: Mapping[str, Mapping[str, dict[str, Any]]] | None = None,
                  confirmation_by_goal: Mapping[str, bool] | None = None) -> dict[str, Any]:
    """Validate and bind a contract to mock capabilities, holding uncertainty.

    READY_FOR_DRY_RUN means only that an inspectable plan can be displayed. It
    never grants provider execution. Missing slots come from the registry, not
    from the model's `missing` array.
    """
    registry = registry or CapabilityRegistry.load(REGISTRY_V2_PATH)
    trusted_context = trusted_context or {}
    results_by_goal = results_by_goal or {}
    external_bindings_by_goal = external_bindings_by_goal or {}
    confirmation_by_goal = confirmation_by_goal or {}
    decision = check_contract(raw, user_text)
    if decision.status != ACCEPT or decision.contract is None:
        result = _shell("HOLD_GUARD", list(decision.codes) or [decision.status], registry.version)
        result["guard_status"] = decision.status
        return result
    contract = decision.contract
    if contract["kind"] == "non_actionable":
        return _shell("NO_ACTION", registry_version=registry.version)
    if _meta_only(user_text):
        return _shell("HOLD_NON_ACTIONABLE_CONTEXT", ["quoted_or_meta_intent"], registry.version)

    result = _shell("READY_FOR_DRY_RUN", registry_version=registry.version)
    result["guard_status"] = decision.status
    seen: set[str] = set()
    blocked_ids: set[str] = set()
    forbidden = {action for action, pattern in _FORBIDS.items() if pattern.search(user_text)}
    for goal in contract["goals"]:
        ident = goal["id"]
        step_codes: list[str] = []
        cap_status, cap = registry.resolve(goal)
        if cap is None:
            step_codes.append(cap_status)
        else:
            result["resolved_capabilities"].append({"goal": ident, "capability_id": cap.capability_id})
        dependencies = goal.get("depends_on", [])
        result["dependencies"].append({"goal": ident, "depends_on": dependencies,
                                       "condition": goal.get("condition")})
        if any(source not in seen for source in dependencies):
            step_codes.append("HOLD_INVALID_DEPENDENCY")
        if any(source in blocked_ids for source in dependencies):
            step_codes.append("HOLD_DEPENDENCY_BLOCKED")
        if goal.get("condition"):
            condition = goal["condition"]
            if condition["goal"] not in dependencies or condition["test"]["field"] != "results_count":
                step_codes.append("HOLD_UNMAPPED_CONDITION")
            elif condition["goal"] not in results_by_goal:
                step_codes.append("HOLD_UNRESOLVED_REFERENCE")
        seen.add(ident)
        if cap is None:
            blocked_ids.add(ident)
            result["blocked_steps"].append({"goal": ident, "codes": sorted(set(step_codes))})
            continue

        supplied: set[str] = set()
        bindings: dict[str, Any] = {}
        for section, slot_key, items in (("fact", "field", goal["facts"]),
                                         ("temporal", "role", goal.get("temporal", []))):
            for item in items:
                name = item[slot_key]
                slot = f"{section}.{name}"
                supplied.add(slot)
                if (slot not in cap.required_slots and slot not in cap.optional_slots and
                        (section != "temporal" or name not in cap.supported_temporal_roles)):
                    step_codes.append("HOLD_UNMAPPED_SLOT")
                    result["semantic_conflicts"].append({"goal": ident, "slot": slot, "reason": "not_registered"})
                if "value_ref" in item:
                    resolved, error = _reference(item["value_ref"], results_by_goal, dependencies, cap)
                    if error:
                        step_codes.append("HOLD_UNRESOLVED_REFERENCE")
                        result["unresolved_references"].append({"goal": ident, "slot": slot, "reason": error})
                    else:
                        result["resolved_references"].append({"goal": ident, "slot": slot, **resolved})
                        bindings[slot] = resolved
                    continue
                value = item.get("value")
                evidence = ground_value(name, value, user_text, trusted_context=trusted_context,
                                        provider_values=provider_values, capability_values=capability_values)
                bindings[slot] = {"value": value, "operator": item["op"], "provenance": evidence.as_dict()}
                if evidence.source not in _TRUSTED:
                    step_codes.append("HOLD_UNGROUNDED_FACT")
                    result["provenance_issues"].append({"goal": ident, "slot": slot,
                                                       "evidence": evidence.as_dict()})
                if section == "temporal":
                    if name == "arrival_time" and item["op"] == "eq" and re.search(
                            r"\b(?:entro|prima di|prima delle|by|before)\b", user_text, re.I):
                        step_codes.append("HOLD_TEMPORAL_ROLE_CONFLICT")
                        result["semantic_conflicts"].append({"goal": ident, "slot": slot,
                                                             "reason": "deadline_misread_as_equality"})
                    if goal["action"] == "contact" and name in {"action_date", "action_time"} and any(
                            str(value).casefold() in str(fact.get("value", "")).casefold()
                            for fact in goal["facts"] if fact["field"] == "message"):
                        step_codes.append("HOLD_TEMPORAL_ROLE_CONFLICT")
                        result["semantic_conflicts"].append({"goal": ident, "slot": slot,
                                                             "reason": "message_time_used_as_action_time"})
        if goal["action"] == "notify":
            expected_message = _reminder_content(user_text)
            if expected_message is not None:
                actual_messages = [" ".join(str(item.get("value", "")).strip().casefold().split())
                                   for item in goal["facts"] if item["field"] == "message"]
                if actual_messages != [expected_message]:
                    step_codes.append("HOLD_MESSAGE_CONTENT_INCOMPLETE")
                    result["semantic_conflicts"].append({"goal": ident, "slot": "fact.message",
                                                         "reason": "explicit_reminder_clause_not_preserved"})
        for key, value in goal.get("modifiers", {}).items():
            slot = f"modifiers.{key}"
            supplied.add(slot)
            if key not in cap.supported_modifiers or (key == "sort" and value.get("field") not in _SORT_FIELDS):
                step_codes.append("HOLD_UNMAPPED_SLOT")
                result["semantic_conflicts"].append({"goal": ident, "slot": slot, "reason": "modifier_not_registered"})
            if key != "sort":
                evidence = ground_value(key, value, user_text, trusted_context=trusted_context)
                if evidence.source not in _TRUSTED:
                    step_codes.append("HOLD_UNGROUNDED_FACT")
                    result["provenance_issues"].append({"goal": ident, "slot": slot,
                                                       "evidence": evidence.as_dict()})
            bindings[slot] = value
        for context_item in contract.get("context", []):
            field = context_item["field"]
            if field not in _CONTEXT_FIELDS:
                step_codes.append("HOLD_UNMAPPED_SLOT")
                result["semantic_conflicts"].append({"goal": ident, "slot": f"context.{field}",
                                                     "reason": "context_field_not_registered"})
            evidence = ground_value(field, context_item.get("value"), user_text,
                                    trusted_context=trusted_context)
            if evidence.source not in _TRUSTED:
                step_codes.append("HOLD_UNGROUNDED_FACT")
                result["provenance_issues"].append({"goal": ident, "slot": f"context.{field}",
                                                   "evidence": evidence.as_dict()})
        for missing in goal.get("missing", []):
            step_codes.append("HOLD_MISSING_INFORMATION")
            result["missing_slots"].append({"goal": ident, "slot": missing["field"],
                                            "reason": "declared_by_interpreter"})
        for slot, binding in external_bindings_by_goal.get(ident, {}).items():
            if not slot.startswith("external.") or slot not in cap.required_slots | cap.optional_slots:
                step_codes.append("HOLD_UNMAPPED_SLOT")
                result["semantic_conflicts"].append({"goal": ident, "slot": slot,
                                                     "reason": "external_slot_not_registered"})
                continue
            if (not isinstance(binding, dict) or binding.get("trusted") is not True or
                    binding.get("source") not in {"TRUSTED_CONVERSATION_CONTEXT", "PROVIDER_RESULT", "CAPABILITY_RESULT"} or
                    binding.get("value") is None):
                step_codes.append("HOLD_UNGROUNDED_FACT")
                result["provenance_issues"].append({"goal": ident, "slot": slot,
                                                   "reason": "external_binding_untrusted"})
                continue
            supplied.add(slot)
            bindings[slot] = {"value": binding["value"], "provenance": binding["source"]}
        for slot in sorted(cap.required_slots - supplied):
            step_codes.append("HOLD_MISSING_INFORMATION")
            result["missing_slots"].append({"goal": ident, "slot": slot,
                                            "reason": "required_by_capability"})

        policy = goal.get("policy", {})
        for action in policy.get("forbid", []):
            if action == goal["action"]:
                step_codes.append("HOLD_POLICY_CONFLICT")
        if goal["action"] in forbidden:
            step_codes.append("HOLD_POLICY_CONFLICT")
        if cap.consequentiality == "read":
            for action in forbidden:
                if (goal["target"]["type"] in _FORBID_TYPES[action] and
                        action not in policy.get("forbid", [])):
                    step_codes.append("HOLD_POLICY_REQUIRED")
        if cap.confirmation_required:
            if goal["action"] not in policy.get("confirm_before", []):
                step_codes.append("HOLD_POLICY_REQUIRED")
            if confirmation_by_goal.get(ident) is not True:
                step_codes.append("HOLD_CONFIRMATION_PENDING")
        result["policy_state"].append({"goal": ident,
                                       "declared": policy,
                                       "confirmation_verified": confirmation_by_goal.get(ident) is True,
                                       "required": sorted(cap.required_policies)})
        result["goal_bindings"].append({"goal": ident, "capability": cap.capability_id,
                                        "slots": bindings})
        if step_codes:
            blocked_ids.add(ident)
            result["blocked_steps"].append({"goal": ident, "capability": cap.capability_id,
                                            "codes": sorted(set(step_codes))})
        else:
            result["execution_plan_steps"].append({"goal": ident, "capability": cap.capability_id,
                                                   "binding": bindings, "mode": "DRY_RUN_ONLY"})

    result["reason_codes"] = sorted({code for step in result["blocked_steps"] for code in step["codes"]})
    if result["reason_codes"]:
        result["planner_status"] = "HOLD"
    return result
