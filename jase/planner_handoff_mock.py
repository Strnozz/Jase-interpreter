"""Typed, atomic mock-only handoff for already validated Planner dry-run steps.

It creates inspectable request drafts. There are no provider clients or execution
paths in this module; every result has execution_permitted=False.
"""
from __future__ import annotations

from typing import Any

from .capability_registry_v1 import CapabilityRegistry, REGISTRY_V3_PATH
from .provenance_v1 import (CAPABILITY_RESULT, DETERMINISTIC_DERIVATION,
                            PROVIDER_RESULT, TRUSTED_CONVERSATION_CONTEXT, USER_EXPLICIT)

VERSION = "mock-handoff-1"
_TRUSTED = {USER_EXPLICIT, TRUSTED_CONVERSATION_CONTEXT, DETERMINISTIC_DERIVATION,
            PROVIDER_RESULT, CAPABILITY_RESULT}
_TEXT_FACTS = {"location", "origin", "destination", "availability", "access", "diet", "cuisine",
               "specialization", "condition", "message"}
_BOOL_FACTS = {"open_now", "direct"}
_INT_FACTS = {"stops", "quantity", "guests"}
_CONTEXT_TEXT = {"user_location", "vehicle_location", "pet", "lodging_owned"}
_CONTEXT_BOOL = {"has_car"}
_CONTEXT_INT = {"party_size"}
_OPERATORS = {"eq", "near", "contains", "lt", "lte", "gt", "gte", "before", "after"}


def _result(status: str, codes: list[str] | None = None) -> dict[str, Any]:
    return {"handoff_version": VERSION, "status": status, "reason_codes": sorted(set(codes or [])),
            "mock_request_drafts": [], "provider_called": False, "execution_permitted": False}


def _typed(slot: str, value: Any) -> bool:
    if slot.startswith("fact."):
        name = slot[5:]
        if name in _TEXT_FACTS:
            return isinstance(value, str) and bool(value.strip()) and len(value) <= 512
        if name in _BOOL_FACTS:
            return type(value) is bool
        if name in _INT_FACTS:
            return type(value) is int and 0 <= value <= 100000
        if name == "price":
            return type(value) in {int, float} and 0 <= value <= 100000000
        return False
    if slot.startswith("temporal."):
        return isinstance(value, str) and bool(value.strip()) and len(value) <= 128
    if slot.startswith("context."):
        name = slot[8:]
        if name in _CONTEXT_BOOL:
            return type(value) is bool
        if name in _CONTEXT_INT:
            return type(value) is int and 1 <= value <= 1000
        if name in _CONTEXT_TEXT:
            return isinstance(value, str) and bool(value.strip()) and len(value) <= 512
    return False


def prepare_mock_handoff(plan: dict[str, Any], *, registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    """Produce only typed mock request drafts, or hold the entire plan."""
    registry = registry or CapabilityRegistry.load(REGISTRY_V3_PATH)
    if plan.get("execution_permitted") is not False:
        return _result("HOLD", ["planner_permission_not_false"])
    if plan.get("registry_version") != registry.version:
        return _result("HOLD", ["registry_version_mismatch"])
    if plan.get("planner_status") == "NO_ACTION":
        return _result("NO_ACTION")
    if (plan.get("planner_status") != "READY_FOR_DRY_RUN" or plan.get("blocked_steps") or
            plan.get("reason_codes")):
        return _result("HOLD", ["planner_not_ready"])
    steps = plan.get("execution_plan_steps")
    bindings = plan.get("goal_bindings")
    if not isinstance(steps, list) or not steps or not isinstance(bindings, list) or len(bindings) != len(steps):
        return _result("HOLD", ["step_binding_count_mismatch"])
    bound_by_goal = {item.get("goal"): item for item in bindings if isinstance(item, dict)}
    if len(bound_by_goal) != len(steps):
        return _result("HOLD", ["duplicate_or_missing_goal_binding"])
    policies = {item.get("goal"): item for item in plan.get("policy_state", []) if isinstance(item, dict)}
    dependencies = {item.get("goal"): item for item in plan.get("dependencies", []) if isinstance(item, dict)}
    if len(policies) != len(steps) or len(dependencies) != len(steps):
        return _result("HOLD", ["policy_or_dependency_mapping_incomplete"])
    by_id = {cap.capability_id: cap for cap in registry.capabilities}
    drafts: list[dict[str, Any]] = []
    problems: list[str] = []
    for step in steps:
        goal_id, cap_id = step.get("goal"), step.get("capability")
        cap = by_id.get(cap_id)
        binding = bound_by_goal.get(goal_id)
        if (cap is None or binding is None or binding.get("capability") != cap_id or
                step.get("mode") != "DRY_RUN_ONLY" or step.get("binding") != binding.get("slots") or
                goal_id not in policies or goal_id not in dependencies):
            problems.append("unmapped_or_mismatched_capability")
            continue
        if (cap.consequentiality != "read" or cap.confirmation_required or
                cap.execution_status != "mock_only" or cap.provider_binding not in {"mock_catalog", "mock_compare"}):
            problems.append("write_or_nonmock_capability")
            continue
        slots = binding.get("slots")
        if not isinstance(slots, dict) or not cap.required_slots.issubset(slots):
            problems.append("required_slot_absent")
            continue
        clean: dict[str, Any] = {}
        for slot, item in slots.items():
            if slot.startswith("modifiers."):
                problems.append("modifier_mapping_unverified")
                continue
            permitted = (slot in cap.required_slots or slot in cap.optional_slots or
                         (slot.startswith("temporal.") and slot[9:] in cap.supported_temporal_roles) or
                         slot.startswith("context."))
            if not permitted or not isinstance(item, dict):
                problems.append("unmapped_slot")
                continue
            evidence = item.get("provenance")
            if not isinstance(evidence, dict) or evidence.get("source") not in _TRUSTED:
                problems.append("untrusted_slot")
                continue
            value = item.get("value")
            if evidence.get("value") != value:
                problems.append("binding_evidence_mismatch")
                continue
            if not _typed(slot, value) or item.get("operator") not in _OPERATORS:
                problems.append("invalid_slot_type_or_operator")
                continue
            clean[slot] = {"value": value, "operator": item["operator"],
                           "provenance": evidence["source"]}
            if item.get("currency"):
                if slot != "fact.price" or item["currency"] not in {"EUR", "USD", "GBP"}:
                    problems.append("currency_not_supported")
                else:
                    clean[slot]["currency"] = item["currency"]
        drafts.append({"goal": goal_id, "capability_id": cap_id, "operation": "read",
                       "mock_binding": cap.provider_binding, "slots": clean,
                       "policy": policies[goal_id]["declared"],
                       "depends_on": dependencies[goal_id]["depends_on"],
                       "condition": dependencies[goal_id]["condition"], "mode": "DRAFT_ONLY"})
    if problems:
        return _result("HOLD", problems)
    result = _result("READY_FOR_MOCK_DRAFT")
    result["mock_request_drafts"] = drafts
    return result
