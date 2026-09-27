"""Isolated, fail-closed GoalContract 1.3 handoff prototype.

There is no provider registry or Planner in this repository. A caller must pass
explicit mock capability bindings; READY_FOR_PLANNING is never an execution
permit. This module does not call providers or modify the frontend.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .semantic_guard_v1_3 import ACCEPT
from .semantic_guard_v1_3_routing_v4 import check_contract

READY_FOR_PLANNING = "READY_FOR_PLANNING"
NO_ACTION = "NO_ACTION"
HOLD_GUARD = "HOLD_GUARD"
HOLD_NO_CAPABILITY = "HOLD_NO_CAPABILITY"
HOLD_AMBIGUOUS_CAPABILITY = "HOLD_AMBIGUOUS_CAPABILITY"
HOLD_UNMAPPED_SLOT = "HOLD_UNMAPPED_SLOT"
HOLD_MISSING_INFORMATION = "HOLD_MISSING_INFORMATION"


@dataclass(frozen=True)
class CapabilitySpec:
    capability_id: str
    action: str
    target_type: str
    target_names: frozenset[str] | None = None
    fact_params: Mapping[str, str] = field(default_factory=dict)
    temporal_params: Mapping[str, str] = field(default_factory=dict)
    modifier_params: Mapping[str, str] = field(default_factory=dict)
    sort_fields: frozenset[str] = frozenset()
    policy_actions: frozenset[str] = frozenset()
    result_params: Mapping[str, str] = field(default_factory=dict)
    accepts_references: bool = False
    accepts_ordinal_references: bool = False
    supports_dependencies: bool = False
    condition_ops: frozenset[str] = frozenset()
    required_slots: frozenset[str] = frozenset()

    def matches(self, goal: dict[str, Any]) -> bool:
        return (self.action == goal["action"] and self.target_type == goal["target"]["type"] and
                (self.target_names is None or goal["target"]["name"].casefold() in self.target_names))


@dataclass(frozen=True)
class Handoff:
    status: str
    codes: tuple[str, ...]
    bindings: tuple[dict[str, Any], ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {"status": self.status, "codes": list(self.codes),
                "bindings": list(self.bindings), "execution_permitted": False}


def inspect_handoff(raw: str | dict[str, Any], user_text: str,
                    capabilities: tuple[CapabilitySpec, ...] = (),
                    context_params: Mapping[str, str] | None = None) -> Handoff:
    """Resolve every declared slot to an explicit interface or HOLD.

    No default bindings or aliases exist. Conditions bind to the SOURCE goal's
    result interface; references require source result support and target
    reference support. Missing information always holds the entire handoff.
    """
    decision = check_contract(raw, user_text)
    if decision.status != ACCEPT or decision.contract is None:
        return Handoff(HOLD_GUARD, decision.codes or ("guard_not_accept",))
    contract = decision.contract
    if contract["kind"] == "non_actionable":
        return Handoff(NO_ACTION, ())
    mapped_context = context_params or {}
    for item in contract.get("context", []):
        if item["field"] not in mapped_context:
            return Handoff(HOLD_UNMAPPED_SLOT, (f"context.{item['field']}",))

    selected: dict[str, CapabilitySpec] = {}
    bindings = []
    for goal in contract["goals"]:
        matches = [spec for spec in capabilities if spec.matches(goal)]
        if not matches:
            return Handoff(HOLD_NO_CAPABILITY, (f"{goal['id']}:{goal['action']}:{goal['target']['type']}",))
        if len(matches) != 1:
            return Handoff(HOLD_AMBIGUOUS_CAPABILITY, (goal["id"],))
        spec = matches[0]
        if goal.get("missing"):
            return Handoff(HOLD_MISSING_INFORMATION,
                           tuple(f"{goal['id']}.missing.{m['field']}" for m in goal["missing"]))
        supplied_slots = ({f"fact.{item['field']}" for item in goal["facts"]} |
                          {f"temporal.{item['role']}" for item in goal.get("temporal", [])} |
                          {f"context.{item['field']}" for item in contract.get("context", [])})
        absent_slots = spec.required_slots - supplied_slots
        if absent_slots:
            return Handoff(HOLD_MISSING_INFORMATION,
                           tuple(f"{goal['id']}.required.{slot}" for slot in sorted(absent_slots)))
        facts, temporal, modifiers = {}, {}, {}
        dependencies = goal.get("depends_on", [])
        if dependencies and (not spec.supports_dependencies or
                             any(source not in selected for source in dependencies)):
            return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.depends_on",))
        for item in goal["facts"]:
            field_name = item["field"]
            if field_name not in spec.fact_params:
                return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.facts.{field_name}",))
            if "value_ref" in item:
                source = item["value_ref"]["goal"]
                ref_field = item["value_ref"]["field"]
                if not spec.accepts_references or source not in selected or \
                        ref_field not in selected[source].result_params:
                    return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.reference.{source}.{ref_field}",))
                if item["value_ref"].get("ordinal") is not None and not spec.accepts_ordinal_references:
                    return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.reference.ordinal",))
            facts[field_name] = spec.fact_params[field_name]
        for item in goal.get("temporal", []):
            role = item["role"]
            if role not in spec.temporal_params:
                return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.temporal.{role}",))
            temporal[role] = spec.temporal_params[role]
        for key, value in goal.get("modifiers", {}).items():
            if key not in spec.modifier_params:
                return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.modifiers.{key}",))
            if key == "sort" and value["field"] not in spec.sort_fields:
                return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.sort.{value['field']}",))
            modifiers[key] = spec.modifier_params[key]
        for key in ("forbid", "confirm_before"):
            for action in goal.get("policy", {}).get(key, []):
                if action not in spec.policy_actions:
                    return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.policy.{key}.{action}",))
        condition = goal.get("condition")
        if condition:
            source = condition["goal"]
            field_name = condition["test"]["field"]
            op = condition["test"]["op"]
            if (source not in selected or source not in dependencies or
                    field_name not in selected[source].result_params or
                    op not in spec.condition_ops):
                return Handoff(HOLD_UNMAPPED_SLOT, (f"{goal['id']}.condition.{source}.{field_name}",))
        selected[goal["id"]] = spec
        bindings.append({"goal": goal["id"], "capability": spec.capability_id,
                         "fact_params": facts, "temporal_params": temporal,
                         "modifier_params": modifiers,
                         "dependencies": dependencies,
                         "condition_source": condition["goal"] if condition else None})
    return Handoff(READY_FOR_PLANNING, (), tuple(bindings))
