"""Versioned GoalContract parsing and V1.2 static/semantic validation.

No provider, planner or execution policy is inferred here. Existing V1.1 Guard
continues to own V1.1 semantic checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .canon import extract_contract

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_11 = json.loads((ROOT / "schema/goal_contract_v1_1.schema.json").read_text(encoding="utf-8"))
SCHEMA_12 = json.loads((ROOT / "schema/goal_contract_v1_2.schema.json").read_text(encoding="utf-8"))
VALIDATOR_11 = Draft202012Validator(SCHEMA_11)
VALIDATOR_12 = Draft202012Validator(SCHEMA_12)
NUMERIC_FIELDS = {"price", "budget", "cost", "distance", "radius", "duration", "rating", "stars", "guests", "quantity", "age", "weight", "ram", "storage"}
UNIT_FIELDS = {"distance", "radius", "duration", "weight", "ram", "storage"}
MONEY_FIELDS = {"price", "budget", "cost"}
NUMERIC_OPERATORS = {"lt", "lte", "gt", "gte", "within", "between"}


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    message: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "message": self.message}


def parse_contract(raw: str | dict[str, Any]) -> dict[str, Any]:
    """Parse without upgrading versions or changing the user's data."""
    return raw if isinstance(raw, dict) else extract_contract(raw)


def _fact_issues(fact: dict, path: str) -> list[Issue]:
    out: list[Issue] = []
    field, op = fact["field"], fact["op"]
    value = fact.get("value")
    if op == "between" and "value_to" not in fact:
        out.append(Issue("between_missing_end", path, "between requires value_to"))
    if op != "between" and "value_to" in fact:
        out.append(Issue("unexpected_value_to", path, "value_to is only valid with between"))
    if op == "between" and "value_to" in fact and (
        type(value) not in (int, float) or type(fact["value_to"]) not in (int, float)
        or value > fact["value_to"]
    ):
        out.append(Issue("invalid_numeric_range", path, "between requires ordered numeric endpoints"))
    if op in NUMERIC_OPERATORS and "value" in fact and type(value) not in (int, float):
        out.append(Issue("numeric_operator_type", path, f"{op} requires a numeric literal"))
    if field in NUMERIC_FIELDS and "value" in fact and type(value) not in (int, float):
        # Some properties are open-text in V19; review rather than silently coerce.
        out.append(Issue("numeric_field_type", path, f"{field} requires a numeric literal"))
    if field in UNIT_FIELDS and type(value) in (int, float) and "unit" not in fact:
        out.append(Issue("unit_missing", path, f"{field} requires an explicit unit"))
    if field in MONEY_FIELDS and type(value) in (int, float) and "currency" not in fact:
        out.append(Issue("currency_missing", path, f"{field} requires an explicit currency"))
    if "value_ref" in fact and ("unit" in fact or "currency" in fact):
        out.append(Issue("ref_unit_unresolved", path, "units of referenced values must be resolved downstream"))
    return out


def _hard_numeric_conflicts(facts: list[dict]) -> set[str]:
    """Detect explicit impossible numeric intervals, without guessing semantics."""
    by_field: dict[tuple[str, str | None, str | None], list[dict]] = {}
    for fact in facts:
        if fact["strength"] == "hard" and type(fact.get("value")) in (int, float):
            key = (fact["field"], fact.get("unit"), fact.get("currency"))
            by_field.setdefault(key, []).append(fact)
    conflict: set[str] = set()
    for (field, _unit, _currency), items in by_field.items():
        lower, lower_strict = float("-inf"), False
        upper, upper_strict = float("inf"), False
        for fact in items:
            op, value = fact["op"], fact["value"]
            if op in ("gt", "gte", "eq", "between"):
                v = value
                strict = op == "gt"
                if v > lower or (v == lower and strict):
                    lower, lower_strict = v, strict
            if op in ("lt", "lte", "eq", "between"):
                v = fact["value_to"] if op == "between" and "value_to" in fact else value
                strict = op == "lt"
                if v < upper or (v == upper and strict):
                    upper, upper_strict = v, strict
        if lower > upper or (lower == upper and (lower_strict or upper_strict)):
            conflict.add(field)
    return conflict


def validate_contract(contract: dict[str, Any], *, version: str | None = None) -> list[Issue]:
    """Return structural and semantic issues; never repair a candidate silently."""
    if not isinstance(contract, dict):
        return [Issue("not_object", "$", "contract must be a JSON object")]
    version = version or contract.get("schema_version")
    if version not in ("1.1", "1.2"):
        return [Issue("unsupported_version", "$.schema_version", f"unsupported version {version!r}")]
    validator = VALIDATOR_11 if version == "1.1" else VALIDATOR_12
    structural = sorted(validator.iter_errors(contract), key=lambda e: (list(map(str, e.path)), e.message))
    if structural:
        return [Issue("schema", "$" + "".join(f"[{part!r}]" for part in e.path), e.message)
                for e in structural]
    if version == "1.1":
        return []  # V1.1 semantic ownership remains with jase.guard.check_contract.

    out: list[Issue] = []
    goals = contract["goals"]
    if contract["kind"] == "task" and not goals:
        out.append(Issue("task_without_goal", "$.goals", "task requires at least one goal"))
    if contract["kind"] == "non_actionable" and goals:
        out.append(Issue("non_actionable_with_goals", "$.goals", "non_actionable request cannot have executable goals"))
    ids = [g["id"] for g in goals]
    if len(ids) != len(set(ids)):
        out.append(Issue("duplicate_goal_id", "$.goals", "goal IDs must be unique"))
    by_id = {g["id"]: g for g in goals}
    graph = {g["id"]: set(g.get("depends_on", [])) for g in goals}
    positions = {g["id"]: i for i, g in enumerate(goals)}

    def check_ref(ref: dict, owner: str, path: str) -> None:
        source = ref["goal"]
        if source not in by_id:
            out.append(Issue("unknown_reference", path, f"unknown goal {source}"))
        elif source == owner or source not in graph[owner] or positions[source] >= positions[owner]:
            out.append(Issue("reference_without_prior_dependency", path, f"{source} must be a prior dependency"))

    for i, goal in enumerate(goals):
        gid = goal["id"]
        path = f"$.goals[{i}]"
        for dep in graph[gid]:
            if dep not in by_id:
                out.append(Issue("unknown_dependency", path + ".depends_on", dep))
            elif dep == gid or positions[dep] >= i:
                out.append(Issue("non_prior_dependency", path + ".depends_on", dep))
        if "condition" in goal:
            check_ref({"goal": goal["condition"]["goal"]}, gid, path + ".condition")
            out.extend(_fact_issues(goal["condition"]["test"], path + ".condition.test"))
            if "value_ref" in goal["condition"]["test"]:
                check_ref(goal["condition"]["test"]["value_ref"], gid,
                          path + ".condition.test.value_ref")
        seen_facts: set[str] = set()
        present_fields = set()
        for j, fact in enumerate(goal["facts"]):
            fp = f"{path}.facts[{j}]"
            out.extend(_fact_issues(fact, fp))
            present_fields.add(fact["field"])
            key = json.dumps({k: v for k, v in fact.items() if k != "strength"}, sort_keys=True, ensure_ascii=False)
            if key in seen_facts:
                out.append(Issue("duplicate_fact", fp, "same fact repeated"))
            seen_facts.add(key)
            if "value_ref" in fact:
                check_ref(fact["value_ref"], gid, fp + ".value_ref")
        for field in _hard_numeric_conflicts(goal["facts"]):
            out.append(Issue("contradictory_hard_facts", path + ".facts", field))
        missing_fields = [m["field"] for m in goal.get("missing", [])]
        if len(missing_fields) != len(set(missing_fields)):
            out.append(Issue("duplicate_missing", path + ".missing", "field repeated"))
        for m in goal.get("missing", []):
            if m["field"] in present_fields:
                out.append(Issue("missing_but_present", path + ".missing", m["field"]))
            if goal["action"] in ("find", "compare") and m["blocks"] == "action":
                out.append(Issue("wrong_block_phase", path + ".missing", "read-only action has no later commit phase"))
        policy = goal.get("policy", {})
        forbidden = set(policy.get("forbid", []))
        confirmed = set(policy.get("confirm_before", []))
        if forbidden & confirmed:
            out.append(Issue("contradictory_policy", path + ".policy", "same action forbidden and confirmation-gated"))
        if goal["action"] in forbidden:
            out.append(Issue("forbidden_requested_action", path + ".policy", "goal requests an action it forbids"))

    # Cycles are reported even when they also violate the prior-dependency rule.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            out.append(Issue("dependency_cycle", "$.goals", f"cycle involving {node}"))
            return
        if node in visited:
            return
        visiting.add(node)
        for dep in graph[node]:
            if dep in graph:
                visit(dep)
        visiting.remove(node)
        visited.add(node)

    for gid in ids:
        visit(gid)
    for i, fact in enumerate(contract.get("context", [])):
        out.extend(_fact_issues(fact, f"$.context[{i}]"))
        if "value_ref" in fact:
            out.append(Issue("context_result_ref", f"$.context[{i}]", "context cannot depend on an unplanned goal result"))
    return out


def canonical_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Order-independent canonical form for static V1.2 semantic equality."""
    if validate_contract(contract):
        raise ValueError("contract must validate before canonicalization")
    if contract["schema_version"] != "1.2":
        raise ValueError("V1.2 only")
    result = json.loads(json.dumps(contract, ensure_ascii=False))
    result["goals"] = sorted(result["goals"], key=lambda g: g["id"])
    for goal in result["goals"]:
        goal["facts"] = sorted(goal["facts"], key=lambda f: json.dumps(f, sort_keys=True, ensure_ascii=False))
        for key in ("depends_on", "missing"):
            if key in goal:
                goal[key] = sorted(goal[key], key=lambda x: x if isinstance(x, str) else json.dumps(x, sort_keys=True))
        if "policy" in goal:
            for key in goal["policy"]:
                goal["policy"][key] = sorted(goal["policy"][key])
    if "context" in result:
        result["context"] = sorted(result["context"], key=lambda f: json.dumps(f, sort_keys=True, ensure_ascii=False))
    return result
