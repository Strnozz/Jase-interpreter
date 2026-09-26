"""Conservative V1.2→V1.3 projection; never modify a source contract."""
from __future__ import annotations

import copy
from typing import Any

from .goal_contract_v1_2 import validate_contract as validate_v12
from .semantic_contract_v1_3 import TEMPORAL_FIELDS, validate_contract as validate_v13

USER_CONTEXT_FIELDS = {"party_size", "has_car", "ticket_owned", "lodging_owned"}


class MigrationHold(ValueError):
    """The old representation does not determine one safe V1.3 interpretation."""


def upgrade_contract(source: dict[str, Any]) -> dict[str, Any]:
    if source.get("schema_version") != "1.2" or validate_v12(source):
        raise MigrationHold("source is not a valid GoalContract 1.2")
    out: dict[str, Any] = {"schema_version": "1.3", "kind": source["kind"], "goals": []}
    if source.get("context"):
        out["context"] = []
        for item in source["context"]:
            if item["field"] not in USER_CONTEXT_FIELDS or "value_ref" in item:
                raise MigrationHold(f"unknown context owner: {item['field']}")
            out["context"].append({"scope": "user", **copy.deepcopy(item)})
    for original in source["goals"]:
        goal = {key: copy.deepcopy(value) for key, value in original.items()
                if key not in {"ranking", "facts"}}
        goal["facts"] = []
        temporal = []
        for fact in original["facts"]:
            field = fact["field"]
            if field not in TEMPORAL_FIELDS:
                goal["facts"].append(copy.deepcopy(fact))
                continue
            if field in {"arrival_time", "departure_time"}:
                role = field
            elif original["action"] == "notify":
                role = "notification_date" if field == "date" else "notification_time"
            elif original["action"] in {"find", "compare"}:
                if field == "time" and original["target"].get("type") == "transport":
                    raise MigrationHold("transport time lacks arrival/departure role")
                role = "search_date" if field == "date" else "search_time"
            else:
                role = "action_date" if field == "date" else "action_time"
            converted = {key: copy.deepcopy(value) for key, value in fact.items() if key != "field"}
            converted["role"] = role
            temporal.append(converted)
        if temporal:
            goal["temporal"] = temporal
        modifiers: dict[str, Any] = {}
        if "ranking" in original:
            modifiers["sort"] = copy.deepcopy(original["ranking"])
        if original["action"] == "compare":
            count_facts = [f for f in goal["facts"] if f["field"] == "quantity" and
                           f["op"] == "eq" and type(f.get("value")) is int]
            if len(count_facts) == 1:
                modifiers["compare_count"] = count_facts[0]["value"]
                goal["facts"].remove(count_facts[0])
        if modifiers:
            goal["modifiers"] = modifiers
        out["goals"].append(goal)
    problems = validate_v13(out)
    if problems:
        raise MigrationHold("V1.3 projection held: " + ", ".join(sorted({p.code for p in problems})))
    return out
