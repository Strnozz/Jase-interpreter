"""Documented, narrow semantic normalizations for offline equality checks.

Never use a canonicalized contract as an execution contract. In particular,
model-in-target vs model-as-fact and eq vs contains are NOT equated here.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from typing import Any

from .goal_contract_v1_2 import validate_contract as validate_v12
from .semantic_contract_v1_3 import validate_contract as validate_v13

TYPE_ALIASES = {"medical_appointment": "appointment"}
TIME_FIELDS = {"time", "arrival_time", "departure_time"}
TIME_ROLES = {"search_time", "arrival_time", "departure_time", "action_time", "notification_time"}


def _text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _time(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    normalized = _text(value)
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", normalized)
    if match and int(match.group(1)) < 24 and int(match.group(2)) < 60:
        return f"{int(match.group(1)):02d}:{match.group(2)}"
    return normalized


def canonicalize(contract: dict[str, Any]) -> dict[str, Any]:
    """Return a new comparison-only V1.2/V1.3 representation."""
    version = contract.get("schema_version")
    issues = validate_v12(contract) if version == "1.2" else validate_v13(contract)
    if issues:
        raise ValueError("contract invalid before canonicalization")
    result = copy.deepcopy(contract)
    result["goals"] = sorted(result["goals"], key=lambda g: g["id"])
    for goal in result["goals"]:
        target = goal["target"]
        target["name"] = _text(target["name"])
        if "type" in target:
            target["type"] = TYPE_ALIASES.get(target["type"], target["type"])
        for fact in goal["facts"]:
            if "value" in fact:
                fact["value"] = _time(fact["value"]) if fact["field"] in TIME_FIELDS else (
                    _text(fact["value"]) if isinstance(fact["value"], str) else fact["value"])
        for item in goal.get("temporal", []):
            if "value" in item:
                item["value"] = _time(item["value"]) if item["role"] in TIME_ROLES else (
                    _text(item["value"]) if isinstance(item["value"], str) else item["value"])
        for key in ("facts", "temporal", "missing", "depends_on"):
            if key in goal:
                goal[key] = sorted(goal[key], key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
        if "policy" in goal:
            for key in goal["policy"]:
                goal["policy"][key] = sorted(goal["policy"][key])
    if "context" in result:
        for item in result["context"]:
            if isinstance(item.get("value"), str):
                item["value"] = _text(item["value"])
        result["context"] = sorted(result["context"], key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))
    return result


def compare(expected: dict[str, Any], predicted: Any) -> dict[str, Any]:
    """Expose literal, legacy V1.2, and narrow canonical equality separately."""
    literal = isinstance(predicted, dict) and expected == predicted
    if not isinstance(predicted, dict):
        return {"literal_equal": False, "canonical_equal": False, "valid": False}
    try:
        same = canonicalize(expected) == canonicalize(predicted)
    except ValueError:
        same = False
    return {"literal_equal": literal, "canonical_equal": same,
            "valid": not (validate_v12(predicted) if predicted.get("schema_version") == "1.2"
                          else validate_v13(predicted))}
