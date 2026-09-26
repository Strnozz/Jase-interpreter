"""Build the additive, role-typed GoalContract 1.3 schema from 1.2."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "schema/goal_contract_v1_2.schema.json"
DEST = ROOT / "schema/goal_contract_v1_3.schema.json"

TARGET_TYPES = [
    "place", "accommodation", "real_estate", "transport", "vehicle", "product",
    "electronics", "flower", "professional_service", "person", "appointment",
    "medical_appointment", "reminder", "food", "ticket", "media", "education",
    "animal", "service", "drink",
]
TEMPORAL_ROLES = [
    "search_date", "search_time", "arrival_time", "departure_time", "action_date", "action_time",
    "notification_date", "notification_time",
]


def build() -> dict:
    schema = deepcopy(json.loads(SOURCE.read_text(encoding="utf-8")))
    schema["$id"] = "https://jase.local/schemas/goal-contract-v1_3.json"
    schema["title"] = "Jase GoalContract v1.3"
    schema["properties"]["schema_version"] = {"const": "1.3"}
    defs = schema["$defs"]
    defs["action"] = {"enum": [*defs["action"]["enum"], "reschedule"]}
    defs["target"]["required"] = ["name", "type"]
    defs["target"]["properties"]["type"] = {"enum": TARGET_TYPES}
    defs["temporal"] = {
        "type": "object", "additionalProperties": False,
        "required": ["role", "op", "strength"],
        "properties": {
            "role": {"enum": TEMPORAL_ROLES},
            "op": {"enum": ["eq", "before", "after", "between"]},
            "value": {"$ref": "#/$defs/scalar"},
            "value_ref": {"$ref": "#/$defs/value_ref"},
            "value_to": {"$ref": "#/$defs/scalar"},
            "strength": {"enum": ["hard", "soft"]},
        },
        "oneOf": [{"required": ["value"]}, {"required": ["value_ref"]}],
    }
    defs["modifiers"] = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "sort": {"$ref": "#/$defs/ranking"},
            "limit": {"type": "integer", "minimum": 1},
            "compare_count": {"type": "integer", "minimum": 2},
        },
        "minProperties": 1,
    }
    defs["context_v13"] = {
        "type": "object", "additionalProperties": False,
        "required": ["scope", "field", "op", "value"],
        "properties": {
            "scope": {"enum": ["user", "conversation", "resource"]},
            "field": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$"},
            "op": {"$ref": "#/$defs/operator"},
            "value": {"$ref": "#/$defs/scalar"},
        },
    }
    schema["properties"]["context"] = {"type": "array", "items": {"$ref": "#/$defs/context_v13"}}
    goal = defs["goal"]["properties"]
    goal.pop("ranking")
    goal["modifiers"] = {"$ref": "#/$defs/modifiers"}
    goal["temporal"] = {"type": "array", "items": {"$ref": "#/$defs/temporal"}}
    return schema


if __name__ == "__main__":
    data = build()
    if DEST.exists():
        raise FileExistsError("Versioned schema exists; edit it intentionally")
    DEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(DEST)
