"""Conservative, opt-in repair of a narrow malformed V1.3 origin slot.

The raw model output remains the benchmark result. A repaired candidate may
only proceed to Guard inspection, never directly to a provider. Unrecognized
shapes are held instead of being guessed into valid contracts.
"""
from __future__ import annotations

import copy
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .semantic_contract_v1_3 import validate_contract


@dataclass(frozen=True)
class RepairResult:
    status: str
    codes: tuple[str, ...]
    contract: dict[str, Any] | None


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold())
                    .encode("ascii", "ignore").decode("ascii").split())


def repair_candidate(raw: str | dict[str, Any], user_text: str) -> RepairResult:
    try:
        candidate = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return RepairResult("HOLD", ("invalid_json",), None)
    if not isinstance(candidate, dict):
        return RepairResult("HOLD", ("not_object",), None)
    if candidate.get("schema_version") != "1.3" or not isinstance(candidate.get("goals"), list):
        return RepairResult("UNCHANGED", (), candidate)
    if not any("spatial" in goal for goal in candidate["goals"] if isinstance(goal, dict)):
        return RepairResult("UNCHANGED", (), candidate)
    repaired = copy.deepcopy(candidate)
    text = _plain(user_text)
    for goal in repaired["goals"]:
        if not isinstance(goal, dict):
            return RepairResult("HOLD", ("malformed_goal",), None)
        spatial = goal.pop("spatial", None)
        if spatial is None:
            continue
        if not isinstance(spatial, list) or len(spatial) != 1:
            return RepairResult("HOLD", ("unsupported_spatial_shape",), None)
        item = spatial[0]
        if (not isinstance(item, dict) or set(item) != {"field", "op", "value", "strength"} or
                item.get("field") != "origin" or item.get("op") != "eq" or
                item.get("strength") != "hard" or not isinstance(item.get("value"), str)):
            return RepairResult("HOLD", ("unsupported_spatial_shape",), None)
        value = _plain(item["value"])
        if len(value) < 3 or not re.search(r"\bda\s+" + re.escape(value) + r"\b", text):
            return RepairResult("HOLD", ("spatial_origin_not_grounded",), None)
        existing = [fact for fact in goal.get("facts", []) if fact.get("field") == "origin"]
        if existing and existing != [item]:
            return RepairResult("HOLD", ("spatial_origin_conflict",), None)
        if not existing:
            goal.setdefault("facts", []).append(item)
    if validate_contract(repaired, user_text=user_text):
        return RepairResult("HOLD", ("repair_did_not_validate",), None)
    return RepairResult("REPAIRED", ("spatial_origin_to_fact",), repaired)
