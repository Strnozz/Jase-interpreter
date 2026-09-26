"""GoalContract 1.3 structural and deterministic semantic checks.

These checks prove only stated invariants. They cannot prove that an LLM copied
every user fact or selected the user's intended action; grounding is a separate
Guard concern. V1.2 validation and frozen metrics remain unchanged.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .goal_contract_v1_2 import (_fact_issues, _hard_numeric_conflicts,
                                 validate_contract as validate_v12)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema/goal_contract_v1_3.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)
CONSEQUENTIAL = {"book", "buy", "contact", "rent", "order", "hire", "cancel", "reschedule"}
SINGLE_VALUED = {"date", "time", "arrival_time", "departure_time", "origin", "destination",
                 "guests", "quantity", "price", "budget", "model", "selection", "recipient"}
TEMPORAL_FIELDS = {"date", "time", "arrival_time", "departure_time", "time_of_day"}
TARGET_FAMILY = {"medical_appointment": "appointment", "electronics": "product", "flower": "product"}
KNOWN_TYPES = {"place", "accommodation", "real_estate", "transport", "vehicle", "product",
               "professional_service", "person", "appointment", "reminder", "food", "ticket",
               "media", "education", "animal", "service", "drink"}
COUNT_WORDS = {"due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6}


@dataclass(frozen=True)
class SemanticIssue:
    code: str
    path: str
    detail: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "path": self.path, "detail": self.detail}


def _norm(value: Any, *, time: bool = False) -> Any:
    if isinstance(value, str):
        s = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
        if time:
            match = re.fullmatch(r"(\d{1,2}):(\d{2})", s)
            if match and int(match.group(1)) < 24 and int(match.group(2)) < 60:
                return f"{int(match.group(1)):02d}:{match.group(2)}"
        return s
    return value


def _text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode().split())


def _hard_eq_conflicts(items: list[dict], slot_key: str, path: str) -> list[SemanticIssue]:
    by_slot: dict[str, set[str]] = {}
    for item in items:
        if item.get("strength") != "hard" or item.get("op") != "eq" or "value" not in item:
            continue
        slot = item[slot_key]
        if slot_key == "field" and slot not in SINGLE_VALUED:
            continue
        value = _norm(item["value"], time=(slot in {"time", "arrival_time", "departure_time",
                                           "action_time", "notification_time", "search_time"}))
        by_slot.setdefault(slot, set()).add(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return [SemanticIssue("conflicting_hard_equality", path, slot)
            for slot, values in by_slot.items() if len(values) > 1]


def validate_contract(contract: Any, *, user_text: str = "") -> list[SemanticIssue]:
    """Validate V1.3, or add opt-in semantic checks to valid legacy V1.2."""
    if not isinstance(contract, dict):
        return [SemanticIssue("not_object", "$", "contract must be an object")]
    version = contract.get("schema_version")
    if version == "1.2":
        old = validate_v12(contract)
        if old:
            return [SemanticIssue("schema_or_v12:" + x.code, x.path, x.message) for x in old]
    elif version == "1.3":
        errors = sorted(VALIDATOR.iter_errors(contract), key=lambda e: (list(map(str, e.path)), e.message))
        if errors:
            return [SemanticIssue("schema", "$" + "".join(f"[{p!r}]" for p in e.path), e.message)
                    for e in errors]
    else:
        return [SemanticIssue("unsupported_version", "$.schema_version", str(version))]

    issues: list[SemanticIssue] = []
    goals = contract["goals"]
    if contract["kind"] == "task" and not goals:
        issues.append(SemanticIssue("task_without_goal", "$.goals", "task needs a goal"))
    if contract["kind"] == "non_actionable" and goals:
        issues.append(SemanticIssue("non_actionable_with_goals", "$.goals", "non-actionable has goals"))
    ids = [g["id"] for g in goals]
    if len(ids) != len(set(ids)):
        issues.append(SemanticIssue("duplicate_goal_id", "$.goals", "goal IDs must be unique"))
    positions = {gid: i for i, gid in enumerate(ids)}
    text = _text(user_text)
    for i, goal in enumerate(goals):
        path = f"$.goals[{i}]"
        action = goal["action"]
        target = goal["target"]
        typ = target.get("type", "")
        family = TARGET_FAMILY.get(typ, typ)
        facts = goal["facts"]
        temporal = goal.get("temporal", []) if version == "1.3" else []
        issues.extend(_hard_eq_conflicts(facts, "field", path + ".facts"))
        issues.extend(_hard_eq_conflicts(temporal, "role", path + ".temporal"))
        if version == "1.3":
            seen_facts: set[str] = set()
            present = {f["field"] for f in facts} | {t["role"] for t in temporal}
            missing_fields = [m["field"] for m in goal.get("missing", [])]
            for j, fact in enumerate(facts):
                fp = f"{path}.facts[{j}]"
                issues.extend(SemanticIssue(x.code, x.path, x.message) for x in _fact_issues(fact, fp))
                key = json.dumps({k: v for k, v in fact.items() if k != "strength"},
                                 ensure_ascii=False, sort_keys=True)
                if key in seen_facts:
                    issues.append(SemanticIssue("duplicate_fact", fp, "same fact repeated"))
                seen_facts.add(key)
            for field in _hard_numeric_conflicts(facts):
                issues.append(SemanticIssue("contradictory_hard_facts", path + ".facts", field))
            if len(missing_fields) != len(set(missing_fields)):
                issues.append(SemanticIssue("duplicate_missing", path + ".missing", "field repeated"))
            for missing in goal.get("missing", []):
                field = missing["field"]
                if field in present or ("action_" + field) in present or ("notification_" + field) in present:
                    issues.append(SemanticIssue("missing_but_present", path + ".missing", field))
                if action in {"find", "compare"} and missing["blocks"] == "action":
                    issues.append(SemanticIssue("wrong_block_phase", path + ".missing", field))
            for j, item in enumerate(temporal):
                tp = f"{path}.temporal[{j}]"
                if item["op"] == "between" and "value_to" not in item:
                    issues.append(SemanticIssue("between_missing_end", tp, item["role"]))
                if item["op"] != "between" and "value_to" in item:
                    issues.append(SemanticIssue("unexpected_value_to", tp, item["role"]))
            if typ not in KNOWN_TYPES and typ not in TARGET_FAMILY:
                issues.append(SemanticIssue("unsupported_target_type", path + ".target.type", typ))
            if any(f["field"] in TEMPORAL_FIELDS for f in facts):
                issues.append(SemanticIssue("temporal_fact_without_role", path + ".facts",
                                            "use typed temporal constraints"))
            allowed = ({"search_date", "search_time", "arrival_time", "departure_time"}
                       if action in {"find", "compare"} else
                       {"notification_date", "notification_time"} if action == "notify" else
                       {"action_date", "action_time"})
            if action == "book" and family == "transport":
                allowed |= {"arrival_time", "departure_time"}
            for t in temporal:
                if t["role"] not in allowed:
                    issues.append(SemanticIssue("temporal_role_action_mismatch", path + ".temporal", t["role"]))
            modifiers = goal.get("modifiers", {})
            if "compare_count" in modifiers and action != "compare":
                issues.append(SemanticIssue("compare_count_action_mismatch", path + ".modifiers", action))
            if action == "compare" and "limit" in modifiers:
                issues.append(SemanticIssue("compare_uses_limit", path + ".modifiers", "use compare_count"))
            if action == "order" and "sort" in modifiers:
                issues.append(SemanticIssue("order_sort_confusion", path + ".modifiers", "sort is a find modifier"))
            for dep in goal.get("depends_on", []):
                if dep not in positions or positions[dep] >= i:
                    issues.append(SemanticIssue("non_prior_dependency", path + ".depends_on", dep))
            for fact in facts + temporal:
                ref = fact.get("value_ref")
                if ref and (ref["goal"] not in positions or positions[ref["goal"]] >= i or
                            ref["goal"] not in goal.get("depends_on", [])):
                    issues.append(SemanticIssue("invalid_reference", path, str(ref)))
            condition = goal.get("condition")
            if condition and (condition["goal"] not in positions or positions[condition["goal"]] >= i or
                              condition["goal"] not in goal.get("depends_on", [])):
                issues.append(SemanticIssue("invalid_condition_dependency", path + ".condition", str(condition)))

        if action == "notify" and family != "reminder":
            issues.append(SemanticIssue("notify_target_mismatch", path + ".target", typ))
        if action == "order" and family == "transport":
            issues.append(SemanticIssue("unsupported_action_target", path, "order transport; use find/sort or book"))
        if action == "hire" and family not in {"professional_service", "person"}:
            issues.append(SemanticIssue("unsupported_action_target", path, f"hire {typ}"))
        if action == "reschedule" and family != "appointment":
            issues.append(SemanticIssue("unsupported_action_target", path, f"reschedule {typ}"))
        if version == "1.3" and action in {"cancel", "reschedule"} and family == "appointment":
            known = {f["field"] for f in facts} | {t["role"] for t in temporal}
            absent = {m["field"] for m in goal.get("missing", [])}
            if not ({"appointment_id", "booking_id", "original_date", "original_time"} & (known | absent)):
                issues.append(SemanticIssue("missing_required_information", path,
                                            "appointment identity or original slot"))
        if action == "rent" and re.search(r"\b(bici|bicicletta|scooter|moto|automobile|auto|furgone|camper)\b",
                                           _text(target["name"])) and family != "vehicle":
            issues.append(SemanticIssue("vehicle_target_mismatch", path + ".target", typ))
        policy = goal.get("policy", {})
        confirmed = set(policy.get("confirm_before", []))
        forbidden = set(policy.get("forbid", []))
        if confirmed & forbidden or action in forbidden:
            issues.append(SemanticIssue("policy_conflict", path + ".policy", action))
        if confirmed and action in CONSEQUENTIAL and action not in confirmed:
            issues.append(SemanticIssue("policy_action_mismatch", path + ".policy", action))
        if action == "contact" and not (any(f["field"] == "message" for f in facts) or
                                        any(m["field"] == "message" for m in goal.get("missing", []))):
            issues.append(SemanticIssue("missing_required_information", path, "contact.message"))
        if action == "notify" and not (any(f["field"] == "message" for f in facts) or
                                       any(m["field"] in {"message", "referent"} for m in goal.get("missing", []))):
            issues.append(SemanticIssue("missing_required_information", path, "notify.message"))
        if action == "book" and "tavolo" in _text(target["name"]):
            present = {f["field"] for f in facts}
            present |= {t["role"] for t in temporal}
            absent = {m["field"] for m in goal.get("missing", [])}
            required = ({"date", "time", "guests"} if version == "1.2" else
                        {"action_date", "action_time", "guests"})
            if any(item not in present and item.replace("action_", "") not in absent for item in required):
                issues.append(SemanticIssue("missing_required_information", path, "table booking"))
        if action == "book" and re.search(r"\b(volo|aereo|flight)\b", _text(target["name"])):
            present = {f["field"] for f in facts} | {t["role"] for t in temporal}
            absent = {m["field"] for m in goal.get("missing", [])}
            date_slot = "date" if version == "1.2" else "action_date"
            if ("origin" not in present and "origin" not in absent) or (date_slot not in present and "date" not in absent):
                issues.append(SemanticIssue("missing_required_information", path, "flight booking"))
        if action == "compare" and user_text:
            found = re.search(r"\b(?:confronta|compara|confrontami)\s+(due|tre|quattro|cinque|sei|[2-6])\b", text)
            if found:
                count = COUNT_WORDS.get(found.group(1)) or int(found.group(1))
                actual = (goal.get("modifiers", {}).get("compare_count") if version == "1.3" else
                          next((f.get("value") for f in facts if f["field"] == "quantity" and f["op"] == "eq"), None))
                if actual != count:
                    issues.append(SemanticIssue("cardinality_lost", path, str(count)))
        if version == "1.2" and family == "transport" and action == "find":
            if any(f["field"] == "time" and f["op"] in {"before", "after"} for f in facts):
                issues.append(SemanticIssue("temporal_role_ambiguous", path + ".facts", "transport time"))
    return issues
