"""Opt-in Planner 1.3 → typed real read-only train timetable handoff.

No booking, purchase, account, payment, passenger data or side-effect path.
The user text is used to prove constraints and provenance, never to silently
replace a model-emitted semantic value.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from .capability_registry_v1 import CapabilityRegistry, REGISTRY_TRAIN_V1_PATH
from .planner_v1_3 import plan_contract
from .train_provider_v1 import (TimeConstraint, TrainSearchProvider, TrainSearchRequest,
                                UnresolvedStation, UnsupportedConstraint)

VERSION = "train-vertical-slice-1"
TZ = ZoneInfo("Europe/Rome")
MONTHS = {"gennaio": 1, "febbraio": 2, "marzo": 3, "aprile": 4, "maggio": 5,
          "giugno": 6, "luglio": 7, "agosto": 8, "settembre": 9, "ottobre": 10,
          "novembre": 11, "dicembre": 12}
WEEKDAYS = {"lunedi": 0, "martedi": 1, "mercoledi": 2, "giovedi": 3,
            "venerdi": 4, "sabato": 5, "domenica": 6}
TRUSTED = {"USER_EXPLICIT", "TRUSTED_CONVERSATION_CONTEXT", "DETERMINISTIC_DERIVATION"}
DAYPART = {"mattina": ("06:00", "11:59"), "mattino": ("06:00", "11:59"),
           "pomeriggio": ("12:00", "17:59"), "sera": ("18:00", "23:59"),
           "serata": ("18:00", "23:59")}


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", text.casefold()).encode("ascii", "ignore").decode().split())


def _date(value: str, anchor: date) -> date:
    raw = _norm(value)
    if raw == "oggi":
        return anchor
    if raw == "domani":
        return anchor + timedelta(days=1)
    if raw == "dopodomani":
        return anchor + timedelta(days=2)
    if raw in WEEKDAYS:
        return anchor + timedelta(days=(WEEKDAYS[raw] - anchor.weekday()) % 7)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return date.fromisoformat(raw)
    match = re.fullmatch(r"(\d{1,2})\s+([a-z]+)(?:\s+(\d{4}))?", raw)
    if match and match[2] in MONTHS:
        year = int(match[3]) if match[3] else anchor.year
        result = date(year, MONTHS[match[2]], int(match[1]))
        if not match[3] and result < anchor:
            result = date(year + 1, MONTHS[match[2]], int(match[1]))
        return result
    raise UnsupportedConstraint("unresolved_travel_date")


def _clock(value: Any) -> str:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value))
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        raise UnsupportedConstraint("invalid_time_value")
    return f"{int(match[1]):02d}:{match[2]}"


def _shell(status: str, codes: list[str], planner: dict | None = None) -> dict:
    return {"version": VERSION, "status": status, "reason_codes": sorted(set(codes)),
            "planner": planner, "provider_request": None, "provider_results": [],
            "provider_called": False, "provider_read_permitted": False,
            "execution_permitted": False, "external_side_effect_permitted": False}


def _evidence(slots: dict, key: str) -> tuple[Any, dict]:
    item = slots.get(key)
    if not isinstance(item, dict) or not isinstance(item.get("provenance"), dict):
        raise UnsupportedConstraint("missing_provenance:" + key)
    provenance = item["provenance"]
    if provenance.get("source") not in TRUSTED or provenance.get("value") != item.get("value"):
        raise UnsupportedConstraint("untrusted_provider_input:" + key)
    return item["value"], provenance


def _declared_coverage(goal: dict, user_text: str) -> list[str]:
    """Detect high-impact explicit requirements omitted from the model output."""
    text = _norm(user_text)
    fields = {item["field"] for item in goal["facts"]}
    temporal = {item["role"] for item in goal.get("temporal", [])}
    codes = []
    if re.search(r"\b(?:regional[ei])\b", text) and "train_category" not in fields and "regionale" not in _norm(goal["target"]["name"]):
        codes.append("HOLD_OMITTED_TRAIN_CATEGORY")
    if re.search(r"\b(?:alta velocita|frecciarossa|intercity)\b", text) and not ({"train_category", "train_service"} & fields):
        codes.append("HOLD_OMITTED_TRAIN_SERVICE")
    if re.search(r"\b(?:dirett[oaie]|senza cambi|senza coincidenze)\b", text) and not ({"stops", "direct"} & fields):
        codes.append("HOLD_OMITTED_DIRECT_CONSTRAINT")
    if re.search(r"\b(?:euro|costa|costino|prezzo|caro|economico)\b", text) and "price" not in fields and goal.get("modifiers", {}).get("sort", {}).get("field") != "price":
        codes.append("HOLD_OMITTED_PRICE_CONSTRAINT")
    if re.search(r"\b(?:arriv(?:are|i|o)|essere\s+(?:a|in))\b.{0,40}\b(?:entro|prima delle|non oltre)\b", text) and "arrival_time" not in temporal:
        codes.append("HOLD_OMITTED_ARRIVAL_DEADLINE")
    if re.search(r"\b(?:mattina|mattino|pomeriggio|sera|serata)\b", text) and not ({"search_time", "departure_time", "arrival_time"} & temporal):
        codes.append("HOLD_OMITTED_DAYPART")
    return codes


def run_train_search(raw: str | dict, user_text: str, *, request_at: datetime,
                     provider: TrainSearchProvider | None = None,
                     trusted_context: dict | None = None) -> dict:
    """Return HOLD or normalized timetable results; never permit a write."""
    if request_at.tzinfo is None:
        return _shell("HOLD", ["request_timestamp_without_timezone"])
    registry = CapabilityRegistry.load(REGISTRY_TRAIN_V1_PATH)
    planner = plan_contract(raw, user_text, registry=registry, trusted_context=trusted_context)
    if planner["planner_status"] == "NO_ACTION":
        return _shell("NO_ACTION", [], planner)
    if planner["planner_status"] != "READY_FOR_DRY_RUN":
        return _shell("HOLD", planner["reason_codes"] or [planner["planner_status"]], planner)
    try:
        contract = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError):
        return _shell("HOLD", ["invalid_contract_json"], planner)
    if len(contract.get("goals", [])) != 1 or len(planner["execution_plan_steps"]) != 1:
        return _shell("HOLD", ["HOLD_MULTI_GOAL_UNSUPPORTED"], planner)
    goal = contract["goals"][0]
    step = planner["execution_plan_steps"][0]
    if (step["capability"] != "find.train" or goal["action"] != "find" or
            goal["target"]["type"] != "transport"):
        return _shell("HOLD", ["HOLD_NO_READ_ONLY_TRAIN_CAPABILITY"], planner)
    if goal.get("depends_on") or goal.get("condition") or goal.get("missing"):
        return _shell("HOLD", ["HOLD_DEPENDENCY_OR_MISSING_INFORMATION"], planner)
    if goal.get("policy", {}).get("confirm_before"):
        return _shell("HOLD", ["HOLD_CONFIRMATION_POLICY_FOR_READ"], planner)
    coverage = _declared_coverage(goal, user_text)
    if coverage:
        return _shell("HOLD", coverage, planner)
    slots = step["binding"]
    provenance: dict[str, dict] = {}
    try:
        origin, provenance["origin"] = _evidence(slots, "fact.origin")
        destination, provenance["destination"] = _evidence(slots, "fact.destination")
        date_value, date_evidence = _evidence(slots, "temporal.search_date")
        travel_date = _date(str(date_value), request_at.astimezone(TZ).date())
        provenance["travel_date"] = {"source": "DETERMINISTIC_DERIVATION",
                                      "derived_from": date_evidence,
                                      "request_date": request_at.astimezone(TZ).date().isoformat(),
                                      "value": travel_date.isoformat()}
        if not isinstance(origin, str) or not isinstance(destination, str):
            raise UnsupportedConstraint("station_name_not_text")
        allowed_facts = {"origin", "destination", "stops", "direct", "train_category", "duration",
                         "price", "train_service", "travel_class"}
        facts = {item["field"]: item for item in goal["facts"]}
        unknown = set(facts) - allowed_facts
        if unknown:
            raise UnsupportedConstraint("unmapped_fact:" + ",".join(sorted(unknown)))
        if {"price", "train_service", "travel_class"} & set(facts):
            raise UnsupportedConstraint("price_brand_or_class_unavailable")
        direct = False
        if "stops" in facts:
            value, provenance["direct_only"] = _evidence(slots, "fact.stops")
            if value != 0 or facts["stops"]["op"] != "eq":
                raise UnsupportedConstraint("only_zero_changes_supported")
            direct = True
        if "direct" in facts:
            value, provenance["direct_only"] = _evidence(slots, "fact.direct")
            if value is not True:
                raise UnsupportedConstraint("direct_value_not_true")
            direct = True
        category = None
        if "train_category" in facts:
            category, provenance["train_category"] = _evidence(slots, "fact.train_category")
            if category not in {"regional", "regionale"}:
                raise UnsupportedConstraint("train_category_not_available")
            category = "regional"
        if "regionale" in _norm(goal["target"]["name"]) and category is None:
            raise UnsupportedConstraint("regional_qualifier_not_bound_as_filter")
        max_duration = None
        if "duration" in facts:
            value, provenance["max_duration_minutes"] = _evidence(slots, "fact.duration")
            fact = facts["duration"]
            if type(value) not in (int, float) or fact["op"] not in {"lt", "lte"} or fact.get("unit") not in {"minute", "minutes", "min"}:
                raise UnsupportedConstraint("duration_unit_or_operator_unsupported")
            max_duration = int(value) - (1 if fact["op"] == "lt" else 0)
        constraints = []
        for item in goal.get("temporal", []):
            role = item["role"]
            if role == "search_date":
                continue
            value, evidence = _evidence(slots, "temporal." + role)
            if role == "search_time":
                daypart = DAYPART.get(_norm(str(value)))
                if item["op"] != "eq" or daypart is None:
                    raise UnsupportedConstraint("unmapped_search_time")
                constraints.append(TimeConstraint("departure_time", "between", *daypart, inclusive=True))
                provenance["departure_daypart"] = {"source": "DETERMINISTIC_DERIVATION", "derived_from": evidence,
                                                    "value": daypart}
            elif role in {"departure_time", "arrival_time"}:
                if item["op"] not in {"eq", "before", "after", "between"}:
                    raise UnsupportedConstraint("temporal_operator_unsupported")
                inclusive = bool(re.search(r"\b(?:entro|non oltre|non prima)\b", _norm(user_text)))
                end = _clock(item["value_to"]) if item["op"] == "between" else None
                constraints.append(TimeConstraint(role, item["op"], _clock(value), end, inclusive))
                provenance[role] = evidence
            else:
                raise UnsupportedConstraint("unmapped_temporal_role")
        modifiers = goal.get("modifiers", {})
        if set(modifiers) - {"limit", "sort"}:
            raise UnsupportedConstraint("unmapped_modifier")
        limit = modifiers.get("limit", 10)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise UnsupportedConstraint("invalid_limit")
        if "limit" in modifiers:
            _, provenance["limit"] = _evidence(slots, "modifiers.limit")
        else:
            provenance["limit"] = {"source": "CAPABILITY_DEFAULT", "value": 10}
        sorting = modifiers.get("sort", {"field": "departure_time", "direction": "asc"})
        if sorting["field"] not in {"departure_time", "arrival_time", "duration"}:
            raise UnsupportedConstraint("sort_field_unavailable")
        if "sort" in modifiers:
            term = {"departure_time": r"partenz|parte|parti", "arrival_time": r"arriv",
                    "duration": r"durata|percorso|veloc"}[sorting["field"]]
            normalized_text = _norm(user_text)
            direction_term = (r"\b(?:prima|presto|minore|breve|crescente|iniziale)\b" if sorting["direction"] == "asc"
                              else r"\b(?:tardi|tardiva|maggiore|lunga|decrescente|ultima)\b")
            if not re.search(term, normalized_text) or not re.search(direction_term, normalized_text):
                raise UnsupportedConstraint("sort_not_grounded_in_request")
            provenance["sort"] = {"source": "DETERMINISTIC_DERIVATION", "derived_from": user_text,
                                  "value": sorting}
        else:
            provenance["sort"] = {"source": "CAPABILITY_DEFAULT", "value": sorting}
        if provider is None:
            return _shell("HOLD", ["provider_not_configured"], planner)
        # Resolve before creating a read permission. No guessed station IDs.
        origin_ids = provider.resolve_station(origin)
        destination_ids = provider.resolve_station(destination)
        provenance["origin_stop_ids"] = {"source": "DETERMINISTIC_DERIVATION", "derived_from": provenance["origin"], "value": origin_ids}
        provenance["destination_stop_ids"] = {"source": "DETERMINISTIC_DERIVATION", "derived_from": provenance["destination"], "value": destination_ids}
        request = TrainSearchRequest(origin, destination, travel_date, tuple(constraints), direct, category,
                                     max_duration, limit, sorting["field"], sorting["direction"])
        results = provider.search(request)
    except UnresolvedStation as exc:
        return _shell("HOLD", ["HOLD_UNRESOLVED_REFERENCE:" + str(exc)], planner)
    except (UnsupportedConstraint, ValueError, KeyError) as exc:
        return _shell("HOLD", ["HOLD_UNSUPPORTED_CONSTRAINT:" + str(exc)], planner)
    outcome = _shell("READY_RESULTS", [], planner)
    outcome["provider_request"] = {"origin": origin, "destination": destination,
                                   "travel_date": travel_date.isoformat(),
                                   "time_constraints": [vars(item) for item in constraints],
                                   "direct_only": direct, "train_category": category,
                                   "max_duration_minutes": max_duration, "limit": limit,
                                   "sort": sorting, "field_provenance": provenance}
    outcome["provider_results"] = [item.as_dict() for item in results]
    outcome["provider_called"] = True
    outcome["provider_read_permitted"] = True
    outcome["provider_source_kind"] = provider.source_kind
    outcome["coverage"] = getattr(provider, "coverage", "unknown")
    if not results and not direct:
        # The pilot reconstructs only direct trips. An empty subset cannot
        # assert that no connecting journey exists for an unrestricted query.
        outcome["status"] = "HOLD_INCOMPLETE_COVERAGE"
        outcome["reason_codes"] = ["no_direct_result_does_not_exclude_connections"]
    return outcome
