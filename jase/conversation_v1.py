"""Opt-in conversational pre-contract state and deterministic requirements loop.

This module does not call an LLM, a provider, or an execution API. Stage A may
propose semantic updates; the registry and this state machine decide progress.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .capability_registry_v1 import Capability, CapabilityRegistry, REGISTRY_TRAIN_V1_PATH
from .provenance_v1 import (DETERMINISTIC_DERIVATION, MODEL_INFERENCE,
                            USER_EXPLICIT, ground_value)
from .semantic_contract_v1_3 import validate_contract

PROTOCOL_VERSION = "conversation-v1"
MODES = frozenset({"CHAT", "QUESTION", "META", "TRANSLATION", "HYPOTHETICAL",
                   "ACTIONABLE_GOAL", "CONTINUATION", "ANSWER_TO_CLARIFICATION",
                   "CONFIRMATION", "REJECTION", "CANCELLATION"})
OPERATIONS = frozenset({"NO_GOAL", "OPEN_DRAFT", "UPDATE_DRAFT", "CANCEL_DRAFT",
                        "CONFIRM_DRAFT", "REJECT_DRAFT"})
NON_ACTION = frozenset({"CHAT", "QUESTION", "META", "TRANSLATION", "HYPOTHETICAL"})
TRUSTED = frozenset({USER_EXPLICIT, DETERMINISTIC_DERIVATION,
                     "TRUSTED_CONVERSATION_CONTEXT", "PROVIDER_RESULT"})
TEMPORAL = frozenset({"search_date", "search_time", "departure_time", "arrival_time",
                      "action_date", "action_time", "notification_date", "notification_time"})
CONSEQUENTIAL = frozenset({"book", "buy", "contact", "rent", "order", "hire", "cancel", "reschedule"})


@dataclass
class GoalDraft:
    draft_id: str
    revision: int
    status: str
    action: str
    target: dict[str, str]
    slots: dict[str, dict[str, Any]] = field(default_factory=dict)
    superseded_slots: list[dict[str, Any]] = field(default_factory=list)
    policy: dict[str, list[str]] = field(default_factory=dict)
    candidate_capability: str | None = None
    unresolved_references: list[dict[str, Any]] = field(default_factory=list)
    confirmation: dict[str, Any] = field(default_factory=lambda: {"status": "NOT_REQUESTED", "binding": None})


@dataclass
class ConversationState:
    version: str = PROTOCOL_VERSION
    conversation_id: str = "local"
    turn_index: int = 0
    drafts: dict[str, GoalDraft] = field(default_factory=dict)
    active_draft_ids: list[str] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    result_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    latest_mode: str | None = None
    turn_refs: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ConversationState":
        if raw.get("version") != PROTOCOL_VERSION:
            raise ValueError("Unsupported conversation state version")
        return cls(version=raw["version"], conversation_id=raw["conversation_id"],
                   turn_index=raw["turn_index"],
                   drafts={key: GoalDraft(**value) for key, value in raw["drafts"].items()},
                   active_draft_ids=list(raw["active_draft_ids"]),
                   clarification=raw.get("clarification"),
                   result_references=dict(raw.get("result_references", {})),
                   latest_mode=raw.get("latest_mode"), turn_refs=list(raw.get("turn_refs", [])))


def validate_decision(raw: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Invalid conversational protocol version")
    if raw.get("mode") not in MODES or raw.get("operation") not in OPERATIONS:
        raise ValueError("Unknown mode or operation")
    if raw["mode"] in NON_ACTION and raw["operation"] != "NO_GOAL":
        raise ValueError("Non-action discourse cannot create a goal")
    if raw["mode"] == "ACTIONABLE_GOAL" and raw["operation"] not in {"OPEN_DRAFT", "UPDATE_DRAFT"}:
        raise ValueError("Actionable mode needs a draft operation")
    if raw["operation"] == "OPEN_DRAFT":
        target = raw.get("target")
        if (not isinstance(raw.get("action"), str) or not isinstance(target, dict)
                or set(target) != {"name", "type"} or
                any(not isinstance(target[key], str) or not target[key] for key in target)):
            raise ValueError("Opening a draft needs action and typed target")
    updates = raw.get("updates", [])
    if not isinstance(updates, list):
        raise ValueError("Updates must be a list")
    for item in updates:
        if (not isinstance(item, dict) or set(item) - {"slot", "value", "value_to", "op"}
                or not isinstance(item.get("slot"), str) or "value" not in item
                or item.get("op", "eq") not in {"eq", "neq", "before", "after", "between", "lt", "lte", "gt", "gte", "near", "contains", "excludes", "within", "in", "not_in"}):
            raise ValueError("Invalid semantic update")
        if item.get("op") == "between" and "value_to" not in item:
            raise ValueError("Between update needs an end value")
        if not item["slot"].startswith(("fact.", "temporal.", "external.", "modifier.", "policy.")):
            raise ValueError("Unsupported slot namespace")
    reference = raw.get("reference")
    if reference is not None:
        if (not isinstance(reference, dict) or reference.get("kind") not in {"ordinal", "cheapest"}
                or (reference["kind"] == "ordinal" and
                    (not isinstance(reference.get("ordinal"), int) or reference["ordinal"] < 1))):
            raise ValueError("Invalid result reference")
    additional = raw.get("additional_goals", [])
    if additional and raw["operation"] != "OPEN_DRAFT":
        raise ValueError("Additional goals require OPEN_DRAFT")
    if not isinstance(additional, list):
        raise ValueError("Additional goals must be a list")
    for goal in additional:
        if (not isinstance(goal, dict) or set(goal) != {"action", "target", "updates"}
                or not isinstance(goal["action"], str) or not goal["action"]
                or not isinstance(goal["target"], dict) or set(goal["target"]) != {"name", "type"}
                or not isinstance(goal["updates"], list)):
            raise ValueError("Invalid additional goal")
        validate_decision({"protocol_version": PROTOCOL_VERSION, "mode": "ACTIONABLE_GOAL",
                           "operation": "OPEN_DRAFT", "action": goal["action"],
                           "target": goal["target"], "updates": goal["updates"]})
    return raw


def _ground(slot: str, value: Any, text: str, timestamp: datetime | None) -> dict[str, Any]:
    field_name = slot.rsplit(".", 1)[-1]
    if slot == "policy.forbid" and value in {"book", "buy", "rent", "hire"}:
        forms = {"book": r"prenot", "buy": r"compr|acquist|ordin", "rent": r"nolegg|affitt",
                 "hire": r"assum|ingagg|incaric"}
        if re.search(r"\b(?:non|senza|niente|nessuna?)\b.{0,24}\b(?:" + forms[value] + r")\w*", text, re.I):
            return {"value": value, "op": "eq", "provenance": DETERMINISTIC_DERIVATION,
                    "derived_from": "explicit_negative_policy"}
    if slot in {"temporal.search_date", "temporal.action_date", "temporal.notification_date"} and timestamp:
        relative = {"oggi": 0, "domani": 1, "dopodomani": 2}
        if isinstance(value, str):
            match = re.search(r"\b(oggi|domani|dopodomani)\b", text, re.I)
            if match and value.casefold() == match.group(1).casefold():
                concrete = (timestamp.date() + timedelta(days=relative[match.group(1).casefold()])).isoformat()
                return {"value": concrete, "op": "eq", "provenance": DETERMINISTIC_DERIVATION,
                        "derived_from": match.group(1), "source_span": [match.start(), match.end()]}
    evidence = ground_value(field_name, value, text)
    return {"value": value, "op": "eq", "provenance": evidence.source,
            "source_span": list(evidence.span) if evidence.span else None,
            "note": evidence.note}


def _material_digest(draft: GoalDraft) -> str:
    data = {"action": draft.action, "target": draft.target, "slots": draft.slots,
            "policy": draft.policy}
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _active(state: ConversationState, draft_id: str | None) -> GoalDraft:
    if draft_id:
        if draft_id not in state.active_draft_ids:
            raise ValueError("Draft is not active")
        return state.drafts[draft_id]
    if len(state.active_draft_ids) != 1:
        raise ValueError("Contextual reply needs one unambiguous active draft")
    return state.drafts[state.active_draft_ids[0]]


def apply_decision(state: ConversationState, raw: dict[str, Any], user_text: str,
                   *, timestamp: datetime | None = None) -> GoalDraft | None:
    decision = validate_decision(raw)
    state.turn_index += 1
    turn_id = f"t{state.turn_index}"
    state.turn_refs.append({"turn_id": turn_id, "text": user_text})
    state.latest_mode = decision["mode"]
    operation = decision["operation"]
    if operation == "NO_GOAL":
        return None
    if operation == "OPEN_DRAFT":
        draft_id = f"d{len(state.drafts) + 1}"
        draft = GoalDraft(draft_id, 1, "COLLECTING", decision["action"], decision["target"])
        state.drafts[draft_id] = draft
        state.active_draft_ids.append(draft_id)
    else:
        draft = _active(state, decision.get("draft_id"))
    if operation in {"CANCEL_DRAFT", "REJECT_DRAFT"}:
        draft.status = "CANCELLED"
        draft.confirmation = {"status": "DENIED", "binding": None}
        state.active_draft_ids.remove(draft.draft_id)
        state.clarification = None
        return draft
    if operation == "CONFIRM_DRAFT":
        pending = draft.confirmation
        if pending["status"] != "PENDING" or pending["binding"] != _material_digest(draft):
            raise ValueError("Confirmation does not match an exact pending draft")
        draft.confirmation = {"status": "GRANTED", "binding": pending["binding"], "turn_id": turn_id}
        return draft
    if operation == "UPDATE_DRAFT" and decision.get("action") and decision["action"] != draft.action:
        raise ValueError("Action change requires an explicit new draft")
    if operation == "UPDATE_DRAFT" and decision.get("target") and decision["target"] != draft.target:
        raise ValueError("Target change requires an explicit new draft")
    changed = False
    reference = decision.get("reference")
    if reference:
        candidates = [item for item in state.result_references.values()
                      if item.get("draft_id") == draft.draft_id and item.get("trusted") is True]
        if reference["kind"] == "ordinal":
            matches = [item for item in candidates if item.get("ordinal") == reference["ordinal"]]
        else:
            priced = [item for item in candidates if isinstance(item.get("price"), (int, float))]
            matches = [item for item in priced if item["price"] == min(
                (other["price"] for other in priced), default=float("inf"))]
        if len(matches) != 1:
            draft.unresolved_references.append({"turn_id": turn_id, "reference": reference,
                                                "reason": "not_uniquely_resolved"})
        else:
            match = matches[0]
            slot = "fact.selection"
            previous = draft.slots.get(slot)
            if previous is not None:
                draft.superseded_slots.append({"slot": slot, **previous, "superseded_at": turn_id})
            draft.slots[slot] = {"value": match["jase_result_ref"], "op": "eq",
                                 "provenance": "PROVIDER_RESULT", "turn_id": turn_id,
                                 "provider_result_ref": match.get("provider_result_ref")}
            changed = True
    for update in decision.get("updates", []):
        slot = update["slot"]
        grounded = _ground(slot, update["value"], user_text, timestamp)
        grounded["op"] = update.get("op", "eq")
        if "value_to" in update:
            grounded["value_to"] = update["value_to"]
        grounded["turn_id"] = turn_id
        if grounded["provenance"] not in TRUSTED:
            # Preserve tentative interpretations for debugging, never for finalization.
            grounded["tentative"] = True
        previous = draft.slots.get(slot)
        if previous != grounded:
            if previous is not None:
                draft.superseded_slots.append({"slot": slot, **previous, "superseded_at": turn_id})
            draft.slots[slot] = grounded
            changed = True
            if slot == "policy.forbid" and grounded["provenance"] in TRUSTED:
                actions = set(draft.policy.get("forbid", []))
                actions.add(str(grounded["value"]))
                draft.policy["forbid"] = sorted(actions)
    if changed:
        draft.revision += 1
        if draft.confirmation["status"] in {"PENDING", "GRANTED"}:
            draft.confirmation = {"status": "SUPERSEDED", "binding": None}
    draft.status = "COLLECTING"
    for goal in decision.get("additional_goals", []):
        other_id = f"d{len(state.drafts) + 1}"
        other = GoalDraft(other_id, 1, "COLLECTING", goal["action"], goal["target"])
        for update in goal["updates"]:
            grounded = _ground(update["slot"], update["value"], user_text, timestamp)
            grounded["op"] = update.get("op", "eq")
            grounded["turn_id"] = turn_id
            if "value_to" in update:
                grounded["value_to"] = update["value_to"]
            if grounded["provenance"] not in TRUSTED:
                grounded["tentative"] = True
            other.slots[update["slot"]] = grounded
        state.drafts[other_id] = other
        state.active_draft_ids.append(other_id)
    return draft


def register_results(state: ConversationState, draft_id: str, results: list[dict[str, Any]]) -> None:
    """Import normalized read-only results with stable session-local identities."""
    if draft_id not in state.active_draft_ids:
        raise ValueError("Results require an active draft")
    for ordinal, item in enumerate(results, 1):
        ref = item.get("jase_result_ref")
        if not isinstance(ref, str) or not ref or item.get("trusted") is not True:
            raise ValueError("Untrusted or unidentified result")
        if ref in state.result_references:
            raise ValueError("Duplicate result identity")
        state.result_references[ref] = {"jase_result_ref": ref, "draft_id": draft_id,
                                         "ordinal": ordinal, "provider": item.get("provider"),
                                         "provider_result_ref": item.get("provider_result_ref"),
                                         "price": item.get("price"), "trusted": True}


def _resolve_capability(registry: CapabilityRegistry, draft: GoalDraft) -> Capability | None:
    candidate = {"action": draft.action, "target": draft.target}
    _, cap = registry.resolve(candidate)
    return cap


def clarification_plan(state: ConversationState, draft_id: str, *,
                       registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    registry = registry or CapabilityRegistry.load(REGISTRY_TRAIN_V1_PATH)
    draft = _active(state, draft_id)
    cap = _resolve_capability(registry, draft)
    plan: dict[str, Any] = {"draft_id": draft_id, "capability_id": cap.capability_id if cap else None,
                            "ask_user": [], "discover_read_only": [], "derive": [],
                            "ambiguous": [], "blocked": [], "optional_absent": [],
                            "reason_codes": [], "ready_to_finalize": False}
    if cap is None:
        plan["blocked"].append("HOLD_NO_CAPABILITY")
        return plan
    draft.candidate_capability = cap.capability_id
    if cap.confirmation_required:
        actions = set(draft.policy.get("confirm_before", []))
        actions.add(draft.action)
        draft.policy["confirm_before"] = sorted(actions)
    for slot in sorted(cap.required_slots):
        present = draft.slots.get(slot)
        if present and present["provenance"] in TRUSTED and not present.get("tentative"):
            continue
        if present:
            plan["ambiguous"].append(slot)
            continue
        if slot == "fact.selection" and draft.action in CONSEQUENTIAL:
            discovery = [spec for spec in registry.capabilities if spec.action == "find" and
                         spec.target_type == draft.target["type"] and
                         spec.matches("find", draft.target["type"], draft.target["name"])]
            if len(discovery) == 1:
                prerequisite_missing = sorted(s for s in discovery[0].required_slots if
                    s not in draft.slots or draft.slots[s]["provenance"] not in TRUSTED)
                plan["ask_user"].extend(prerequisite_missing)
                plan["discover_read_only"].append({"slot": slot, "capability_id": discovery[0].capability_id,
                                                    "requires": prerequisite_missing})
                continue
        plan["ask_user"].append(slot)
    plan["ask_user"] = sorted(set(plan["ask_user"]))
    plan["optional_absent"] = sorted(cap.optional_slots - draft.slots.keys())
    if cap.confirmation_required:
        binding = _material_digest(draft)
        if draft.confirmation != {"status": "GRANTED", "binding": binding} and not (
                draft.confirmation.get("status") == "GRANTED" and draft.confirmation.get("binding") == binding):
            plan["reason_codes"].append("CONFIRMATION_REQUIRED")
    if plan["ambiguous"]:
        plan["reason_codes"].append("AMBIGUOUS_INFORMATION")
    if plan["ask_user"]:
        plan["reason_codes"].append("MISSING_USER_INFORMATION")
    if plan["discover_read_only"]:
        plan["reason_codes"].append("PREREQUISITE_DISCOVERY")
    plan["ready_to_finalize"] = not (plan["ask_user"] or plan["discover_read_only"] or
                                      plan["ambiguous"] or plan["blocked"] or plan["reason_codes"])
    draft.status = ("READY_TO_FINALIZE" if plan["ready_to_finalize"] else
                    "DISCOVERING" if plan["discover_read_only"] and not plan["ask_user"] else
                    "AWAITING_USER" if plan["ask_user"] or plan["ambiguous"] else "COLLECTING")
    state.clarification = plan if not plan["ready_to_finalize"] else None
    return plan


def render_clarification(plan: dict[str, Any]) -> str:
    """Deterministic fallback wording; an optional Stage B may only rephrase."""
    names = {"fact.origin": "da dove parti", "fact.destination": "dove vuoi arrivare",
             "fact.location": "in quale zona", "external.party_size": "per quante persone",
             "temporal.search_date": "per quale giorno", "temporal.action_date": "per quale giorno",
             "temporal.action_time": "a che ora", "temporal.notification_time": "a che ora",
             "fact.message": "che cosa devo ricordarti", "fact.selection": "quale risultato scegli"}
    fields = [names.get(slot, f"il valore di {slot}") for slot in plan["ask_user"] + plan["ambiguous"]]
    if fields:
        return "Mi dici " + " e ".join(fields[:2]) + "?"
    if plan["discover_read_only"]:
        return "Cerco prima le opzioni disponibili; poi potrai scegliere."
    if "CONFIRMATION_REQUIRED" in plan["reason_codes"]:
        return "Vuoi confermare questa operazione con i dettagli mostrati?"
    return "Non posso proseguire con questa richiesta."


def request_confirmation(draft: GoalDraft) -> None:
    draft.confirmation = {"status": "PENDING", "binding": _material_digest(draft)}


def finalize_draft(state: ConversationState, draft_id: str, *,
                   registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    registry = registry or CapabilityRegistry.load(REGISTRY_TRAIN_V1_PATH)
    plan = clarification_plan(state, draft_id, registry=registry)
    if not plan["ready_to_finalize"]:
        raise ValueError("Draft is not ready to finalize")
    draft = state.drafts[draft_id]
    facts = []
    temporal = []
    modifiers: dict[str, Any] = {}
    for slot, item in sorted(draft.slots.items()):
        namespace, field_name = slot.split(".", 1)
        if namespace == "fact":
            fact = {"field": field_name, "op": item["op"], "value": item["value"], "strength": "hard"}
            if "value_to" in item:
                fact["value_to"] = item["value_to"]
            facts.append(fact)
        elif namespace == "temporal":
            time_item = {"role": field_name, "op": item["op"], "value": item["value"], "strength": "hard"}
            if "value_to" in item:
                time_item["value_to"] = item["value_to"]
            temporal.append(time_item)
        elif namespace == "modifier":
            modifiers[field_name] = item["value"]
    goal: dict[str, Any] = {"id": "g1", "action": draft.action, "target": draft.target, "facts": facts}
    if temporal:
        goal["temporal"] = temporal
    if modifiers:
        goal["modifiers"] = modifiers
    cap = _resolve_capability(registry, draft)
    if cap and cap.confirmation_required:
        draft.policy.setdefault("confirm_before", [draft.action])
    if draft.policy:
        goal["policy"] = draft.policy
    if any(slot.startswith("external.") for slot in draft.slots):
        contract_context = [{"scope": "conversation", "field": slot.split(".", 1)[1],
                             "op": "eq", "value": item["value"]}
                            for slot, item in sorted(draft.slots.items()) if slot.startswith("external.")]
    contract = {"schema_version": "1.3", "kind": "task", "goals": [goal]}
    if any(slot.startswith("external.") for slot in draft.slots):
        contract["context"] = contract_context
    issues = validate_contract(contract)
    if issues:
        raise ValueError("Final GoalContract invalid: " + ", ".join(issue.code for issue in issues))
    draft.status = "FINALIZED"
    state.active_draft_ids.remove(draft_id)
    state.clarification = None
    return contract


def planner_context(state: ConversationState, draft_id: str) -> dict[str, Any]:
    """Bind previously grounded user values for the existing dry-run Planner.

    This does not grant an execution permission. The caller must still invoke
    Planner 1.3 and inspect HOLD/READY and its bound provenance.
    """
    draft = state.drafts[draft_id]
    trusted_context = {slot.split(".", 1)[1]: item["value"] for slot, item in draft.slots.items()
                       if item["provenance"] in TRUSTED}
    external = {slot: {"value": item["value"], "trusted": True,
                       "source": "TRUSTED_CONVERSATION_CONTEXT"}
                for slot, item in draft.slots.items() if slot.startswith("external.") and
                item["provenance"] in TRUSTED}
    return {"trusted_context": trusted_context, "external_bindings_by_goal": {"g1": external},
            "confirmation_by_goal": {"g1": draft.confirmation.get("status") == "GRANTED" and
                                     draft.confirmation.get("binding") == _material_digest(draft)}}


def plan_finalized(state: ConversationState, draft_id: str, contract: dict[str, Any], *,
                   registry: CapabilityRegistry | None = None) -> dict[str, Any]:
    """Use the historical fail-closed Planner after deterministic finalization."""
    from .planner_v1_3 import plan_contract

    draft = state.drafts[draft_id]
    if draft.status != "FINALIZED":
        raise ValueError("Planner requires a finalized draft")
    context = planner_context(state, draft_id)
    user_text = "\n".join(turn["text"] for turn in state.turn_refs)
    result = plan_contract(contract, user_text, registry=registry or CapabilityRegistry.load(REGISTRY_TRAIN_V1_PATH),
                           **context)
    if result.get("execution_permitted") is not False:
        raise RuntimeError("Planner safety boundary changed")
    return result
