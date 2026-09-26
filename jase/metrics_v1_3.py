"""Offline typed-role metrics; exact equality remains separate from safety labels."""
from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .canonical_semantics import canonicalize
from .migrate_contract_v1_3 import MigrationHold, upgrade_contract
from .semantic_contract_v1_3 import validate_contract
from .semantic_guard_v1_3 import ACCEPT, check_contract


def project(contract: Any) -> tuple[dict | None, str | None]:
    if not isinstance(contract, dict):
        return None, "not_object"
    if contract.get("schema_version") == "1.2":
        try:
            return upgrade_contract(contract), None
        except (MigrationHold, TypeError, KeyError) as exc:
            return None, f"v12_projection_hold:{exc}"
    if contract.get("schema_version") == "1.3":
        issues = validate_contract(contract)
        return (contract, None) if not issues else (None, "v13_invalid:" + ",".join(i.code for i in issues))
    return None, "unsupported_version"


def _items(goal: dict, key: str) -> set[str]:
    return {json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in goal.get(key, [])}


def score(expected: dict, predicted_raw: Any, user_text: str) -> dict[str, Any]:
    """Score without treating a wrong exact match as necessarily dangerous."""
    gold, gold_hold = project(expected)
    predicted, candidate_hold = project(predicted_raw)
    result = {"eligible": gold is not None, "gold_projection_hold": gold_hold,
              "candidate_projection_hold": candidate_hold, "native_v13":
              isinstance(predicted_raw, dict) and predicted_raw.get("schema_version") == "1.3"}
    if gold is None:
        return result
    result.update({"literal_equal": False, "canonical_equal": False,
                   "action_correct": 0, "target_correct": 0, "goal_count_correct": False,
                   "fact_tp": 0, "fact_gold": sum(len(g["facts"]) for g in gold["goals"]),
                   "fact_pred": 0, "temporal_tp": 0,
                   "temporal_gold": sum(len(g.get("temporal", [])) for g in gold["goals"]),
                   "temporal_pred": 0, "modifier_correct": 0, "context_correct": False,
                   "policy_correct": 0, "missing_correct": 0,
                   "semantic_invariant_valid": False, "guard_accept": False,
                   "guard_status": None, "guard_codes": [],
                   "gold_goals": len(gold["goals"])})
    try:
        decision = check_contract(predicted_raw, user_text)
        result["guard_accept"] = decision.status == ACCEPT
        result["guard_status"] = decision.status
        result["guard_codes"] = list(decision.codes)
    except (KeyError, TypeError, ValueError):
        result["guard_status"] = "REJECT_INVALID"
        result["guard_codes"] = ["guard_exception"]
    if predicted is None:
        return result
    issues = validate_contract(predicted, user_text=user_text)
    result["semantic_invariant_valid"] = not issues
    result["semantic_issue_codes"] = [i.code for i in issues]
    result["literal_equal"] = gold == predicted
    try:
        result["canonical_equal"] = canonicalize(gold) == canonicalize(predicted)
    except ValueError:
        pass
    result["goal_count_correct"] = len(gold["goals"]) == len(predicted["goals"])
    got_by_id = {g["id"]: g for g in predicted["goals"]}
    result["fact_pred"] = sum(len(g["facts"]) for g in predicted["goals"])
    result["temporal_pred"] = sum(len(g.get("temporal", [])) for g in predicted["goals"])
    for want in gold["goals"]:
        got = got_by_id.get(want["id"])
        if not got:
            continue
        result["action_correct"] += want["action"] == got["action"]
        result["target_correct"] += want["target"] == got["target"]
        result["fact_tp"] += len(_items(want, "facts") & _items(got, "facts"))
        result["temporal_tp"] += len(_items(want, "temporal") & _items(got, "temporal"))
        result["modifier_correct"] += want.get("modifiers", {}) == got.get("modifiers", {})
        result["policy_correct"] += want.get("policy", {}) == got.get("policy", {})
        result["missing_correct"] += _items(want, "missing") == _items(got, "missing")
    result["context_correct"] = {json.dumps(x, ensure_ascii=False, sort_keys=True)
                                 for x in gold.get("context", [])} == {
                                     json.dumps(x, ensure_ascii=False, sort_keys=True)
                                     for x in predicted.get("context", [])}
    result["gold_goals"] = len(gold["goals"])
    return result


def aggregate(rows: list[dict]) -> dict[str, Any]:
    eligible = [r for r in rows if r["eligible"]]
    counts = Counter()
    statuses = Counter()
    for r in eligible:
        for key in ("literal_equal", "canonical_equal", "goal_count_correct", "action_correct",
                    "target_correct", "fact_tp", "fact_gold", "fact_pred", "temporal_tp",
                    "temporal_gold", "temporal_pred", "modifier_correct", "context_correct",
                    "policy_correct", "missing_correct", "semantic_invariant_valid", "guard_accept",
                    "gold_goals"):
            counts[key] += r[key]
        statuses[r["guard_status"]] += 1
        counts["accepted_not_canonical"] += r["guard_accept"] and not r["canonical_equal"]
        counts["held_canonical"] += not r["guard_accept"] and r["canonical_equal"]
    def ratio(a, b):
        return round(a / b, 4) if b else None
    return {"cases": len(rows), "eligible": len(eligible),
            "gold_projection_holds": len(rows) - len(eligible),
            "candidate_projection_holds": sum(bool(r.get("candidate_projection_hold")) for r in eligible),
            "counts": dict(counts), "guard_statuses": dict(statuses),
            "rates": {"canonical_exact": ratio(counts["canonical_equal"], len(eligible)),
                      "action": ratio(counts["action_correct"], counts["gold_goals"]),
                      "target": ratio(counts["target_correct"], counts["gold_goals"]),
                      "fact_precision": ratio(counts["fact_tp"], counts["fact_pred"]),
                      "fact_recall": ratio(counts["fact_tp"], counts["fact_gold"]),
                      "temporal_precision": ratio(counts["temporal_tp"], counts["temporal_pred"]),
                      "temporal_recall": ratio(counts["temporal_tp"], counts["temporal_gold"]),
                      "guard_coverage": ratio(counts["guard_accept"], len(eligible))},
            "warning": "accepted_not_canonical is a triage proxy, not a material-risk judgment"}
