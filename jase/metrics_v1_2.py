"""Static GoalContract 1.2 comparison; no runtime acceptance is inferred."""
from __future__ import annotations

import json
import unicodedata
from collections import Counter
from typing import Any

from .canon import norm_currency, norm_unit
from .goal_contract_v1_2 import validate_contract


def _norm(value: Any) -> Any:
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    if isinstance(value, dict):
        return {key: _norm(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [_norm(item) for item in value]
    return value


def _fact_key(fact: dict, *, strength: bool = True) -> str:
    row = {k: v for k, v in fact.items() if strength or k != "strength"}
    if "unit" in row:
        row["unit"] = norm_unit(row["unit"])
    if "currency" in row:
        row["currency"] = norm_currency(row["currency"])
    return json.dumps(_norm(row), sort_keys=True, ensure_ascii=False)


def canonical_v12(contract: dict) -> dict:
    """Ignore field order, fact order, empty optionals and local goal renaming."""
    if validate_contract(contract):
        raise ValueError("invalid V1.2 contract")
    goals = sorted(contract["goals"], key=lambda g: (
        _norm(g["action"]), _norm(g["target"]["name"]),
        _norm(g["target"].get("type", "")), g["id"]))
    rename = {goal["id"]: f"g{index}" for index, goal in enumerate(goals, 1)}
    out = {"schema_version": "1.2", "kind": contract["kind"], "goals": []}
    for goal in goals:
        cur = json.loads(json.dumps(goal, ensure_ascii=False))
        cur["id"] = rename[goal["id"]]
        cur["facts"] = sorted(cur["facts"], key=_fact_key)
        for fact in cur["facts"]:
            if "value_ref" in fact:
                fact["value_ref"]["goal"] = rename[fact["value_ref"]["goal"]]
            if "unit" in fact:
                fact["unit"] = norm_unit(fact["unit"])
            if "currency" in fact:
                fact["currency"] = norm_currency(fact["currency"])
        if cur.get("depends_on"):
            cur["depends_on"] = sorted(rename[x] for x in cur["depends_on"])
        else:
            cur.pop("depends_on", None)
        if "condition" in cur:
            cur["condition"]["goal"] = rename[cur["condition"]["goal"]]
            if "value_ref" in cur["condition"]["test"]:
                ref = cur["condition"]["test"]["value_ref"]
                ref["goal"] = rename[ref["goal"]]
        if cur.get("missing"):
            cur["missing"] = sorted(cur["missing"], key=lambda m: (m["field"], m["blocks"], m["reason"]))
        else:
            cur.pop("missing", None)
        if "policy" in cur:
            for key in list(cur["policy"]):
                cur["policy"][key] = sorted(cur["policy"][key])
                if not cur["policy"][key]:
                    del cur["policy"][key]
            if not cur["policy"]:
                del cur["policy"]
        out["goals"].append(cur)
    if contract.get("context"):
        out["context"] = sorted(contract["context"], key=_fact_key)
    return _norm(out)


def _multiset_scores(expected: list[str], predicted: list[str]) -> dict[str, int]:
    a, b = Counter(expected), Counter(predicted)
    tp = sum((a & b).values())
    return {"tp": tp, "fp": sum(b.values()) - tp, "fn": sum(a.values()) - tp}


def compare_v12(expected: dict, predicted: Any) -> dict[str, Any]:
    """Compare two V1.2 contracts without assuming the predicted one can execute."""
    if validate_contract(expected):
        raise ValueError("gold contract is invalid")
    if not isinstance(predicted, dict):
        return {"schema_valid": False, "complete_success": False, "issues": ["not_object"]}
    issues = validate_contract(predicted)
    if issues:
        return {"schema_valid": False, "complete_success": False,
                "issues": [x.code for x in issues]}
    e, p = canonical_v12(expected), canonical_v12(predicted)
    eg, pg = e["goals"], p["goals"]
    # Independent of original local ID spelling; align by canonical rank.
    aligned = list(zip(eg, pg))
    ef = [_fact_key(f) for g in eg for f in g["facts"]]
    pf = [_fact_key(f) for g in pg for f in g["facts"]]
    all_facts = _multiset_scores(ef, pf)
    eh = [_fact_key(f, strength=False) for g in eg for f in g["facts"] if f["strength"] == "hard"]
    ph = [_fact_key(f, strength=False) for g in pg for f in g["facts"] if f["strength"] == "hard"]
    es = [_fact_key(f, strength=False) for g in eg for f in g["facts"] if f["strength"] == "soft"]
    ps = [_fact_key(f, strength=False) for g in pg for f in g["facts"] if f["strength"] == "soft"]
    em = [json.dumps(m, sort_keys=True) for g in eg for m in g.get("missing", [])]
    pm = [json.dumps(m, sort_keys=True) for g in pg for m in g.get("missing", [])]
    ed = [f"{g['id']}->{d}" for g in eg for d in g.get("depends_on", [])]
    pd = [f"{g['id']}->{d}" for g in pg for d in g.get("depends_on", [])]
    er = [json.dumps(f["value_ref"], sort_keys=True) for g in eg for f in g["facts"] if "value_ref" in f]
    pr = [json.dumps(f["value_ref"], sort_keys=True) for g in pg for f in g["facts"] if "value_ref" in f]
    return {"schema_valid": True, "complete_success": e == p,
            "goal_count_correct": len(eg) == len(pg),
            "action_correct": sum(a["action"] == b["action"] for a, b in aligned),
            "action_total": max(len(eg), len(pg)),
            "target_correct": sum(a["target"] == b["target"] for a, b in aligned),
            "target_total": max(len(eg), len(pg)),
            "facts": all_facts,
            "hard": _multiset_scores(eh, ph), "soft": _multiset_scores(es, ps),
            "context": _multiset_scores([_fact_key(f) for f in e.get("context", [])],
                                        [_fact_key(f) for f in p.get("context", [])]),
            "missing": _multiset_scores(em, pm),
            "dependencies": _multiset_scores(ed, pd),
            "references": _multiset_scores(er, pr),
            "policy_correct": sum(a.get("policy", {}) == b.get("policy", {}) for a, b in aligned),
            "policy_total": max(len(eg), len(pg)),
            "condition_correct": sum(a.get("condition") == b.get("condition") for a, b in aligned),
            "condition_total": max(len(eg), len(pg)),
            "ranking_correct": sum(a.get("ranking") == b.get("ranking") for a, b in aligned),
            "ranking_total": max(len(eg), len(pg)),
            "hallucinated_fact_count": all_facts["fp"],
            "schema_valid_but_wrong": e != p,
            "issues": []}


def aggregate_v12(rows: list[dict]) -> dict[str, Any]:
    """All-case success separate from fact metrics conditional on valid schema."""
    n = len(rows)
    valid = [r for r in rows if r.get("schema_valid")]

    def ratio(a: int, b: int) -> float | None:
        return a / b if b else None

    def score(key: str) -> dict[str, float | int | None]:
        counts = {part: sum(r[key][part] for r in valid) for part in ("tp", "fp", "fn")}
        return {**counts, "precision": ratio(counts["tp"], counts["tp"] + counts["fp"]),
                "recall": ratio(counts["tp"], counts["tp"] + counts["fn"])}

    return {"cases": n, "schema_valid_rate": ratio(len(valid), n),
            "complete_success_rate": ratio(sum(r.get("complete_success", False) for r in rows), n),
            "schema_valid_but_wrong": sum(r.get("schema_valid_but_wrong", False) for r in rows),
            "goal_count_accuracy": ratio(sum(r.get("goal_count_correct", False) for r in valid), len(valid)),
            "action_accuracy": ratio(sum(r["action_correct"] for r in valid),
                                     sum(r["action_total"] for r in valid)),
            "target_accuracy": ratio(sum(r["target_correct"] for r in valid),
                                     sum(r["target_total"] for r in valid)),
            "facts": score("facts"), "hard": score("hard"), "soft": score("soft"),
            "context": score("context"), "missing": score("missing"),
            "dependencies": score("dependencies"), "references": score("references"),
            "explicit_policy_accuracy": ratio(sum(r["policy_correct"] for r in valid),
                                              sum(r["policy_total"] for r in valid)),
            "condition_accuracy": ratio(sum(r["condition_correct"] for r in valid),
                                        sum(r["condition_total"] for r in valid)),
            "ranking_accuracy": ratio(sum(r["ranking_correct"] for r in valid),
                                      sum(r["ranking_total"] for r in valid)),
            "not_available": ["capability_accuracy (no registry)",
                              "plan_accuracy (Planner-owned)",
                              "runtime_confirmation_correctness (no policy engine)",
                              "accepted_but_wrong (no V1.2 Guard deployed)"]}
