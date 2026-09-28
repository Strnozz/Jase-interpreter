"""Curated Interpreter-only failures from frozen train slice; never training data."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "benchmarks/train_slice_v1/blind.jsonl"
REPLAY = ROOT / "evaluation/train_slice_v1_after/replay.jsonl"
OUT = ROOT / "evaluation/train_slice_v1_after/V27_INTERPRETER_FAILURE_CANDIDATES.json"

# Inspected cases with an explicit train request and a model-level semantic or
# structural mistake. Generic "travel" prompts with inferred train gold are excluded.
REVIEW = {
    "train-v1-011": ("TEMPORAL_ROLE_ERROR", ["goals[0].temporal: search_time/after -> departure_time/after"],
                     "The model assigned a boarding threshold to search_time; Planner cannot assert the departure role without reinterpreting the utterance."),
    "train-v1-012": ("FACT_HALLUCINATION", ["context[0].field=visit", "context[1].field=location"],
                     "The model appended unsupported context objects despite a correctly represented arrival deadline; stripping them would rewrite an invalid contract."),
    "train-v1-018": ("TEMPORAL_ROLE_ERROR", ["goals[0].temporal: missing search_time=mattina"],
                     "The morning departure window is absent from the model contract; the safety gate holds rather than adding it invisibly."),
    "train-v1-021": ("FACT_OMISSION", ["goals[0].facts: missing direct/no_changes", "goals[0].modifiers.exclude invalid"],
                     "The requested no-change constraint is absent and replaced by a malformed modifier; no provider query can preserve it."),
    "train-v1-033": ("POLICY_ERROR", ["goals[0].policy.forbid: missing book"],
                     "The explicit no-book instruction is absent; Guard blocks the omission rather than assuming consent."),
    "train-v1-035": ("ACTION_ERROR", ["goals[0].action: order instead of find+sort"],
                     "The model interpreted Italian 'ordina' as purchasing rather than sorting; Planner rejects consequential transport order."),
}


def rows(path):
    return {r["id"]: r for r in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)}


def main():
    if OUT.exists():
        raise FileExistsError("Candidate corpus already versioned")
    panel, replay = rows(PANEL), rows(REPLAY)
    out = []
    for ident, (category, fields, reason) in REVIEW.items():
        case = replay[ident]
        out.append({"case_id": ident, "raw_user_request": panel[ident]["text"],
                    "qwen_output": case["interpreter"]["raw"],
                    "reviewed_intended_interpretation": panel[ident]["contract"],
                    "interpretation_review_status": "AGENT_PROVISIONAL_PENDING_INDEPENDENT_HUMAN_REVIEW",
                    "failure_category": category, "exact_wrong_or_missing_fields": fields,
                    "why_planner_cannot_safely_reconstruct": reason,
                    "deterministic_normalization_could_solve": False,
                    "capability_semantics_could_solve": False,
                    "requires_improved_language_interpretation": True,
                    "train_on_this_benchmark_case": False})
    OUT.write_text(json.dumps({"version": "V27_INTERPRETER_FAILURE_CANDIDATES_V1",
                               "status": "REVIEW_CANDIDATES_NOT_TRAINING_DATA",
                               "cases": out}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(len(out))


if __name__ == "__main__":
    main()
