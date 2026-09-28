"""Frozen V37 conversation replay with predicted state; no provider calls.

Use --legacy-v26 for the historical one-shot baseline. Its metrics are limited
to actionable detection and first-goal semantics because V26 has no GoalDraft.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.conversation_v1 import (ConversationState, apply_decision, clarification_plan,
                                  finalize_draft, plan_finalized, validate_decision)  # noqa: E402
from jase.foundation import chat_template_kwargs, load_base  # noqa: E402
from jase.multimodel import load_config, read_jsonl, sha256  # noqa: E402

PANEL = ROOT / "benchmarks/v37_conversational/blind.jsonl"
NOW = datetime(2026, 9, 28, 12, tzinfo=ZoneInfo("Europe/Rome"))
ACTIONABLE_MODES = {"ACTIONABLE_GOAL", "CONTINUATION", "ANSWER_TO_CLARIFICATION",
                    "CONFIRMATION", "REJECTION", "CANCELLATION"}


def model_state(state: ConversationState) -> dict:
    active = [state.drafts[ident] for ident in state.active_draft_ids]
    if not active:
        return {}
    def compact(draft):
        return {"draft_id": draft.draft_id, "action": draft.action, "target": draft.target,
                "slots": {slot: {"value": value["value"], "provenance": value["provenance"]}
                          for slot, value in draft.slots.items()},
                "confirmation": draft.confirmation}
    data = {"active_draft": compact(active[0])} if len(active) == 1 else {
        "active_drafts": [compact(draft) for draft in active]}
    if state.clarification:
        data["clarification"] = {"ask_user": state.clarification["ask_user"],
                                 "discover_read_only": state.clarification["discover_read_only"]}
    if state.result_references:
        data["results"] = list(state.result_references.values())
    return data


def normalized(value):
    return str(value).strip().casefold() if value is not None else None


def score_turn(gold: dict, predicted: dict | None, state_status: str, actual_next: str) -> dict:
    if predicted is None:
        return {"mode_correct": False, "operation_correct": False, "actionable_correct": False,
                "action_correct": False, "target_correct": False, "slot_recall": 0,
                "slot_expected": len(gold["slot_updates"]), "slot_hallucinations": 0,
                "next_correct": False, "material_false_ready_candidate": False,
                "false_action": False, "false_nonaction": gold["mode"] in ACTIONABLE_MODES}
    expected_actionable = gold["mode"] in ACTIONABLE_MODES
    actual_actionable = predicted["operation"] != "NO_GOAL"
    actual = {item["slot"].split(".", 1)[-1]: item["value"] for item in predicted.get("updates", [])}
    expected = gold["slot_updates"]
    recalled = sum(normalized(actual.get(key)) == normalized(value) for key, value in expected.items())
    extra = sum(key not in expected for key in actual)
    reference = predicted.get("reference")
    if "result_ordinal" in expected and reference:
        recalled += reference.get("ordinal") == expected["result_ordinal"]
        extra = max(0, extra - 1)
    if "result_rank" in expected and reference:
        recalled += reference.get("kind") == "cheapest"
        extra = max(0, extra - 1)
    false_ready = actual_next in {"READY_TO_FINALIZE", "READY_FOR_DRY_RUN"} and gold["expected_next"] in {
        "NO_ACTION", "ASK_USER", "DISCOVER_READ_ONLY", "RESOLVE_RESULT_REFERENCE",
        "CHECK_CONFIRMATION_BINDING"}
    return {"mode_correct": predicted["mode"] == gold["mode"],
            "operation_correct": predicted["operation"] == gold["operation"],
            "actionable_correct": expected_actionable == actual_actionable,
            "action_correct": normalized(predicted.get("action")) == normalized(gold.get("action")),
            "target_correct": normalized((predicted.get("target") or {}).get("name")) == normalized(gold.get("target")),
            "slot_recall": recalled, "slot_expected": len(expected), "slot_hallucinations": extra,
            "next_correct": actual_next == gold["expected_next"],
            "material_false_ready_candidate": false_ready,
            "false_action": not expected_actionable and actual_actionable,
            "false_nonaction": expected_actionable and not actual_actionable,
            "state_status": state_status}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--output", required=True)
    ap.add_argument("--legacy-v26", action="store_true")
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    output = (ROOT / args.output).resolve()
    if ROOT not in run.parents or ROOT not in output.parents:
        ap.error("Run/output must be inside repository")
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("Adapter run is not complete")
    cfg = load_config(manifest["model"])
    adapter = (ROOT / (args.adapter or manifest["adapter"])).resolve()
    if ROOT not in adapter.parents or not (adapter / "adapter_config.json").is_file():
        ap.error("Invalid adapter")
    panel = read_jsonl(PANEL)
    panel_meta = json.loads((PANEL.parent / "manifest.json").read_text(encoding="utf-8"))
    if sha256(PANEL) != panel_meta["panel_sha256"]:
        raise RuntimeError("V37 frozen panel hash mismatch")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel

    output.mkdir(parents=True, exist_ok=True)
    cases = output / "cases.jsonl"
    done = {row["id"]: row for row in read_jsonl(cases)} if cases.exists() else {}
    model, tokenizer = load_base(cfg)
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    system = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()

    def infer(content: str) -> tuple[str, int, float, int]:
        turns = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        encoded = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=True,
            return_tensors="pt", add_generation_prompt=True, **chat_template_kwargs(cfg)).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        with torch.inference_mode():
            ids = model.generate(**encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        torch.cuda.synchronize()
        generated = ids[0, encoded["input_ids"].shape[-1]:]
        return tokenizer.decode(generated, skip_special_tokens=True), len(generated), round(
            time.perf_counter() - started, 3), torch.cuda.max_memory_allocated()

    with cases.open("a", encoding="utf-8", newline="\n") as handle:
        for index, scenario in enumerate(panel, 1):
            if scenario["id"] in done:
                continue
            state = ConversationState(conversation_id=scenario["id"])
            previous_text: list[str] = []
            turn_records = []
            for turn_index, turn in enumerate(scenario["turns"], 1):
                user_text = turn["user"]
                if args.legacy_v26:
                    content = "\n".join(f"Utente {i}: {text}" for i, text in
                                        enumerate([*previous_text, user_text], 1))
                else:
                    content = json.dumps({"task": "semantic_update", "user": user_text,
                                          "state": model_state(state)},
                                         ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                raw, output_tokens, latency, peak_vram = infer(content)
                parsed, error = None, None
                try:
                    parsed = json.loads(raw)
                    if args.legacy_v26:
                        if not isinstance(parsed, dict) or parsed.get("kind") not in {"task", "non_actionable"}:
                            raise ValueError("Invalid legacy GoalContract")
                    else:
                        parsed = validate_decision(parsed)
                except (ValueError, TypeError, KeyError) as exc:
                    error = repr(exc)
                    parsed = None
                actual_next = "INVALID"
                state_status = "INVALID"
                planner = None
                if args.legacy_v26 and parsed is not None:
                    first = parsed.get("goals", [{}])[0] if parsed.get("goals") else {}
                    parsed = {"mode": "ACTIONABLE_GOAL" if parsed["kind"] == "task" else "QUESTION",
                              "operation": "OPEN_DRAFT" if parsed["kind"] == "task" else "NO_GOAL",
                              "action": first.get("action"), "target": first.get("target"),
                              "updates": []}
                    actual_next = "LEGACY_ONE_SHOT"
                    state_status = "NO_CONVERSATION_STATE"
                elif parsed is not None:
                    try:
                        draft = apply_decision(state, parsed, user_text, timestamp=NOW)
                        if draft is None or draft.status == "CANCELLED":
                            actual_next = "NO_ACTION"
                            state_status = draft.status if draft else "NO_GOAL"
                        elif parsed["operation"] == "CONFIRM_DRAFT":
                            actual_next = "CHECK_CONFIRMATION_BINDING"
                            state_status = draft.confirmation["status"]
                        else:
                            plan = clarification_plan(state, draft.draft_id)
                            if draft.unresolved_references:
                                actual_next = "RESOLVE_RESULT_REFERENCE"
                            elif plan["ask_user"] or plan["ambiguous"]:
                                actual_next = "ASK_USER"
                            elif plan["discover_read_only"]:
                                actual_next = "DISCOVER_READ_ONLY"
                            elif "CONFIRMATION_REQUIRED" in plan["reason_codes"]:
                                actual_next = "REQUEST_CONFIRMATION"
                            elif plan["ready_to_finalize"]:
                                contract = finalize_draft(state, draft.draft_id)
                                planner = plan_finalized(state, draft.draft_id, contract)
                                actual_next = "READY_TO_FINALIZE"
                            else:
                                actual_next = "BLOCKED"
                            state_status = draft.status
                    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                        error = repr(exc)
                        actual_next = "STATE_REJECTED"
                        state_status = "STATE_REJECTED"
                metric = score_turn(turn["gold"], parsed, state_status, actual_next)
                turn_records.append({"turn_index": turn_index, "user": user_text, "gold": turn["gold"],
                                     "raw": raw, "parsed": parsed, "error": error,
                                     "state": state.as_dict() if not args.legacy_v26 else None,
                                     "actual_next": actual_next, "planner": planner,
                                     "metric": metric, "output_tokens": output_tokens,
                                     "latency_seconds": latency, "peak_vram_bytes": peak_vram})
                previous_text.append(user_text)
            record = {"id": scenario["id"], "family": scenario["family"], "turns": turn_records}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0:
                print(f"{index}/{len(panel)}", flush=True)
    produced = read_jsonl(cases)
    if len(produced) != len(panel):
        raise RuntimeError("V37 replay incomplete")
    turns = [turn for case in produced for turn in case["turns"]]
    metrics = [turn["metric"] for turn in turns]
    counts = Counter()
    for turn in turns:
        counts["turns"] += 1
        counts["json_valid"] += turn["parsed"] is not None
        counts["schema_valid"] += turn["parsed"] is not None and turn["error"] is None
        for key in ("mode_correct", "operation_correct", "actionable_correct", "action_correct",
                    "target_correct", "next_correct", "material_false_ready_candidate",
                    "false_action", "false_nonaction", "slot_recall", "slot_expected",
                    "slot_hallucinations"):
            counts[key] += turn["metric"].get(key, 0)
    summary = {"run": str(run.relative_to(ROOT)), "adapter": str(adapter.relative_to(ROOT)),
               "model": cfg["name"], "legacy_v26": args.legacy_v26,
               "panel_sha256": sha256(PANEL), "cases_sha256": sha256(cases),
               "scenarios": len(produced), "turns": len(turns), "counts": dict(counts),
               "tokens_per_second": round(sum(t["output_tokens"] for t in turns) /
                                          max(.001, sum(t["latency_seconds"] for t in turns)), 2),
               "peak_vram_bytes": max(t["peak_vram_bytes"] for t in turns),
               "limitations": ["Agent-authored V37 gold; no independent human review",
                               "V26 one-shot representation lacks a GoalDraft, so conversational metrics are asymmetric",
                               "No provider calls or consequential execution; false READY candidates require case review"]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
