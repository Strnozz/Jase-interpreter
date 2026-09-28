"""Single-pass V27 semantic-proposal regression on frozen V33-V36 panels.

This is a draft-level projection, not a GoalContract exact-score replacement.
Historical V26 GoalContracts remain in their existing benchmark outputs.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.conversation_v1 import ConversationState, apply_decision, clarification_plan, validate_decision  # noqa: E402
from jase.foundation import chat_template_kwargs, load_base  # noqa: E402
from jase.metrics_v1_3 import project  # noqa: E402
from jase.multimodel import load_config, read_jsonl, sha256  # noqa: E402

RUN = ROOT / "training/runs/qwen35-9b-v27/20260928T114238Z-d2316823"
ADAPTER = RUN / "best_adapter"
OUTPUT = ROOT / "benchmarks/outputs/v27-v33-v36-draft-regression"
PANEL_FILES = [ROOT / f"benchmarks/v{v}/blind.jsonl" for v in range(33, 37)]


def normalized(value: object) -> str:
    return str(value).strip().casefold() if value is not None else ""


def goal_slots(goal: dict) -> set[tuple[str, str, str]]:
    slots = {(f"fact.{item['field']}", normalized(item.get("op", "eq")),
              normalized(item.get("value"))) for item in goal.get("facts", [])}
    slots |= {(f"temporal.{item['role']}", normalized(item.get("op", "eq")),
               normalized(item.get("value"))) for item in goal.get("temporal", [])}
    for name in goal.get("policy", {}).get("forbid", []):
        slots.add(("policy.forbid", "eq", normalized(name)))
    return slots


def proposed_slots(decision: dict) -> set[tuple[str, str, str]]:
    return {(item["slot"], normalized(item.get("op", "eq")), normalized(item["value"]))
            for item in decision.get("updates", [])}


def main() -> int:
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise RuntimeError("V27 run is not complete")
    cfg = load_config(manifest["model"])
    panel_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in PANEL_FILES}
    rows = [(row, panel.name) for panel in PANEL_FILES for row in read_jsonl(panel)]
    if len(rows) != 242 or len({row["id"] for row, _ in rows}) != len(rows):
        raise RuntimeError("Historical panel identity/count mismatch")
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    from peft import PeftModel

    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases_path = OUTPUT / "cases.jsonl"
    completed = {row["id"] for row in read_jsonl(cases_path)} if cases_path.exists() else set()
    model, tokenizer = load_base(cfg)
    model = PeftModel.from_pretrained(model, str(ADAPTER))
    model.eval()
    prompt = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()
    with cases_path.open("a", encoding="utf-8", newline="\n") as handle:
        for index, (row, panel_name) in enumerate(rows, 1):
            if row["id"] in completed:
                continue
            user_text = row["text"]
            content = json.dumps({"task": "semantic_update", "user": user_text, "state": {}},
                                 ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            messages = [{"role": "system", "content": prompt}, {"role": "user", "content": content}]
            encoded = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True,
                return_tensors="pt", add_generation_prompt=True, **chat_template_kwargs(cfg)).to("cuda")
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                ids = model.generate(**encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            torch.cuda.synchronize()
            generated = ids[0, encoded["input_ids"].shape[-1]:]
            raw = tokenizer.decode(generated, skip_special_tokens=True)
            decision, error = None, None
            try:
                decision = validate_decision(json.loads(raw))
            except (ValueError, TypeError, KeyError) as exc:
                error = repr(exc)
            state = ConversationState(conversation_id=row["id"])
            plan = None
            if decision is not None:
                try:
                    draft = apply_decision(state, decision, user_text)
                    if draft and draft.status != "CANCELLED":
                        plan = clarification_plan(state, draft.draft_id)
                except (ValueError, TypeError, KeyError) as exc:
                    error = "state:" + repr(exc)
            gold, gold_hold = project(row["contract"])
            if gold is None:
                raise RuntimeError(f"Gold projection failed for {row['id']}: {gold_hold}")
            expected_task = gold["kind"] == "task"
            actual_task = decision is not None and decision["operation"] != "NO_GOAL"
            first = gold["goals"][0] if gold["goals"] else None
            proposed_count = (1 + len(decision.get("additional_goals", []))) if actual_task else 0
            wanted_slots = goal_slots(first) if first else set()
            got_slots = proposed_slots(decision) if actual_task else set()
            metric = {"actionable_correct": expected_task == actual_task,
                      "false_action": bool(actual_task and not expected_task),
                      "false_nonaction": bool(expected_task and not actual_task),
                      "first_action_correct": bool(first and actual_task and decision.get("action") == first["action"]),
                      "first_target_correct": bool(first and actual_task and decision.get("target") == first["target"]),
                      "goal_count_correct": proposed_count == len(gold["goals"]),
                      "slot_tp": len(wanted_slots & got_slots), "slot_gold": len(wanted_slots),
                      "slot_pred": len(got_slots),
                      "ready_candidate": bool(plan and plan["ready_to_finalize"]),
                      "ask_user": bool(plan and plan["ask_user"]),
                      "discover_read_only": bool(plan and plan["discover_read_only"]),
                      "unknown_capability": bool(plan and plan["blocked"])}
            record = {"id": row["id"], "panel": panel_name, "family": row.get("family"),
                      "text": user_text, "gold_kind": gold["kind"],
                      "gold_first_action": first["action"] if first else None,
                      "gold_first_target": first["target"] if first else None,
                      "gold_goal_count": len(gold["goals"]), "gold_slots": sorted(wanted_slots),
                      "raw": raw, "decision": decision, "error": error, "plan": plan,
                      "metric": metric, "output_tokens": len(generated),
                      "latency_seconds": round(time.perf_counter() - start, 3),
                      "peak_vram_bytes": torch.cuda.max_memory_allocated()}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 20 == 0:
                print(f"{index}/{len(rows)}", flush=True)
    produced = read_jsonl(cases_path)
    if len(produced) != len(rows) or {r["id"] for r in produced} != {r["id"] for r, _ in rows}:
        raise RuntimeError("Legacy projection replay incomplete")
    metrics = [r["metric"] for r in produced]
    counts = Counter({"cases": len(produced), "json_valid": sum(_json_object(r["raw"]) for r in produced),
                      "protocol_valid": sum(r["decision"] is not None for r in produced),
                      "runtime_accepted": sum(r["decision"] is not None and r["error"] is None for r in produced)})
    for metric in metrics:
        for key, value in metric.items():
            counts[key] += value
    by_panel = {f"v{v}": {"cases": sum(r["panel"] == "blind.jsonl" and r["id"].startswith(f"v{v}-")
                                         for r in produced),
                          "false_action": sum(r["id"].startswith(f"v{v}-") and r["metric"]["false_action"]
                                              for r in produced),
                          "false_nonaction": sum(r["id"].startswith(f"v{v}-") and r["metric"]["false_nonaction"]
                                                 for r in produced)} for v in range(33, 37)}
    summary = {"run": str(RUN.relative_to(ROOT)), "adapter": str(ADAPTER.relative_to(ROOT)),
               "panel_sha256": panel_hashes, "cases_sha256": sha256(cases_path),
               "counts": dict(counts), "by_panel": by_panel,
               "tokens_per_second": round(sum(r["output_tokens"] for r in produced) /
                                          max(0.001, sum(r["latency_seconds"] for r in produced)), 2),
               "peak_vram_bytes": max(r["peak_vram_bytes"] for r in produced),
               "limitations": ["Draft-level projection only: no V27 GoalContract canonical-exact comparison",
                               "V26 historical outputs use one-shot GoalContract protocol",
                               "Gold agent-authored and not independently reviewed",
                               "READY candidates require case-level material-risk review"]}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


def _json_object(raw: str) -> bool:
    try:
        return isinstance(json.loads(raw), dict)
    except (ValueError, TypeError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
