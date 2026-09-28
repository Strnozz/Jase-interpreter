"""Build V27 Stage-A/Stage-B supervision without using any benchmark requests.

Legacy V26 train/valid are only a semantic source. The V37 panel is an exclusion
set and is never a source of targets. Generated replies are separate by split.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.capability_registry_v1 import CapabilityRegistry, REGISTRY_TRAIN_V1_PATH  # noqa: E402
from jase.conversation_v1 import (ConversationState, GoalDraft, clarification_plan,
                                  render_clarification)  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.provenance_v1 import MODEL_INFERENCE, ground_value  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402

SEED = 2041
DEST = ROOT / "data/v27_conversational"
SYSTEM = (ROOT / "V27_CONVERSATIONAL_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
REGISTRY = CapabilityRegistry.load(REGISTRY_TRAIN_V1_PATH)
CITY = ["Belluno", "Matera", "Piacenza", "Rieti", "Pordenone", "Aosta", "Terni", "Teramo",
        "Agrigento", "Potenza", "Pistoia", "Monza", "Vercelli", "Arezzo", "Benevento", "Trapani",
        "Varese", "Forlì", "Nuoro", "Avellino", "Macerata", "Isernia", "Fermo", "Sondrio"]
DAY = ["martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica", "lunedì"]
NONACTION_MODES = {"TRANSLATION": [
    "Come si dice in inglese «{phrase}»?", "Traduci per il mio esercizio «{phrase}».",
    "Dammi una resa francese di «{phrase}».", "Qual è la traduzione tedesca di «{phrase}»?"],
    "META": ["Se scrivessi «{phrase}», quale sarebbe il tuo ragionamento?",
             "Per un manuale descrivi cosa significa chiedere: «{phrase}».",
             "In un esempio di dialogo, interpreta la frase «{phrase}».",
             "Spiegami il processo dietro la richiesta «{phrase}»."],
    "HYPOTHETICAL": ["Se un giorno volessi dire «{phrase}», cosa dovrei sapere?",
                     "Immagina un caso fittizio in cui qualcuno dica «{phrase}».",
                     "In teoria, che cosa accadrebbe dopo la frase «{phrase}»?",
                     "Supponiamo che la richiesta fosse «{phrase}»: come funzionerebbe?"],
    "QUESTION": ["Quali informazioni servono generalmente per cercare un {noun}?",
                 "Come si distingue una richiesta di {noun} da una spiegazione?",
                 "Che cosa significa confrontare opzioni di {noun}?",
                 "Perché si chiede conferma prima di prenotare un {noun}?"],
    "CHAT": ["Oggi ho visto un bel {noun} in centro e mi è tornato in mente un viaggio.",
             "Ieri parlavamo di {noun} e mi sono divertito molto.",
             "La mia famiglia racconta sempre storie sui {noun} quando ci incontriamo.",
             "Che giornata movimentata, persino il discorso sui {noun} mi ha fatto sorridere."]}
NOUNS = ["treno", "ristorante", "hotel", "volo", "idraulico", "bicicletta", "appuntamento",
         "promemoria", "farmacia", "furgone", "veterinario", "messaggio", "camera", "elettricista",
         "museo", "taxi", "negozio", "dentista", "biglietto", "corso"]


def compact(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def row(task: str, user: str, state: dict, answer: dict | str, category: str, source: str) -> dict:
    prompt = compact({"task": task, "user": user, "state": state}) if task == "semantic_update" else compact({
        "task": task, "plan": state})
    target = compact(answer) if isinstance(answer, dict) else answer
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": prompt},
                         {"role": "assistant", "content": target}],
            "category": category, "source": source}


def decision(mode: str, operation: str, updates: list[dict] | None = None,
             action: str | None = None, target: dict | None = None, draft_id: str | None = None) -> dict:
    output = {"protocol_version": "conversation-v1", "mode": mode,
              "operation": operation, "updates": updates or []}
    if action:
        output["action"] = action
    if target:
        output["target"] = target
    if draft_id:
        output["draft_id"] = draft_id
    return output


def contract_updates(goal: dict, contract: dict) -> list[dict]:
    updates = []
    for fact in goal.get("facts", []):
        if "value" in fact and isinstance(fact["value"], (str, int, float, bool)):
            item = {"slot": "fact." + fact["field"], "value": fact["value"], "op": fact["op"]}
            if "value_to" in fact:
                item["value_to"] = fact["value_to"]
            updates.append(item)
    for temporal in goal.get("temporal", []):
        if "value" in temporal and isinstance(temporal["value"], (str, int, float, bool)):
            item = {"slot": "temporal." + temporal["role"],
                    "value": temporal["value"], "op": temporal["op"]}
            if "value_to" in temporal:
                item["value_to"] = temporal["value_to"]
            updates.append(item)
    for name, value in goal.get("modifiers", {}).items():
        if isinstance(value, (str, int, float, bool)):
            updates.append({"slot": "modifier." + name, "value": value})
    for context in contract.get("context", []):
        if (context.get("field") in {"party_size", "guests", "has_car"} and
                isinstance(context.get("value"), (str, int, float, bool))):
            updates.append({"slot": "external." + context["field"], "value": context["value"]})
    for forbidden in goal.get("policy", {}).get("forbid", []):
        updates.append({"slot": "policy.forbid", "value": forbidden})
    return updates


def legacy_examples(split: str) -> dict[str, list[dict]]:
    source = ROOT / "data/v26_9b" / f"{split}.jsonl"
    rng = random.Random(SEED + (0 if split == "train" else 1))
    buckets: dict[str, list[dict]] = {key: [] for key in ("complete", "incomplete", "nonaction", "hard")}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        request = item["messages"][1]["content"]
        contract = json.loads(item["messages"][2]["content"])
        if len(request) > 240:
            continue
        if contract["kind"] == "non_actionable":
            mode = ("TRANSLATION" if re.search(r"traduc|traduzion|in inglese|in francese", request, re.I)
                    else "HYPOTHETICAL" if re.search(r"\bse\b|ipotetic|suppon", request, re.I)
                    else "META" if re.search(r"esempio|cosa faresti|spieg", request, re.I)
                    else "QUESTION")
            buckets["nonaction"].append(row("semantic_update", request, {},
                decision(mode, "NO_GOAL"), "nonaction", "v26_semantic_projection"))
            continue
        if len(contract["goals"]) != 1:
            continue  # multi-goal requests are supervised in a separate protocol batch, never flattened
        if validate_contract(contract, user_text=request):
            continue
        goal = contract["goals"][0]
        if REGISTRY.resolve(goal)[1] is None:
            continue
        if goal.get("condition") or goal.get("depends_on"):
            continue
        updates = contract_updates(goal, contract)
        if any(ground_value(u["slot"].split(".")[-1], u["value"], request).source == MODEL_INFERENCE
               for u in updates):
            continue  # do not teach unsupported values as if the user supplied them
        target = goal["target"]
        # The old corpus has some grammatical noise. Keep a conservative subset.
        if "\ufffd" in request or len(request.split()) < 5:
            continue
        answer = decision("ACTIONABLE_GOAL", "OPEN_DRAFT", updates, goal["action"], target)
        category = "incomplete" if goal.get("missing") else "complete"
        buckets[category].append(row("semantic_update", request, {}, answer,
                                     category, "v26_semantic_projection"))
    for values in buckets.values():
        rng.shuffle(values)
    return buckets


def generated_nonaction(split: str) -> list[dict]:
    rng = random.Random(SEED + (12 if split == "train" else 13))
    phrases = [f"cercami un {noun} a {city}" for noun in NOUNS for city in CITY]
    rng.shuffle(phrases)
    examples = []
    for mode_index, (mode, templates) in enumerate(NONACTION_MODES.items()):
        for i in range(150 if split == "train" else 32):
            noun = NOUNS[(i * 7 + (0 if split == "train" else 3)) % len(NOUNS)]
            phrase = phrases[(i + mode_index * 151) % len(phrases)]
            prompt = templates[(i + (0 if split == "train" else 1)) % len(templates)].format(
                noun=noun, phrase=phrase)
            examples.append(row("semantic_update", prompt, {}, decision(mode, "NO_GOAL"),
                                "nonaction", "generated_discourse"))
    return examples


def generated_incomplete(split: str) -> list[dict]:
    """Independent, registry-compatible incomplete requests across domains."""
    rng = random.Random(SEED + (30 if split == "train" else 31))
    places = CITY if split == "train" else ["Cremona", "Grosseto", "Caltanissetta", "Chieti",
                                             "Pavia", "Lecco", "Taranto", "Sassari"]
    outputs = []
    for i, city in enumerate(places):
        for day in DAY:
            other = places[(i + 7) % len(places)]
            hour = f"{10 + i % 9}:00"
            variants = [
                (f"Per {day} vorrei vedere i treni per {city} dopo le {hour}; mi dici le opzioni?",
                 "find", "treno", "transport", [
                     {"slot": "fact.destination", "value": city},
                     {"slot": "temporal.search_date", "value": day},
                     {"slot": "temporal.departure_time", "value": hour, "op": "after"}]),
                (f"Cercami un treno in partenza da {city} {day} mattina, senza cambi se possibile.",
                 "find", "treno", "transport", [
                     {"slot": "fact.origin", "value": city},
                     {"slot": "temporal.search_date", "value": day},
                     {"slot": "fact.stops", "value": 0}]),
                (f"Mi servono collegamenti ferroviari tra {city} e {other}, magari diretti.",
                 "find", "treno", "transport", [
                     {"slot": "fact.origin", "value": city},
                     {"slot": "fact.destination", "value": other},
                     {"slot": "fact.stops", "value": 0}]),
                (f"Per {day} sera vorrei prenotare un ristorante a {city}; trovi tu le opzioni?",
                 "book", "ristorante", "place", [
                     {"slot": "fact.location", "value": city},
                     {"slot": "temporal.action_date", "value": day}]),
                (f"Mi trovi un idraulico disponibile {day} per una perdita d'acqua?",
                 "find", "idraulico", "professional_service", [
                     {"slot": "fact.availability", "value": day}]),
                (f"Fammi vedere gli hotel a {city} per un viaggio di lavoro; non prenotare ancora.",
                 "find", "hotel", "accommodation", [
                     {"slot": "fact.location", "value": city},
                     {"slot": "policy.forbid", "value": "book"}]),
                (f"Vorrei un volo per {city} {day}, senza acquisti per ora.",
                 "find", "volo", "transport", [
                     {"slot": "fact.destination", "value": city},
                     {"slot": "temporal.search_date", "value": day},
                     {"slot": "policy.forbid", "value": "buy"}]),
                (f"Ricordami {day} di chiamare il medico per il controllo.",
                 "notify", "promemoria", "reminder", [
                     {"slot": "fact.message", "value": "chiamare il medico per il controllo"},
                     {"slot": "temporal.notification_date", "value": day}]),
            ]
            for text, action, name, kind, updates in variants:
                # The hotel example is a complete search and is assigned elsewhere.
                if action == "find" and name == "hotel":
                    continue
                outputs.append(row("semantic_update", text, {},
                    decision("ACTIONABLE_GOAL", "OPEN_DRAFT", updates, action,
                             {"name": name, "type": kind}),
                    "incomplete", "generated_capability_contrast"))
    rng.shuffle(outputs)
    return outputs


def generated_selections(split: str) -> list[dict]:
    rng = random.Random(SEED + (40 if split == "train" else 41))
    places = CITY if split == "train" else ["Cremona", "Grosseto", "Chieti", "Sassari"]
    forms = ["Preferisco la seconda opzione per {city}.", "Mi interessa la proposta numero due a {city}.",
             "Fra i risultati di {city}, scegli il secondo.", "Mostrami la numero 2 delle offerte per {city}.",
             "A {city} prendiamo in considerazione il secondo risultato.",
             "Per {city} vorrei vedere quello più economico.",
             "Delle opzioni per {city}, quale costa meno?", "Considera il prezzo più basso a {city}."]
    examples = []
    for i, city in enumerate(places):
        for form_index, form in enumerate(forms):
            kind = "ordinal" if form_index < 5 else "cheapest"
            target = {"name": "hotel" if i % 2 else "ristorante",
                      "type": "accommodation" if i % 2 else "place"}
            state = {"active_draft": {"draft_id": "d1", "action": "book", "target": target},
                     "results": [{"jase_result_ref": f"session-{i}-1", "ordinal": 1, "price": 115},
                                 {"jase_result_ref": f"session-{i}-2", "ordinal": 2, "price": 89},
                                 {"jase_result_ref": f"session-{i}-3", "ordinal": 3, "price": 140}]}
            answer = decision("CONTINUATION", "UPDATE_DRAFT", draft_id="d1")
            answer["reference"] = ({"kind": "ordinal", "ordinal": 2} if kind == "ordinal"
                                   else {"kind": "cheapest"})
            examples.append(row("semantic_update", form.format(city=city), state,
                                answer, "selection", "generated_result_reference"))
    rng.shuffle(examples)
    return examples


def generated_multi_goal(split: str) -> list[dict]:
    rng = random.Random(SEED + (50 if split == "train" else 51))
    places = CITY if split == "train" else ["Cremona", "Grosseto", "Chieti", "Sassari",
                                             "Pavia", "Lecco", "Taranto", "Caltanissetta"]
    examples = []
    for i, city in enumerate(places):
        for day in DAY:
            origin = places[(i + 5) % len(places)]
            variants = [
                (f"Per {day} cerca alberghi a {city} e treni in partenza da {origin} verso {city}.",
                 "find", {"name": "hotel", "type": "accommodation"},
                 [{"slot": "fact.location", "value": city}],
                 "find", {"name": "treno", "type": "transport"},
                 [{"slot": "fact.origin", "value": origin}, {"slot": "fact.destination", "value": city},
                  {"slot": "temporal.search_date", "value": day}]),
                (f"A {city} vorrei vedere ristoranti e farmacie per {day}; mostrami entrambe le ricerche.",
                 "find", {"name": "ristorante", "type": "place"},
                 [{"slot": "fact.location", "value": city},
                  {"slot": "temporal.search_date", "value": day}],
                 "find", {"name": "farmacia", "type": "place"},
                 [{"slot": "fact.location", "value": city},
                  {"slot": "temporal.search_date", "value": day}]),
            ]
            for text, action, target, updates, second_action, second_target, second_updates in variants:
                answer = decision("ACTIONABLE_GOAL", "OPEN_DRAFT", updates, action, target)
                answer["additional_goals"] = [{"action": second_action,
                                               "target": second_target, "updates": second_updates}]
                examples.append(row("semantic_update", text, {}, answer,
                                    "multi_goal", "generated_independent_goals"))
    rng.shuffle(examples)
    return examples


def missing_slot(item: dict) -> str | None:
    answer = json.loads(item["messages"][-1]["content"])
    goal = answer.get("target")
    if not goal:
        return None
    text = item["messages"][1]["content"]
    # Derive requirements from the same runtime Registry used by the conversation loop.
    _, cap = REGISTRY.resolve({"action": answer["action"], "target": goal})
    if not cap:
        return None
    present = {u["slot"] for u in answer["updates"]}
    available = sorted(cap.required_slots - present)
    return next((slot for slot in available if slot in {
        "fact.origin", "fact.destination", "fact.location", "external.party_size",
        "temporal.search_date", "temporal.action_date", "temporal.action_time",
        "temporal.notification_time", "fact.message"}), None)


def contextual_and_corrections(source_rows: list[dict], split: str) -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED + (20 if split == "train" else 21))
    contextual, corrections = [], []
    candidates = source_rows[:]
    rng.shuffle(candidates)
    for idx, item in enumerate(candidates):
        answer = json.loads(item["messages"][-1]["content"])
        if answer.get("operation") != "OPEN_DRAFT":
            continue
        action, target = answer["action"], answer["target"]
        known = {u["slot"]: {"value": u["value"], "provenance": "USER_EXPLICIT"}
                 for u in answer["updates"]}
        slot = missing_slot(item)
        if slot:
            value = (CITY[idx % len(CITY)] if slot in {"fact.origin", "fact.destination", "fact.location"}
                     else DAY[idx % len(DAY)] if slot.endswith("date")
                     else f"{8 + idx % 12}:00" if slot.endswith("time")
                     else 2 + idx % 5 if slot == "external.party_size"
                     else "comprare il latte")
            phrase = (f"Da {value}." if slot == "fact.origin" else
                      f"A {value}." if slot in {"fact.destination", "fact.location"} else
                      f"Per {value}." if slot == "external.party_size" else
                      f"{value}." if slot.endswith("date") else
                      f"Alle {value}." if slot.endswith("time") else
                      f"Di {value}.")
            state = {"active_draft": {"draft_id": "d1", "action": action, "target": target,
                                      "slots": known}, "clarification": {"ask_user": [slot]}}
            contextual.append(row("semantic_update", phrase, state,
                decision("ANSWER_TO_CLARIFICATION", "UPDATE_DRAFT",
                         [{"slot": slot, "value": value}], draft_id="d1"),
                "contextual", "registry_generated_context"))
        mutable = next((u for u in answer["updates"] if u["slot"] in {
            "fact.origin", "fact.destination", "fact.location", "temporal.search_date",
            "temporal.action_date", "external.party_size"}), None)
        if mutable:
            slot = mutable["slot"]
            new_value = CITY[(idx + 9) % len(CITY)] if slot.startswith("fact.") else (
                DAY[(idx + 2) % len(DAY)] if slot.startswith("temporal.") else 3 + idx % 4)
            if new_value != mutable["value"]:
                phrase = f"Anzi, {new_value} invece di {mutable['value']}."
                state = {"active_draft": {"draft_id": "d1", "action": action, "target": target,
                                          "slots": known}}
                corrections.append(row("semantic_update", phrase, state,
                    decision("CONTINUATION", "UPDATE_DRAFT", [{"slot": slot, "value": new_value}],
                             draft_id="d1"), "correction", "generated_state_transition"))
        if idx % 3 == 0:
            state = {"active_draft": {"draft_id": "d1", "action": action, "target": target,
                                      "slots": known}}
            phrase = ["Lascia perdere questa richiesta.", "Non voglio più procedere.",
                      "Annulla il piano, ho cambiato idea.", "Fermiamoci qui, niente più ricerca."][idx % 4]
            corrections.append(row("semantic_update", phrase, state,
                decision("CANCELLATION", "CANCEL_DRAFT", draft_id="d1"),
                "cancellation", "generated_state_transition"))
        if idx % 4 == 0 and action in {"book", "contact", "hire", "rent"}:
            state = {"active_draft": {"draft_id": "d1", "action": action, "target": target,
                                      "slots": known, "confirmation": {"status": "PENDING"}}}
            phrase = ["Sì, confermo esattamente questa scelta.", "Va bene, procedi con quella.",
                      "Confermo i dettagli mostrati."][idx % 3]
            corrections.append(row("semantic_update", phrase, state,
                decision("CONFIRMATION", "CONFIRM_DRAFT", draft_id="d1"),
                "confirmation", "generated_state_transition"))
    rng.shuffle(contextual)
    rng.shuffle(corrections)
    return contextual, corrections


def plan_verbalization(source_rows: list[dict], split: str) -> list[dict]:
    result = []
    for idx, item in enumerate(source_rows):
        answer = json.loads(item["messages"][-1]["content"])
        if answer.get("operation") != "OPEN_DRAFT":
            continue
        state = ConversationState()
        draft = GoalDraft("d1", 1, "COLLECTING", answer["action"], answer["target"])
        draft.slots = {u["slot"]: {"value": u["value"], "provenance": "USER_EXPLICIT"}
                       for u in answer["updates"]}
        state.drafts["d1"] = draft
        state.active_draft_ids.append("d1")
        plan = clarification_plan(state, "d1", registry=REGISTRY)
        if not (plan["ask_user"] or plan["discover_read_only"]):
            continue
        # Stage B gets authoritative field names; it cannot add or remove them.
        response = {"reply": render_clarification(plan)}
        result.append(row("verbalize_plan", "", {"action": draft.action, "target": draft.target,
            "known_slots": {name: value["value"] for name, value in draft.slots.items()},
            "ask_user": plan["ask_user"], "discover_read_only": plan["discover_read_only"],
            "reason_codes": plan["reason_codes"]},
            response, "verbalization", "registry_generated_feedback"))
    return result


def audit_leakage(rows: list[dict]) -> dict:
    panel = [json.loads(line) for line in (ROOT / "benchmarks/v37_conversational/blind.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    forbidden = [" ".join(turn["user"] for turn in case["turns"]) for case in panel]
    forbidden += [turn["user"] for case in panel for turn in case["turns"]]
    for path in sorted(ROOT.glob("benchmarks/v*/blind.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            forbidden.append(item.get("text", item.get("request", "")))
    prior = {normalize_text(text) for text in forbidden if text}
    forbidden_tokens = [token_set(text) for text in forbidden if text]
    forbidden_tokens = [words for words in forbidden_tokens if len(words) >= 5]
    nearest = 0.0
    exact = 0
    for item in rows:
        user = json.loads(item["messages"][1]["content"]).get("user", "")
        if not user:
            continue
        if normalize_text(user) in prior:
            exact += 1
        words = token_set(user)
        if len(words) < 5:
            continue
        nearest = max(nearest, *(len(words & other) / len(words | other)
                                  for other in forbidden_tokens))
    if exact or nearest >= 0.85:
        raise ValueError(f"Benchmark leakage: exact={exact}, max_jaccard={nearest:.3f}")
    return {"exact_benchmark_overlap": exact, "max_benchmark_token_jaccard": round(nearest, 6)}


def build_split(split: str) -> tuple[list[dict], dict]:
    rng = random.Random(SEED + (100 if split == "train" else 101))
    legacy = legacy_examples(split)
    contextual, corrections = contextual_and_corrections(
        legacy["incomplete"] + legacy["complete"], split)
    nonaction = generated_nonaction(split) + legacy["nonaction"]
    verbalization = plan_verbalization(legacy["incomplete"] + legacy["complete"], split)
    if split == "train":
        quotas = {"nonaction": 650, "incomplete": 540, "contextual": 530,
                  "correction": 550, "selection": 180, "multi_goal": 180,
                  "complete": 650, "verbalization": 500}
    else:
        quotas = {"nonaction": 160, "incomplete": 110, "contextual": 110,
                  "correction": 100, "selection": 28, "multi_goal": 32,
                  "complete": 120, "verbalization": 90}
    sources = {"nonaction": nonaction, "incomplete": legacy["incomplete"] + generated_incomplete(split),
               "contextual": contextual, "correction": corrections,
               "selection": generated_selections(split),
               "multi_goal": generated_multi_goal(split),
               "complete": legacy["complete"], "verbalization": verbalization}
    output = []
    counts = {}
    for category, target in quotas.items():
        candidates = sources[category]
        rng.shuffle(candidates)
        seen = set()
        selected = []
        for candidate in candidates:
            key = candidate["messages"][1]["content"]
            if key in seen:
                continue
            seen.add(key)
            selected.append(candidate)
            if len(selected) == target:
                break
        counts[category] = {"target": target, "actual": len(selected), "available": len(candidates)}
        if len(selected) < target:
            raise ValueError(f"Insufficient {category} for {split}: {counts[category]}")
        output += selected
    rng.shuffle(output)
    return output, counts


def main() -> None:
    train, train_counts = build_split("train")
    valid, valid_counts = build_split("valid")
    train_prompt = {row["messages"][1]["content"] for row in train}
    valid = [row for row in valid if row["messages"][1]["content"] not in train_prompt]
    leakage = audit_leakage(train + valid)
    DEST.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid)):
        (DEST / f"{name}.jsonl").write_text("".join(compact(row) + "\n" for row in rows),
                                              encoding="utf-8", newline="\n")
    manifest = {"version": "V27_CONVERSATIONAL", "seed": SEED,
                "source": "V26 train/valid semantic projection plus independent contextual and discourse generation",
                "v37_sha256": hashlib.sha256((ROOT / "benchmarks/v37_conversational/blind.jsonl").read_bytes()).hexdigest(),
                "splits": {"train": {"rows": len(train), "sha256": hashlib.sha256((DEST / "train.jsonl").read_bytes()).hexdigest(),
                                     "categories": dict(Counter(row["category"] for row in train)), "quotas": train_counts},
                           "valid": {"rows": len(valid), "sha256": hashlib.sha256((DEST / "valid.jsonl").read_bytes()).hexdigest(),
                                     "categories": dict(Counter(row["category"] for row in valid)), "quotas": valid_counts}},
                "leakage": leakage, "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "human_review": "not independently reviewed", "note": "Stage-A and Stage-B tasks share one model; no benchmark gold used as training target"}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"train": len(train), "valid": len(valid), "leakage": leakage,
                      "train_counts": train_counts, "valid_counts": valid_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
