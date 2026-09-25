#!/usr/bin/env python3
"""Pipeline di produzione: testo utente → GoalContract verificato.

    user text → Interpreter (LoRA) → Semantic Guard → esito

L'esito è SEMPRE uno di questi, mai un contratto "probabilmente giusto":

    accepted    contratto valido, può andare al Planner
    repaired    valido dopo riparazioni deterministiche (elencate)
    clarify     fermato: serve una domanda all'utente
    rejected    output inutilizzabile, dopo N tentativi

Il retry è l'unico meccanismo non deterministico ed è limitato: se il
modello continua a produrre contratti che il Guard ferma, si chiede
chiarimento. Meglio una domanda in più che un'azione sbagliata nel mondo.

    python scripts/interpret.py --adapter adapters/jase-v2 \
        "Trovami un panino con la mortadella a Milano sotto i 10 euro"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.canon import ExtractionError, extract_contract  # noqa: E402
from jase.guard import (  # noqa: E402
    ACCEPT, CLARIFY, REJECT, REPAIRED, check_contract, clarification_contract,
)

SYSTEM = (ROOT / "TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()


def interpret(model, tokenizer, text: str, *, max_tokens=420, retries=2,
              temps=(0.0, 0.3, 0.6)) -> dict:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    # Il retry vale SOLO per i guasti strutturali (JSON rotto, output
    # degenerato). Per un errore semantico no: se il Guard dice che il
    # contratto contraddice il testo, rigenerare a temperatura più alta non
    # lo rende più giusto, lo rende più fortunato — è campionare finché il
    # controllo non si distrae. Misurato sul gold: con il retry su tutto, la
    # quota di contratti fermati andava a ZERO e i contratti sbagliati che
    # arrivavano al Planner salivano da 12 a 16. Il guadagno sui casi
    # mandatory (0.929 → 0.941) era pagato con l'invariante più importante.
    attempts = []
    last_candidate = None
    for attempt in range(retries + 1):
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": text}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        sampler = make_sampler(temp=temps[min(attempt, len(temps) - 1)])
        out = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens,
                       sampler=sampler, verbose=False)
        try:
            candidate = extract_contract(out)
            last_candidate = candidate
        except ExtractionError as exc:
            attempts.append({"attempt": attempt, "status": REJECT,
                             "reason": f"extraction:{exc.code}"})
            continue

        res = check_contract(candidate, text)
        attempts.append({"attempt": attempt, "status": res.status,
                         "issues": [i.as_dict() for i in res.issues],
                         "repairs": res.repairs})
        if res.status == CLARIFY:
            # errore semantico: si chiede, non si ritenta
            break
        if res.ok:
            return {"status": "accepted" if res.status == ACCEPT else "repaired",
                    "contract": res.contract, "repairs": res.repairs,
                    "warnings": [i.as_dict() for i in res.issues
                                 if i.severity == "warn"],
                    "attempts": attempts}

    # fail-closed: nessun tentativo ha prodotto un contratto affidabile
    last = attempts[-1] if attempts else {}
    codes = {i.get("code") for i in last.get("issues", [])}
    if "hallucinated_target" in codes or "empty_target" in codes:
        reason, missing = "target_missing", ["target"]
    elif "hallucinated_raw_action" in codes:
        reason, missing = "action_ambiguous", ["action"]
    elif codes & {"hallucinated_value", "hallucinated_currency",
                  "operator_polarity_mismatch",
                  "operator_strictness_mismatch", "number_role_mismatch",
                  "conflicting_predicates"}:
        reason, missing = "conflicting_constraints", ["constraints"]
    elif "unaccounted_fact" in codes:
        reason, missing = "goal_unintelligible", ["facts"]
    else:
        reason, missing = "goal_unintelligible", ["goal"]
    return {"status": "clarify",
            "contract": clarification_contract(text, reason, missing,
                                               candidate=last_candidate),
            "attempts": attempts,
            "message": "Non riesco a interpretare la richiesta in modo "
                       "affidabile: meglio chiedere un chiarimento che "
                       "procedere con un contratto incerto."}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?",
                    default="Trovami un panino con la mortadella a Milano sotto i 10 euro")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    from mlx_lm import load
    model, tokenizer = load(args.model, adapter_path=args.adapter)
    result = interpret(model, tokenizer, args.request,
                       max_tokens=args.max_tokens, retries=args.retries)
    if not args.verbose:
        result.pop("attempts", None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in ("accepted", "repaired") else 2


if __name__ == "__main__":
    raise SystemExit(main())
