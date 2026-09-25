#!/usr/bin/env python3
"""Evaluator v2 — valutazione semantica su benchmark gold o dataset.

Misura DUE cose distinte, che il v1 confondeva:

  RAW      quanto è bravo il modello da solo
  GUARDED  quanto è affidabile la pipeline completa
           (modello + Semantic Guard fail-closed)

La seconda è quella che conta per la produzione: un contratto che il Guard
ferma non è un successo, ma non è nemmeno un danno — è una richiesta di
chiarimento. Un contratto sbagliato che passa, quello sì è un danno.

Uso tipico:
    python scripts/eval_v2.py --adapter adapters/jase-v2 \
        --data benchmarks/gold_v1.jsonl --report reports/v2-iter600.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from jase.canon import ExtractionError, extract_contract  # noqa: E402
from jase.guard import ACCEPT, CLARIFY, REJECT, REPAIRED, check_contract  # noqa: E402
from jase.lexicon import SEMANTICS_VERSION  # noqa: E402
from jase.metrics import aggregate, compare, failure_taxonomy  # noqa: E402

TRAIN_SYSTEM = (ROOT / "TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()


def load_cases(path: Path) -> list[dict]:
    """Accetta sia il formato gold (id/text/contract/tags) sia il chat jsonl."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            user = next(m["content"] for m in row["messages"] if m["role"] == "user")
            expected = json.loads(row["messages"][-1]["content"])
            cases.append({"id": f"row{len(cases)}", "text": user,
                          "contract": expected, "tags": [], "mandatory": False})
        else:
            cases.append(row)
    return cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="benchmarks/gold_v1.jsonl")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=0, help="0 = tutti")
    ap.add_argument("--max-tokens", type=int, default=420)
    ap.add_argument("--report", default=None)
    ap.add_argument("--label", default=None)
    ap.add_argument("--seed", type=int, default=2026,
                    help="seed esplicito anche per eventuali retry della pipeline")
    ap.add_argument("--pipeline", action="store_true",
                    help="valuta la PIPELINE di produzione (scripts/interpret.py, "
                         "con i suoi tentativi) invece del singolo colpo. È ciò "
                         "che arriva davvero al Planner: i gate di release "
                         "parlano di quello, non di una generazione sola.")
    ap.add_argument("--system", default=None,
                    help="file di system prompt alternativo (diagnostica)")
    ap.add_argument("--no-failures", action="store_true",
                    help="non scrivere l'elenco dei casi falliti nel report. "
                         "Obbligatorio sul release holdout: se i testi dei "
                         "casi falliti finiscono in un file, prima o poi "
                         "qualcuno li legge e li corregge, e l'holdout "
                         "smette di misurare la generalizzazione.")
    ap.add_argument("--benchmark-role",
                    choices=("dev_regression", "release_holdout"),
                    default="dev_regression",
                    help="ruolo statistico del dataset: il gold_v1 storico è "
                         "dev/regression, non un holdout indipendente")
    args = ap.parse_args()
    if args.benchmark_role == "release_holdout" and not args.no_failures:
        ap.error("release_holdout richiede --no-failures")

    import mlx.core as mx
    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    system = (Path(args.system).read_text(encoding="utf-8").strip()
              if args.system else TRAIN_SYSTEM)
    cases = load_cases(ROOT / args.data if not Path(args.data).is_absolute()
                       else Path(args.data))
    if args.limit:
        cases = cases[:args.limit]

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    sampler = make_sampler(temp=0.0)  # deterministico: due run devono coincidere
    mx.random.seed(args.seed)

    raw_rows, guarded_rows = [], []
    guard_status: dict[str, int] = {}
    extraction_errors: dict[str, int] = {}
    per_tag: dict[str, list] = {}
    failures: list[dict] = []
    mandatory_total = mandatory_ok = 0
    decisions = {"clarification_true_positive": 0, "clarification_false_positive": 0,
                 "clarification_false_negative": 0, "clarification_true_negative": 0,
                 "actionable": 0, "actionable_but_wrong": 0}
    t0 = time.time()

    for i, c in enumerate(cases):
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": c["text"]}]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        pipeline_status = None
        if args.pipeline:
            from interpret import interpret as _interpret
            pres = _interpret(model, tokenizer, c["text"],
                              max_tokens=args.max_tokens, retries=2)
            # La pipeline ha QUATTRO esiti, non uno. Appiattirli tutti su
            # "ecco un contratto" faceva contare come accettato anche ciò
            # che era stato fermato: la quota di contratti bloccati usciva a
            # zero per costruzione e i contratti sbagliati sembravano
            # aumentare. Misurare male è peggio che non misurare.
            pipeline_status = {"accepted": ACCEPT, "repaired": REPAIRED,
                               "clarify": CLARIFY,
                               "rejected": REJECT}.get(pres["status"], REJECT)
            out = json.dumps(pres.get("contract") or {}, ensure_ascii=False)
        else:
            out = generate(model, tokenizer, prompt=prompt,
                           max_tokens=args.max_tokens, sampler=sampler,
                           verbose=False)

        expected = c["contract"]
        # ---------- RAW
        try:
            got = extract_contract(out)
            raw = compare(expected, got, c["text"], c.get("equivalences"))
        except ExtractionError as exc:
            extraction_errors[exc.code] = extraction_errors.get(exc.code, 0) + 1
            got = None
            raw = compare(expected, {"schema_version": "1.0", "goals": [],
                                     "clarification": {"required": False,
                                                       "reason": None,
                                                       "missing_fields": []}},
                          c["text"])
            raw["errors"] = sorted(set(raw["errors"] + [f"extraction:{exc.code}"]))
        raw_rows.append(raw)

        # ---------- GUARDED
        res = check_contract(got, c["text"]) if got is not None else None
        status = res.status if res else REJECT
        if pipeline_status is not None:
            status = pipeline_status
            if status in (CLARIFY, REJECT):
                res = None
        guard_status[status] = guard_status.get(status, 0) + 1
        if res and res.ok:
            gr = compare(expected, res.contract, c["text"],
                         c.get("equivalences"))
        else:
            # il Guard ha fermato tutto: nessun contratto raggiunge il Planner.
            # Non è un successo semantico, ma non è un contratto sbagliato.
            gr = dict(raw)
            gr["errors"] = sorted(set(gr["errors"] + [f"guard:{status}"]))
            gr["semantic_exact"] = 0
        guarded_rows.append(gr)

        expected_clarify = bool(expected.get("clarification", {}).get("required"))
        delivered = res.contract if res and res.ok else (got or {})
        asks_clarify = (status in (CLARIFY, REJECT) or
                       bool(delivered.get("clarification", {}).get("required")))
        decision_key = ("true_positive" if expected_clarify and asks_clarify else
                        "false_negative" if expected_clarify else
                        "false_positive" if asks_clarify else "true_negative")
        decisions["clarification_" + decision_key] += 1
        if status in (ACCEPT, REPAIRED) and not asks_clarify:
            decisions["actionable"] += 1
            decisions["actionable_but_wrong"] += int(not gr["semantic_exact"])

        ok = bool(gr["semantic_exact"])
        if c.get("mandatory"):
            mandatory_total += 1
            mandatory_ok += ok
        for t in c.get("tags", []):
            per_tag.setdefault(t, []).append(gr)

        if not ok and not args.no_failures and len(failures) < 60:
            failures.append({
                "id": c.get("id"), "text": c["text"], "tags": c.get("tags", []),
                "mandatory": c.get("mandatory", False),
                "guard": status,
                "guard_issues": [i.code for i in res.issues] if res else [],
                "guard_repairs": res.repairs if res else [],
                "errors": gr["errors"],
                "expected": expected,
                "got": got,
            })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(cases)}", file=sys.stderr, flush=True)

    elapsed = time.time() - t0
    report = {
        "label": args.label or (args.adapter or "base"),
        "model": args.model,
        "adapter": args.adapter,
        "data": args.data,
        "benchmark_role": args.benchmark_role,
        "semantics_version": SEMANTICS_VERSION,
        "pipeline": args.pipeline,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "data_sha256": hashlib.sha256((ROOT / args.data).read_bytes()).hexdigest(),
        "system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "guard_sha256": hashlib.sha256((ROOT / "jase/guard.py").read_bytes()).hexdigest(),
        "decisions": decisions,
        "system_prompt_chars": len(system),
        "examples": len(cases),
        "seconds": round(elapsed, 1),
        "guard": {
            "status": guard_status,
            "extraction_errors": extraction_errors,
            "blocked_rate": round(
                (guard_status.get(CLARIFY, 0) + guard_status.get(REJECT, 0))
                / max(1, len(cases)), 4),
            "repaired_rate": round(guard_status.get(REPAIRED, 0) / max(1, len(cases)), 4),
            "accepted_rate": round(guard_status.get(ACCEPT, 0) / max(1, len(cases)), 4),
        },
        "raw": aggregate(raw_rows),
        "guarded": aggregate(guarded_rows),
        "release_gates": {
            "mandatory_total": mandatory_total,
            "mandatory_passed": mandatory_ok,
            "mandatory_rate": round(mandatory_ok / max(1, mandatory_total), 4),
            "json_validity": round(
                1 - sum(extraction_errors.values()) / max(1, len(cases)), 4),
            "no_wrong_contract_reaches_planner": None,  # calcolato sotto
        },
        "by_tag": {
            t: {"n": len(rows),
                "semantic_exact_rate": round(
                    sum(r["semantic_exact"] for r in rows) / len(rows), 4),
                "facts": aggregate(rows)["facts"],
                "goals": aggregate(rows)["goals"],
                "accepted_but_wrong": sum(
                    not r["semantic_exact"] and not any(
                        e.startswith("guard:") for e in r["errors"]) for r in rows)}
            for t, rows in sorted(per_tag.items())
        },
        "failure_taxonomy": failure_taxonomy(guarded_rows),
        "failures": failures,
    }

    # invariante fail-closed: tra i contratti ACCETTATI dal Guard, quanti
    # erano semanticamente sbagliati? Questo numero deve tendere a 0: un
    # contratto fermato è recuperabile, uno sbagliato che passa no.
    accepted_wrong = 0
    for gr in guarded_rows:
        blocked = any(e.startswith("guard:") for e in gr["errors"])
        if not blocked and not gr["semantic_exact"]:
            accepted_wrong += 1
    report["release_gates"]["accepted_but_wrong"] = accepted_wrong
    report["release_gates"]["no_wrong_contract_reaches_planner"] = accepted_wrong == 0

    out_path = Path(args.report) if args.report else None
    if out_path:
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                            encoding="utf-8")

    summary = {k: report[k] for k in
               ("label", "examples", "seconds", "guard", "release_gates")}
    summary["guarded"] = report["guarded"]
    summary["failure_taxonomy"] = dict(list(report["failure_taxonomy"].items())[:12])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if out_path:
        print(f"\nreport completo: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
