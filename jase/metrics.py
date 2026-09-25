"""Evaluator semantico v2.

Confronta un contratto predetto con quello atteso al livello che conta: i
FATTI dell'utente e il loro ruolo. Una differenza cosmetica
(`target.raw = "panino"` vs `"un panino"`) non pesa come una semantica
(prezzo finito in `attributes` invece che in `constraints`).

Gerarchia di confronto, dal più permissivo al più severo:
    fact_key   (property, value, unit, currency)  → è lo stesso fatto?
    + bucket                                      → nel ruolo giusto?
    + operator                                    → con la relazione giusta?
    + reference                                   → rispetto alla cosa giusta?
"""
from __future__ import annotations

from typing import Any

from .canon import canon_contract, canon_predicate, fact_key, norm_text
from .guard import value_is_grounded

BUCKETS = ("attributes", "constraints", "preferences")


def _equiv_map(equivalences: list[list[str]] | None) -> dict[str, str]:
    """Classi di proprietà considerate intercambiabili PER QUESTO caso.

    Serve dove non esiste una sola risposta canonica: `"32gb"` su un
    dispositivo sconosciuto può essere RAM o archiviazione, e nessun umano
    saprebbe dirlo. Il benchmark dichiara l'ambiguità invece di premiare una
    delle due a caso — e dichiararla è anche il modo di non nasconderla.
    """
    out: dict[str, str] = {}
    for group in equivalences or []:
        canon = norm_text(sorted(group)[0])
        for name in group:
            out[norm_text(name)] = canon
    return out


def _flatten(goal: dict[str, Any],
             equiv: dict[str, str] | None = None) -> list[tuple[str, dict[str, Any]]]:
    out = []
    for b in BUCKETS:
        for p in goal.get(b) or []:
            if isinstance(p, dict):
                cp = canon_predicate(p)
                if equiv:
                    cp["property"] = equiv.get(cp["property"], cp["property"])
                out.append((b, cp))
    return out


def _match_facts(exp: list[tuple[str, dict]], got: list[tuple[str, dict]]):
    """Allinea i fatti attesi con quelli predetti.

    Ritorna (matched, missing, spurious, value_errors) dove `matched` è una
    lista di coppie ((bucket_exp, pred_exp), (bucket_got, pred_got)).
    """
    used_got = [False] * len(got)
    matched: list[tuple[tuple[str, dict], tuple[str, dict]]] = []
    value_errors: list[tuple[tuple[str, dict], tuple[str, dict]]] = []
    unmatched_exp: list[tuple[str, dict]] = []

    # passata 1 — stesso fatto esatto (property+value+unit+currency)
    for e in exp:
        ek = fact_key(e[1])
        hit = -1
        for i, g in enumerate(got):
            if used_got[i]:
                continue
            if fact_key(g[1]) == ek:
                hit = i
                break
        if hit >= 0:
            used_got[hit] = True
            matched.append((e, got[hit]))
        else:
            unmatched_exp.append(e)

    # passata 2 — stessa property, valore diverso → errore di valore, non fatto mancante
    still: list[tuple[str, dict]] = []
    for e in unmatched_exp:
        prop = e[1]["property"]
        hit = -1
        for i, g in enumerate(got):
            if used_got[i]:
                continue
            if g[1]["property"] == prop:
                hit = i
                break
        if hit >= 0:
            used_got[hit] = True
            value_errors.append((e, got[hit]))
        else:
            still.append(e)

    spurious = [got[i] for i, u in enumerate(used_got) if not u]
    return matched, still, spurious, value_errors


def _align_goals(exp_goals: list[dict], got_goals: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Allinea i goal per target (non per indice): l'ordine non è semantico."""
    pairs: list[tuple[dict | None, dict | None]] = []
    used = [False] * len(got_goals)
    for eg in exp_goals:
        ename = norm_text((eg.get("target") or {}).get("name"))
        hit = -1
        for i, gg in enumerate(got_goals):
            if used[i]:
                continue
            if norm_text((gg.get("target") or {}).get("name")) == ename:
                hit = i
                break
        if hit < 0:  # nessun target uguale: si ripiega sull'ordine
            for i, gg in enumerate(got_goals):
                if not used[i]:
                    hit = i
                    break
        if hit >= 0:
            used[hit] = True
            pairs.append((eg, got_goals[hit]))
        else:
            pairs.append((eg, None))
    for i, gg in enumerate(got_goals):
        if not used[i]:
            pairs.append((None, gg))
    return pairs


def compare(expected: dict[str, Any], predicted: dict[str, Any],
            user_text: str = "",
            equivalences: list[list[str]] | None = None) -> dict[str, Any]:
    """Confronto semantico di due contratti. Ritorna i contatori grezzi.

    `equivalences` dichiara, solo per questo caso, gruppi di property che
    valgono come la stessa cosa (vedi `_equiv_map`)."""
    equiv = _equiv_map(equivalences)
    r: dict[str, Any] = {
        "goal_count_ok": int(len(expected.get("goals") or []) == len(predicted.get("goals") or [])),
        "action_ok": 0, "action_total": 0,
        "target_name_ok": 0, "target_type_ok": 0, "target_total": 0,
        "fact_tp": 0, "fact_fn": 0, "fact_fp": 0, "fact_value_err": 0,
        "bucket_ok": 0, "bucket_total": 0,
        "operator_ok": 0, "operator_total": 0,
        "unit_ok": 0, "unit_total": 0,
        "currency_ok": 0, "currency_total": 0,
        "reference_ok": 0, "reference_total": 0,
        "hallucinated_ungrounded": 0,
        "duplicate_facts": 0,
        "clarification_ok": 0,
        "gate_ok": 0, "gate_total": 0,
        "semantic_exact": 0,
        "cosmetic_exact": 0,
        "errors": [],
    }
    for bucket in BUCKETS:
        r[f"{bucket}_tp"] = 0
        r[f"{bucket}_fp"] = 0
        r[f"{bucket}_fn"] = 0

    pairs = _align_goals(expected.get("goals") or [], predicted.get("goals") or [])

    for eg, gg in pairs:
        if eg is None or gg is None:
            # goal in eccesso o mancante: tutti i suoi fatti contano come errore
            side = eg or gg
            n = sum(len(side.get(b) or []) for b in BUCKETS)
            if eg is None:
                r["fact_fp"] += n
                r["errors"].append("extra_goal")
            else:
                r["fact_fn"] += n
                r["errors"].append("missing_goal")
            r["target_total"] += 1
            r["action_total"] += 1
            continue

        r["action_total"] += 1
        if norm_text(eg.get("action")) == norm_text(gg.get("action")):
            r["action_ok"] += 1
        else:
            r["errors"].append("action")

        et, gt = eg.get("target") or {}, gg.get("target") or {}
        r["target_total"] += 1
        if norm_text(et.get("name")) == norm_text(gt.get("name")):
            r["target_name_ok"] += 1
        else:
            r["errors"].append("target_name")
        if norm_text(et.get("type")) == norm_text(gt.get("type")):
            r["target_type_ok"] += 1
        else:
            r["errors"].append("target_type")

        # gate: presenza, condizione e azioni dei due rami (§11)
        from .canon import canon_gate
        ege, gge = canon_gate(eg.get("gate")), canon_gate(gg.get("gate"))
        if ege is not None or gge is not None:
            r["gate_total"] += 1
            if ege == gge:
                r["gate_ok"] += 1
            elif ege is None:
                r["errors"].append("gate_spurious")
            elif gge is None:
                r["errors"].append("gate_missing")
            else:
                if ege["when"] != gge["when"]:
                    r["errors"].append("gate_condition")
                if ege["then"] != gge["then"]:
                    r["errors"].append("gate_then")
                if ege["otherwise"] != gge["otherwise"]:
                    r["errors"].append("gate_otherwise")

        exp_f, got_f = _flatten(eg, equiv), _flatten(gg, equiv)

        # duplicati nel predetto
        seen_keys = set()
        for _b, p in got_f:
            k = fact_key(p)
            if k in seen_keys:
                r["duplicate_facts"] += 1
                r["errors"].append("duplicate_fact")
            seen_keys.add(k)

        matched, missing, spurious, value_errors = _match_facts(exp_f, got_f)

        r["fact_tp"] += len(matched)
        r["fact_fn"] += len(missing)
        r["fact_fp"] += len(spurious)
        r["fact_value_err"] += len(value_errors)
        if missing:
            r["errors"].append("missing_fact")
        if spurious:
            r["errors"].append("hallucinated_fact")
        if value_errors:
            r["errors"].append("value_error")

        for (eb, ep), (gb, gp) in matched:
            r["bucket_total"] += 1
            if eb == gb:
                r["bucket_ok"] += 1
                r[f"{eb}_tp"] += 1
            else:
                r["errors"].append(f"bucket:{eb}->{gb}")
                r[f"{eb}_fn"] += 1
                r[f"{gb}_fp"] += 1
            r["operator_total"] += 1
            if ep["operator"] == gp["operator"]:
                r["operator_ok"] += 1
            else:
                r["errors"].append(f"operator:{ep['operator']}->{gp['operator']}")
            if ep["unit"] is not None or gp["unit"] is not None:
                r["unit_total"] += 1
                r["unit_ok"] += int(ep["unit"] == gp["unit"])
                if ep["unit"] != gp["unit"]:
                    r["errors"].append("unit")
            if ep["currency"] is not None or gp["currency"] is not None:
                r["currency_total"] += 1
                r["currency_ok"] += int(ep["currency"] == gp["currency"])
                if ep["currency"] != gp["currency"]:
                    r["errors"].append("currency")
            if ep["reference"] is not None or gp["reference"] is not None:
                r["reference_total"] += 1
                r["reference_ok"] += int(ep["reference"] == gp["reference"])
                if ep["reference"] != gp["reference"]:
                    r["errors"].append("reference")

        for eb, ep in missing:
            r[f"{eb}_fn"] += 1
        for gb, gp in spurious:
            r[f"{gb}_fp"] += 1
            if user_text and not value_is_grounded(gp["value"], user_text):
                r["hallucinated_ungrounded"] += 1
                r["errors"].append("ungrounded_value")
        for (eb, ep), (gb, gp) in value_errors:
            r[f"{eb}_fn"] += 1
            r[f"{gb}_fp"] += 1
            if user_text and not value_is_grounded(gp["value"], user_text):
                r["hallucinated_ungrounded"] += 1
                r["errors"].append("ungrounded_value")

    ec = expected.get("clarification") or {}
    gc = predicted.get("clarification") or {}
    if bool(ec.get("required")) == bool(gc.get("required")):
        if not ec.get("required") or norm_text(ec.get("reason")) == norm_text(gc.get("reason")):
            r["clarification_ok"] = 1
    if not r["clarification_ok"]:
        r["errors"].append("clarification")

    if equiv:
        ce, cg = canon_contract(expected), canon_contract(predicted)
        for cc in (ce, cg):
            for g in cc["goals"]:
                for b in BUCKETS:
                    for pp in g[b]:
                        pp["property"] = equiv.get(pp["property"], pp["property"])
                    g[b].sort(key=lambda d: tuple(str(d[k]) for k in sorted(d)))
        r["semantic_exact"] = int(ce == cg)
    else:
        r["semantic_exact"] = int(canon_contract(expected) == canon_contract(predicted))
    r["cosmetic_exact"] = int(expected == predicted)
    r["errors"] = sorted(set(r["errors"]))
    return r


def aggregate(rows: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Somma i contatori e calcola i tassi. `rows` = output di `compare`."""
    n = max(1, len(rows))
    keys = [k for k in rows[0] if isinstance(rows[0][k], int)] if rows else []
    tot = {k: sum(r.get(k, 0) for r in rows) for k in keys}

    def rate(num: str, den: str) -> float:
        d = tot.get(den, 0)
        return round(tot.get(num, 0) / d, 4) if d else 1.0

    def per_ex(k: str) -> float:
        return round(tot.get(k, 0) / n, 4)

    prec = lambda tp, fp: round(tp / (tp + fp), 4) if (tp + fp) else 1.0  # noqa: E731
    rec = lambda tp, fn: round(tp / (tp + fn), 4) if (tp + fn) else 1.0   # noqa: E731

    out: dict[str, Any] = {
        "examples": len(rows),
        "goals": {
            "goal_count_rate": per_ex("goal_count_ok"),
            "action_accuracy": rate("action_ok", "action_total"),
            "target_name_accuracy": rate("target_name_ok", "target_total"),
            "target_type_accuracy": rate("target_type_ok", "target_total"),
        },
        "facts": {
            "precision": prec(tot.get("fact_tp", 0), tot.get("fact_fp", 0) + tot.get("fact_value_err", 0)),
            "recall": rec(tot.get("fact_tp", 0), tot.get("fact_fn", 0) + tot.get("fact_value_err", 0)),
            "tp": tot.get("fact_tp", 0),
            "fp": tot.get("fact_fp", 0),
            "fn": tot.get("fact_fn", 0),
            "value_errors": tot.get("fact_value_err", 0),
        },
        "roles": {
            "bucket_accuracy": rate("bucket_ok", "bucket_total"),
            "operator_accuracy": rate("operator_ok", "operator_total"),
            "unit_accuracy": rate("unit_ok", "unit_total"),
            "currency_accuracy": rate("currency_ok", "currency_total"),
            "reference_accuracy": rate("reference_ok", "reference_total"),
        },
        "buckets": {},
        "fidelity": {
            "missing_fact_rate": per_ex("fact_fn"),
            "hallucinated_fact_rate": per_ex("fact_fp"),
            "ungrounded_value_rate": per_ex("hallucinated_ungrounded"),
            "duplicate_fact_rate": per_ex("duplicate_facts"),
        },
        "contract": {
            "semantic_exact_rate": per_ex("semantic_exact"),
            "cosmetic_exact_rate": per_ex("cosmetic_exact"),
            "clarification_accuracy": per_ex("clarification_ok"),
            "gate_accuracy": rate("gate_ok", "gate_total"),
        },
    }
    for b in BUCKETS:
        tp, fp, fn = tot.get(f"{b}_tp", 0), tot.get(f"{b}_fp", 0), tot.get(f"{b}_fn", 0)
        out["buckets"][b] = {"precision": prec(tp, fp), "recall": rec(tp, fn),
                             "tp": tp, "fp": fp, "fn": fn}
    if extra:
        out.update(extra)
    return out


def failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
    tax: dict[str, int] = {}
    for r in rows:
        for e in r.get("errors", []):
            tax[e] = tax.get(e, 0) + 1
    return dict(sorted(tax.items(), key=lambda kv: -kv[1]))
