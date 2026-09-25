#!/usr/bin/env python3
"""Controlla contaminazione fra training/validation e benchmark.

Esempi:
  python scripts/leakage_check.py data/v17/train.jsonl benchmarks/gold_v1.jsonl
  python scripts/leakage_check.py data/v19_clean/train.jsonl benchmarks/gold_v1.jsonl \
      --near-threshold 0.80 --fail-on-near

Per default fallisce soltanto sugli exact duplicate. I near duplicate vengono
stampati per revisione perché una somiglianza lessicale non implica leakage.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.leakage import load_examples, normalize_text, token_jaccard, token_set  # noqa: E402


def contract_key(c):
    if c is None:
        return None
    return json.dumps(c, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("train")
    ap.add_argument("against")
    ap.add_argument("--near-threshold", type=float, default=0.80)
    ap.add_argument("--min-tokens", type=int, default=4)
    ap.add_argument("--max-near", type=int, default=40)
    ap.add_argument("--fail-on-near", action="store_true")
    ap.add_argument("--json-report", default=None)
    args = ap.parse_args()

    train_path = Path(args.train)
    against_path = Path(args.against)
    if not train_path.is_absolute():
        train_path = ROOT / train_path
    if not against_path.is_absolute():
        against_path = ROOT / against_path

    train = load_examples(train_path)
    test = load_examples(against_path)

    by_norm: dict[str, list[int]] = defaultdict(list)
    token_index: dict[str, set[int]] = defaultdict(set)
    train_tokens: list[set[str]] = []
    for i, ex in enumerate(train):
        n = normalize_text(ex.text)
        by_norm[n].append(i)
        ts = token_set(ex.text)
        train_tokens.append(ts)
        for tok in ts:
            token_index[tok].add(i)

    exact = []
    near = []
    for ex in test:
        n = normalize_text(ex.text)
        for i in by_norm.get(n, []):
            tr = train[i]
            exact.append({
                "against_id": ex.id,
                "against_text": ex.text,
                "train_text": tr.text,
                "same_contract": contract_key(ex.contract) == contract_key(tr.contract),
            })

        tt = token_set(ex.text)
        if len(tt) < args.min_tokens:
            continue
        # Usa soltanto i token più informativi. Unione su parole comuni
        # ("un", "a", "mi"...) rendeva il near-check O(gold×train).
        ranked = sorted(
            ((len(token_index.get(tok, ())), tok) for tok in tt),
            key=lambda x: x[0])
        informative = [tok for df, tok in ranked
                       if df and df <= max(50, int(len(train) * 0.05))][:6]
        if not informative:
            informative = [tok for df, tok in ranked if df][:3]
        candidates: set[int] = set()
        for tok in informative:
            candidates.update(token_index.get(tok, ()))
        for i in candidates:
            tr = train[i]
            if normalize_text(tr.text) == n or len(train_tokens[i]) < args.min_tokens:
                continue
            score = token_jaccard(ex.text, tr.text)
            if score >= args.near_threshold:
                near.append({
                    "score": round(score, 4),
                    "against_id": ex.id,
                    "against_text": ex.text,
                    "train_text": tr.text,
                })

    near.sort(key=lambda x: x["score"], reverse=True)
    # dedup del near report
    seen = set()
    unique_near = []
    for row in near:
        k = (normalize_text(row["against_text"]), normalize_text(row["train_text"]))
        if k in seen:
            continue
        seen.add(k)
        unique_near.append(row)
    near = unique_near

    print(f"train={len(train)}  against={len(test)}")
    print(f"exact_duplicate={len(exact)}")
    for row in exact[:30]:
        same = " stesso-contract" if row["same_contract"] else ""
        print(f"  EXACT {row['against_id'] or '-'}:{same} {row['against_text']!r}")
    if len(exact) > 30:
        print(f"  ... +{len(exact)-30} exact")

    print(f"near_duplicate>={args.near_threshold:.2f}={len(near)}")
    for row in near[:args.max_near]:
        print(f"  NEAR {row['score']:.2f} {row['against_id'] or '-'}: "
              f"{row['against_text']!r}  ~  {row['train_text']!r}")

    report = {
        "train": str(train_path), "against": str(against_path),
        "train_examples": len(train), "against_examples": len(test),
        "exact_count": len(exact), "near_count": len(near),
        "near_threshold": args.near_threshold,
        "exact": exact, "near": near,
    }
    if args.json_report:
        rp = Path(args.json_report)
        if not rp.is_absolute():
            rp = ROOT / rp
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 1 if exact or (args.fail_on_near and near) else 0


if __name__ == "__main__":
    raise SystemExit(main())
