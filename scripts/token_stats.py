#!/usr/bin/env python3
"""Distribuzione della lunghezza in token degli esempi di training.

Serve a rispondere a una domanda che altrimenti resta un'opinione: quanti
esempi vengono TRONCATI da `max_seq_length`? Un esempio troncato perde i
fatti finali della frase, e il modello impara a perderli — il che somiglia
tantissimo all'errore `missing_fact` che domina i report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/v6/train.jsonl")
    limits = [int(x) for x in (sys.argv[2:] or ["448", "512", "640", "768"])]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("mlx-community/Qwen3.5-2B-4bit")

    lens = []
    for line in data.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        # apply_chat_template(tokenize=True) in questa versione non restituisce
        # gli id: si passa per la stringa, che è anche più verificabile
        text = tok.apply_chat_template(msgs, tokenize=False)
        lens.append(len(tok(text)["input_ids"]))
    lens.sort()
    n = len(lens)
    out = {
        "file": str(data.relative_to(ROOT)),
        "n": n,
        "min": lens[0], "median": lens[n // 2],
        "p90": lens[int(n * .90)], "p99": lens[int(n * .99)], "max": lens[-1],
        "share_within": {str(c): round(sum(1 for x in lens if x <= c) / n, 4)
                         for c in limits},
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
