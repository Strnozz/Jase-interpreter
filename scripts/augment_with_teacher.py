#!/usr/bin/env python3
"""Optional dataset augmentation through ANY OpenAI-compatible chat endpoint.

Set:
  TEACHER_BASE_URL=http://127.0.0.1:8080/v1
  TEACHER_MODEL=your-model
  TEACHER_API_KEY=optional

The teacher only paraphrases a user request. The original, deterministic GoalContract
is kept as ground truth. This is deliberately safer than asking the teacher to label data.
"""
import argparse, json, os, random
from pathlib import Path
import requests


def endpoint(base):
    return base.rstrip("/") + "/chat/completions"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/train.jsonl")
    ap.add_argument("--output", default="data/train_augmented.jsonl")
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    base = os.getenv("TEACHER_BASE_URL")
    model = os.getenv("TEACHER_MODEL")
    key = os.getenv("TEACHER_API_KEY", "")
    if not base or not model:
        raise SystemExit("Set TEACHER_BASE_URL and TEACHER_MODEL first.")

    rows = [json.loads(x) for x in Path(args.input).read_text(encoding="utf-8").splitlines() if x.strip()]
    if args.limit: rows = rows[:args.limit]
    headers = {"Content-Type":"application/json"}
    if key: headers["Authorization"] = f"Bearer {key}"

    with Path(args.output).open("w", encoding="utf-8") as out:
        for i, row in enumerate(rows, 1):
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            user_idx = next(j for j,m in enumerate(row["messages"]) if m["role"] == "user")
            original = row["messages"][user_idx]["content"]
            prompt = (
                f"Rewrite the following user request in {args.variants} natural, semantically equivalent ways. "
                "Do not add, remove, weaken or strengthen any constraint. Return ONLY a JSON array of strings.\n\n"
                f"REQUEST: {original}"
            )
            payload = {"model":model,"messages":[{"role":"user","content":prompt}],"temperature":0.8,"max_tokens":500}
            r = requests.post(endpoint(base), headers=headers, json=payload, timeout=120)
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().replace("```json","").replace("```","").strip()
            try:
                variants = json.loads(text)
            except Exception:
                print(f"WARN row {i}: teacher returned non-JSON; skipped")
                continue
            for variant in variants[:args.variants]:
                clone = json.loads(json.dumps(row))
                clone["messages"][user_idx]["content"] = str(variant)
                out.write(json.dumps(clone, ensure_ascii=False) + "\n")
            if i % 25 == 0: print(f"augmented {i}/{len(rows)}")

if __name__ == "__main__":
    main()
