#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from jsonschema import Draft202012Validator
from mlx_lm import load, generate

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
SCHEMA = json.loads((ROOT / "schema/goal_contract_v1.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)


def extract_json(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b >= a: text = text[a:b+1]
    return json.loads(text)


def canon(x):
    if isinstance(x, dict):
        return {k: canon(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [canon(v) for v in x]
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/test.jsonl")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x.strip()][:args.limit]
    parse_ok = schema_ok = exact = 0
    goal_count_ok = action_ok = target_ok = 0
    failures = []

    for idx, row in enumerate(rows):
        user = next(m["content"] for m in row["messages"] if m["role"] == "user")
        expected = json.loads(row["messages"][-1]["content"])
        system = next(
            (m["content"] for m in row["messages"] if m["role"] == "system"),
            SYSTEM
        )
        messages = [{"role":"system","content":system},{"role":"user","content":user}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        text = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        try:
            got = extract_json(text); parse_ok += 1
            if not list(VALIDATOR.iter_errors(got)): schema_ok += 1
            if canon(got) == canon(expected): exact += 1
            eg, gg = expected.get("goals", []), got.get("goals", [])
            if len(eg) == len(gg): goal_count_ok += 1
            if eg and gg and eg[0].get("action") == gg[0].get("action"): action_ok += 1
            if eg and gg and eg[0].get("target",{}).get("name") == gg[0].get("target",{}).get("name"): target_ok += 1
        except Exception as e:
            failures.append({"index":idx,"request":user,"error":str(e),"output":text[:800]})

    n = max(1, len(rows))
    result = {
        "examples": len(rows),
        "json_parse_rate": parse_ok/n,
        "schema_valid_rate": schema_ok/n,
        "exact_contract_rate": exact/n,
        "goal_count_rate": goal_count_ok/n,
        "first_action_rate": action_ok/n,
        "first_target_rate": target_ok/n,
        "parse_failures": failures[:10]
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
