#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from mlx_lm import load, generate

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()


def extract_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end >= start:
        text = text[start:end+1]
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("request", nargs="?", default="Trovami un panino con la mortadella a Milano sotto i 10 euro")
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--max-tokens", type=int, default=500)
    args = ap.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": args.request},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
    try:
        obj = extract_json(output)
        print(json.dumps(obj, ensure_ascii=False, indent=2))
    except Exception:
        print(output)

if __name__ == "__main__":
    main()
