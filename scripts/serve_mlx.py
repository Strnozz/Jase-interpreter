#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from jsonschema import Draft202012Validator
from mlx_lm import load, generate
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
SYSTEM = (ROOT / "SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
SCHEMA = json.loads((ROOT / "schema/goal_contract_v1.schema.json").read_text(encoding="utf-8"))
VALIDATOR = Draft202012Validator(SCHEMA)

class InterpretRequest(BaseModel):
    query: str


def extract_json(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b >= a:
        text = text[a:b+1]
    return json.loads(text)


def create_app(model_id: str, adapter: str | None):
    app = FastAPI(title="Jase Interpreter", version="0.1")
    model, tokenizer = load(model_id, adapter_path=adapter)

    @app.get("/health")
    def health():
        return {"ok": True, "model": model_id, "adapter": adapter}

    @app.post("/interpret")
    def interpret(req: InterpretRequest):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": req.query},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        raw = generate(model, tokenizer, prompt=prompt, max_tokens=500, verbose=False)
        try:
            obj = extract_json(raw)
        except Exception as e:
            raise HTTPException(502, detail={"error": "invalid_json_from_model", "raw": raw, "reason": str(e)})
        errors = [e.message for e in VALIDATOR.iter_errors(obj)]
        if errors:
            raise HTTPException(502, detail={"error": "schema_validation_failed", "issues": errors[:10], "raw": obj})
        return obj

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.5-2B-4bit")
    ap.add_argument("--adapter", default="adapters/jase-v1")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()
    adapter = args.adapter if args.adapter and Path(args.adapter).exists() else None
    uvicorn.run(create_app(args.model, adapter), host=args.host, port=args.port)

if __name__ == "__main__":
    main()
