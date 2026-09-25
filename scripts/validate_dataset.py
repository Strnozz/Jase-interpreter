#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
try:  # jsonschema >= 4.x
    from jsonschema import Draft202012Validator as _Validator
except ImportError:  # 3.x: Draft7 copre const e $ref, come in jase/guard.py
    from jsonschema import Draft7Validator as _Validator

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "schema/goal_contract_v1_1.schema.json").read_text(encoding="utf-8"))
VALIDATOR = _Validator(SCHEMA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    total = 0
    failures = 0
    for filename in ["train.jsonl", "valid.jsonl", "test.jsonl"]:
        p = Path(args.data) / filename
        if not p.exists():
            print(f"SKIP {p} (missing)")
            continue
        local = 0
        for lineno, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            local += 1; total += 1
            try:
                row = json.loads(line)
                msgs = row["messages"]
                assert msgs[-1]["role"] == "assistant"
                obj = json.loads(msgs[-1]["content"])
                errors = list(VALIDATOR.iter_errors(obj))
                if errors:
                    raise ValueError("; ".join(e.message for e in errors[:3]))
            except Exception as e:
                failures += 1
                print(f"FAIL {filename}:{lineno}: {e}")
        print(f"OK checked {local} rows in {filename}")
    print(f"Total={total}, failures={failures}")
    raise SystemExit(1 if failures else 0)

if __name__ == "__main__":
    main()
