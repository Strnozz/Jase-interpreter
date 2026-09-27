"""Fresh-process 4-bit base + PEFT adapter reload and one V1.2 inference."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.goal_contract_v1_2 import parse_contract, validate_contract as validate_v12  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract as validate_v13  # noqa: E402
from jase.multimodel import load_config  # noqa: E402
from jase.foundation import chat_template_kwargs, load_base  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = (ROOT / args.run).resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    cfg = load_config(manifest["model"])
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    import torch
    from peft import PeftModel

    model, tokenizer = load_base(cfg)
    adapter = ROOT / manifest["adapter"]
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    prompt_text = (ROOT / cfg["system_prompt_file"]).read_text(encoding="utf-8").strip()
    request = "Trovami un ristorante giapponese stasera sotto i 40 euro"
    turns = [{"role": "system", "content": prompt_text}, {"role": "user", "content": request}]
    encoded = tokenizer.apply_chat_template(turns, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=True, **chat_template_kwargs(cfg)).to("cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        ids = model.generate(**encoded, max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
    torch.cuda.synchronize()
    output_ids = ids[0, encoded["input_ids"].shape[-1]:]
    raw = tokenizer.decode(output_ids, skip_special_tokens=True)
    if not raw.strip():
        raise RuntimeError("Reloaded adapter produced empty output")
    try:
        contract = parse_contract(raw)
        validator = validate_v13 if cfg["schema"].endswith("v1_3.schema.json") else validate_v12
        issues = [item.as_dict() for item in validator(contract)]
    except (ValueError, TypeError, KeyError) as exc:
        contract, issues = None, [{"code": "parse", "message": repr(exc)}]
    report = {"status": "complete", "adapter": str(adapter.relative_to(ROOT)),
              "request": request, "output": raw, "contract": contract, "issues": issues,
              "schema_valid": not issues, "latency_seconds": round(time.perf_counter() - started, 3),
              "output_tokens": len(output_ids), "peak_vram_bytes": torch.cuda.max_memory_allocated()}
    (run / "post_train_inference.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
