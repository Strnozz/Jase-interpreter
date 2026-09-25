"""Verify official checkpoint metadata and freeze its commit before downloading weights."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen35-9b")
    args = ap.parse_args()
    cfg = load_config(args.model)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoTokenizer
    info = model_info(cfg["model_id"], files_metadata=True)
    license_name = (info.card_data or {}).get("license") if isinstance(info.card_data, dict) else getattr(info.card_data, "license", None)
    if info.id != cfg["model_id"] or license_name != "apache-2.0":
        raise RuntimeError(f"Unexpected model owner or license: {info.id} / {license_name}")
    weights = [s for s in info.siblings if s.rfilename.endswith(".safetensors")]
    total = sum(s.size or 0 for s in weights)
    configuration = AutoConfig.from_pretrained(cfg["model_id"], revision=info.sha)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], revision=info.sha)
    prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": "You are Jase Interpreter."},
        {"role": "user", "content": "Trovami un hotel a Roma"}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    report = {"model": info.id, "revision": info.sha, "license": license_name,
              "safetensors_files": len(weights), "weight_bytes": total,
              "architecture": getattr(configuration, "architectures", None),
              "model_type": configuration.model_type,
              "chat_template_ok": bool(prompt), "has_processor": any(s.rfilename == "preprocessor_config.json" for s in info.siblings)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = ROOT / "docs" / "QWEN35_9B_PREFLIGHT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
