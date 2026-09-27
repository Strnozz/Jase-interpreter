"""Foundation-model loading, chat formatting, and architecture-specific LoRA selection.

The legacy Qwen path intentionally retains its previous defaults. New model
profiles must declare their loader, chat-template arguments and LoRA strategy.
"""
from __future__ import annotations

import re


def chat_template_kwargs(cfg: dict) -> dict:
    return dict(cfg.get("chat_template_kwargs", {"enable_thinking": False}))


def load_base(cfg: dict):
    import torch
    from transformers import (AutoModelForCausalLM, AutoModelForMultimodalLM,
                              AutoTokenizer, BitsAndBytesConfig)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required")
    if torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RuntimeError("At least 14 GiB VRAM required for this profile")
    q = cfg["quantization"]
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if q["compute_dtype"] == "bfloat16" and dtype != torch.bfloat16:
        raise RuntimeError("This profile requires bf16; validate fp16 separately before changing config")
    if q["bits"] != 4:
        raise ValueError("Only 4-bit QLoRA profiles are supported")
    kwargs = {}
    if "trust_remote_code" in cfg:
        kwargs["trust_remote_code"] = cfg["trust_remote_code"]
    tok = AutoTokenizer.from_pretrained(cfg.get("tokenizer_id", cfg["model_id"]),
                                        revision=cfg.get("tokenizer_revision", cfg["revision"]), **kwargs)
    if tok.pad_token_id is None:
        if cfg.get("pad_strategy", "model_or_eos") != "model_or_eos":
            raise ValueError("Tokenizer has no pad token and profile forbids eos fallback")
        tok.pad_token = tok.eos_token
    classes = {"AutoModelForMultimodalLM": AutoModelForMultimodalLM,
               "AutoModelForCausalLM": AutoModelForCausalLM}
    name = cfg["model_class"]
    if name not in classes:
        raise ValueError(f"Unsupported model class {name!r}")
    model = classes[name].from_pretrained(
        cfg["model_id"], revision=cfg["revision"], device_map={"": 0},
        dtype=dtype, attn_implementation=cfg.get("attention_implementation", "sdpa"),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type=q["type"],
            bnb_4bit_use_double_quant=q["double_quant"],
            bnb_4bit_compute_dtype=dtype), **kwargs)
    return model, tok


def select_lora_targets(model, cfg: dict) -> list[str]:
    """Inspect real quantized modules; never infer targets from another model."""
    import bitsandbytes as bnb

    lora = cfg["lora"]
    last_n = lora["last_n_layers"]
    strategy = lora.get("target_strategy", "qwen_text_last_n")
    names = [name for name, module in model.named_modules()
             if isinstance(module, bnb.nn.Linear4bit)]
    if strategy == "qwen_text_last_n":
        names = [name for name in names
                 if any(f".{part}." in f".{name}." for part in ("language_model", "text_model"))]
        if not names or any("visual" in n or "vision" in n for n in names):
            raise RuntimeError("Could not isolate quantized Qwen text layers")
        pattern = re.compile(r"(?:^|\.)(?:language_model|text_model)\.layers\.(\d+)\.")
        allowed = None
    elif strategy == "causal_last_n":
        pattern = re.compile(r"(?:^|\.)model\.layers\.(\d+)\.")
        allowed = set(lora["target_leaf_names"])
        names = [name for name in names if pattern.search(name) and
                 name.rsplit(".", 1)[-1] in allowed]
        if not names or any("vision" in n or "visual" in n for n in names):
            raise RuntimeError("No matching quantized causal layers found")
    else:
        raise ValueError(f"Unknown LoRA target strategy {strategy!r}")
    layers = sorted({int(m.group(1)) for name in names if (m := pattern.search(name))})
    if len(layers) < last_n:
        raise RuntimeError(f"Expected {last_n} quantized text layers, found {layers}")
    chosen = set(layers[-last_n:])
    selected = [name for name in names if (m := pattern.search(name)) and
                int(m.group(1)) in chosen]
    if strategy == "causal_last_n":
        # Every selected layer must expose each configured leaf exactly once.
        for layer in chosen:
            found = [n.rsplit(".", 1)[-1] for n in selected
                     if (m := pattern.search(n)) and int(m.group(1)) == layer]
            if len(found) != len(allowed) or set(found) != allowed:
                raise RuntimeError(f"Layer {layer} LoRA topology differs from config: {found}")
    if not selected:
        raise RuntimeError("No trainable quantized language-model modules selected")
    return selected
