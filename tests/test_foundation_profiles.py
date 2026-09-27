from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from jase.foundation import chat_template_kwargs
from jase.multimodel import encode_supervised, load_config, read_jsonl

ROOT = Path(__file__).resolve().parents[1]


class FoundationProfileTests(unittest.TestCase):
    def test_legacy_qwen_chat_and_mask_are_byte_for_byte_unchanged(self):
        os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
        from transformers import AutoTokenizer
        cfg = load_config("qwen35-9b-v26")
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], revision=cfg["revision"],
                                            local_files_only=True)
        row = read_jsonl(ROOT / "data/v26_9b/train.jsonl")[0]
        legacy = encode_supervised(tok, row, cfg["max_length"])
        configured = encode_supervised(tok, row, cfg["max_length"], chat_template_kwargs(cfg))
        self.assertEqual(legacy, configured)
        self.assertEqual(chat_template_kwargs(cfg), {"enable_thinking": False})
        self.assertGreater(legacy["labels"].count(-100), 0)
        self.assertEqual(legacy["labels"][-1], legacy["input_ids"][-1])

    def test_granite_uses_its_native_template_and_same_semantic_row(self):
        os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
        from transformers import AutoTokenizer
        cfg = load_config("granite41-8b-v26-challenger")
        tok = AutoTokenizer.from_pretrained(cfg["tokenizer_id"],
                                            revision=cfg["tokenizer_revision"],
                                            local_files_only=True)
        row = read_jsonl(ROOT / "data/v26_9b/train.jsonl")[0]
        encoded = encode_supervised(tok, row, cfg["max_length"], chat_template_kwargs(cfg))
        self.assertEqual(chat_template_kwargs(cfg), {})
        self.assertLessEqual(len(encoded["input_ids"]), 512)
        self.assertTrue(all(x == -100 for x in encoded["labels"][:encoded["labels"].count(-100)]))
        self.assertEqual(json.loads(row["messages"][-1]["content"])["schema_version"], "1.3")


if __name__ == "__main__":
    unittest.main()
