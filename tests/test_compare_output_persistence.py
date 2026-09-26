from __future__ import annotations

import json
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from evaluation import compare


FAKE_OUTPUT = json.dumps({"schema_version": "1.1", "goals": [{
    "action": "find", "raw_action": "trova", "target": {"name": "hotel", "type": "place", "raw": "hotel"},
    "attributes": [], "constraints": [], "preferences": []}],
    "clarification": {"required": False, "reason": None, "missing_fields": []}})


def fake_backend(_cfg, _adapter):
    return lambda _text: {"output": FAKE_OUTPUT, "latency_s": .01, "ttft_s": .001,
                          "input_tokens": 10, "output_tokens": 10,
                          "peak_vram_bytes": 1, "process_ram_bytes_after": 1}


class CompareOutputTests(unittest.TestCase):
    def test_dev_outputs_saved_only_when_requested_and_release_sealed(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(compare, "load_backend", fake_backend):
            dev = Path(temp) / "dev"
            argv = ["compare.py", "--models", "qwen35-9b", "--data", "data/v20_quality/dev.jsonl",
                    "--sample", "1", "--include-outputs", "--output", str(dev)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(), 0)
            report = json.loads(dev.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertIn("cases", report["models"]["qwen35-9b"])
            self.assertIn("output", report["models"]["qwen35-9b"]["cases"][0])

            holdout = Path(temp) / "holdout"
            argv = ["compare.py", "--models", "qwen35-9b", "--data", "benchmarks/release_holdout_v1.jsonl",
                    "--limit", "1", "--output", str(holdout)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                self.assertEqual(compare.main(), 0)
            report = json.loads(holdout.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertNotIn("cases", report["models"]["qwen35-9b"])


if __name__ == "__main__":
    unittest.main()
