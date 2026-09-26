from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from jase.goal_contract_v1_2 import validate_contract
from jase.leakage import normalize_text

ROOT = Path(__file__).resolve().parents[1]


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class V20DatasetTests(unittest.TestCase):
    def test_migrated_data_and_original_hashes(self):
        manifest = json.loads((ROOT / "data/v20/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["training_readiness"], "NOT_READY")
        self.assertEqual(sum(manifest["source_rows"].values()),
                         sum(manifest["automatically_converted"].values()) +
                         manifest["requires_review"] + manifest["rejected"])
        for split in ("train", "valid"):
            path = ROOT / f"data/v20/{split}.jsonl"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["output_sha256"][split])
            self.assertEqual(len(rows(path)), manifest["automatically_converted"][split])
            original = ROOT / f"data/v19_clean/{split}.jsonl"
            self.assertEqual(hashlib.sha256(original.read_bytes()).hexdigest(),
                             manifest["source_sha256"][split])
        train = rows(ROOT / "data/v20/train.jsonl")
        valid = rows(ROOT / "data/v20/valid.jsonl")
        def keys(items):
            return {normalize_text(next(m["content"] for m in row["messages"] if m["role"] == "user"))
                    for row in items}
        self.assertFalse(keys(train) & keys(valid))
        for row in train + valid:
            self.assertFalse(validate_contract(json.loads(row["messages"][-1]["content"])))

    def test_sealed_benchmarks_are_valid_and_disjoint(self):
        manifest = json.loads((ROOT / "benchmarks/v20/manifest.json").read_text(encoding="utf-8"))
        train_keys = set()
        for split in ("train", "valid"):
            for row in rows(ROOT / f"data/v20/{split}.jsonl"):
                train_keys.add(normalize_text(row["messages"][1]["content"]))
        panel_keys = set()
        for name in ("hard_dev", "holdout"):
            path = ROOT / f"benchmarks/v20/{name}.jsonl"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                             manifest["panels"][name]["sha256"])
            for row in rows(path):
                self.assertFalse(validate_contract(row["contract"]), row["id"])
                key = normalize_text(row["text"])
                self.assertNotIn(key, train_keys)
                self.assertNotIn(key, panel_keys)
                panel_keys.add(key)
        audit = json.loads((ROOT / "data/v20/audit.json").read_text(encoding="utf-8"))
        self.assertEqual(audit["leakage"]["train_to_valid"]["near_candidates_ge_0_85"], 0)
        for name in ("train", "valid", "hard_dev", "holdout"):
            path = ROOT / (f"data/v20/{name}.jsonl" if name in ("train", "valid")
                           else f"benchmarks/v20/{name}.jsonl")
            self.assertEqual(audit["input_sha256"][name], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
