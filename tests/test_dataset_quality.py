"""Regression tests for label loss, split leakage and dev coverage."""
import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_dataset_v2 import (Fact, Scenario, _telegraph, contrastive_variants,
                                 render, to_contract)
from jase.dataset_quality import (audit_splits, clean_rows, grouped_split,
                                 text_key)


def contract(action="find", clarify=False):
    sc = Scenario(action, "cercami", "cercami", "hotel", "accommodation", "un hotel")
    c = to_contract(sc)
    if clarify:
        c["clarification"] = {"required": True, "reason": "target_missing",
                              "missing_fields": ["target"]}
    return c


class DatasetQualityTests(unittest.TestCase):
    def test_telegraphic_action_does_not_depend_on_hidden_scenario(self):
        outputs = []
        for action in ("find", "buy", "book"):
            sc = Scenario(action, "verbo", "verbo", "hotel", "accommodation", "un hotel")
            text, raw = render(random.Random(1), sc, style="telegraphic")
            outputs.append((text, to_contract(sc, raw)))
            self.assertEqual(sc.action, action)  # rendering does not mutate parent
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    def test_explicit_action_preserved(self):
        self.assertEqual(contract("book")["goals"][0]["action"], "book")

    def test_temporal_operator_and_quantity_survive_telegraph(self):
        before = Fact("time", "constraints", "before", "9:00", "prima delle 9")
        after = Fact("time", "constraints", "after", "9:00", "dopo le 9")
        self.assertNotEqual(_telegraph(before), _telegraph(after))
        q = Fact("quantity", "constraints", "gte", 3, "almeno 3 biglietti")
        self.assertEqual(_telegraph(q), "almeno 3 biglietti")

    def test_contrastive_derivations_keep_family(self):
        sc = Scenario("find", "cercami", "cercami", "hotel", "accommodation", "un hotel",
                      [Fact("price", "constraints", "lt", 80, "sotto 80 euro")], family="root")
        variants = contrastive_variants(random.Random(1), sc)
        self.assertTrue(variants)
        self.assertTrue(all(v.family == sc.family for v in variants))

    def test_semantic_symbols_are_not_erased_by_dedup(self):
        self.assertNotEqual(text_key("hotel <80€"), text_key("hotel >80€"))
        self.assertNotEqual(text_key("hotel 80€"), text_key("hotel 80"))

    def test_transitive_duplicate_families_stay_together(self):
        rows = [("same", contract(), "a"), ("same", contract(), "b"),
                ("second", contract(), "b"), ("second", contract(), "c"),
                ("third", contract(), "c")]
        cleaned, report, rejected = clean_rows(rows)
        self.assertEqual(len({r[2] for r in cleaned}), 1)
        self.assertEqual(report["duplicate_rows_removed"], 2)
        self.assertFalse(rejected)

    def test_all_conflicting_labels_quarantined(self):
        rows = [("hotel", contract("buy"), "a"), ("hotel", contract(), "b")]
        cleaned, report, rejected = clean_rows(rows)
        self.assertFalse(cleaned)
        self.assertEqual(len(rejected), 2)
        self.assertEqual(report["conflicting_prompts"], 1)

    def test_clarification_in_both_splits_and_deterministic(self):
        rows = [(f"request {i}", contract(clarify=i < 10), f"f{i}") for i in range(40)]
        a = grouped_split(rows, 42)
        self.assertEqual(a, grouped_split(rows, 42))
        audit = audit_splits(*a)
        self.assertTrue(audit["passed"])
        self.assertGreater(audit["valid"]["tags"]["clarification"], 0)
        self.assertGreater(audit["train"]["tags"]["clarification"], 0)


if __name__ == "__main__":
    unittest.main()
