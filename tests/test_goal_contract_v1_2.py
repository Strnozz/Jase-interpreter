from __future__ import annotations

import copy
import unittest

from jase.goal_contract_v1_2 import validate_contract
from jase.metrics_v1_2 import aggregate_v12, compare_v12


def base():
    return {"schema_version": "1.2", "kind": "task", "goals": [
        {"id": "g1", "action": "find", "target": {"name": "ristorante", "type": "place"},
         "facts": [{"field": "price", "op": "lte", "value": 40,
                    "currency": "EUR", "strength": "hard"},
                   {"field": "seating", "op": "eq", "value": "outdoor", "strength": "soft"}]}
    ]}


class GoalContract12Tests(unittest.TestCase):
    def codes(self, contract):
        return {x.code for x in validate_contract(contract)}

    def test_valid_and_semantic_reordering(self):
        a, b = base(), base()
        b["goals"][0]["facts"].reverse()
        b["context"] = []
        self.assertEqual(self.codes(a), set())
        self.assertTrue(compare_v12(a, b)["complete_success"])

    def test_hard_soft_are_not_equivalent(self):
        a, b = base(), base()
        b["goals"][0]["facts"][1]["strength"] = "hard"
        result = compare_v12(a, b)
        self.assertFalse(result["complete_success"])
        self.assertEqual(result["soft"]["fn"], 1)
        self.assertEqual(result["hard"]["fp"], 1)

    def test_duplicate_unknown_and_cyclic_goals(self):
        c = base()
        c["goals"].append(copy.deepcopy(c["goals"][0]))
        self.assertIn("duplicate_goal_id", self.codes(c))
        c = base()
        c["goals"][0]["depends_on"] = ["g99"]
        self.assertIn("unknown_dependency", self.codes(c))
        c = base()
        c["goals"].append({"id": "g2", "action": "find", "target": {"name": "hotel"},
                           "facts": [], "depends_on": ["g1"]})
        c["goals"][0]["depends_on"] = ["g2"]
        self.assertIn("dependency_cycle", self.codes(c))

    def test_references_need_a_prior_dependency(self):
        c = base()
        c["goals"].append({"id": "g2", "action": "find", "target": {"name": "hotel"},
                           "facts": [{"field": "location", "op": "near",
                                      "value_ref": {"goal": "g1", "field": "arrival_airport"},
                                      "strength": "hard"}]})
        self.assertIn("reference_without_prior_dependency", self.codes(c))
        c["goals"][1]["depends_on"] = ["g1"]
        self.assertEqual(self.codes(c), set())
        c["goals"][1]["condition"] = {"goal": "g1", "test": {
            "field": "price", "op": "eq", "value_ref": {"goal": "g99", "field": "price"}}}
        self.assertIn("unknown_reference", self.codes(c))

    def test_numeric_unit_currency_and_range(self):
        c = base()
        del c["goals"][0]["facts"][0]["currency"]
        self.assertIn("currency_missing", self.codes(c))
        c = base()
        c["goals"][0]["facts"].append({"field": "distance", "op": "lte",
                                          "value": 5, "strength": "hard"})
        self.assertIn("unit_missing", self.codes(c))
        c = base()
        c["goals"][0]["facts"][0]["op"] = "between"
        self.assertIn("between_missing_end", self.codes(c))

    def test_policy_missing_and_impossible_constraints(self):
        c = base()
        c["goals"][0]["policy"] = {"forbid": ["book"], "confirm_before": ["book"]}
        self.assertIn("contradictory_policy", self.codes(c))
        c = base()
        c["goals"][0]["missing"] = [{"field": "price", "blocks": "search",
                                       "reason": "unspecified"}]
        self.assertIn("missing_but_present", self.codes(c))
        c = base()
        c["goals"][0]["facts"].append({"field": "price", "op": "gte", "value": 60,
                                          "currency": "EUR", "strength": "hard"})
        self.assertIn("contradictory_hard_facts", self.codes(c))

    def test_kind_and_invalid_action(self):
        c = base()
        c["kind"] = "non_actionable"
        self.assertIn("non_actionable_with_goals", self.codes(c))
        c["goals"] = []
        self.assertEqual(self.codes(c), set())
        c = base()
        c["goals"][0]["action"] = "pay_without_consent"
        self.assertIn("schema", self.codes(c))

    def test_aggregate_keeps_schema_failure_in_denominator(self):
        good = compare_v12(base(), base())
        bad = {"schema_valid": False, "complete_success": False, "issues": ["schema"]}
        summary = aggregate_v12([good, bad])
        self.assertEqual(summary["schema_valid_rate"], .5)
        self.assertEqual(summary["complete_success_rate"], .5)

    def test_generated_dependency_chains_and_backward_edges(self):
        for length in range(2, 12):
            goals = []
            for index in range(1, length + 1):
                goal = {"id": f"g{index}", "action": "find",
                        "target": {"name": f"entity_{index}"}, "facts": []}
                if index > 1:
                    goal["depends_on"] = [f"g{index - 1}"]
                goals.append(goal)
            contract = {"schema_version": "1.2", "kind": "task", "goals": goals}
            self.assertEqual(self.codes(contract), set())
            broken = copy.deepcopy(contract)
            broken["goals"][0]["depends_on"] = [f"g{length}"]
            self.assertIn("dependency_cycle", self.codes(broken))


if __name__ == "__main__":
    unittest.main()
