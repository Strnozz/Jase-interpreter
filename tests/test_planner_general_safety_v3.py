from __future__ import annotations

import unittest

from jase.planner_v1_3 import plan_contract


def fact(field, value, op="eq"):
    return {"field": field, "value": value, "op": op, "strength": "hard"}


def temporal(role, value, op="eq"):
    return {"role": role, "value": value, "op": op, "strength": "hard"}


def task(action, typ, name, facts, *, times=(), policy=None):
    goal = {"id": "g1", "action": action, "target": {"type": typ, "name": name},
            "facts": facts}
    if times:
        goal["temporal"] = list(times)
    if policy:
        goal["policy"] = policy
    return {"schema_version": "1.3", "kind": "task", "goals": [goal]}


class PlannerGeneralSafetyTests(unittest.TestCase):
    def test_unique_hour_normalizes_without_inventing_time(self):
        contract = task("find", "transport", "treno", [fact("origin", "Pisa"),
            fact("destination", "Livorno")], times=[temporal("arrival_time", "14:00", "before")])
        outcome = plan_contract(contract, "Trova treni da Pisa a Livorno che arrivino entro le 14.")
        self.assertEqual(outcome["planner_status"], "READY_FOR_DRY_RUN")
        evidence = outcome["goal_bindings"][0]["slots"]["temporal.arrival_time"]["provenance"]
        self.assertEqual(evidence["source"], "DETERMINISTIC_DERIVATION")
        self.assertFalse(outcome["execution_permitted"])
        outcome = plan_contract(contract, "Trova treni da Pisa a Livorno dopo le 10 e prima delle 14.")
        self.assertIn("HOLD_UNGROUNDED_FACT", outcome["reason_codes"])

    def test_day_part_cannot_be_calendar_date(self):
        contract = task("find", "transport", "treno", [fact("origin", "Pisa"),
            fact("destination", "Livorno")], times=[temporal("search_date", "mattina")])
        outcome = plan_contract(contract, "Mostrami i treni del mattina da Pisa a Livorno.")
        self.assertIn("HOLD_TEMPORAL_ROLE_CONFLICT", outcome["reason_codes"])

    def test_explicit_availability_cannot_be_dropped(self):
        contract = task("find", "place", "museo", [fact("location", "Bergamo", "near")])
        outcome = plan_contract(contract, "Cerca un museo visitabile a Bergamo.")
        self.assertIn("HOLD_AVAILABILITY_MISSING", outcome["reason_codes"])
        contract["goals"][0]["facts"].append(fact("availability", "visitabile"))
        outcome = plan_contract(contract, "Cerca un museo visitabile a Bergamo.")
        self.assertNotIn("HOLD_AVAILABILITY_MISSING", outcome["reason_codes"])

    def test_city_cannot_be_dropped_from_landmark(self):
        contract = task("find", "accommodation", "ostello", [fact("location", "stazione", "near")])
        outcome = plan_contract(contract, "A Ferrara cerca un ostello vicino alla stazione.")
        self.assertIn("HOLD_LOCATION_SCOPE_INCOMPLETE", outcome["reason_codes"])
        contract["goals"][0]["facts"][0]["value"] = "stazione di Ferrara"
        outcome = plan_contract(contract, "Cerca un ostello vicino alla stazione di Ferrara.")
        self.assertNotIn("HOLD_LOCATION_SCOPE_INCOMPLETE", outcome["reason_codes"])

    def test_deferred_selection_requires_policy(self):
        contract = task("find", "accommodation", "ostello", [fact("location", "Ferrara", "near")])
        outcome = plan_contract(contract, "Cerca un ostello a Ferrara; decido io quale scegliere.")
        self.assertIn("HOLD_POLICY_REQUIRED", outcome["reason_codes"])
        contract["goals"][0]["policy"] = {"forbid": ["book"]}
        outcome = plan_contract(contract, "Cerca un ostello a Ferrara; decido io quale scegliere.")
        self.assertNotIn("HOLD_POLICY_REQUIRED", outcome["reason_codes"])


if __name__ == "__main__":
    unittest.main()
