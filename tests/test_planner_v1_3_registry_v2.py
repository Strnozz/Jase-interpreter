from __future__ import annotations

import unittest

from jase.capability_registry_v1 import CapabilityRegistry, REGISTRY_V2_PATH
from jase.planner_v1_3 import plan_contract


def goal(action, typ, name, facts, **extra):
    return {"id": "g1", "action": action, "target": {"type": typ, "name": name},
            "facts": [{"field": k, "op": "near" if k == "location" else "eq",
                       "value": v, "strength": "hard"} for k, v in facts.items()], **extra}


def task(item):
    return {"schema_version": "1.3", "kind": "task", "goals": [item]}


class RegistryV2PlannerTests(unittest.TestCase):
    def test_v2_is_mock_only_and_has_no_write_wildcard(self):
        registry = CapabilityRegistry.load(REGISTRY_V2_PATH)
        self.assertEqual(registry.version, "2")
        self.assertTrue(all(c.execution_status == "mock_only" for c in registry.capabilities))
        self.assertEqual(registry.resolve(goal("find", "place", "farmacia", {}))[0], "RESOLVED")
        self.assertEqual(registry.resolve(goal("book", "place", "farmacia", {}))[0], "HOLD_NO_CAPABILITY")
        self.assertEqual(registry.resolve(goal("find", "product", "farmacia", {}))[0], "HOLD_NO_CAPABILITY")

    def test_registered_read_only_bus_can_be_planned(self):
        item = goal("find", "transport", "autobus", {"origin": "Potenza", "destination": "Melfi"})
        result = plan_contract(task(item), "Trova autobus da Potenza a Melfi.",
                               registry=CapabilityRegistry.load(REGISTRY_V2_PATH))
        self.assertEqual(result["planner_status"], "READY_FOR_DRY_RUN")
        self.assertEqual(result["registry_version"], "2")
        self.assertFalse(result["execution_permitted"])

    def test_no_buy_policy_required_for_local_shop(self):
        item = goal("find", "place", "farmacia", {"location": "Asti"})
        result = plan_contract(task(item), "Trova una farmacia ad Asti, senza acquistare nulla.")
        self.assertIn("HOLD_POLICY_REQUIRED", result["reason_codes"])
        item["policy"] = {"forbid": ["buy"]}
        result = plan_contract(task(item), "Trova una farmacia ad Asti, senza acquistare nulla.")
        self.assertEqual(result["planner_status"], "READY_FOR_DRY_RUN")
        self.assertFalse(result["execution_permitted"])

    def test_deferred_choice_requires_no_booking_policy(self):
        item = goal("find", "place", "biblioteca", {"location": "Arona"})
        text = "Mostrami le biblioteche ad Arona; sceglierò dopo, senza riservare ora."
        result = plan_contract(task(item), text)
        self.assertIn("HOLD_POLICY_REQUIRED", result["reason_codes"])
        item["policy"] = {"forbid": ["book"]}
        result = plan_contract(task(item), text)
        self.assertEqual(result["planner_status"], "READY_FOR_DRY_RUN")

    def test_reminder_message_must_preserve_explicit_clause(self):
        item = goal("notify", "reminder", "promemoria", {"message": "chiamare l'ufficio"},
                    temporal=[{"role": "notification_date", "op": "eq", "value": "giovedì",
                               "strength": "hard"},
                              {"role": "notification_time", "op": "eq", "value": "08:15",
                               "strength": "hard"}])
        text = "Giovedì alle 08:15 ricordami di chiamare l'ufficio."
        result = plan_contract(task(item), text)
        self.assertEqual(result["planner_status"], "READY_FOR_DRY_RUN")
        item["facts"][0]["value"] = "l'ufficio"
        result = plan_contract(task(item), text)
        self.assertIn("HOLD_MESSAGE_CONTENT_INCOMPLETE", result["reason_codes"])
        self.assertFalse(result["execution_permitted"])


if __name__ == "__main__":
    unittest.main()
