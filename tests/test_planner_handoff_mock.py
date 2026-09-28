from __future__ import annotations

import copy
import unittest

from jase.capability_registry_v1 import CapabilityRegistry, REGISTRY_V3_PATH
from jase.planner_handoff_mock import prepare_mock_handoff
from jase.planner_v1_3 import plan_contract


def fact(field, value, op="eq"):
    return {"field": field, "value": value, "op": op, "strength": "hard"}


def task(action, typ, name, facts, **other):
    return {"schema_version": "1.3", "kind": "task", "goals": [
        {"id": "g1", "action": action, "target": {"type": typ, "name": name},
         "facts": facts, **other}]}


class MockHandoffTests(unittest.TestCase):
    def test_registry_extension_remains_mock_only(self):
        registry = CapabilityRegistry.load(REGISTRY_V3_PATH)
        self.assertEqual(registry.version, "3")
        self.assertEqual(registry.resolve({"action": "find", "target":
                         {"type": "professional_service", "name": "veterinario"}})[0], "RESOLVED")
        self.assertEqual(registry.resolve({"action": "hire", "target":
                         {"type": "professional_service", "name": "veterinario"}})[0], "HOLD_NO_CAPABILITY")
        self.assertTrue(all(cap.execution_status == "mock_only" for cap in registry.capabilities))
        status, cap = registry.resolve({"action": "find", "target":
                                        {"type": "professional_service", "name": "tecnico"}})
        self.assertEqual(status, "RESOLVED")
        self.assertEqual(cap.capability_id, "find.technician")

    def test_read_query_draft_preserves_slots_and_policy(self):
        contract = task("find", "transport", "treno", [fact("origin", "Sassari"),
            fact("destination", "Olbia")], policy={"forbid": ["buy"]})
        plan = plan_contract(contract, "Cerca treni da Sassari a Olbia, senza comprare biglietti.")
        self.assertEqual(plan["planner_status"], "READY_FOR_DRY_RUN")
        handoff = prepare_mock_handoff(plan)
        self.assertEqual(handoff["status"], "READY_FOR_MOCK_DRAFT")
        self.assertEqual(handoff["mock_request_drafts"][0]["policy"], {"forbid": ["buy"]})
        self.assertEqual(handoff["mock_request_drafts"][0]["slots"]["fact.origin"]["value"], "Sassari")
        self.assertFalse(handoff["provider_called"])
        self.assertFalse(handoff["execution_permitted"])

    def test_context_is_preserved_in_draft(self):
        contract = task("find", "transport", "treno", [fact("origin", "Sassari"),
            fact("destination", "Olbia")])
        contract["context"] = [{"scope": "user", "field": "has_car", "op": "eq", "value": False}]
        plan = plan_contract(contract, "Non ho auto: cerca treni da Sassari a Olbia.",
                             trusted_context={"has_car": False})
        self.assertEqual(plan["planner_status"], "READY_FOR_DRY_RUN")
        handoff = prepare_mock_handoff(plan)
        self.assertEqual(handoff["status"], "READY_FOR_MOCK_DRAFT")
        self.assertIs(handoff["mock_request_drafts"][0]["slots"]["context.has_car"]["value"], False)

    def test_write_capability_is_never_handed_off(self):
        contract = task("contact", "person", "Dario", [fact("message", "Sono in arrivo")],
                        policy={"confirm_before": ["contact"]})
        plan = plan_contract(contract, "Scrivi a Dario 'Sono in arrivo' dopo il mio sì.",
                             external_bindings_by_goal={"g1": {"external.provider_ref": {
                                 "value": "contact-1", "trusted": True,
                                 "source": "TRUSTED_CONVERSATION_CONTEXT"}}},
                             confirmation_by_goal={"g1": True})
        self.assertEqual(plan["planner_status"], "READY_FOR_DRY_RUN")
        handoff = prepare_mock_handoff(plan)
        self.assertEqual(handoff["status"], "HOLD")
        self.assertIn("write_or_nonmock_capability", handoff["reason_codes"])
        self.assertEqual(handoff["mock_request_drafts"], [])

    def test_tampered_binding_or_registry_fails_atomically(self):
        contract = task("find", "transport", "treno", [fact("origin", "Sassari"),
            fact("destination", "Olbia")])
        plan = plan_contract(contract, "Cerca treni da Sassari a Olbia.")
        changed = copy.deepcopy(plan)
        changed["goal_bindings"][0]["slots"]["fact.origin"]["value"] = "Roma"
        self.assertIn("binding_evidence_mismatch",
                      prepare_mock_handoff(changed)["reason_codes"])
        changed = copy.deepcopy(plan)
        changed["registry_version"] = "2"
        self.assertIn("registry_version_mismatch", prepare_mock_handoff(changed)["reason_codes"])
        changed = copy.deepcopy(plan)
        changed["goal_bindings"][0]["slots"]["fact.origin"]["value"] = 123
        changed["execution_plan_steps"][0]["binding"]["fact.origin"]["value"] = 123
        changed["goal_bindings"][0]["slots"]["fact.origin"]["provenance"]["value"] = 123
        self.assertIn("invalid_slot_type_or_operator", prepare_mock_handoff(changed)["reason_codes"])
        self.assertEqual(prepare_mock_handoff(changed)["mock_request_drafts"], [])

    def test_unknown_modifier_mapping_holds(self):
        contract = task("find", "transport", "treno", [fact("origin", "Sassari"),
            fact("destination", "Olbia")], modifiers={"limit": 2})
        plan = plan_contract(contract, "Mostrami due treni da Sassari a Olbia.")
        self.assertEqual(plan["planner_status"], "READY_FOR_DRY_RUN")
        handoff = prepare_mock_handoff(plan)
        self.assertEqual(handoff["status"], "HOLD")
        self.assertIn("modifier_mapping_unverified", handoff["reason_codes"])

    def test_no_action_is_not_a_provider_request(self):
        plan = plan_contract({"schema_version": "1.3", "kind": "non_actionable", "goals": []},
                             "Spiegami il significato della frase.")
        handoff = prepare_mock_handoff(plan)
        self.assertEqual(handoff["status"], "NO_ACTION")
        self.assertEqual(handoff["mock_request_drafts"], [])
        self.assertFalse(handoff["execution_permitted"])

    def test_no_hire_pronoun_cannot_be_replaced_by_no_contact(self):
        contract = task("find", "professional_service", "idraulico",
                        [fact("location", "Bergamo", "near")], policy={"forbid": ["contact"]})
        plan = plan_contract(contract, "Trova un idraulico a Bergamo ma non incaricarlo.")
        self.assertIn("HOLD_POLICY_REQUIRED", plan["reason_codes"])
        self.assertEqual(prepare_mock_handoff(plan)["status"], "HOLD")

    def test_stay_period_requires_availability_not_search_date(self):
        contract = task("find", "accommodation", "camera", [fact("location", "Teramo", "near")],
                        temporal=[{"role": "search_date", "op": "eq", "value": "weekend",
                                   "strength": "hard"}], policy={"forbid": ["book"]})
        plan = plan_contract(contract, "Cerca una camera a Teramo per il weekend, senza prenotare.")
        self.assertIn("HOLD_AVAILABILITY_MISSING", plan["reason_codes"])
        contract["goals"][0]["facts"].append(fact("availability", "weekend"))
        plan = plan_contract(contract, "Cerca una camera a Teramo per il weekend, senza prenotare.")
        self.assertNotIn("HOLD_AVAILABILITY_MISSING", plan["reason_codes"])


if __name__ == "__main__":
    unittest.main()
