from __future__ import annotations

import copy
import unittest

from jase.planner_v1_3 import plan_contract


def fact(field: str, value, op="eq") -> dict:
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def temporal(role: str, value, op="eq") -> dict:
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def goal(action: str, typ: str, name: str, facts=None, **other) -> dict:
    return {"id": other.pop("id", "g1"), "action": action,
            "target": {"type": typ, "name": name}, "facts": facts or [], **other}


def contract(*goals, kind="task") -> dict:
    return {"schema_version": "1.3", "kind": kind, "goals": list(goals)}


class PlannerDryRunTests(unittest.TestCase):
    def test_no_action_and_no_execution_even_when_ready(self):
        result = plan_contract(contract(kind="non_actionable"), "Spiegami un concetto.")
        self.assertEqual(result["planner_status"], "NO_ACTION")
        self.assertFalse(result["execution_permitted"])
        request = contract(goal("find", "transport", "treno", [
            fact("origin", "Bari"), fact("destination", "Lecce")]))
        result = plan_contract(request, "Cerca treni da Bari a Lecce.")
        self.assertEqual(result["planner_status"], "READY_FOR_DRY_RUN")
        self.assertFalse(result["execution_permitted"])
        self.assertEqual(result["execution_plan_steps"][0]["mode"], "DRY_RUN_ONLY")

    def test_missing_slots_authoritative_over_model_missing(self):
        request = contract(goal("book", "accommodation", "hotel", [fact("location", "Siena")],
                                policy={"confirm_before": ["book"]}))
        result = plan_contract(request, "Prenota un hotel a Siena dopo il mio ok.")
        self.assertIn("HOLD_MISSING_INFORMATION", result["reason_codes"])
        slots = {x["slot"] for x in result["missing_slots"]}
        self.assertIn("fact.selection", slots)
        self.assertIn("external.check_in", slots)
        self.assertIn("external.check_out", slots)

    def test_unknown_and_ambiguous_capability_are_distinct(self):
        request = contract(goal("find", "product", "ristorante"))
        result = plan_contract(request, "Cerca un ristorante.")
        self.assertIn("HOLD_NO_CAPABILITY", result["reason_codes"])
        from jase.capability_registry_v1 import Capability, CapabilityRegistry
        registry = CapabilityRegistry.load()
        a = next(c for c in registry.capabilities if c.capability_id == "find.train")
        registry = CapabilityRegistry("1", registry.capabilities + (
            Capability.from_dict({**a.as_dict(), "capability_id": "duplicate.train"}),))
        request = contract(goal("find", "transport", "treno", [fact("origin", "Bari"), fact("destination", "Lecce")]))
        result = plan_contract(request, "Cerca treni da Bari a Lecce.", registry=registry)
        self.assertIn("HOLD_AMBIGUOUS_CAPABILITY", result["reason_codes"])

    def test_unregistered_alias_and_field_hold(self):
        request = contract(goal("find", "transport", "volo", [
            fact("origin", "Pisa"), fact("destination", "Dublino"), fact("route", "nonstop")]))
        result = plan_contract(request, "Cerca un volo nonstop da Pisa a Dublino.")
        self.assertEqual(result["planner_status"], "HOLD_GUARD")
        self.assertIn("unknown_planner_field", result["reason_codes"])

    def test_reference_resolution_requires_trusted_result_set(self):
        first = goal("find", "accommodation", "hotel", [fact("location", "Siena")])
        second = goal("book", "accommodation", "hotel", [
            {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "result", "ordinal": 2}, "strength": "hard"}],
            id="g2", depends_on=["g1"], policy={"confirm_before": ["book"]})
        request = contract(first, second)
        text = "Cerca hotel a Siena e prenota il secondo da lunedì a martedì per 2 ospiti dopo il mio ok."
        unresolved = plan_contract(request, text)
        self.assertIn("HOLD_UNRESOLVED_REFERENCE", unresolved["reason_codes"])
        resolved = plan_contract(request, text, results_by_goal={"g1": [
            {"id": "hotel-A", "trusted": True}, {"id": "hotel-B", "trusted": True}]},
            confirmation_by_goal={"g2": True})
        self.assertEqual(resolved["resolved_references"][0]["result_id"], "hotel-B")
        self.assertFalse(resolved["execution_permitted"])
        bad = copy.deepcopy(request)
        bad["goals"][1]["facts"][0]["value_ref"]["ordinal"] = 3
        result = plan_contract(bad, text, results_by_goal={"g1": [{"id": "a", "trusted": True}]})
        self.assertEqual(result["planner_status"], "HOLD_GUARD")
        self.assertIn("ordinal_result_reference_missing_v4", result["reason_codes"])

    def test_cheapest_requires_unique_price(self):
        from jase.capability_registry_v1 import Capability, CapabilityRegistry
        registry = CapabilityRegistry.load()
        original = next(c for c in registry.capabilities if c.capability_id == "book.hotel")
        modified = Capability.from_dict({**original.as_dict(),
            "supported_reference_types": sorted(original.supported_reference_types | {"cheapest_result"})})
        registry = CapabilityRegistry("1", tuple(c if c != original else modified for c in registry.capabilities))
        first = goal("find", "accommodation", "hotel", [fact("location", "Siena")])
        second = goal("book", "accommodation", "hotel", [
            {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "cheapest"}, "strength": "hard"}],
            id="g2", depends_on=["g1"], policy={"confirm_before": ["book"]})
        text = "Cerca hotel a Siena e prenota il meno costoso da lunedì a martedì per 2 ospiti dopo il mio ok."
        results = {"g1": [{"id": "A", "price": 60, "trusted": True}, {"id": "B", "price": 40, "trusted": True}]}
        outcome = plan_contract(contract(first, second), text, registry=registry,
                                results_by_goal=results, confirmation_by_goal={"g2": True})
        self.assertEqual(outcome["resolved_references"][0]["result_id"], "B")
        results["g1"][0]["price"] = 40
        outcome = plan_contract(contract(first, second), text, registry=registry, results_by_goal=results)
        self.assertIn("HOLD_UNRESOLVED_REFERENCE", outcome["reason_codes"])

    def test_confirmation_forbid_and_deferred_state(self):
        request = contract(goal("book", "transport", "volo", [fact("origin", "Bari"),
            fact("destination", "Oslo"), fact("selection", "AZ123")],
            temporal=[temporal("action_date", "domani")], policy={"confirm_before": ["book"]}))
        text = "Prenota il volo AZ123 da Bari a Oslo domani, ma solo dopo il mio sì."
        result = plan_contract(request, text)
        self.assertIn("HOLD_CONFIRMATION_PENDING", result["reason_codes"])
        forbidden = plan_contract(request, "Trova il volo AZ123 da Bari a Oslo domani, senza prenotare.")
        self.assertNotEqual(forbidden["planner_status"], "READY_FOR_DRY_RUN")

    def test_quoted_and_hypothetical_content_are_not_actions(self):
        request = contract(goal("rent", "vehicle", "auto", [fact("location", "Como")],
                                temporal=[temporal("action_date", "domani")]))
        translated = plan_contract(request, "Traduci 'rent a car in Como tomorrow', senza agire.")
        self.assertEqual(translated["planner_status"], "HOLD_NON_ACTIONABLE_CONTEXT")
        hypothetical = plan_contract(request, "Se qualcuno mi chiedesse di affittare un'auto a Como domani, cosa significa?")
        self.assertEqual(hypothetical["planner_status"], "HOLD_NON_ACTIONABLE_CONTEXT")

    def test_untraceable_location_and_temporal_role(self):
        request = contract(goal("find", "accommodation", "hotel", [fact("location", "Como")]))
        result = plan_contract(request, "Cerca un hotel.")
        self.assertIn("HOLD_UNGROUNDED_FACT", result["reason_codes"])
        self.assertEqual(result["provenance_issues"][0]["evidence"]["source"], "MODEL_INFERENCE")
        request = contract(goal("find", "transport", "treno", [fact("origin", "Bari"),
            fact("destination", "Lecce")], temporal=[temporal("arrival_time", "09:30")]))
        result = plan_contract(request, "Cerca un treno da Bari a Lecce che arrivi entro le 09:30.")
        self.assertIn("HOLD_TEMPORAL_ROLE_CONFLICT", result["reason_codes"])

    def test_message_date_cannot_be_send_date(self):
        request = contract(goal("contact", "person", "Marco", [fact("message", "Ci vediamo giovedì")],
            temporal=[temporal("action_date", "giovedì")],
            policy={"confirm_before": ["contact"]}))
        result = plan_contract(request, "Manda a Marco 'Ci vediamo giovedì' dopo il mio ok.")
        self.assertIn("HOLD_TEMPORAL_ROLE_CONFLICT", result["reason_codes"])

    def test_external_provider_binding_must_be_trusted(self):
        request = contract(goal("contact", "person", "Marco", [fact("message", "Arrivo alle sette")],
                                policy={"confirm_before": ["contact"]}))
        text = "Manda a Marco 'Arrivo alle sette' dopo il mio ok."
        absent = plan_contract(request, text)
        self.assertIn("HOLD_MISSING_INFORMATION", absent["reason_codes"])
        untrusted = plan_contract(request, text, external_bindings_by_goal={"g1": {
            "external.provider_ref": {"value": "contact-7", "source": "MODEL_INFERENCE", "trusted": True}}})
        self.assertIn("HOLD_UNGROUNDED_FACT", untrusted["reason_codes"])
        trusted = plan_contract(request, text, external_bindings_by_goal={"g1": {
            "external.provider_ref": {"value": "contact-7", "source": "TRUSTED_CONVERSATION_CONTEXT", "trusted": True}}},
            confirmation_by_goal={"g1": True})
        self.assertEqual(trusted["planner_status"], "READY_FOR_DRY_RUN")
        self.assertFalse(trusted["execution_permitted"])

    def test_dependencies_and_unknown_condition(self):
        first = goal("find", "transport", "treno", [fact("origin", "Bari"), fact("destination", "Lecce")])
        second = goal("find", "transport", "volo", [fact("origin", "Bari"), fact("destination", "Lecce")],
                      id="g2", depends_on=["g1"], condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})
        result = plan_contract(contract(first, second), "Se non trovi treni Bari-Lecce, cerca un volo.")
        self.assertEqual(result["planner_status"], "HOLD_GUARD")
        self.assertIn("alias_requires_planner_mapping", result["reason_codes"])
        second["condition"]["test"]["field"] = "results_count"
        result = plan_contract(contract(first, second), "Se non trovi treni Bari-Lecce, cerca un volo.")
        self.assertIn("HOLD_UNRESOLVED_REFERENCE", result["reason_codes"])


if __name__ == "__main__":
    unittest.main()
