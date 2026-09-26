from __future__ import annotations

import copy
import unittest

from jase.canonical_semantics import compare
from jase.migrate_contract_v1_3 import MigrationHold, upgrade_contract
from jase.metrics_v1_3 import aggregate, score
from jase.semantic_contract_v1_3 import validate_contract
from jase.semantic_guard_v1_3 import (ACCEPT, HOLD_MISSING_REQUIRED_INFORMATION,
                                      HOLD_POLICY_MISMATCH, HOLD_SEMANTIC_CONFLICT,
                                      HOLD_UNSUPPORTED_ACTION, check_contract)


def goal(action="find", name="treno", typ="transport", facts=None, **extra):
    return {"id": "g1", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def fact(field, value, op="eq", strength="hard"):
    return {"field": field, "op": op, "value": value, "strength": strength}


def contract(g, version="1.3", **extra):
    return {"schema_version": version, "kind": "task", "goals": [g], **extra}


class SemanticContract13Tests(unittest.TestCase):
    def codes(self, c, text=""):
        return {x.code for x in validate_contract(c, user_text=text)}

    def test_typed_sort_time_and_context_are_valid(self):
        c = contract(goal(temporal=[{"role": "arrival_time", "op": "before", "value": "13:00",
                                     "strength": "hard"}],
                          modifiers={"sort": {"field": "price", "direction": "asc"}, "limit": 3}),
                     context=[{"scope": "user", "field": "has_car", "op": "eq", "value": False}])
        self.assertEqual(self.codes(c), set())
        self.assertEqual(check_contract(c, "Trova treni che arrivino prima delle 13").status, ACCEPT)

    def test_conflicting_single_value_equalities_hold_both_versions(self):
        for version in ("1.2", "1.3"):
            if version == "1.2":
                g = goal("book", "visita dermatologica", "appointment",
                         [fact("date", "martedì"), fact("date", "oggi")])
            else:
                g = goal("book", "visita dermatologica", "appointment", temporal=[
                    {"role": "action_date", "op": "eq", "value": x, "strength": "hard"}
                    for x in ("martedì", "oggi")])
            result = check_contract(contract(g, version), "Prenota la visita martedì.")
            self.assertEqual(result.status, HOLD_SEMANTIC_CONFLICT)
            self.assertIn("conflicting_hard_equality", result.codes)

    def test_equivalent_clock_format_does_not_conflict(self):
        g = goal("book", "visita", "appointment", temporal=[
            {"role": "action_time", "op": "eq", "value": x, "strength": "hard"}
            for x in ("8:00", "08:00")])
        self.assertNotIn("conflicting_hard_equality", self.codes(contract(g)))

    def test_confirmation_is_bound_to_consequential_goal(self):
        c = contract(goal("book", "visita", "appointment",
                          policy={"confirm_before": ["contact"]}))
        self.assertIn("policy_action_mismatch", self.codes(c))
        self.assertEqual(check_contract(c, "Prenota la visita, dopo mia conferma.").status,
                         HOLD_POLICY_MISMATCH)

    def test_sort_is_modifier_and_order_transport_is_held(self):
        c = contract(goal("order", "mezzi", "transport",
                          modifiers={"sort": {"field": "price", "direction": "asc"}}))
        self.assertIn("unsupported_action_target", self.codes(c))
        self.assertEqual(check_contract(c, "Ordina i mezzi per prezzo.").status,
                         HOLD_UNSUPPORTED_ACTION)

    def test_required_information_and_cardinality(self):
        c = contract(goal("book", "tavolo", "place"))
        self.assertEqual(check_contract(c, "Prenota un tavolo.").status,
                         HOLD_MISSING_REQUIRED_INFORMATION)
        c = contract(goal("compare", "fotocamera", "product", modifiers={"compare_count": 3}))
        self.assertNotIn("cardinality_lost", self.codes(c, "Confronta tre fotocamere."))
        c["goals"][0]["modifiers"]["compare_count"] = 2
        self.assertIn("cardinality_lost", self.codes(c, "Confronta tre fotocamere."))

    def test_guard_grounding_for_rental_and_context(self):
        c = contract(goal("find", "bicicletta", "vehicle"))
        self.assertEqual(check_contract(c, "Mi serve a noleggio una bicicletta.").status,
                         HOLD_UNSUPPORTED_ACTION)
        c = contract(goal("find", "ristorante", "place"))
        self.assertIn("missing_user_context",
                      check_contract(c, "Siamo in tre senza auto; cerca un ristorante.").codes)

    def test_migration_preserves_roles_and_source(self):
        old = contract(goal("find", "treno", "transport", [fact("destination", "Pisa"),
            fact("arrival_time", "12:00", "before")], ranking={"field": "price", "direction": "asc"}),
            "1.2", context=[{"field": "has_car", "op": "eq", "value": False}])
        before = copy.deepcopy(old)
        upgraded = upgrade_contract(old)
        self.assertEqual(old, before)
        self.assertEqual(upgraded["goals"][0]["temporal"][0]["role"], "arrival_time")
        self.assertEqual(upgraded["goals"][0]["modifiers"]["sort"]["field"], "price")
        self.assertEqual(upgraded["context"][0]["scope"], "user")
        self.assertEqual(self.codes(upgraded), set())

    def test_migration_holds_ambiguous_transport_time(self):
        old = contract(goal("find", "treno", "transport", [fact("time", "12:00", "before")]), "1.2")
        with self.assertRaises(MigrationHold):
            upgrade_contract(old)

    def test_canonical_time_alias_only(self):
        a = contract(goal("notify", "promemoria", "reminder", [fact("message", "chiamare Anna"),
            fact("time", "08:00")]), "1.2")
        b = copy.deepcopy(a)
        b["goals"][0]["facts"][1]["value"] = "8:00"
        self.assertFalse(compare(a, b)["literal_equal"])
        self.assertTrue(compare(a, b)["canonical_equal"])
        b["goals"][0]["facts"][0]["op"] = "contains"
        self.assertFalse(compare(a, b)["canonical_equal"])

    def test_v13_numeric_and_missing_invariants(self):
        c = contract(goal("find", "albergo", "accommodation", [
            fact("price", 100, "lt"), fact("price", 150, "gt")]))
        self.assertIn("currency_missing", self.codes(c))
        self.assertIn("contradictory_hard_facts", self.codes(c))
        c = contract(goal("book", "tavolo", "place", [fact("guests", 2)],
                          missing=[{"field": "guests", "blocks": "search", "reason": "unspecified"}]))
        self.assertIn("missing_but_present", self.codes(c))

    def test_reschedule_requires_appointment_identity_and_provider(self):
        c = contract(goal("reschedule", "visita", "appointment", temporal=[
            {"role": "action_date", "op": "eq", "value": "venerdì", "strength": "hard"}]))
        self.assertIn("missing_required_information", self.codes(c))
        c["goals"][0]["missing"] = [
            {"field": "appointment_id", "blocks": "action", "reason": "unspecified"}]
        self.assertEqual(self.codes(c), set())
        self.assertEqual(check_contract(c, "Sposta la visita a venerdì.").status,
                         HOLD_UNSUPPORTED_ACTION)

    def test_metric_handles_invalid_and_conservative_projection(self):
        gold = contract(goal("find", "treno", "transport", [fact("destination", "Pisa")],
                             modifiers={"sort": {"field": "price", "direction": "asc"}}))
        good = score(gold, copy.deepcopy(gold), "Trova un treno per Pisa")
        bad = score(gold, None, "Trova un treno per Pisa")
        self.assertTrue(good["canonical_equal"])
        self.assertFalse(bad["canonical_equal"])
        summary = aggregate([good, bad])
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["counts"]["canonical_equal"], 1)


if __name__ == "__main__":
    unittest.main()
