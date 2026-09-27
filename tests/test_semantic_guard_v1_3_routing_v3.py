from __future__ import annotations

import copy
import unittest

from jase.semantic_guard_v1_3 import ACCEPT
from jase.semantic_guard_v1_3_routing_v3 import check_contract


def fact(field, value):
    return {"field": field, "op": "eq", "value": value, "strength": "hard"}


def temporal(role, value):
    return {"role": role, "op": "eq", "value": value, "strength": "hard"}


def goal(action, name, typ, facts=None, **more):
    return {"id": "g1", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **more}


def task(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


class RoutingV3Tests(unittest.TestCase):
    def test_message_preview_requires_confirmation_without_capturing_message_hour(self):
        text = "Mostrami il testo prima di spedirlo a Lina: 'L'appuntamento sarà alle 14'."
        good = task(goal("contact", "Lina", "person", [fact("message", "L'appuntamento sarà alle 14")],
                         policy={"confirm_before": ["contact"]}))
        self.assertEqual(check_contract(good, text).status, ACCEPT)
        bad = copy.deepcopy(good)
        bad["goals"][0].pop("policy")
        self.assertIn("explicit_confirmation_missing_v3", check_contract(bad, text).codes)
        # A quoted command without an actual requested side effect stays inert.
        self.assertEqual(check_contract(task(), "Nel racconto 'mostrami prima di spedirlo' è dialogo.").status,
                         ACCEPT)

    def test_reminder_uses_explicit_scheduling_hour_not_content_hour(self):
        text = "Mercoledì alle 07 ricordami: 'prelevare il materiale alle 12'."
        good = task(goal("notify", "promemoria", "reminder", [fact("message", "prelevare il materiale alle 12")],
                         temporal=[temporal("notification_date", "mercoledì"),
                                   temporal("notification_time", "07:00")]))
        self.assertEqual(check_contract(good, text).status, ACCEPT)
        bad = copy.deepcopy(good)
        bad["goals"][0]["temporal"][1]["value"] = "12:00"
        self.assertIn("explicit_reminder_time_mismatch_v3", check_contract(bad, text).codes)

    def test_unknown_hire_provider_is_held_even_with_confirmation(self):
        text = "Ingaggia un fabbro a Orvieto, ma chiedimi il via libera prima."
        bad = task(goal("hire", "fabbro", "professional_service", [fact("location", "Orvieto")],
                        policy={"confirm_before": ["hire"]}))
        self.assertIn("provider_identity_missing_v3", check_contract(bad, text).codes)
        good = copy.deepcopy(bad)
        good["goals"][0]["missing"] = [{"field": "provider", "blocks": "action", "reason": "unspecified"}]
        self.assertEqual(check_contract(good, text).status, ACCEPT)

    def test_explicit_purchase_and_hire_prohibitions(self):
        food_text = "Ordina i risultati delle torte per prezzo; non fare acquisti."
        food = task(goal("find", "torta", "food", modifiers={"sort": {"field": "price", "direction": "asc"}}))
        self.assertIn("explicit_no_purchase_policy_missing_v3", check_contract(food, food_text).codes)
        food["goals"][0]["policy"] = {"forbid": ["buy", "order"]}
        self.assertEqual(check_contract(food, food_text).status, ACCEPT)
        hire_text = "Cerca un muratore, senza ingaggiare nessuno."
        service = task(goal("find", "muratore", "professional_service"))
        self.assertIn("explicit_no_hire_policy_missing_v3", check_contract(service, hire_text).codes)
        service["goals"][0]["policy"] = {"forbid": ["hire"]}
        self.assertEqual(check_contract(service, hire_text).status, ACCEPT)

    def test_search_without_execution_is_not_forced_to_confirmation(self):
        text = "Cerca un ostello a Lodi; prima di prenotare voglio vederlo."
        find = task(goal("find", "ostello", "accommodation", [fact("location", "Lodi")],
                         policy={"forbid": ["book"]}))
        self.assertEqual(check_contract(find, text).status, ACCEPT)


if __name__ == "__main__":
    unittest.main()
