from __future__ import annotations

import copy
import unittest

from jase.semantic_guard_v1_3 import ACCEPT
from jase.semantic_guard_v1_3_routing import check_contract


def fact(field, value, op="eq"):
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def goal(action, target, typ, facts=None, **extra):
    return {"id": "g1", "action": action, "target": {"name": target, "type": typ},
            "facts": facts or [], **extra}


def contract(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


class RoutingGuardTests(unittest.TestCase):
    def test_quoted_command_is_not_executed_but_quoted_message_is_actionable(self):
        text = "Per un racconto, immagina qualcuno che dica 'prenota una visita'."
        wrong = contract(goal("book", "visita", "appointment"))
        self.assertIn("quoted_command_misread_as_action", check_contract(wrong, text).codes)
        self.assertEqual(check_contract(contract(), text).status, ACCEPT)
        message = contract(goal("contact", "Luca", "person", [fact("message", "Arrivo alle 18")]))
        self.assertEqual(check_contract(message, "Invia a Luca 'Arrivo alle 18'.").status, ACCEPT)

    def test_explicit_no_booking_and_direct_flight_are_grounded(self):
        hotel = contract(goal("find", "albergo", "accommodation"))
        text = "Mostrami alberghi economici, senza prenotare."
        self.assertIn("explicit_no_booking_policy_missing", check_contract(hotel, text).codes)
        hotel["goals"][0]["policy"] = {"forbid": ["book"]}
        self.assertEqual(check_contract(hotel, text).status, ACCEPT)
        flight = contract(goal("find", "volo", "transport", [fact("destination", "Berlino")]))
        text = "Trova un volo diretto per Berlino."
        self.assertIn("direct_flight_constraint_missing", check_contract(flight, text).codes)
        flight["goals"][0]["facts"].append(fact("stops", 0))
        self.assertEqual(check_contract(flight, text).status, ACCEPT)

    def test_search_then_book_needs_second_goal_and_result_reference(self):
        text = "Trova una camera a Torino, poi prenota la prima."
        found = goal("find", "camera", "accommodation", [fact("location", "Torino", "near")])
        self.assertIn("requested_booking_goal_missing", check_contract(contract(found), text).codes)
        booked = goal("book", "camera", "accommodation")
        booked["id"] = "g2"
        booked["depends_on"] = ["g1"]
        self.assertIn("booking_result_dependency_missing", check_contract(contract(found, booked), text).codes)
        booked["facts"] = [{"field": "selection", "op": "eq", "strength": "hard",
                             "value_ref": {"goal": "g1", "field": "result", "ordinal": 1}}]
        self.assertEqual(check_contract(contract(found, booked), text).status, ACCEPT)

    def test_field_alias_and_result_count_are_held_for_mapping_review(self):
        order = contract(goal("order", "pizza", "food", [fact("delivery", "Torino")]))
        self.assertIn("alias_requires_planner_mapping", check_contract(order, "Ordina pizza a Torino.").codes)
        order["goals"][0]["facts"][0]["field"] = "delivery_location"
        self.assertEqual(check_contract(order, "Ordina pizza a Torino.").status, ACCEPT)
        primary = goal("find", "autobus", "transport", [fact("destination", "Torino")])
        fallback = copy.deepcopy(primary)
        fallback["id"] = "g2"
        fallback["target"]["name"] = "treno"
        fallback["depends_on"] = ["g1"]
        fallback["condition"] = {"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}}
        text = "Cerca un autobus; se non trovi risultati, cerca un treno."
        self.assertIn("alias_requires_planner_mapping", check_contract(contract(primary, fallback), text).codes)
        fallback["condition"]["test"]["field"] = "results_count"
        self.assertEqual(check_contract(contract(primary, fallback), text).status, ACCEPT)

    def test_compare_cardinality_uses_modifier(self):
        c = contract(goal("compare", "zaino", "product"))
        text = "Confrontane esattamente tre."
        self.assertIn("compare_cardinality_missing", check_contract(c, text).codes)
        c["goals"][0]["modifiers"] = {"compare_count": 3}
        self.assertEqual(check_contract(c, text).status, ACCEPT)


if __name__ == "__main__":
    unittest.main()
