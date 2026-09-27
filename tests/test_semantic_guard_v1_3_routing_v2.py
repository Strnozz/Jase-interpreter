from __future__ import annotations

import copy
import unittest

from jase.semantic_guard_v1_3 import ACCEPT
from jase.semantic_guard_v1_3_routing_v2 import check_contract
from jase.semantic_repair_v1_3 import repair_candidate


def fact(field, value, op="eq"):
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def time(role, value, op="eq"):
    return {"role": role, "op": op, "value": value, "strength": "hard"}


def goal(action, name, typ, facts=None, **extra):
    return {"id": "g1", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def contract(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


class RoutingV2Tests(unittest.TestCase):
    def test_story_quote_is_held_but_message_to_send_is_actionable(self):
        text = "Nel copione c'e' la battuta 'prenota un taxi'. Mi dici se suona naturale?"
        self.assertEqual(check_contract(contract(), text).status, ACCEPT)
        self.assertIn("quoted_example_misread_as_action",
                      check_contract(contract(goal("book", "taxi", "transport")), text).codes)
        send = contract(goal("contact", "Lia", "person", [fact("message", "Prenota un taxi")]))
        self.assertEqual(check_contract(send, "Manda a Lia 'Prenota un taxi'.").status, ACCEPT)

    def test_origin_arrival_and_grounded_spatial_repair(self):
        text = "Parto da Parma: cerca un treno per Milano che arrivi entro le 10:15."
        good = contract(goal("find", "treno", "transport",
                             [fact("origin", "Parma"), fact("destination", "Milano")],
                             temporal=[time("arrival_time", "10:15", "before")]))
        self.assertEqual(check_contract(good, text).status, ACCEPT)
        no_origin = copy.deepcopy(good)
        no_origin["goals"][0]["facts"] = [fact("destination", "Milano")]
        self.assertIn("explicit_origin_missing", check_contract(no_origin, text).codes)
        no_origin["goals"][0]["spatial"] = [fact("origin", "Parma")]
        result = repair_candidate(no_origin, text)
        self.assertEqual(result.status, "REPAIRED")
        self.assertEqual(check_contract(result.contract, text).status, ACCEPT)
        no_origin["goals"][0]["spatial"][0]["value"] = "Torino"
        self.assertIn("spatial_origin_not_grounded", repair_candidate(no_origin, text).codes)
        wrong_time = copy.deepcopy(good)
        wrong_time["goals"][0]["temporal"][0]["role"] = "search_time"
        self.assertIn("arrival_deadline_role_missing", check_contract(wrong_time, text).codes)

    def test_open_now_walking_access_and_limit(self):
        pharmacy = contract(goal("find", "farmacia", "place", [fact("location", "Parma", "near")]))
        self.assertIn("open_now_missing", check_contract(pharmacy, "Trova una farmacia aperta ora a Parma.").codes)
        pharmacy["goals"][0]["facts"].append(fact("open_now", True))
        self.assertEqual(check_contract(pharmacy, "Trova una farmacia aperta ora a Parma.").status, ACCEPT)
        walking = contract(goal("find", "ristorante", "place", [fact("location", "Pisa", "near")]))
        self.assertIn("walking_access_missing",
                      check_contract(walking, "Trova un ristorante a Pisa raggiungibile a piedi.").codes)
        self.assertIn("result_limit_missing_v2",
                      check_contract(walking, "Cerca tre ristoranti a Pisa.").codes)

    def test_no_book_synonym_and_zero_result_fallback(self):
        hotel = contract(goal("find", "hotel", "accommodation"))
        text = "Fammi vedere due hotel, ma non riservare camere."
        decision = check_contract(hotel, text)
        self.assertIn("explicit_no_booking_policy_missing_v2", decision.codes)
        self.assertIn("result_limit_missing_v2", decision.codes)
        hotel["goals"][0]["policy"] = {"forbid": ["book"]}
        hotel["goals"][0]["modifiers"] = {"limit": 2}
        self.assertEqual(check_contract(hotel, text).status, ACCEPT)
        first = goal("find", "treno", "transport", [fact("destination", "Roma")])
        second = goal("find", "autobus", "transport", [fact("destination", "Roma")])
        second["id"] = "g2"
        second["depends_on"] = ["g1"]
        text = "Trova un treno; solo se non ce ne sono, cerca un autobus."
        self.assertIn("zero_results_fallback_missing", check_contract(contract(first, second), text).codes)
        second["condition"] = {"goal": "g1", "test": {"field": "results_count", "op": "eq", "value": 0}}
        self.assertEqual(check_contract(contract(first, second), text).status, ACCEPT)

    def test_restaurant_booking_requires_slots_or_missing_markers(self):
        book = contract(goal("book", "ristorante giapponese", "place"))
        self.assertIn("restaurant_booking_slots_missing",
                      check_contract(book, "Prenota il ristorante giapponese.").codes)
        book["goals"][0]["missing"] = [
            {"field": name, "blocks": "action", "reason": "unspecified"}
            for name in ("date", "time", "guests")]
        self.assertEqual(check_contract(book, "Prenota il ristorante giapponese.").status, ACCEPT)


if __name__ == "__main__":
    unittest.main()
