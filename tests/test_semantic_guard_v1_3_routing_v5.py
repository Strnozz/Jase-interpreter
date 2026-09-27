from __future__ import annotations

import copy
import unittest

from jase.semantic_guard_v1_3 import ACCEPT
from jase.semantic_guard_v1_3_routing_v5 import check_contract


def goal(action, name, typ, **extra):
    return {"id": "g1", "action": action, "target": {"name": name, "type": typ},
            "facts": [], **extra}


def contract(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


class RoutingV5Tests(unittest.TestCase):
    def test_delayed_consent_needs_policy(self):
        text = "Prenota l'hotel solo dopo che ti avrò detto di procedere."
        bad = contract(goal("book", "hotel", "accommodation"))
        self.assertIn("explicit_delayed_consent_missing_v5", check_contract(bad, text).codes)
        good = copy.deepcopy(bad)
        good["goals"][0]["policy"] = {"confirm_before": ["book"]}
        self.assertEqual(check_contract(good, text).status, ACCEPT)

    def test_no_booking_is_scoped_to_booking_targets(self):
        text = "Cerca un albergo, senza fare una prenotazione per ora."
        bad = contract(goal("find", "albergo", "accommodation"))
        self.assertIn("explicit_no_book_policy_missing_v5", check_contract(bad, text).codes)
        good = copy.deepcopy(bad)
        good["goals"][0]["policy"] = {"forbid": ["book"]}
        self.assertEqual(check_contract(good, text).status, ACCEPT)
        self.assertEqual(check_contract(contract(goal("find", "lampada", "product")), text).status,
                         ACCEPT)

    def test_narrative_is_not_converted_into_action(self):
        text = "In un romanzo qualcuno dice: 'Prenota l'hotel solo dopo che ti avrò detto di procedere.'"
        self.assertEqual(check_contract(contract(), text).status, ACCEPT)


if __name__ == "__main__":
    unittest.main()
