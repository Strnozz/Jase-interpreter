from __future__ import annotations

import copy
import unittest

from jase.semantic_guard_v1_3 import ACCEPT
from jase.semantic_guard_v1_3_routing_v4 import check_contract


def f(field, value, op="eq"):
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def g(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def c(*goals):
    return {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
            "goals": list(goals)}


class RoutingV4Tests(unittest.TestCase):
    def test_indirect_approval_for_side_effect(self):
        text = "Prima di cancellare la seduta, mostrami quale hai trovato."
        bad = c(g("cancel", "seduta", "appointment", [f("original_date", "8 maggio")]))
        self.assertIn("explicit_approval_missing_v4", check_contract(bad, text).codes)
        good = copy.deepcopy(bad)
        good["goals"][0]["policy"] = {"confirm_before": ["cancel"]}
        self.assertEqual(check_contract(good, text).status, ACCEPT)
        self.assertEqual(check_contract(c(), "Nel testo si legge 'cancella la seduta'; è una citazione.").status,
                         ACCEPT)

    def test_ordinal_result_requires_dependency_and_reference(self):
        text = "Cerca una lampada e compra la seconda dopo il mio sì."
        find = g("find", "lampada", "product", idx=1)
        buy = g("buy", "lampada", "product", [f("selection", "seconda")], idx=2,
                depends_on=["g1"], policy={"confirm_before": ["buy"]})
        self.assertIn("ordinal_result_reference_missing_v4", check_contract(c(find, buy), text).codes)
        buy["facts"] = [{"field": "selection", "op": "eq", "strength": "hard",
                         "value_ref": {"goal": "g1", "field": "result", "ordinal": 2}}]
        self.assertEqual(check_contract(c(find, buy), text).status, ACCEPT)

    def test_compare_count_and_price_sort_preserved(self):
        text = "Metti a confronto tre alberghi a Pisa dal più economico al più caro."
        bad = c(g("compare", "albergo", "accommodation", [f("location", "Pisa", "near")]))
        decision = check_contract(bad, text)
        self.assertIn("compare_count_missing_v4", decision.codes)
        self.assertIn("explicit_price_sort_missing_v4", decision.codes)
        good = copy.deepcopy(bad)
        good["goals"][0]["modifiers"] = {"compare_count": 3,
                                          "sort": {"field": "price", "direction": "asc"}}
        self.assertEqual(check_contract(good, text).status, ACCEPT)

    def test_rail_mode_and_indirect_rent_prohibition(self):
        text = "Consulta i collegamenti ferroviari fra Faenza e Forlì."
        generic = c(g("find", "trasporto", "transport", [f("origin", "Faenza"),
                                                       f("destination", "Forlì")]))
        self.assertIn("rail_mode_not_bound_to_target_v4", check_contract(generic, text).codes)
        specific = copy.deepcopy(generic)
        specific["goals"][0]["target"]["name"] = "treno"
        self.assertEqual(check_contract(specific, text).status, ACCEPT)
        rent_text = "Confronta le biciclette a noleggio, senza prenderne una per ora."
        vehicle = c(g("compare", "bicicletta", "vehicle", modifiers={"compare_count": 2}))
        self.assertIn("explicit_no_rent_policy_missing_v4", check_contract(vehicle, rent_text).codes)
        vehicle["goals"][0]["policy"] = {"forbid": ["rent"]}
        self.assertEqual(check_contract(vehicle, rent_text).status, ACCEPT)


if __name__ == "__main__":
    unittest.main()
