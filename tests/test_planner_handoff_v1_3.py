from __future__ import annotations

import copy
import unittest

from jase.planner_handoff_v1_3 import (CapabilitySpec, HOLD_GUARD, HOLD_MISSING_INFORMATION,
                                        HOLD_NO_CAPABILITY, HOLD_UNMAPPED_SLOT, NO_ACTION,
                                        READY_FOR_PLANNING, inspect_handoff)


def f(field, value, op="eq"):
    return {"field": field, "op": op, "value": value, "strength": "hard"}


def g(action, name, typ, facts=None, *, idx=1, **extra):
    return {"id": f"g{idx}", "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def c(*goals, context=None):
    result = {"schema_version": "1.3", "kind": "task" if goals else "non_actionable",
              "goals": list(goals)}
    if context:
        result["context"] = context
    return result


class PlannerHandoffTests(unittest.TestCase):
    def test_no_default_capability_or_execution_permission(self):
        request = c(g("find", "farmacia", "place", [f("location", "Tivoli", "near")]))
        self.assertEqual(inspect_handoff(request, "Cerca una farmacia a Tivoli.").status,
                         HOLD_NO_CAPABILITY)
        spec = CapabilitySpec("mock.place.find", "find", "place", fact_params={"location": "nearby"})
        result = inspect_handoff(request, "Cerca una farmacia a Tivoli.", (spec,))
        self.assertEqual(result.status, READY_FOR_PLANNING)
        self.assertFalse(result.as_dict()["execution_permitted"])
        self.assertEqual(inspect_handoff(c(), "In un racconto qualcuno chiede una farmacia.").status,
                         NO_ACTION)

    def test_each_field_and_temporal_role_needs_an_explicit_binding(self):
        request = c(g("find", "volo", "transport", [f("origin", "Bari"), f("destination", "Oslo"),
                                                   f("stops", 0)],
                      temporal=[{"role": "search_date", "op": "eq", "value": "sabato", "strength": "hard"}]))
        spec = CapabilitySpec("mock.flight.search", "find", "transport",
                              fact_params={"origin": "from", "destination": "to"},
                              temporal_params={"search_date": "departure_day"})
        result = inspect_handoff(request, "Cerca un volo diretto Bari-Oslo per sabato.", (spec,))
        self.assertEqual(result.status, HOLD_UNMAPPED_SLOT)
        self.assertIn("stops", result.codes[0])
        mapped = CapabilitySpec("mock.flight.search", "find", "transport",
                                fact_params={"origin": "from", "destination": "to", "stops": "max_stops"},
                                temporal_params={"search_date": "departure_day"})
        self.assertEqual(inspect_handoff(request, "Cerca un volo diretto Bari-Oslo per sabato.",
                                         (mapped,)).status, READY_FOR_PLANNING)
        bad_alias = copy.deepcopy(request)
        bad_alias["goals"][0]["facts"][2]["field"] = "direct"
        self.assertEqual(inspect_handoff(bad_alias, "Cerca un volo diretto Bari-Oslo per sabato.",
                                         (mapped,)).status, HOLD_GUARD)

    def test_confirmation_and_missing_inputs_hold_even_with_capability(self):
        request = c(g("hire", "idraulico", "professional_service", [f("location", "Rieti", "near")],
                      missing=[{"field": "provider", "blocks": "action", "reason": "unspecified"}],
                      policy={"confirm_before": ["hire"]}))
        spec = CapabilitySpec("mock.hire", "hire", "professional_service",
                              fact_params={"location": "area"}, policy_actions=frozenset({"hire"}))
        text = "Ingaggia un idraulico a Rieti solo dopo il mio assenso."
        self.assertEqual(inspect_handoff(request, text, (spec,)).status, HOLD_MISSING_INFORMATION)
        request["goals"][0].pop("missing")
        self.assertEqual(inspect_handoff(request, text, (spec,)).status, HOLD_GUARD)

    def test_dependencies_conditions_references_and_policy_are_bound(self):
        first = g("find", "lampada", "product", [f("location", "Vasto", "near")], idx=1)
        second = g("buy", "lampada", "product", [{"field": "selection", "op": "eq", "strength": "hard",
                                                   "value_ref": {"goal": "g1", "field": "result", "ordinal": 1}}],
                   idx=2, depends_on=["g1"], policy={"confirm_before": ["buy"]})
        request = c(first, second)
        search = CapabilitySpec("mock.product.find", "find", "product",
                                fact_params={"location": "area"}, result_params={"result": "items"})
        buy = CapabilitySpec("mock.product.buy", "buy", "product",
                             fact_params={"selection": "item_id"}, accepts_references=True,
                             accepts_ordinal_references=True, supports_dependencies=True,
                             policy_actions=frozenset({"buy"}))
        text = "Cerca una lampada a Vasto, poi compra la prima dopo il mio sì."
        result = inspect_handoff(request, text, (search, buy))
        self.assertEqual(result.status, READY_FOR_PLANNING)
        self.assertEqual(len(result.bindings), 2)
        no_refs = copy.deepcopy(buy)
        object.__setattr__(no_refs, "accepts_references", False)
        self.assertEqual(inspect_handoff(request, text, (search, no_refs)).status,
                         HOLD_UNMAPPED_SLOT)
        no_result = copy.deepcopy(search)
        object.__setattr__(no_result, "result_params", {})
        self.assertEqual(inspect_handoff(request, text, (no_result, buy)).status,
                         HOLD_UNMAPPED_SLOT)

    def test_fallback_requires_result_and_condition_support(self):
        train = g("find", "treno", "transport", [f("destination", "Aosta")], idx=1)
        bus = g("find", "autobus", "transport", [f("destination", "Aosta")], idx=2,
                depends_on=["g1"],
                condition={"goal": "g1", "test": {"field": "results_count", "op": "eq", "value": 0}})
        request = c(train, bus)
        text = "Se non trovi treni per Aosta, cerca autobus per Aosta."
        train_spec = CapabilitySpec("mock.train.find", "find", "transport",
                                    target_names=frozenset({"treno"}),
                                    fact_params={"destination": "to"},
                                    result_params={"results_count": "count"})
        bus_spec = CapabilitySpec("mock.bus.find", "find", "transport",
                                  target_names=frozenset({"autobus"}),
                                  fact_params={"destination": "to"},
                                  supports_dependencies=True)
        self.assertEqual(inspect_handoff(request, text, (train_spec, bus_spec)).status,
                         HOLD_UNMAPPED_SLOT)
        bus_spec = CapabilitySpec("mock.bus.find", "find", "transport",
                                  target_names=frozenset({"autobus"}),
                                  fact_params={"destination": "to"},
                                  supports_dependencies=True, condition_ops=frozenset({"eq"}))
        self.assertEqual(inspect_handoff(request, text, (train_spec, bus_spec)).status,
                         READY_FOR_PLANNING)

    def test_capability_required_slot_holds_when_model_omits_missing(self):
        request = c(g("book", "hotel", "accommodation", [f("location", "Piombino", "near")],
                      policy={"confirm_before": ["book"]}))
        spec = CapabilitySpec("mock.hotel.book", "book", "accommodation",
                              fact_params={"location": "city"},
                              policy_actions=frozenset({"book"}),
                              required_slots=frozenset({"temporal.action_date"}))
        result = inspect_handoff(request, "Prenota un hotel a Piombino dopo il mio ok.", (spec,))
        self.assertEqual(result.status, HOLD_MISSING_INFORMATION)
        self.assertIn("temporal.action_date", result.codes[0])


if __name__ == "__main__":
    unittest.main()
