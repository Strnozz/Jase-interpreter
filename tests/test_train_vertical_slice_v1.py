"""Safety and semantic binding tests for the opt-in train vertical slice."""
from __future__ import annotations

import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

from benchmarks.build_v28_blind import c, f, g, t
from jase.train_vertical_slice_v1 import run_train_search


NOW = datetime(2026, 9, 28, 12, tzinfo=ZoneInfo("Europe/Rome"))


class RecordingProvider:
    provider_id = "recording_provider"
    source_kind = "MOCK"
    coverage = "unit_test"

    def __init__(self):
        self.requests = []

    def resolve_station(self, value):
        if value not in {"Lecco", "Monza"}:
            raise AssertionError("unexpected station")
        return (value,)

    def search(self, request):
        self.requests.append(request)
        return []


class TrainVerticalSliceTests(unittest.TestCase):
    def test_arrival_deadline_preserved_and_only_read_permitted(self):
        provider = RecordingProvider()
        request = "Domani devo essere a Monza entro le 10:00: cerca un treno da Lecco."
        contract = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")],
                       temporal=[t("search_date", "domani"), t("arrival_time", "10:00", "before")]))
        result = run_train_search(contract, request, request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD_INCOMPLETE_COVERAGE")
        self.assertTrue(result["provider_read_permitted"])
        self.assertFalse(result["execution_permitted"])
        self.assertFalse(result["external_side_effect_permitted"])
        self.assertEqual(len(provider.requests), 1)
        self.assertEqual(provider.requests[0].travel_date.isoformat(), "2026-09-29")
        constraint = provider.requests[0].time_constraints[0]
        self.assertEqual((constraint.role, constraint.operator, constraint.start, constraint.inclusive),
                         ("arrival_time", "before", "10:00", True))

    def test_missing_date_and_non_action_never_call_provider(self):
        provider = RecordingProvider()
        no_date = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")]))
        result = run_train_search(no_date, "Cerca treni da Lecco a Monza.", request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(len(provider.requests), 0)
        meta = run_train_search(c(), "Traduci 'cerca treni da Lecco a Monza'.", request_at=NOW, provider=provider)
        self.assertEqual(meta["status"], "NO_ACTION")
        self.assertEqual(len(provider.requests), 0)

    def test_price_and_write_requests_stay_closed(self):
        provider = RecordingProvider()
        priced = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza"),
                                                   f("price", 10, "lte", currency="EUR")],
                     temporal=[t("search_date", "domani")]))
        result = run_train_search(priced, "Domani cerca un treno da Lecco a Monza entro 10 euro.",
                                  request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD")
        booking = c(g("book", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")],
                      temporal=[t("action_date", "domani")], policy={"confirm_before": ["book"]}))
        result = run_train_search(booking, "Prenota da Lecco a Monza domani.", request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(len(provider.requests), 0)

    def test_explicit_morning_not_dropped_from_provider_query(self):
        provider = RecordingProvider()
        contract = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")],
                       temporal=[t("search_date", "domani")]))
        result = run_train_search(contract, "Domani mattina cerca un treno da Lecco a Monza.",
                                  request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD")
        self.assertIn("HOLD_OMITTED_DAYPART", result["reason_codes"])
        self.assertEqual(provider.requests, [])

    def test_zero_direct_results_do_not_claim_complete_search(self):
        provider = RecordingProvider()
        contract = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")],
                       temporal=[t("search_date", "domani")]))
        result = run_train_search(contract, "Domani cerca un treno da Lecco a Monza.",
                                  request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD_INCOMPLETE_COVERAGE")
        self.assertTrue(result["provider_called"])
        self.assertFalse(result["execution_permitted"])

    def test_wrong_sort_direction_cannot_reach_provider(self):
        provider = RecordingProvider()
        contract = c(g("find", "treno", "transport", [f("origin", "Lecco"), f("destination", "Monza")],
                       temporal=[t("search_date", "domani")],
                       modifiers={"sort": {"field": "duration", "direction": "desc"}}))
        result = run_train_search(contract, "Domani trova treni da Lecco a Monza ordinati dalla durata minore.",
                                  request_at=NOW, provider=provider)
        self.assertEqual(result["status"], "HOLD")
        self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()
