"""Contract checks for rail-only filtering, temporal operators and result refs."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
import zipfile
from datetime import date
from pathlib import Path

from jase.train_provider_v1 import (GtfsSnapshotTrainProvider, TimeConstraint,
                                    TrainSearchRequest, UnsupportedConstraint,
                                    resolve_result_reference)


class TrainProviderContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "fixture.zip"
        tables = {
            "stops.txt": "stop_id,stop_name,stop_lat,stop_lon\nA,Lecco,45,9\nB,Monza,45,9\nC,Varese,45,9\nD,Varese Nord,45,9\n",
            "routes.txt": "route_id,route_short_name,route_type\nR1,R1,2\nBUS,Bus,3\n",
            "trips.txt": "trip_id,route_id,service_id\nT1,R1,S\nT2,R1,S\nT3,BUS,S\nT4,R1,S\n",
            "stop_times.txt": "trip_id,arrival_time,departure_time,stop_id,stop_sequence\n"
                              "T1,08:00:00,08:00:00,A,1\nT1,08:45:00,08:45:00,B,2\n"
                              "T2,09:00:00,09:00:00,A,1\nT2,09:40:00,09:40:00,B,2\n"
                              "T3,08:20:00,08:20:00,A,1\nT3,08:30:00,08:30:00,B,2\n"
                              "T4,17:30:00,17:30:00,A,1\nT4,18:15:00,18:15:00,B,2\n",
            "calendar_dates.txt": "service_id,date,exception_type\nS,20260929,1\nS,20261025,1\n",
        }
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in tables.items():
                archive.writestr(name, content)
        self.provider = GtfsSnapshotTrainProvider(path, expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

    def tearDown(self):
        self.tmp.cleanup()

    def test_read_only_rail_schedule_and_stable_references(self):
        request = TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29), limit=3)
        first = self.provider.search(request)
        second = self.provider.search(request)
        self.assertEqual(len(first), 3)
        self.assertEqual([x.jase_result_ref for x in first], [x.jase_result_ref for x in second])
        self.assertEqual(first[0].departure_datetime[11:16], "08:00")
        self.assertEqual(first[1].departure_datetime[11:16], "09:00")
        self.assertEqual(resolve_result_reference(first, ordinal=2).provider_result_ref,
                         first[1].provider_result_ref)
        self.assertEqual(resolve_result_reference(first, ordinal=1).provider_result_ref,
                         first[0].provider_result_ref)
        self.assertEqual(resolve_result_reference(first, ordinal=3).provider_result_ref,
                         first[2].provider_result_ref)
        self.assertEqual(resolve_result_reference(first, departure_time="17:30").provider_result_ref,
                         first[2].provider_result_ref)
        self.assertEqual(len([item for item in first if item.number_of_changes == 0]), 3)
        self.assertNotIn("price", first[0].as_dict())

    def test_departure_and_arrival_roles_not_collapsed(self):
        after = TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29),
                                   time_constraints=(TimeConstraint("departure_time", "after", "08:00"),))
        before = TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29),
                                    time_constraints=(TimeConstraint("arrival_time", "before", "09:00"),))
        self.assertEqual([x.departure_datetime[11:16] for x in self.provider.search(after)], ["09:00", "17:30"])
        self.assertEqual([x.departure_datetime[11:16] for x in self.provider.search(before)], ["08:00"])
        inclusive = TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29),
                                       time_constraints=(TimeConstraint("arrival_time", "before", "08:45", inclusive=True),))
        strict = TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29),
                                    time_constraints=(TimeConstraint("arrival_time", "before", "08:45"),))
        self.assertEqual(len(self.provider.search(inclusive)), 1)
        self.assertEqual(len(self.provider.search(strict)), 0)

    def test_unsupported_or_unavailable_information_is_not_fabricated(self):
        with self.assertRaises(UnsupportedConstraint):
            self.provider.search(TrainSearchRequest("Lecco", "Monza", date(2026, 9, 29),
                                                    sort_field="price"))
        with self.assertRaises(UnsupportedConstraint):
            resolve_result_reference(self.provider.search(TrainSearchRequest(
                "Lecco", "Monza", date(2026, 9, 29))), cheapest=True)

    def test_city_station_group_is_explicit_and_deterministic(self):
        self.assertEqual(self.provider.resolve_station("Varese"), ("C", "D"))

    def test_duration_arrival_sort_and_limit_apply_after_filters(self):
        shortest = self.provider.search(TrainSearchRequest(
            "Lecco", "Monza", date(2026, 9, 29), sort_field="duration", limit=1))
        self.assertEqual(len(shortest), 1)
        self.assertEqual(shortest[0].departure_datetime[11:16], "09:00")
        latest_arrival = self.provider.search(TrainSearchRequest(
            "Lecco", "Monza", date(2026, 9, 29), sort_field="arrival_time",
            sort_direction="desc", limit=1))
        self.assertEqual(latest_arrival[0].departure_datetime[11:16], "17:30")

    def test_ambiguous_dst_transition_day_holds(self):
        with self.assertRaisesRegex(UnsupportedConstraint, "dst_transition_day"):
            self.provider.search(TrainSearchRequest("Lecco", "Monza", date(2026, 10, 25)))


if __name__ == "__main__":
    unittest.main()
