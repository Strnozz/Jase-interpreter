"""Typed, read-only GTFS train timetable provider (no bookings or writes)."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo("Europe/Rome")
SOURCE_URL = "https://www.dati.lombardia.it/download/3z4k-mxz9"
EXPECTED_SOURCE_SHA256 = "b4296f145b42ccb35c26085470ff4b3fd5dffe533251c0aab312312a73820ad6"


class UnsupportedConstraint(ValueError):
    pass


class UnresolvedStation(ValueError):
    pass


@dataclass(frozen=True)
class TimeConstraint:
    role: str  # departure_time or arrival_time
    operator: str  # after, before, between, eq
    start: str  # HH:MM
    end: str | None = None


@dataclass(frozen=True)
class TrainSearchRequest:
    origin: str
    destination: str
    travel_date: date
    time_constraints: tuple[TimeConstraint, ...] = ()
    direct_only: bool = False
    train_category: str | None = None
    max_duration_minutes: int | None = None
    limit: int = 10
    sort_field: str = "departure_time"
    sort_direction: str = "asc"
    query_ref: str = ""


@dataclass(frozen=True)
class TrainSearchResult:
    jase_result_ref: str
    provider_result_ref: str
    provider: str
    query_ref: str
    origin: str
    destination: str
    departure_datetime: str
    arrival_datetime: str
    duration_minutes: int
    number_of_changes: int
    train_segments: tuple[dict, ...]
    train_categories: tuple[str, ...]
    source_metadata: dict

    def as_dict(self) -> dict:
        return asdict(self)


class TrainSearchProvider(Protocol):
    provider_id: str
    source_kind: str

    def resolve_station(self, value: str) -> tuple[str, ...]: ...
    def search(self, request: TrainSearchRequest) -> list[TrainSearchResult]: ...


def _norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKD", value.casefold()).encode("ascii", "ignore").decode().split())


def _minutes(value: str) -> int:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", value)
    if not match or int(match[1]) > 23 or int(match[2]) > 59:
        raise UnsupportedConstraint("invalid_hhmm")
    return int(match[1]) * 60 + int(match[2])


def _seconds(value: str) -> int:
    h, m, s = (int(part) for part in value.split(":"))
    return h * 3600 + m * 60 + s


def _time_ok(value: datetime, constraint: TimeConstraint) -> bool:
    current = value.hour * 60 + value.minute
    bound = _minutes(constraint.start)
    if constraint.operator == "after":
        return current > bound
    if constraint.operator == "before":
        return current < bound
    if constraint.operator == "eq":
        return current == bound
    if constraint.operator == "between":
        if constraint.end is None:
            raise UnsupportedConstraint("range_end_missing")
        return bound <= current <= _minutes(constraint.end)
    raise UnsupportedConstraint("unknown_temporal_operator")


class GtfsSnapshotTrainProvider:
    """Read the official licensed Trenord GTFS snapshot. Direct train trips only.

    Source is a real provider feed, retrieved via read-only GET and verified by
    SHA-256. It is static schedule data, not live running/price/availability.
    """

    provider_id = "trenord_gtfs_lombardia"
    source_kind = "REAL_PROVIDER"
    coverage = "direct_train_trips_in_feed_only"

    def __init__(self, zip_path: Path, *, expected_sha256: str = EXPECTED_SOURCE_SHA256):
        self.zip_path = Path(zip_path)
        self.source_sha256 = hashlib.sha256(self.zip_path.read_bytes()).hexdigest()
        if self.source_sha256 != expected_sha256:
            raise ValueError("GTFS source hash mismatch; inspect new snapshot before use")
        self.stops: dict[str, str] = {}
        self.by_name: dict[str, list[str]] = defaultdict(list)
        self.routes: dict[str, dict] = {}
        self.trips: dict[str, dict] = {}
        self.times_by_trip: dict[str, list[tuple[int, str, int, int]]] = defaultdict(list)
        self.by_origin_stop: dict[str, set[str]] = defaultdict(set)
        self.service_by_date: dict[str, set[str]] = defaultdict(set)
        with zipfile.ZipFile(self.zip_path) as archive:
            def rows(name: str):
                with archive.open(name) as raw:
                    yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))

            for row in rows("stops.txt"):
                self.stops[row["stop_id"]] = row["stop_name"]
                self.by_name[_norm(row["stop_name"])].append(row["stop_id"])
            for row in rows("routes.txt"):
                self.routes[row["route_id"]] = row
            for row in rows("trips.txt"):
                route = self.routes.get(row["route_id"])
                if route and route.get("route_type") == "2":  # GTFS rail only, not replacement buses
                    self.trips[row["trip_id"]] = row
            for row in rows("stop_times.txt"):
                trip_id = row["trip_id"]
                if trip_id not in self.trips:
                    continue
                arrival, departure = row["arrival_time"], row["departure_time"]
                if not arrival or not departure:
                    continue
                entry = (int(row["stop_sequence"]), row["stop_id"], _seconds(arrival), _seconds(departure))
                self.times_by_trip[trip_id].append(entry)
                self.by_origin_stop[row["stop_id"]].add(trip_id)
            for row in rows("calendar_dates.txt"):
                if row["exception_type"] == "1":
                    self.service_by_date[row["date"]].add(row["service_id"])
        for entries in self.times_by_trip.values():
            entries.sort()

    def resolve_station(self, value: str) -> tuple[str, ...]:
        normalized = _norm(value)
        exact = self.by_name.get(normalized)
        if exact:
            return tuple(sorted(exact))
        # Explicit city names are station groups, never a single guessed stop.
        city_groups = {"milano", "como", "varese", "brescia", "novara"}
        if normalized in city_groups:
            ids = [ident for ident, name in self.stops.items()
                   if _norm(name).startswith(normalized + " ")]
            if ids:
                return tuple(sorted(ids))
        raise UnresolvedStation(value)

    def search(self, request: TrainSearchRequest) -> list[TrainSearchResult]:
        if request.train_category not in {None, "regional"}:
            raise UnsupportedConstraint("train_category_not_in_feed")
        if not 1 <= request.limit <= 100:
            raise UnsupportedConstraint("invalid_result_limit")
        if request.sort_field not in {"departure_time", "arrival_time", "duration"} or request.sort_direction not in {"asc", "desc"}:
            raise UnsupportedConstraint("sort_not_supported")
        if any(item.role not in {"departure_time", "arrival_time"} for item in request.time_constraints):
            raise UnsupportedConstraint("temporal_role_not_supported")
        origins, destinations = self.resolve_station(request.origin), set(self.resolve_station(request.destination))
        query_ref = request.query_ref or "train:" + hashlib.sha256(json.dumps(
            {"origin": origins, "destination": sorted(destinations), "date": request.travel_date.isoformat(),
             "time": [asdict(x) for x in request.time_constraints], "category": request.train_category,
             "max_duration": request.max_duration_minutes, "sort": request.sort_field,
             "direction": request.sort_direction, "limit": request.limit, "source": self.source_sha256},
            sort_keys=True).encode()).hexdigest()[:16]
        candidate_trips = set().union(*(self.by_origin_stop.get(stop, set()) for stop in origins))
        found: list[tuple[datetime, datetime, str, str, str, str, str]] = []
        for service_date in (request.travel_date - timedelta(days=1), request.travel_date):
            active = self.service_by_date.get(service_date.strftime("%Y%m%d"), set())
            for trip_id in candidate_trips:
                trip = self.trips.get(trip_id)
                if not trip or trip["service_id"] not in active:
                    continue
                route = self.routes[trip["route_id"]]
                line = route.get("route_short_name", "")
                if request.train_category == "regional" and not re.fullmatch(r"R\d+|RE[ _-]?\d+", line, re.I):
                    continue
                entries = self.times_by_trip[trip_id]
                for i, (seq, origin_id, _, depart_seconds) in enumerate(entries):
                    if origin_id not in origins:
                        continue
                    departure = datetime.combine(service_date, datetime.min.time(), TIMEZONE) + timedelta(seconds=depart_seconds)
                    if departure.date() != request.travel_date:
                        continue
                    for _, destination_id, arrive_seconds, _ in entries[i + 1:]:
                        if destination_id not in destinations or arrive_seconds < depart_seconds:
                            continue
                        arrival = datetime.combine(service_date, datetime.min.time(), TIMEZONE) + timedelta(seconds=arrive_seconds)
                        if arrival.date() != request.travel_date and any(
                                tc.role == "arrival_time" for tc in request.time_constraints):
                            # A clock-only deadline cannot safely select a different calendar day.
                            continue
                        if request.max_duration_minutes is not None and (arrival - departure).total_seconds() > request.max_duration_minutes * 60:
                            continue
                        if any(not _time_ok(departure if tc.role == "departure_time" else arrival, tc)
                               for tc in request.time_constraints):
                            continue
                        found.append((departure, arrival, trip_id, origin_id, destination_id, line, trip["route_id"]))
                        break
        key = {"departure_time": lambda x: x[0], "arrival_time": lambda x: x[1],
               "duration": lambda x: x[1] - x[0]}[request.sort_field]
        found.sort(key=lambda item: (key(item), item[2], item[3], item[4]), reverse=request.sort_direction == "desc")
        results = []
        for rank, (departure, arrival, trip_id, origin_id, destination_id, line, route_id) in enumerate(found[:request.limit], 1):
            results.append(TrainSearchResult(
                jase_result_ref=f"{query_ref}:R{rank:04d}",
                provider_result_ref=f"{trip_id}:{origin_id}:{destination_id}:{departure.isoformat()}",
                provider=self.provider_id, query_ref=query_ref,
                origin=self.stops[origin_id], destination=self.stops[destination_id],
                departure_datetime=departure.isoformat(), arrival_datetime=arrival.isoformat(),
                duration_minutes=int((arrival - departure).total_seconds() // 60), number_of_changes=0,
                train_segments=({"trip_id": trip_id, "route_id": route_id, "line": line,
                                 "origin_stop_id": origin_id, "destination_stop_id": destination_id},),
                train_categories=("regional",) if re.fullmatch(r"R\d+|RE[ _-]?\d+", line, re.I) else (),
                source_metadata={"source_kind": self.source_kind, "source_url": SOURCE_URL,
                                 "feed_sha256": self.source_sha256, "coverage": self.coverage,
                                 "realtime": False, "attribution": "Regione Lombardia / Trenord (CC BY 4.0)"}))
        return results


def resolve_result_reference(results: list[TrainSearchResult], *, ordinal: int | None = None,
                             departure_time: str | None = None, direct_only: bool = False,
                             cheapest: bool = False) -> TrainSearchResult:
    """Resolve a later read-only reference against identified session results."""
    if cheapest:
        raise UnsupportedConstraint("price_unavailable_for_result_reference")
    matches = list(results)
    if direct_only:
        matches = [item for item in matches if item.number_of_changes == 0]
    if departure_time:
        minute = _minutes(departure_time)
        matches = [item for item in matches if _minutes(item.departure_datetime[11:16]) == minute]
    if ordinal is not None:
        if ordinal < 1 or ordinal > len(matches):
            raise UnresolvedStation("result_ordinal_out_of_range")
        return matches[ordinal - 1]
    if len(matches) != 1:
        raise UnresolvedStation("result_reference_ambiguous")
    return matches[0]
