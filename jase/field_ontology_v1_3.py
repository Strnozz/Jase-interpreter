"""Experimental GoalContract 1.3 field vocabulary for Planner handoff.

This is a validation vocabulary, not a provider mapping or execution permit.
Aliases below are held for review because silently rewriting a field may alter
which provider parameter the Planner would receive.
"""
from __future__ import annotations

FACT_FIELDS = frozenset({
    "access", "amenity", "appointment_id", "author", "availability", "booking_id",
    "brand", "breed", "budget", "certification", "color", "condition",
    "cuisine", "delivery_location", "destination", "diet", "display",
    "distance", "duration", "feature", "format", "genre", "guests",
    "ingredient", "language", "location", "message", "mileage", "model",
    "open_now", "origin", "original_date", "original_time", "price",
    "quantity", "ram", "rating", "recipient", "room_type", "selection",
    "seller", "size", "specialization", "spiciness", "stars", "stops",
})

CONTEXT_FIELDS = frozenset({"party_size", "has_car", "lodging_owned"})
RESULT_FIELDS = frozenset({"results_count"})

# These aliases describe known model output, but are not automatic rewrites.
AMBIGUOUS_ALIASES = {
    "delivery": "delivery_location",
    "new_date": "temporal.action_date",
    "move_to": "temporal.action_date",
    "action_date": "temporal.action_date",
    "result_count": "condition.test.results_count",
}


def field_issue(field: str, *, scope: str = "fact") -> str | None:
    allowed = {"fact": FACT_FIELDS, "context": CONTEXT_FIELDS,
               "result": RESULT_FIELDS}[scope]
    if field in allowed:
        return None
    if field in AMBIGUOUS_ALIASES:
        return "alias_requires_planner_mapping"
    return "unknown_planner_field"
