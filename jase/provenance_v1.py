"""Conservative field/value provenance for dry-run planning.

Evidence is not inferred from model confidence. A source span is emitted only
when the original request literally contains the supported value. Derived
booleans and number words have no fabricated span.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

USER_EXPLICIT = "USER_EXPLICIT"
TRUSTED_CONVERSATION_CONTEXT = "TRUSTED_CONVERSATION_CONTEXT"
PROVIDER_RESULT = "PROVIDER_RESULT"
CAPABILITY_RESULT = "CAPABILITY_RESULT"
DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"
MODEL_INFERENCE = "MODEL_INFERENCE"
UNKNOWN = "UNKNOWN"

_NUMBERS = {"uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
            "sei": 6, "sette": 7, "otto": 8, "nove": 9, "dieci": 10}
_OPEN_NOW = re.compile(r"\b(?:apert[oaie]|open)\b.{0,32}\b(?:ora|adesso|now)\b", re.I)
_DIRECT = re.compile(r"\b(?:dirett[oaie]|senza scali|nonstop|non-stop)\b", re.I)


@dataclass(frozen=True)
class Evidence:
    source: str
    field: str
    value: Any
    span: tuple[int, int] | None = None
    note: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"source": self.source, "field": self.field, "value": self.value,
                "span": list(self.span) if self.span else None, "note": self.note}


def _literal_span(value: Any, text: str) -> tuple[int, int] | None:
    if value is None or isinstance(value, bool):
        return None
    needle = str(value)
    if not needle:
        return None
    match = re.search(r"(?<!\w)" + re.escape(needle) + r"(?!\w)", text, re.I)
    return (match.start(), match.end()) if match else None


def ground_value(field: str, value: Any, user_text: str, *,
                 trusted_context: Mapping[str, Any] | None = None,
                 provider_values: Iterable[Mapping[str, Any]] = (),
                 capability_values: Iterable[Mapping[str, Any]] = ()) -> Evidence:
    """Classify a model-emitted scalar without promoting unsupported claims.

    Trusted and provider evidence require an explicit field AND value match.
    Merely seeing a similar noun or an unverified provider name is insufficient.
    """
    if value is None:
        return Evidence(UNKNOWN, field, value)
    span = _literal_span(value, user_text)
    if span:
        return Evidence(USER_EXPLICIT, field, value, span)
    if trusted_context and trusted_context.get(field) == value:
        return Evidence(TRUSTED_CONVERSATION_CONTEXT, field, value)
    for item in provider_values:
        if item.get("field") == field and item.get("value") == value and item.get("trusted") is True:
            return Evidence(PROVIDER_RESULT, field, value)
    for item in capability_values:
        if item.get("field") == field and item.get("value") == value and item.get("trusted") is True:
            return Evidence(CAPABILITY_RESULT, field, value)
    if field == "open_now" and value is True and _OPEN_NOW.search(user_text):
        return Evidence(DETERMINISTIC_DERIVATION, field, value, note="explicit_open_now_phrase")
    if _DIRECT.search(user_text) and ((field == "stops" and type(value) is int and value == 0) or
                                      (field == "direct" and value is True)):
        return Evidence(DETERMINISTIC_DERIVATION, field, value, note="explicit_direct_route_phrase")
    if isinstance(value, int) and not isinstance(value, bool):
        for word, number in _NUMBERS.items():
            if number == value and re.search(r"(?<!\w)" + word + r"(?!\w)", user_text, re.I):
                return Evidence(DETERMINISTIC_DERIVATION, field, value, note=f"number_word:{word}")
    return Evidence(MODEL_INFERENCE, field, value)
