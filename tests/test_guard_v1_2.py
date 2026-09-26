"""Safety regressions for the V1.2 handoff gate."""
from jase.guard_v1_2 import ACCEPT, REJECT, REVIEW, check_contract


def one(action: str, target: str, facts=None, **extra):
    return {"schema_version": "1.2", "kind": "task", "goals": [
        {"id": "g1", "action": action, "target": {"name": target, "type": "place"},
         "facts": facts or [], **extra}]}


def test_incomplete_booking_is_held_even_when_schema_is_valid():
    result = check_contract(one("book", "tavolo"), "Puoi prenotare un tavolo per noi?")
    assert result.status == REVIEW
    assert "table_booking_requires_details" in result.codes


def test_explicit_missing_is_still_not_permission_to_execute():
    contract = one("book", "tavolo", missing=[
        {"field": name, "blocks": "action", "reason": "unspecified"}
        for name in ("date", "time", "guests")])
    result = check_contract(contract, "Vorrei prenotare un tavolo; i dettagli li darò dopo.")
    assert result.status == REVIEW
    assert "declared_missing_input" in result.codes


def test_nonsense_price_is_rejected():
    contract = one("find", "trasporto", [{"field": "price", "op": "lt", "value": 0,
                                            "strength": "hard", "currency": "EUR"}])
    result = check_contract(contract, "Trova il modo meno caro per andare in città.")
    assert result.status == REJECT
    assert "nonpositive_price_ceiling" in result.codes


def test_schema_invalid_policy_is_rejected():
    contract = one("find", "hotel", policy={"forbid": ["book"], "reason": "per ora"})
    assert check_contract(contract, "Cerca un hotel, senza prenotare.").status == REJECT


def test_safe_read_only_search_can_reach_planner():
    contract = one("find", "farmacia", [{"field": "location", "op": "near",
                                         "value": "Udine", "strength": "hard"}])
    assert check_contract(contract, "Cerca una farmacia a Udine.").status == ACCEPT


def test_missing_user_prohibition_is_held():
    contract = one("find", "hotel")
    result = check_contract(contract, "Mostrami un hotel ma non prenotare nulla.")
    assert result.status == REVIEW
    assert "missing_explicit_booking_prohibition" in result.codes
