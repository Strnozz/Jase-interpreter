"""General routing checks, using fresh examples outside frozen panels."""
from jase.guard_v1_2 import ACCEPT, REVIEW
from jase.guard_v1_2_routing import check_contract


def candidate(action, name, typ, facts=None, **extra):
    return {"schema_version": "1.2", "kind": "task", "goals": [{
        "id": "g1", "action": action, "target": {"name": name, "type": typ},
        "facts": facts or [], **extra}]}


def fact(field, value):
    return {"field": field, "op": "eq", "value": value, "strength": "hard"}


def test_reminder_requires_message_and_reminder_route():
    bad = candidate("notify", "deposito", "service", [fact("date", "giovedì")])
    result = check_contract(bad, "Giovedì ricordami di ritirare il casco.")
    assert result.status == REVIEW
    assert {"notify_target_not_reminder", "notify_message_unresolved"} <= set(result.codes)


def test_reminder_with_explicit_missing_reference_remains_held_by_base_guard():
    unresolved = candidate("notify", "promemoria", "reminder", [fact("date", "giovedì")],
        missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}])
    assert check_contract(unresolved, "Ricordami quella cosa giovedì.").status == REVIEW


def test_appointment_route_holds_service_type():
    wrong = candidate("cancel", "visita fisiatrica", "professional_service", [fact("date", "giovedì")])
    assert "appointment_routing_mismatch" in check_contract(wrong, "Disdici la visita fisiatrica giovedì.").codes
    good = candidate("cancel", "visita fisiatrica", "medical_appointment", [fact("date", "giovedì")])
    assert check_contract(good, "Disdici la visita fisiatrica giovedì.").status == ACCEPT


def test_vehicle_and_professional_routing_are_distinct():
    scooter = candidate("rent", "moto da turismo", "product")
    assert "vehicle_routing_mismatch" in check_contract(scooter, "Noleggia una moto da turismo.").codes
    doctor = candidate("find", "medico", "place")
    assert "professional_routing_mismatch" in check_contract(doctor, "Cerca un medico.").codes
    pharmacy = candidate("find", "farmacia", "place")
    assert check_contract(pharmacy, "Cerca una farmacia.").status == ACCEPT


def test_transport_temporal_role_must_be_explicit():
    wrong = candidate("find", "autobus", "transport", [fact("time", "14:20")])
    result = check_contract(wrong, "Un autobus che arrivi alle 14:20.")
    assert "arrival_time_field_missing" in result.codes
    right = candidate("find", "autobus", "transport", [fact("arrival_time", "14:20")])
    assert check_contract(right, "Un autobus che arrivi alle 14:20.").status == ACCEPT
    ambiguous = candidate("find", "autobus", "transport", [{
        "field": "time", "op": "before", "value": "14:20", "strength": "hard"}])
    assert "transport_time_role_ambiguous" in check_contract(ambiguous, "Autobus prima delle 14:20.").codes


def test_no_silent_rewrite_of_valid_contract():
    good = candidate("rent", "furgone", "vehicle", [fact("date", "sabato")])
    result = check_contract(good, "Noleggia un furgone sabato.")
    assert result.status == ACCEPT
    assert result.contract == good


def test_explicit_comparison_count_must_be_retained():
    bad = candidate("compare", "fotocamera", "product")
    result = check_contract(bad, "Confronta tre fotocamere per me.")
    assert "comparison_quantity_missing" in result.codes
    good = candidate("compare", "fotocamera", "product", [fact("quantity", 3)])
    assert check_contract(good, "Confronta tre fotocamere per me.").status == ACCEPT
