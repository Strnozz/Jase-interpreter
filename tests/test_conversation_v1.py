from __future__ import annotations

import unittest
from datetime import datetime

from jase.conversation_v1 import (ConversationState, apply_decision, clarification_plan,
                                  finalize_draft, plan_finalized, planner_context, register_results,
                                  request_confirmation)


def d(mode, operation, **kwargs):
    return {"protocol_version": "conversation-v1", "mode": mode, "operation": operation,
            "updates": kwargs.pop("updates", []), **kwargs}


class ConversationTests(unittest.TestCase):
    def test_train_clarification_then_finalization(self):
        state = ConversationState()
        draft = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="find",
            target={"name": "treno", "type": "transport"}, updates=[
                {"slot": "fact.destination", "value": "Roma"},
                {"slot": "temporal.search_date", "value": "domani"}]),
            "Trovami un treno per Roma domani", timestamp=datetime(2026, 9, 28, 12))
        self.assertEqual(clarification_plan(state, draft.draft_id)["ask_user"], ["fact.origin"])
        with self.assertRaises(ValueError):
            finalize_draft(state, draft.draft_id)
        apply_decision(state, d("ANSWER_TO_CLARIFICATION", "UPDATE_DRAFT", draft_id=draft.draft_id,
            updates=[{"slot": "fact.origin", "value": "Milano"}]), "Da Milano.")
        contract = finalize_draft(state, draft.draft_id)
        self.assertEqual({f["field"]: f["value"] for f in contract["goals"][0]["facts"]},
                         {"destination": "Roma", "origin": "Milano"})
        self.assertEqual(contract["goals"][0]["temporal"][0]["value"], "2026-09-29")
        self.assertIs(plan_finalized(state, draft.draft_id, contract)["execution_permitted"], False)

    def test_nonaction_and_unbound_confirmation(self):
        state = ConversationState()
        self.assertIsNone(apply_decision(state, d("TRANSLATION", "NO_GOAL"),
                                         "Traduci 'trovami un treno per Roma'"))
        with self.assertRaises(ValueError):
            apply_decision(state, d("META", "OPEN_DRAFT", action="find",
                target={"name": "treno", "type": "transport"}), "Spiega cosa faresti")
        with self.assertRaises(ValueError):
            apply_decision(state, d("CONFIRMATION", "CONFIRM_DRAFT"), "Sì")

    def test_booking_discovers_selection_and_asks_for_user_inputs(self):
        state = ConversationState()
        draft = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="book",
            target={"name": "ristorante", "type": "place"}, updates=[
                {"slot": "temporal.action_date", "value": "domani"},
                {"slot": "temporal.action_time", "value": "21:00"}]),
            "Prenotami un ristorante domani alle 21:00", timestamp=datetime(2026, 9, 28, 12))
        plan = clarification_plan(state, draft.draft_id)
        self.assertIn("external.party_size", plan["ask_user"])
        self.assertIn("fact.location", plan["ask_user"])
        self.assertEqual(plan["discover_read_only"][0]["capability_id"], "find.restaurant")
        self.assertNotIn("fact.selection", plan["ask_user"])

    def test_correction_and_cancellation_are_stateful(self):
        state = ConversationState()
        first = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="book",
            target={"name": "ristorante", "type": "place"}, updates=[
                {"slot": "fact.location", "value": "Milano"}]), "Prenota un ristorante a Milano")
        request_confirmation(first)
        apply_decision(state, d("CONTINUATION", "UPDATE_DRAFT", draft_id=first.draft_id,
            updates=[{"slot": "fact.location", "value": "Roma"}]), "Anzi, Roma invece di Milano.")
        self.assertEqual(first.superseded_slots[0]["value"], "Milano")
        self.assertEqual(first.confirmation["status"], "SUPERSEDED")
        second = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="find",
            target={"name": "treno", "type": "transport"}), "E cerca un treno")
        with self.assertRaises(ValueError):
            apply_decision(state, d("CONTINUATION", "UPDATE_DRAFT", updates=[
                {"slot": "fact.origin", "value": "Torino"}]), "Da Torino")
        apply_decision(state, d("CANCELLATION", "CANCEL_DRAFT", draft_id=first.draft_id),
                       "Lascia perdere il ristorante")
        self.assertEqual(first.status, "CANCELLED")
        self.assertIn(second.draft_id, state.active_draft_ids)

    def test_ungrounded_values_never_satisfy_requirements(self):
        state = ConversationState()
        draft = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="find",
            target={"name": "treno", "type": "transport"}, updates=[
                {"slot": "fact.origin", "value": "Bari"}]), "Trova un treno")
        self.assertEqual(draft.slots["fact.origin"]["provenance"], "MODEL_INFERENCE")
        self.assertIn("fact.origin", clarification_plan(state, draft.draft_id)["ambiguous"])

    def test_external_slot_provenance_and_no_execution(self):
        state = ConversationState()
        draft = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="book",
            target={"name": "ristorante", "type": "place"}, updates=[
                {"slot": "external.party_size", "value": 4}]), "Prenota un ristorante per quattro")
        bound = planner_context(state, draft.draft_id)
        self.assertEqual(bound["external_bindings_by_goal"]["g1"]["external.party_size"]["value"], 4)
        self.assertFalse(bound["confirmation_by_goal"]["g1"])

    def test_result_ordinal_is_bound_only_to_registered_trusted_result(self):
        state = ConversationState()
        draft = apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="book",
            target={"name": "ristorante", "type": "place"}), "Prenota un ristorante")
        apply_decision(state, d("CONTINUATION", "UPDATE_DRAFT", draft_id=draft.draft_id,
            reference={"kind": "ordinal", "ordinal": 2}), "Il secondo")
        self.assertNotIn("fact.selection", draft.slots)
        register_results(state, draft.draft_id, [
            {"jase_result_ref": "session-a-1", "provider_result_ref": "p1", "provider": "test", "trusted": True},
            {"jase_result_ref": "session-a-2", "provider_result_ref": "p2", "provider": "test", "trusted": True}])
        apply_decision(state, d("CONTINUATION", "UPDATE_DRAFT", draft_id=draft.draft_id,
            reference={"kind": "ordinal", "ordinal": 2}), "Il secondo")
        self.assertEqual(draft.slots["fact.selection"]["value"], "session-a-2")
        self.assertEqual(draft.slots["fact.selection"]["provenance"], "PROVIDER_RESULT")

    def test_one_utterance_can_open_distinct_goals_without_merging(self):
        state = ConversationState()
        apply_decision(state, d("ACTIONABLE_GOAL", "OPEN_DRAFT", action="find",
            target={"name": "hotel", "type": "accommodation"},
            updates=[{"slot": "fact.location", "value": "Trento"}],
            additional_goals=[{"action": "find", "target": {"name": "treno", "type": "transport"},
                               "updates": [{"slot": "fact.destination", "value": "Trento"}]}]),
            "Cerca un hotel a Trento e un treno per Trento")
        self.assertEqual(state.turn_index, 1)
        self.assertEqual(state.active_draft_ids, ["d1", "d2"])
        self.assertNotIn("fact.destination", state.drafts["d1"].slots)
        self.assertNotIn("fact.location", state.drafts["d2"].slots)


if __name__ == "__main__":
    unittest.main()
