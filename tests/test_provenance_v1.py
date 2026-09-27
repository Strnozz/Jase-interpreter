from __future__ import annotations

import unittest

from jase.provenance_v1 import (CAPABILITY_RESULT, DETERMINISTIC_DERIVATION,
                                MODEL_INFERENCE, PROVIDER_RESULT,
                                TRUSTED_CONVERSATION_CONTEXT, USER_EXPLICIT,
                                ground_value)


class ProvenanceTests(unittest.TestCase):
    def test_exact_user_span_only(self):
        text = "Trovami un alloggio a Perugia."
        evidence = ground_value("location", "Perugia", text)
        self.assertEqual(evidence.source, USER_EXPLICIT)
        self.assertEqual(text[slice(*evidence.span)], "Perugia")
        self.assertEqual(ground_value("location", "Como", text).source, MODEL_INFERENCE)

    def test_context_and_provider_need_field_value_and_trust(self):
        self.assertEqual(ground_value("location", "Terni", "Cerca un hotel", trusted_context={"location": "Terni"}).source,
                         TRUSTED_CONVERSATION_CONTEXT)
        self.assertEqual(ground_value("price", 50, "Cerca un hotel", provider_values=[
            {"field": "price", "value": 50, "trusted": True}]).source, PROVIDER_RESULT)
        self.assertEqual(ground_value("price", 50, "Cerca un hotel", capability_values=[
            {"field": "price", "value": 50, "trusted": True}]).source, CAPABILITY_RESULT)
        self.assertEqual(ground_value("price", 50, "Cerca un hotel", provider_values=[
            {"field": "price", "value": 50, "trusted": False}]).source, MODEL_INFERENCE)

    def test_only_explicit_deterministic_derivations(self):
        self.assertEqual(ground_value("stops", 0, "Voglio un volo senza scali").source,
                         DETERMINISTIC_DERIVATION)
        self.assertEqual(ground_value("stops", 1, "Voglio un volo senza scali").source,
                         MODEL_INFERENCE)
        self.assertEqual(ground_value("open_now", True, "Una farmacia aperta adesso").source,
                         DETERMINISTIC_DERIVATION)
        self.assertEqual(ground_value("limit", 3, "Mostrami tre risultati").source,
                         DETERMINISTIC_DERIVATION)


if __name__ == "__main__":
    unittest.main()
