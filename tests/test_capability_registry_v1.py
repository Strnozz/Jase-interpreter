from __future__ import annotations

import unittest

from jase.capability_registry_v1 import Capability, CapabilityRegistry


def goal(action: str, typ: str, name: str) -> dict:
    return {"action": action, "target": {"type": typ, "name": name}}


class CapabilityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = CapabilityRegistry.load()

    def test_versioned_mock_registry_resolves_only_explicit_names(self):
        self.assertGreaterEqual(len(self.registry.capabilities), 16)
        self.assertTrue(all(c.execution_status == "mock_only" for c in self.registry.capabilities))
        status, cap = self.registry.resolve(goal("find", "transport", "treno regionale"))
        self.assertEqual(status, "RESOLVED")
        self.assertEqual(cap.capability_id, "find.train")
        self.assertEqual(self.registry.resolve(goal("find", "transport", "rapidissimo"))[0],
                         "HOLD_NO_CAPABILITY")
        self.assertEqual(self.registry.resolve(goal("order", "transport", "treno"))[0],
                         "HOLD_NO_CAPABILITY")

    def test_type_is_not_silently_reinterpreted(self):
        self.assertEqual(self.registry.resolve(goal("find", "product", "ristorante"))[0],
                         "HOLD_NO_CAPABILITY")
        self.assertEqual(self.registry.resolve(goal("book", "accommodation", "hotel"))[0], "RESOLVED")
        self.assertEqual(self.registry.resolve(goal("find", "accommodation", "hotel"))[0], "RESOLVED")

    def test_generic_person_contact_still_requires_provider_binding(self):
        status, cap = self.registry.resolve(goal("contact", "person", "Marco"))
        self.assertEqual(status, "RESOLVED")
        self.assertIn("external.provider_ref", cap.required_slots)
        self.assertTrue(cap.confirmation_required)

    def test_ambiguous_registry_fails_closed(self):
        first = self.registry.capabilities[0]
        registry = CapabilityRegistry("1", (first, Capability.from_dict({**first.as_dict(),
            "capability_id": "conflicting.restaurant"})))
        self.assertEqual(registry.resolve(goal("find", "place", "ristorante"))[0],
                         "HOLD_AMBIGUOUS_CAPABILITY")

    def test_reject_non_mock_and_unbounded_wildcard(self):
        base = self.registry.capabilities[0].as_dict()
        with self.assertRaises(ValueError):
            Capability.from_dict({**base, "execution_status": "enabled"})
        with self.assertRaises(ValueError):
            Capability.from_dict({**base, "target_names": ["*"]})


if __name__ == "__main__":
    unittest.main()
