"""Versioned, explicit capability ontology for dry-run Planner 1.3.

The registry describes theoretical interfaces. It does not connect providers or
grant permission to execute. Unknown names and ambiguous mappings fail closed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs/capabilities/v1.json"


@dataclass(frozen=True)
class Capability:
    capability_id: str
    version: str
    action: str
    target_type: str
    target_names: frozenset[str]
    aliases: frozenset[str]
    required_slots: frozenset[str]
    optional_slots: frozenset[str]
    supported_modifiers: frozenset[str]
    supported_temporal_roles: frozenset[str]
    supported_reference_types: frozenset[str]
    required_policies: frozenset[str]
    provider_binding: str
    consequentiality: str
    confirmation_required: bool
    execution_status: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Capability":
        required = {"capability_id", "version", "action", "target_type", "target_names", "aliases",
                    "required_slots", "optional_slots", "supported_modifiers", "supported_temporal_roles",
                    "supported_reference_types", "required_policies", "provider_binding",
                    "consequentiality", "confirmation_required", "execution_status"}
        if set(raw) != required:
            raise ValueError(f"Capability fields mismatch: {sorted(set(raw) ^ required)}")
        if not raw["capability_id"] or not raw["target_names"]:
            raise ValueError("Capability ID and target names are required")
        if "*" in raw["target_names"] and (raw["target_type"] != "person" or raw["action"] != "contact"):
            raise ValueError("Wildcard names are limited to person contact, which still requires binding")
        if raw["execution_status"] != "mock_only":
            raise ValueError("V1 capabilities must remain mock_only")
        if raw["consequentiality"] not in {"read", "write", "destructive"}:
            raise ValueError("Unknown consequentiality")
        for key in ("target_names", "aliases", "required_slots", "optional_slots", "supported_modifiers",
                    "supported_temporal_roles", "supported_reference_types", "required_policies"):
            if not isinstance(raw[key], list) or any(not isinstance(x, str) or not x for x in raw[key]):
                raise ValueError(f"Invalid {key}")
        if set(raw["required_slots"]) & set(raw["optional_slots"]):
            raise ValueError("A slot cannot be both required and optional")
        if raw["confirmation_required"] and raw["consequentiality"] == "read":
            raise ValueError("Read-only capability cannot require execution confirmation")
        data = raw.copy()
        for key in ("target_names", "aliases", "required_slots", "optional_slots", "supported_modifiers",
                    "supported_temporal_roles", "supported_reference_types", "required_policies"):
            data[key] = frozenset(x.casefold() if key in {"target_names", "aliases"} else x for x in raw[key])
        return cls(**data)

    def matches(self, action: str, target_type: str, target_name: str) -> bool:
        return (self.action == action and self.target_type == target_type and
                ("*" in self.target_names or
                 target_name.casefold().strip() in self.target_names | self.aliases))

    def as_dict(self) -> dict[str, Any]:
        return {key: sorted(value) if isinstance(value, frozenset) else value
                for key, value in vars(self).items()}


class CapabilityRegistry:
    def __init__(self, version: str, capabilities: tuple[Capability, ...]):
        if version != "1":
            raise ValueError("Unsupported capability registry version")
        ids = [item.capability_id for item in capabilities]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate capability ID")
        self.version = version
        self.capabilities = capabilities

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "CapabilityRegistry":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if set(raw) != {"version", "capabilities"}:
            raise ValueError("Invalid registry envelope")
        return cls(raw["version"], tuple(Capability.from_dict(x) for x in raw["capabilities"]))

    def resolve(self, goal: dict[str, Any]) -> tuple[str, Capability | None]:
        matches = [spec for spec in self.capabilities if spec.matches(goal["action"],
                   goal["target"]["type"], goal["target"]["name"])]
        if not matches:
            return "HOLD_NO_CAPABILITY", None
        if len(matches) != 1:
            return "HOLD_AMBIGUOUS_CAPABILITY", None
        return "RESOLVED", matches[0]
