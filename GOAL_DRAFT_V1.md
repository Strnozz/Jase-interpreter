# GoalDraft V1

`GoalDraft` is mutable semantic intent, never an execution instruction. Its stable `draft_id` and increasing `revision` identify corrections. It contains candidate action/target, typed `fact.*`, `temporal.*` and modifier slots with source turn and provenance, constraints, unresolved references, candidate capability, missing/ambiguous slots, policy, confirmation binding, and status.

Statuses: `COLLECTING`, `AWAITING_USER`, `DISCOVERING`, `READY_TO_FINALIZE`, `FINALIZED`, `CANCELLED`, `SUPERSEDED`. A model's proposed status is advisory. Deterministic code computes actual status. Superseded values are retained as audit history and do not enter a finalized contract.

Capability requirements classify slots as required, conditionally required, optional, derivable, or discoverable. The existing Registry's `required_slots` and `optional_slots` are the source of truth. A versioned overlay can add conditional/discoverable semantics without changing historical Registry files. For a booking target, a selected result may come from a read-only search; user-specific party size must come from the user or trusted context. No invented location, time, date, party size, or result identity is allowed.

Only a finalized contract is passed to Planner. `FINALIZED` does not mean execution permission.
