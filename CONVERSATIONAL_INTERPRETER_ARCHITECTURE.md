# Conversational Interpreter V1 (V27)

The new path is opt-in. The historical one-shot GoalContract 1.3 path and all V26 artifacts remain unchanged.

1. Stage A receives the latest user message plus a compact, versioned `ConversationState`. Qwen returns a typed *semantic proposal*: discourse mode, an operation on one draft, and user-grounded slot updates. It does not decide what a capability requires.
2. Deterministic code validates the proposal, applies corrections/cancellations, preserves provenance, resolves a candidate capability against the versioned Registry, and computes required, optional, and discoverable information. Unknown values are never filled by defaults.
3. The requirement result becomes a `ClarificationPlan`. A renderer may phrase its requested fields naturally; the field list and priority are deterministic. A second prompt to the same model may be used for wording, but its text cannot change the plan.
4. A draft can be finalized only by deterministic checks: actionable discourse, unique capability, grounded required slots, no unresolved blocking references/conflicts, and policy/confirmation represented. Finalization creates a new GoalContract 1.3 version; Planner 1.3 remains the final fail-closed authority.
5. A read-only prerequisite can discover a selected result. Consequential actions stay disabled. Confirmation is bound to a content digest of the exact proposed action and is invalidated by any material change.

The frozen V37 panel tests state transitions and safety before V27-targeted data generation. V33–V36 remain historical regression diagnostics; their gold is not rewritten. V37 gold is agent-authored pending independent human review. No provider write is reachable from this architecture.
