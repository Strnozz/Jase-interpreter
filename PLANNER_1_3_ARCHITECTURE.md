# Planner 1.3 — isolated dry-run architecture

The new pipeline is opt-in and does not touch the frontend or provider code:

`frozen Interpreter output → Guard V5 → Capability Registry V1 → slot/reference/provenance/temporal/policy/dependency checks → dry-run plan`

`jase/planner_v1_3.py` accepts the **raw** GoalContract, original user request, a versioned mock capability registry, and optionally trusted context, trusted mock result sets, external slot bindings and explicit confirmation state. It never calls a provider. Every return path sets `execution_permitted: false`; `READY_FOR_DRY_RUN` means only that a plan can be inspected. The final execution boundary is absent on purpose.

The Planner returns a versioned object with `planner_status`, `reason_codes`, resolved capabilities, per-goal bindings, resolved and unresolved references, missing slots, provenance issues, semantic conflicts, policy state, dependencies, dry-run steps and blocked steps. The registry, not the model's `missing` list, determines required slots. A model-declared missing value also blocks planning. Unknown action/target names, aliases, facts, modifiers and conditional result fields hold without rewriting.

GoalContract facts are checked against the request and explicit trusted sources by `jase/provenance_v1.py`. Values seen only in model output remain `MODEL_INFERENCE`; no source span is fabricated. `external.*` slots (for example provider reference or hotel dates) are accepted only from trusted Planner bindings and are never silently copied from a public model fact. Ordinal references require a trusted result set with enough entries. `cheapest` requires known prices for every result and one unique minimum. References to a future search result hold until its mock result set exists.

Temporal checks distinguish roles and operators. Arrival deadlines cannot silently become equality at the stated time. A date contained in a contact message cannot become that contact action's date. Quoted/meta and explanatory hypothetical language is held as non-actionable when the model emits an action. These are deliberately conservative checks; they do not claim complete natural-language understanding.

Consequential capabilities require a declaration of confirmation policy and a separate trusted confirmation state. Explicit user prohibitions are checked within relevant target families. Even if all requirements are met, the result remains a **dry run** with execution forbidden. Independent goals can be inspected separately; blocked dependencies block downstream steps.

This is not production-ready. Current gaps include the small mock capability vocabulary, no real provider binding, no authenticated conversation state, no independent human review of V33 gold and no provider-side execution gate. Every false ACCEPT and false HOLD from replay must be reported rather than repaired silently.
