# Clarification protocol V1

Stage A JSON contains `protocol_version`, `mode`, `operation`, `draft_id`, `action`, `target`, `updates`, and `reference`. `mode` distinguishes chat, question, meta, translation, hypothetical, actionable goal, continuation, clarification answer, confirmation, rejection, and cancellation. `operation` is one of `NO_GOAL`, `OPEN_DRAFT`, `UPDATE_DRAFT`, `CANCEL_DRAFT`, `CONFIRM_DRAFT`, or `REJECT_DRAFT`. The model emits observed semantics, not an authoritative `missing` list.

The runtime returns a `ClarificationPlan` with `ask_user`, `discover_read_only`, `derive`, `ambiguous`, `blocked`, and `reason_codes`. Required slots come from the capability Registry. Optional fields are not asked merely because they are absent. Compatible missing fields can be combined into one question. Discovery takes precedence over asking for a selected result when a read-only search can supply candidates.

Stage B receives the frozen plan and may produce a brief Italian question, information message, result presentation, or confirmation request. Its wording is not parsed as semantic state. A deterministic fallback renderer is available if Stage B produces an invalid or irrelevant answer. Planner HOLD reasons are classified into recoverable by user, discovery, provider, or non-recoverable; no HOLD is silently converted to READY.
