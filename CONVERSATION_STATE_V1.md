# ConversationState V1

The state is independent of frontend rendering. It stores a version, conversation ID, monotonic turn index, active draft IDs, drafts with revision and provenance, current clarification request, result references (provider and session identity), last discourse mode, and cancelled/superseded draft IDs. Multiple goals have separate draft IDs and cannot be merged implicitly.

Raw turn text is referenced by turn ID and can be retained in a separate audit log; Stage A receives only the latest turn and relevant structured state. Contextual replies such as “quattro”, “da Milano”, “il secondo”, and “sì” are attached only when the state contains a compatible pending slot, result selection, or confirmation. An unbound “sì” cannot authorize an action.

Confirmation states are `NOT_REQUIRED`, `NOT_REQUESTED`, `PENDING`, `GRANTED`, `DENIED`, `REVOKED`, `SUPERSEDED`. A grant binds a digest of the exact goal revision and material fields. A correction, changed provider result, price, or scope invalidates it. All consequential provider execution remains disabled.
