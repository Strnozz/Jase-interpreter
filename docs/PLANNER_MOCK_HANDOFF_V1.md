# Planner 1.3 mock handoff

`jase.planner_handoff_mock.prepare_mock_handoff` accepts a Planner dry-run result and creates typed **request drafts** only. It never calls a provider. Every result includes `provider_called: false` and `execution_permitted: false`.

The handoff is atomic: a missing or untrusted slot, registry version mismatch, unknown operator/type, unverified modifier, non-mock binding or write/destructive capability produces `HOLD` with zero drafts. `NO_ACTION` also produces zero drafts. Read-only drafts preserve capability ID, provenance source, policy, dependency and typed slot values. They do not contain a provider endpoint or authorization token.

Registry V3 is an explicit mock-only extension of V2 with a pinned parent hash. Exact target names take precedence over aliases; multiple exact matches still fail as ambiguous. The current draft adapter accepts only read-only `mock_catalog` and `mock_compare` interfaces. Sort and other modifiers remain in HOLD until field mapping and grounding are defined and tested. Context slots are now carried through the Planner binding rather than silently discarded.

`READY_FOR_MOCK_DRAFT` means the request can be inspected in a simulated interface. It is not evidence that a real provider supports the slots or operators. Before connecting any provider, the interface needs independent human review of gold, provider capability contracts, authentication/consent design, failure handling and end-to-end safety tests.
