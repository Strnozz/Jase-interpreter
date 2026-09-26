# Experimental routing Guard, offline audit

`jase/guard_v1_2_routing.py` adds a **separate** conservative handoff check after the unchanged Guard 1.2. It only returns `REVIEW` for known routing or field ambiguities; it never repairs output or authorizes a side effect. The frontend and old adapters are untouched. Thirteen direct regression tests pass (the local venv has no pytest).

The rules are based on action semantics: notification needs a reminder target and message; a cancelled visit needs appointment routing; vehicle rental and professional searches need matching target families; explicit comparison cardinality must be a `quantity` fact; a transport time bound must carry arrival/departure role. `medical_appointment` is an explicit appointment routing alias; the original output stays unchanged.

## Paired Guard results on saved V23 generations

| Panel | Base Guard ACCEPT | Routing Guard ACCEPT | Exact-wrong accepted, base → routing | New exact-correct holds |
|---|---:|---:|---:|---:|
| V24 safety, 20 | 15 | 11 | 8 → 4 | 0 |
| V23 transfer, 30 | 21 | 16 | 10 → 5 | 0 |
| V20 hard dev, 16 | 10 | 10 | 2 → 2 | 0 |
| V20 holdout, 12 | 8 | 7 | 3 → 2 | 0 |
| Historical convertible, 198 | 181 | 179 | 22 → 22 | 2 |

On the V24 safety panel, a **provisional AI review** labels five V23 outputs as material errors or routing risks. The base Guard accepted four; the routing Guard accepts **zero of five** and holds all five. Its coverage falls from 15/20 to 11/20. Four exact-correct contracts are still held by both Guards due to explicit missing information or confirmation; this is intentional handoff gating, not automatically a false positive. Four remaining accepted exact disagreements include two ambiguous `camera` golds, a model identifier represented as a separate fact, and an appointment subtype alias. They require independent review and Planner semantics before being scored as safe.

The transfer panel gains five holds of exact-wrong output: missing reminder message, ambiguous transport time, wrong visit routing, omitted comparison quantity, and wrong bicycle rental routing. The historical panel gains two **exact-correct holds** on `rent bici/bicicletta` typed `product`; its migrated gold uses `product`, while the proposed routing ontology uses `vehicle`. These are label/ontology conflicts, so the Guard rule must not be deployed until the older gold is reviewed in a new benchmark version. The historical file is unchanged.

Outputs and reproducible audit: `benchmarks/outputs/guard-routing-v23-safety/summary.json`, `benchmarks/outputs/guard-routing-v23-historical/*.json`, and `evaluation/evaluate_routing_guard.py`. `benchmarks/v24/v23_operational_review_draft.json` is a provisional review of saved V23 predictions, **not** official gold. No independent human adjudication or provider execution test has occurred.

## Contract and data decision

The V23 training corpus has **zero** `appointment` and `vehicle` target types among 4,411 train / 829 validation examples; frequent types are `place`, `accommodation`, `product`, `transport`, and `professional_service`. This is a concrete coverage gap that can explain routing failures. GoalContract 1.2 permits arbitrary `target.type` and `fact.field`, so a schema-valid output can still use a field the Planner cannot interpret. A GoalContract 1.3 proposal is in `docs/GOAL_CONTRACT_V13_ROUTING_PROPOSAL.md`.

Before another 9B training, create a genuinely separate test panel, document a target/field ontology and provider mappings, then add balanced, versioned examples for appointment/vehicle/reminder/identifier cases while replaying prior capabilities. Do not train on any V20/V23/V24 benchmark requests or model outputs. Do not promote this Guard or the current 9B to production on these small, repeatedly inspected panels.
