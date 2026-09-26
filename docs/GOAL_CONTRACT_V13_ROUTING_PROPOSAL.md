# GoalContract 1.3 routing and field vocabulary proposal

Status: design proposal. Current training and benchmark outputs remain GoalContract 1.2; no frontend or Planner migration has been made.

## Problem shown by the frozen audits

GoalContract 1.2 checks JSON structure but accepts any `target.type` and `fact.field`. A notification with target `service` and no `message`, a visit cancellation typed `professional_service`, or an arrival deadline stored as generic `time` can validate while changing the requested operation. The V23 training corpus contains no `appointment` or `vehicle` target types, which is an actual data coverage hole. Some exact benchmark disagreements are merely representational: `medical_appointment` can route as `appointment`, and a named product model can be represented in a `model` fact if the Planner supports it. Two older bicycle rental golds use `product`, contradicting the proposed vehicle route.

## Proposed invariant

The Interpreter produces a *typed, grounded interpretation*. A versioned routing registry maps `action + target.type` to a capability family. Providers may add subtypes only through an explicit registry entry. Unknown subtypes are held for review. The original target and facts remain in the contract; normalization is a separate, auditable view with provenance. Neither a schema validator nor a model may silently turn `REVIEW` into permission to execute.

Candidate core families: `place`, `accommodation`, `real_estate`, `transport`, `vehicle`, `product`, `professional_service`, `person`, `appointment`, `reminder`, `food`, `ticket`, `media`, `education`, `animal`, `service`. Candidate aliases require a documented capability mapping, for example `medical_appointment → appointment`, `electronics → product`, and `flower → product` for ordering. `camera` alone is ambiguous between accommodation and real estate and should not be forced into either category without more context.

Critical fields should be enumerated by purpose: `arrival_time` and `departure_time` for transport bounds; `message` for contact/notification content; `quantity` as a goal fact for cardinality; `model` or a complete `target.name` for product identity. Generic `time` remains valid for appointments and reminders, but is insufficient for a bounded transport search unless a capability defines its role. Hard and soft constraints must never be normalized together. A model identifier in a separate `model` fact is potentially equivalent to one in `target.name`, but only after the Planner proves it joins them into the same provider query.

## Migration and evidence gate

1. Review ambiguous and conflicting gold with a person. Publish corrected cases as a **new** benchmark version, keeping old V20/V23/V24 files immutable and reporting both score series.
2. Specify the registry and any permitted aliases in a versioned machine-readable file. Add contract-to-capability tests for cancellation, rental, reminders, timing, identifiers, and confirmation. Unknown fields/types should produce `REVIEW`, not a guessed provider call.
3. Build a new dataset version with targeted, balanced coverage for absent families and legacy replay. Hold out newly authored requests **before** creating the dataset and audit exact/near overlap. Do not use benchmark outputs as examples.
4. Evaluate base 9B, old Guard, new Guard, and any trained 9B on identical frozen panels. Report exact success, operationally adjudicated wrong acceptance, coverage, and correct contracts held. A smaller wrong-acceptance count is only meaningful alongside coverage and human-reviewed semantics.

The experimental routing Guard implements a narrow subset of these invariants for offline measurement. It is not GoalContract 1.3 and is not production-ready.
