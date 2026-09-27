# Jase Interpreter 9B: V28 independent routing benchmark

V28 was frozen in commit `228fc80` **before** the new Guard/repair code in
`40b30dd`. It contains 42 individually written requests across 12 families.
Panel SHA-256:
`0f0a08f2ab7829c49ab29aff5901e43a6915f4660ef24206e4f17ef384ac2c01`.
There is no exact overlap or Jaccard ≥0.8 overlap with the recorded V25
train/validation and earlier panels (maximum 0.6364). The panel and this review
are agent-authored; independent human gold review remains necessary.

| V28 measure | V24 9B final | V25 9B final |
| --- | ---: | ---: |
| Valid JSON | 41/42 | **42/42** |
| Canonical exact, V24 projected 1.2→1.3 | **15/42** | 13/42 |
| Paired gains / losses for V25 | — | 4 / 6 |
| Raw action before schema projection | 41/46 | **45/46** |
| Raw target type | 43/46 | **45/46** |
| Raw goal count | 40/42 | 40/42 |
| Candidate schema/projection holds | **10/42** | 18/42 |
| Projected fact recall | **41/69** | 26/69 |
| Projected temporal recall | **19/28** | 9/28 |
| Output tokens/s | 18.12 | 18.11 |
| Peak inference VRAM | 8.19 GB | 8.18 GB |

V25's action and target selection improves, but the complete structured result
**regresses** on this more varied panel. V24's projection can hold legitimate
V1.2 expressions, so the exact and projected field metrics do not isolate
adapter quality. Raw action/target counts are the fairer direct comparison.
The V25 result of 13/42 versus 40/100 on V27 shows that the earlier
patterned result did not transfer to these differently phrased cases; neither
targeted panel estimates production prevalence.

The six paired V25 losses include all four train routes with origin and arrival
deadline. V24 gets 4/4; V25 gets 0/4. V25 repeatedly puts the origin in
`temporal` or top-level `context`, or invents another unsupported shape,
instead of `facts.origin`; it sometimes also mislabels the arrival deadline.
The separate conservative repair handles only a grounded `spatial.origin`
shape previously seen on V27. It repairs **0/42** V28 outputs, which shows that
repairing one error shape is not a general fix. V25 also loses a quoted contact
case by treating the time *inside the message* as action time and inventing
`today`, while omitting the requested confirmation. It gains one quoted
non-action case, two appointment cases and one vehicle rental.

V25 gets 0/4 exact in each of conditional fallback, direct flight,
find-then-book, arrival-route and open/access search; V24 also gets 0/4 in
the first three. These are operationally important omissions, not simply
target-name synonyms. V25's four quoted non-actions succeed here, but six
less-explicit V26 quoted examples all failed; wording sensitivity remains.

## Guard and material-risk review

On identical V25 raw predictions, the original Guard accepts 23/42, routing
V1 accepts 19/42, and frozen routing V2 accepts **15/42**. V2 holds four more
outputs with omitted no-book policy/result limit, restaurant booking details,
open-now status or walking access. It newly holds zero of the 13
canonical-exact predictions. Of its 15 ACCEPTs, 13 are canonical exact.
The other two were reviewed case by case:

- `v28-006` is a **clear material error**: confirmation is missing and the
  model turns message content (“alle 16”) into an invented action time/date.
- `v28-040` expresses “ristorante thailandese” as target `ristorante` plus
  a `cuisine=thailandese` fact. This may be equivalent, but Planner mapping
  is unverified; it is not counted as a clear material error.

The reviewed accepted-material lower bound is **1/15** at 35.7% coverage.
Zero newly held exact contracts is an exact-metric false-hold check, not a
proof that every non-exact HOLD is necessary. See the provisional review in
`benchmarks/v28/v25_operational_review_draft.json`. V2 is opt-in and cannot
authorize provider execution; frontend and previous adapters were untouched.

## Decision

Do not promote V25 or the Guard. Before another QLoRA, freeze a separate V29
panel and audit a substantially broader, versioned training corpus. Focus on
schema-valid routing of origin, destination and temporal roles; quoted message
content versus action time; negative policy; result cardinality; and
find-then-act dependencies. Preserve replay of existing capabilities to avoid
another paired regression. Use smoke, resume-capable checkpoints and the same
V29 cases for baseline and candidate. Do not train on benchmark requests or
outputs. Accuracy and accepted material risk, not validation loss alone,
must determine whether the next adapter helps.

Outputs: `benchmarks/outputs/v24-9b-v28-blind/` and
`benchmarks/outputs/v25-9b-v28-blind/`.
