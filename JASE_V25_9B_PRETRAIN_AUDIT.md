# Jase Interpreter 9B V25: pretraining audit

## Decision and architecture

The V24 result was structurally sound but had semantic errors: conflicting dates,
misbound confirmation, action/sort confusion, lost comparison count, and context
used as a filter. V25 changes the representation before additional QLoRA.

GoalContract 1.3 adds explicit temporal roles, result modifiers (`sort`, `limit`,
`compare_count`), scoped context, and `reschedule`. It retains goals, dependencies,
conditions, policies, and missing information. The separate semantic validator
checks hard conflicts, numeric intervals and units, policy binding, target/action
compatibility, required fields, references, and typed time roles. The experimental
Guard consumes these results and returns explicit holds. `ACCEPT` permits Planner
inspection only; no provider action or frontend integration is implied.

Canonical comparison only equates documented safe differences: case/whitespace,
clock zero padding, and `medical_appointment` with `appointment`. It does not
equate target names with model facts, `eq` with `contains`, or different time
roles. V1.2-to-V1.3 migration is conservative: 369/5,005 V24 train rows and
36/905 valid rows are held, chiefly for incomplete old gold. These rows were
excluded from replay, not silently repaired.

The offline V25 audit of the already known V24 predictions changed the
experimental Guard from 3/6 to 0/6 *provisionally labelled* material/routing
risks accepted. Coverage decreased to 17/24; one exact case with an unresolved
reference was held. This is post hoc evidence only. The V25 ambiguous-case
review remains a draft for human adjudication in `docs/V25_AMBIGUOUS_REVIEW.md`.

## Freeze and data

The V26 panel was committed **before** constructing V25 data, in commit
`b17a4d4`. It has 120 Italian requests, six variations in each of 20 semantic
families. SHA-256:

`47e9083090684f3d998597e44bee0a288cfd8950f36e332fb51fee3c49a22389`

Its maximum token-set Jaccard similarity to prior train/validation and panels
is 0.58824; no exact overlap. V26 is targeted and AI-authored, not a random
production sample. Its gold still needs independent human review. Historical
V25 gold and outputs remain frozen.

V25 uses 1,870 train and 276 validation rows. The train mix is 1,370 new
contrastive examples and 500 conservatively projected V24 replay examples
(26.7% replay); validation is 206 new and 70 replay. Train and validation are
disjoint. The generator rejected exact/near benchmark overlaps and 16 near
train/validation candidates. It detected no conflicting labels for duplicate
requests. Dataset hashes and the V26 panel hash are in `data/v25_9b/manifest.json`.

Pinned tokenizer audit: Qwen/Qwen3.5-9B revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`; maximum total length 497
train and 495 validation, below 512. Completion-only labels were verified on
every row. The generated examples are template-derived; validation loss alone
will not establish semantic generalization.

## Run configuration and gates

V25 warm-starts from the preserved V24 adapter. NF4 4-bit, double quantization,
BF16, LoRA rank 16/alpha 32/dropout 0.05 on the last 16 text layers, per-device
batch 1, accumulation 4, learning rate 8e-6, weight decay 0.01, max norm 1,
and length 512 match V24. The 600-step cap covers approximately 1.28 passes
over 1,870 train rows, less compute than V24. Validation/checkpoints occur
every 150 steps, with early checkpoints at 25 and 75. Smoke and adapter reload
precede the full run. Free disk was about 103 GB at audit time.

The restartable pipeline first evaluates V24 on V26, then trains V25 and
evaluates both models on identical V26 gold. It also runs V25 on the frozen
24-case V25 panel as a diagnostic. Benchmarks report native JSON, conservative
V1.2 projection holds, canonical exact, action/target/fact/temporal scores,
Guard coverage, accepted mismatches for triage, throughput, and VRAM. Accepted
exact mismatches are **not** automatically labelled material safety failures;
case-level review is required.

No frontend integration or promotion is part of this run. `reschedule` is
representable for interpretation, but the experimental Guard holds it until a
provider capability is verified. The existing 2B and V19–V24 adapters are
preserved.

## Verification

All 13 new V1.3 tests and nine V1.2-specific tests passed. Full Windows test
discovery ran 39 tests and encountered one existing Linux-only import
(`fcntl` in `scripts/run_v20_experiment.py`); it is unrelated to V25. The
training/evaluation modules compile. The corpus/tokenizer audit passed.
