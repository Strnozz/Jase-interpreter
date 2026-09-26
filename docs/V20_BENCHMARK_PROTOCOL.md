# GoalContract 1.2 benchmark protocol

`benchmarks/v20/hard_dev.jsonl` has 16 authored diagnostic cases and can guide schema/evaluator development. `benchmarks/v20/holdout.jsonl` has 12 independently authored cases. Its hash and size are frozen in `benchmarks/v20/manifest.json`; do not add it to prompts, train or validation, select checkpoints on it, or inspect per-case model predictions after its first release evaluation. `evaluation/evaluate_v20.py` emits aggregate-only results for filenames containing `holdout`.

The gold panels are validated by `jase.goal_contract_v1_2.validate_contract`. The scorer `jase.metrics_v1_2` ignores JSON key order, independent fact order, empty optional arrays, case of string values, and local goal ID renaming when it is unambiguous. It preserves hard-versus-soft distinctions, operators, numeric strictness, dependency/reference structure, confirmation directives and missing-input phase. It does not resolve relative dates without a reference clock or normalize away semantic contradictions.

## Metrics and denominators

- **All requests:** JSON validity, schema validity, full contract success, schema-valid-but-wrong count. Invalid JSON/schema is a failed complete interpretation, never dropped from that denominator.
- **Schema-valid requests:** goal count, action, target, facts precision/recall, hard/soft fact precision/recall, context, missing input, dependencies, result references, explicit policy and condition accuracy. Fact rates are explicitly conditional and must be reported with the number of evaluable cases.
- **Not measurable yet:** concrete capability selection (no registry), executable plan correctness (Planner-owned), runtime confirmation correctness (no policy engine), accepted-but-wrong *after a V1.2 Guard* (no deployed Guard). `schema_valid_but_wrong` is an approximation of semantic risk, not proof that Jase would execute the contract.
- **Attribute accuracy:** no separate V1.2 attribute bucket exists; hard fact accuracy replaces it. This is a deliberate representation change, not a missing column accidentally set to zero.

For a future same-hardware 9B V19/V20 comparison, use the same frozen user requests, decoder settings, final Guard policy and independent semantic gold. Compare shared action/target/fact meaning and all-case success through a **version-neutral projection**, then report V1.2-only abilities (context, missing phase, explicit policy, dependencies/references) separately. Do not compare training losses across tokenizations or claim V1.2 wins merely because its own schema is easier to satisfy. A new V1.2 model must also be compared with the preserved V19 9B adapter on the old dev panel to detect regressions in established tasks. None of these model runs occur in this preparation phase.

## Known panel limitations

The diagnostic and holdout suites are small, hand-authored and do not yet cover every requested class, notably contradictory or impossible requests with a valid blocked contract, provider-specific unsupported actions and previous-result IDs backed by real conversation state. They are a scaffold for benchmark design, **not** sufficient release gates. The migration is highly imbalanced toward `find`; train and validation contain no non-actionable, dependency, confirmation or clarification labels. The pretraining report therefore remains `NOT_READY`.
