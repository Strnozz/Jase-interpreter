# V19 → V20 / GoalContract 1.2 migration

This is a **candidate** dataset, not authorization to train. `data/v19_clean` and the completed 9B adapter/checkpoints are unchanged. `data/v20_quality` is an earlier V1.1 experiment and is **not** this GoalContract 1.2 dataset.

## Deterministic conversion and review rule

`scripts/migrate_v19_to_v20.py` reads the original 27,504 train and 2,310 validation rows. It auto-converts only a V1.1 contract with one `find` goal, no gate, no clarification, no free-text reference, a known target type, V1.1 Guard `ACCEPT`, and V1.2 structural/semantic validity. V1.1 `attributes` and `constraints` become `facts` with `strength=hard`; `preferences` become `strength=soft`. The original operator, value, unit and currency are preserved. Only a local `g1` ID is introduced. No missing inputs, execution policy, capability, dependency, context or provider is inferred.

Everything else goes to `data/v20/review.jsonl` with a reason. Invalid/unparseable source rows would go to `rejected.jsonl`; none occurred. The review file has source IDs, split and user text; the original contract remains in V19 for annotation. Review reasons overlap, so their counts cannot be summed.

| Outcome | Rows |
|---|---:|
| Source train + validation | 29,814 |
| Auto train | 12,022 |
| Auto validation | 803 |
| Requires review | 16,989 |
| Rejected/unparseable | 0 |
| New reviewed train examples | 0 |

Major review reasons: non-`find` actions 11,174; gates 2,371; duplicate normalized prompts 1,384; unknown target type 1,007; multi-goal 968; free-text references 833; clarification 662; numeric price without explicit currency 537; near train/validation overlap 305. Source/target SHA-256 hashes, full reason counts and exact conversion policy are in `data/v20/manifest.json`. The source hashes matched after writing outputs.

## Leakage and coverage

Normalized train/validation prompt overlap is zero. A token-set Jaccard audit first found **305** near duplicates between migrated train and validation (threshold 0.85, including word-order variants with Jaccard 1). These validation examples were quarantined for review, leaving **zero detected near duplicates** under the same audit. The heuristic is not proof of independence: V19 lacks source-family IDs, and synthetic templates may still cross splits. New hard-dev (16 cases) and sealed holdout (12 cases) have no normalized exact or detected near overlap with migrated train/validation. The holdout also has none against V19 train, validation, historical dev or historical release under this audit. IDs, counts and method are in `data/v20/audit.json`; holdout gold was never used to choose a model/checkpoint or edit migration labels.

The auto dataset is **not representative** of the desired Interpreter:

| Semantic category | Auto train | Auto validation | Hard-dev diagnostic | Training action needed |
|---|---:|---:|---|---|
| Single-goal `find` | 12,022 | 803 | yes | Present, but audit template families. |
| Other actions / side effects | 0 | 0 | yes | Human annotation + policy rules. |
| Multi-goal independent/dependent | 0 | 0 | yes | Review V19 multi-goal rows; add dependent cases. |
| Hard facts | 24,733 | see audit | yes | Verify ambiguous hard annotations. |
| Soft preferences | 1,990 | see audit | yes | Increase examples; maintain explicit markers. |
| Context | 0 | 0 | yes | Add reviewed annotations. |
| Missing required/optional data | 0 | 0 | yes | Define domain/provider requirements first. |
| Clarification/progressive search | 0 | 0 | yes | V19 validation had zero clarification cases; stratify new split. |
| Explicit confirmation/prohibition | 0 | 0 | yes | Human labels and runtime policy. |
| Goal dependency/result reference | 0 | 0 | yes | Human labels with stable local refs. |
| Negative / non-actionable | 0 | 0 | yes | Add train/validation negatives. |
| Relative time, numeric limits, currency | present | present | yes | Verify without inventing resolved dates/default currency. |
| Contradictions/impossible requests | 0 | 0 | validation-only currently | Define gold refusal/clarification protocol. |
| Unsupported capability | N/A | N/A | N/A | Requires a real capability registry. |

The full distribution is machine-readable in `data/v20/audit.json`. `benchmarks/v20/hard_dev.jsonl` is a **development diagnostic**, and `benchmarks/v20/holdout.jsonl` is sealed with SHA-256 in `benchmarks/v20/manifest.json`. Neither is in training or validation. The holdout must not be used for target-specific tuning. A later release evaluation should return aggregate metrics only.

## Token budget

`scripts/audit_v20_tokens.py` loaded the exact pinned Qwen3.5-9B tokenizer revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` from local cache, used the final chat template with thinking disabled, and refused any prefix mismatch. For 12,022 auto train examples: input mean **106.51** tokens, target mean **82.08**, total mean **188.59**, p99 total **261**, max **298**, none above 512. The matched V19 targets averaged **98.74** tokens, so the new target is **16.66 tokens (16.9%) shorter** on this restricted subset. Auto validation (803 rows) has total mean **190.30**, p99 **266**, max **290**, none above 512. These numbers do **not** establish that future reviewed multi-goal V20 examples fit 512; rerun the audit after adding them. Exact distributions are in `data/v20/token_audit.json`.

## Remaining migration work

Review action-side-effect rows, gates, multi-goal relations, previous-result references and clarification labels with a written annotation guide. Build an actual policy/capability interface before labeling tool IDs or claiming an action is executable. Add reviewed non-actionable and ambiguous cases to train/validation, then stratify by semantic family rather than by line. Audit near duplicates again, retokenize the final corpus and freeze the dataset/benchmark hashes before any future training. Until then the status is **NOT_READY**.
