# V19 failure analysis before GoalContract 1.2

## Evidence and limits

Sources: completed 9B run `20260925T123205Z-21249614`, `benchmarks/outputs/jase9b-v20-dev.json` (246 unseen dev cases), `benchmarks/outputs/jase9b-release-holdout.json` (27 sealed cases), `benchmarks/outputs/jase9b-posttrain-probes.json` (seven independent requests), the 2B historical reports, and `data/v19_clean`. Dev success was 212/246 (86.18%) versus the historical 2B's 228/246 (92.68%). Intent was 99.61% for both. The 9B had 32 accepted-but-wrong dev contracts versus 15 for the 2B. Its dev facts precision/recall were 98.15%/97.26%, against 99.63%/98.35% for the 2B. Release success was 8/27 versus 6/27, but 9B facts precision was 58.62% and one of 27 outputs failed schema validation. Release facts rates are computed on only 26 schema-valid cases, so they have a different denominator from the historical 2B rates.

The 9B dev evaluator saved aggregate counts but **did not persist per-case rows** despite `--include-outputs`. Thus dev IDs and intersections between failure types cannot be reconstructed. Counts below are *cases carrying that error* from the evaluator taxonomy, not disjoint populations. No per-case release output is exposed or used to edit targets. Probe IDs are available; absence of a measurable count is reported as `N/M`, not zero. This reporting gap is an evaluator bug to fix for future non-sealed dev runs.

## Taxonomy

| Category | Dev cases | Probe / aggregate evidence | Likely cause and owner |
|---|---:|---|---|
| `omitted_fact` | 10 | Release aggregate: 7 | Multi-fact extraction; model and dataset coverage. |
| `hallucinated_fact` | 4 | Release aggregate: 10; probe `multi-step-travel` invented a price condition | Model, permissive grounding and insufficient negative labels. |
| `wrong_action` | 0 observed | Release aggregate: 3 | Ambiguous action phrases; model/annotation. Dev action accuracy remains high. |
| `wrong_target` | name 1, type 8 | Release aggregate: name 4, type 4 | Open target vocabulary and inconsistent type conventions; model/annotation. |
| `wrong_entity_role` | N/M | Release has anaphora/role categories, without case-level predictions | Schema has only free-text `reference`; annotation/evaluator gap. |
| `wrong_attribute` | 4 attributes→constraints, 3 attributes→preferences | Combined bucket errors | Schema overlap and model. |
| `preference_as_constraint` | 4 | Release aggregate: 1 | Three competing predicate buckets; schema/model. |
| `constraint_as_preference` | 3 | Release aggregate: 1 | Three competing predicate buckets; schema/model. |
| `contextual_fact_misclassified` | N/M | V1.1 has no context field | Schema/evaluator gap; cannot infer a count from bucket errors. |
| `missing_clarification` | N/M | Probe `ambiguous-booking` accepts a table booking without essential details; release aggregate has one generic clarification error | Dataset/schema/policy gap. V19 validation has **zero** clarification examples although train has 662. |
| `unnecessary_clarification` | 2 on dev by evaluator | Release aggregate: 2/27 | Model/guard policy; evaluator reports this directly. |
| `premature_execution` | N/M | Probe `ambiguous-booking` is accepted, but no executor exists here | Missing executable policy; schema/runtime boundary. Acceptance alone is not proof of real-world execution. |
| `malformed_gate` | N/M | Probe `multi-step-travel`: invalid `gate` (`when` and branch actions do not match the request) | Model/schema; v1.1 `gate` conflates condition with action. |
| `multi_goal_collapse` | N/M | Dev aggregate has 1 extra goal and multi-goal success 8/10 | Model/annotation; no per-case dev rows to distinguish collapse from other goal-count errors. |
| `dependency_error` | N/M | V1.1 has no goal dependency | Schema/evaluator gap. |
| `anaphora_failure` | N/M | Release aggregate weak on anaphora, no exposed per-case predictions | Schema/context gap. |
| `numerical_value_error` | up to 3 generic value errors | Release aggregate: up to 5 generic value errors | Model/normalization/evaluator; value errors are not typed by domain. |
| `currency_error` | 0 in evaluated schema-valid dev facts | Release aggregate currency accuracy 100% over evaluable facts | No observed currency failure; coverage still needed. |
| `date_time_error` | N/M | Probe `ambiguous-booking` preserves `stasera`, but missing inputs are not identified | Temporal normalization and evaluator gap. |
| `location_error` | N/M | Location is the most frequent V19 property (11,237 occurrences in train); generic value/role errors cannot be attributed specifically | Need field-level evaluator. |
| `confirmation_policy_error` | N/M | Probe `confirmation-gate` keeps only `find`, and cannot encode the user's “do not book until confirmation” as a policy | Schema gap. |
| `unsupported_action_acceptance` | N/M | No capability registry/executor exists in this repository | Runtime capability contract needed before measuring. |
| `schema_valid_but_semantically_wrong` | 32 accepted-but-wrong | Release aggregate: 16 | Primary safety failure; model and incomplete guard. |
| `bucket:preferences→attributes` | 5 | Dev preference success 25/39 versus historical 31/39 | Schema/model; strongest observed dev bucket direction. |
| `extra_goal` | 1 | Dev taxonomy | Model; suggests goal decomposition error. |

Definitions and synthetic examples (not release case text):

| Category | Definition / synthetic example |
|---|---|
| `omitted_fact` | User says “a Roma sotto 40€” but location or budget is absent. |
| `hallucinated_fact` | Contract adds `cuisine=giapponese` when no cuisine was requested. |
| `wrong_action` | User asks to search, output action is `book`. |
| `wrong_target` | User asks for a train, output target is a hotel or wrong type. |
| `wrong_entity_role` | “Scrivi a Marta di Luca” assigns the recipient/message roles incorrectly. |
| `wrong_attribute` | A descriptive fact is placed in another V1.1 bucket. |
| `preference_as_constraint` | “Possibilmente all'aperto” becomes mandatory. |
| `constraint_as_preference` | “Massimo 40€” becomes optional. |
| `contextual_fact_misclassified` | “Sono già a Milano” becomes a target location constraint. |
| `missing_clarification` | “Prenotami un tavolo” is treated as action-ready without date/time/party information. |
| `unnecessary_clarification` | Complete search request is blocked for an optional extra detail. |
| `premature_execution` | A side-effecting goal is passed toward execution before required input/confirmation. No live execution was observed here. |
| `malformed_gate` | Conditional branch lacks required fields or binds the wrong condition/action. |
| `multi_goal_collapse` | Flight and hotel requests are merged into one target/goal. |
| `dependency_error` | Hotel near arrival airport is attempted before the flight result exists. |
| `anaphora_failure` | “Prenota il secondo” points to no prior option or the wrong one. |
| `numerical_value_error` | `<40` becomes `<=40` or price 40 becomes 400. |
| `currency_error` | 40 EUR is emitted as 40 USD or without required currency. |
| `date_time_error` | “Prima delle 14” becomes after 14 or an invented absolute date. |
| `location_error` | Origin and destination are swapped, or a city is invented. |
| `confirmation_policy_error` | “Solo dopo conferma” is omitted or applied to read-only search. |
| `unsupported_action_acceptance` | A job is accepted as executable without a matching registered capability. Registry absent here. |
| `schema_valid_but_semantically_wrong` | JSON/schema valid and Guard accepted, but gold meaning differs. |
| `bucket:preferences→attributes` | Soft preference becomes a V1.1 descriptive attribute. |
| `extra_goal` | Contract adds another job the user did not request. |

The release holdout is only an aggregate diagnostic here. Its text, predictions, and case-specific targets are excluded from migration decisions and from new examples. A new V20 holdout must be authored independently and sealed before inference.

## Dataset and evaluator findings

V19 train has 27,504 rows, including 26,614 single-goal and 890 two-goal rows; 3,856 preference predicates and 2,258 gates. Validation has 2,310 rows, 78 two-goal rows, 343 preference predicates, 113 gates, and **no clarification-required rows**. Train has 662 clarification-required rows. The current fixed 64-case validation sample cannot reveal clarification regressions. The low final validation loss (0.01069) therefore does not establish semantic safety. V19 uses 38,425 constraint predicates and 13,192 attribute predicates, often with no operational distinction in the local runtime.

The current evaluator compares facts canonically but conditionally excludes schema-invalid outputs from precision/recall and does not persist dev case rows. Future reports must show all-case complete success separately from conditional fact rates, retain case-level **dev** diagnostics, and keep sealed holdout predictions private. Train/dev/holdout overlap must be checked before any new training.
