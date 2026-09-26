# Interpreter boundary and current 1.1 system map

This map was made before changing code for GoalContract 1.2. The repository is an Interpreter training/evaluation repository, not the complete Jase runtime. It contains no executable Planner, Discovery capability registry, provider selector, Executor, Verifier, conversation-state store, or payment/booking policy registry. References to those components in `jase/guard.py` and `scripts/interpret.py` are interfaces/intent, not implementations. `scripts/jase_runner.py` is a local **ML experiment job queue**, not the product Planner.

## Existing path

`TRAIN_SYSTEM_PROMPT.txt` + user request → tokenizer/model → `jase.canon.extract_contract` → `jase.guard.check_contract` → accepted/repaired/clarify/rejected (`scripts/interpret.py`). The guard loads `schema/goal_contract_v1_1.schema.json` and performs structural checks, limited grounding and deterministic repairs. `jase.metrics.compare/aggregate` canonicalizes V1.1 facts and evaluates action, target, buckets, operator, units, currency, reference, clarification and gate. `evaluation/compare.py` runs benchmarks. No action is executed in this repository. `scripts/serve_mlx.py` is a legacy alternate HTTP path: it loads `schema/goal_contract_v1.schema.json`, uses `SYSTEM_PROMPT.txt`, and bypasses the semantic Guard. It must not silently be treated as a 1.1 or 1.2 production endpoint.

## Complete GoalContract 1.1 field map

Every object has `additionalProperties: false`. Arrays are ordered JSON sequences, although the evaluator canonicalizes some order. `scalar` accepts string, number, integer, boolean **and null**, which is wider than many operational predicates need. Names of actions, targets and fields are open strings; there is no capability ID or action enum in the schema.

| Path | Schema/type; required? | Meaning and actual local use | Problem |
|---|---|---|---|
| `schema_version` | enum `1.1`; yes | Version, guard/evaluator | Guard also upgrades some legacy versions to 1.1; no explicit 1.2 dispatch. |
| `goals` | array min 1; yes | Desired goals; evaluator counts/aligns by target | No stable IDs, dependencies or previous-result references. |
| `clarification` | object; yes | One global required flag/reason/missing_fields; guard/evaluator | Blocks or releases the whole request, even if search can proceed while booking cannot. |
| `goals[].action` | nonempty string; yes | Action/intent, evaluator and guard | Open action string; no side-effect class or provider-independent capability binding. |
| `goals[].raw_action` | nonempty string; yes | Grounding evidence for the action in guard | Duplicates action; implicit action uses a sentinel. No executor consumes it. |
| `goals[].target` | object; yes | What the action concerns | Target attributes live elsewhere; no entity/result reference. |
| `target.name` | nonempty string; yes | Entity name, guard/evaluator | Open vocabulary with synonyms; target matching can be fragile. |
| `target.type` | string or null; yes | Coarse type, evaluator | Null conflates unknown/unavailable; no type registry. |
| `target.raw` | nonempty string; yes | User-text grounding, guard | Repeats target wording and increases output tokens. |
| `attributes` | predicate array; yes | V1.1 descriptive facts, guard/evaluator | Competes with constraints even when both filter search results. |
| `constraints` | predicate array; yes | Hard filters, guard/evaluator | Overlap with attributes; mixed intrinsic and search criteria. |
| `preferences` | predicate array; yes | Soft criteria, guard/evaluator | Operationally different from hard filters, but a third full predicate bucket causes classification errors. |
| `gate` | gate object or null; optional | `when` result predicate; `then` action; `otherwise` action/null | No source/result reference, no execution policy, easy to make incomplete or invent branches. |
| `predicate.property` | nonempty string; yes | Open field name; guard/evaluator | Normalization conventions are in prose and lexicon, not schema. |
| `predicate.operator` | enum `eq,neq,lt,lte,gt,gte,contains,excludes,before,after,near,within,between,in,not_in`; yes | Comparison relation; guard/evaluator | Schema permits nonsensical field/operator/value combinations. |
| `predicate.value` | scalar including null; yes | Compared value; guard/evaluator | Null has too many possible meanings; structured quantities and temporal values missing. |
| `predicate.value_to` | scalar; optional | Upper end for `between` | Schema does not require it for `between` or forbid it otherwise. |
| `predicate.unit` | string/null; optional | Physical unit; guard/evaluator | No mandatory unit for distance/duration, no compatibility check at schema level. |
| `predicate.currency` | string/null; optional | Currency; guard/evaluator | Optional even for monetary numbers. |
| `predicate.reference` | string/null; optional | Relative anchor; guard/evaluator | Free text cannot point to a stable goal/result/entity. |
| `gate.when` | predicate; yes | Condition | No explicit source of the value being tested. |
| `gate.then` | nonempty action string; yes | Action on true branch | Branch action not tied to goal ID or policy. |
| `gate.otherwise` | action string or null; yes | Action on false branch | Null means no branch; not the same as unresolved/unknown. |
| `clarification.required` | boolean; yes | Ask user before proceeding | Global binary state prevents progressive execution. |
| `clarification.reason` | string/null; yes | Human reason | Free text, measured only loosely; null is overloaded. |
| `clarification.missing_fields` | string array; yes | Needed values | No step/capability affected, no distinction between optional and required. |

## Planned ownership boundary

| Component | Responsibility for 1.2 | Explicitly outside its responsibility |
|---|---|---|
| Interpreter | Extract grounded user intent, target, hard/soft facts, explicit prohibitions and confirmation requests, unresolved references, and goal relationships. | Selecting a provider, supplying unknown booking fields, claiming an action is executable, resolving dates without a reference clock. |
| Deterministic validator/normalizer | Validate IDs, references, operators and types; normalize units/time only with an explicit clock/timezone; preserve raw phrase. | Guessing missing facts or repairing semantic errors by sampling. |
| Planner/policy engine (future) | Turn goals into steps; decide search-before-clarification; determine required inputs and confirmation from registry/provider policies; enforce user prohibitions. | Reinterpreting the user's hard/soft preference. |
| Discovery (future) | Resolve abstract action+target requirements to registered capabilities and providers. | Trusting LLM-produced concrete provider IDs. |
| Executor (future) | Perform permitted steps after dependencies, inputs and confirmation are satisfied. | Running a contract merely because it is valid JSON. |
| Verifier (future) | Check tool results and requested outcome, report errors and provenance. | Assuming LLM output proves real-world success. |

**Decision:** do not make the Interpreter output a concrete plan or provider/tool ID. The existing repository has no registry to validate those labels. GoalContract 1.2 can express goal dependencies and *explicit user policy*, while a future Planner/Discovery derives capability requirements and executable steps. A separately versioned capability binding can be added when real registry IDs exist. This avoids invented training labels and keeps future provider changes out of the model.
