# GoalContract 1.2 design (pretraining, no model migration)

See `V19_FAILURE_ANALYSIS.md` and `JASE_INTERPRETER_V20_ARCHITECTURE.md` for evidence and the exact V1.1 field inventory. This contract is the **Interpreter's request representation**, not an executable plan. The old V1.1 schema, Guard, reports, checkpoints and frontend remain intact.

## Main decisions

1. Replace `attributes`, `constraints`, `preferences` with one `facts` list. Each fact has `strength: hard|soft`. A bare target descriptor such as “ristorante giapponese” is a hard equality fact; a preference explicitly marked “possibilmente” is soft. Intrinsic-vs-selection classification is removed because the local runtime has no distinct behavior for the two V1.1 hard buckets. This addresses 20 observed dev bucket misclassifications without erasing the hard/soft distinction.
2. Keep the target `name` and optional `type`. `name` is the user's entity, `type` is an optional coarse taxonomy hint. No `raw_action`/`target.raw` duplication. Missing `type` means unspecified; JSON `null` is not overloaded as unknown/not-applicable/missing.
3. Add stable local goal IDs and optional `depends_on`. Optional `condition` refers to a **prior goal result**; a validator checks the reference and acyclicity. No free-form `gate` with unbound `then/otherwise` actions. A false branch means the conditional goal is skipped. An alternative requires another explicitly stated goal or reviewed annotation.
4. Add `missing` per goal only for **known required inputs**, with `blocks: search|action`; absent means the Interpreter has no reliable missing-input label, not that execution is safe. The Planner/provider registry may add further requirements. This permits search-before-clarification without asking the model to decide provider-specific fields.
5. Add `policy` only for explicit user instructions: `forbid` (never perform that action under this request) and `confirm_before` (wait for confirmation). System confirmation, communication and payment rules belong to the future policy engine and may only become more restrictive. An accepted contract is never permission to execute.
6. `kind: task|non_actionable` distinguishes requests for Jase to perform a lookup or action from theoretical/cited instructions. `task` requires at least one goal; `non_actionable` requires zero goals. `find` is still a task that can search without external side effects.
7. Optional `context` holds user-provided facts about the user/situation that are not requirements on a target. Literal fact values come from the user text. `value_ref` can refer to a result of another goal; the two value forms are exclusive. Prior-conversation references remain unresolved until a real conversation-state service supplies a stable ID. No guessed `result_id` is trained.
8. Relative dates remain literal strings such as `domani` or `dopo le 20`; a deterministic normalizer needs request timestamp and timezone to resolve them. Numeric quantities are numbers with `unit`/`currency` where needed. No model-calculated dates or implicit currency defaults.
9. Capability requirement/selection and the executable plan are deliberately **not model fields yet**. The repository has no capability registry, Planner or provider IDs. Discovery must derive abstract requirements from action+target and bind registered capabilities. Inventing IDs in V20 training would create unverifiable labels. Capability and plan accuracy are therefore explicitly unavailable until that interface exists.

## Schema shape and field semantics

The normative structural schema is `schema/goal_contract_v1_2.schema.json`; semantic invariants are in `jase/goal_contract_v1_2.py`. Optional fields are omitted when absent. An absent optional array means “no assertion of that category”, never “the downstream operation is authorized”. The object has no prose plan.

```json
{
  "schema_version": "1.2",
  "kind": "task",
  "goals": [
    {
      "id": "g1",
      "action": "find",
      "target": {"name": "ristorante", "type": "place"},
      "facts": [
        {"field": "cuisine", "op": "eq", "value": "giapponese", "strength": "hard"},
        {"field": "price", "op": "lte", "value": 40, "currency": "EUR", "strength": "hard"},
        {"field": "seating", "op": "eq", "value": "all'aperto", "strength": "soft"}
      ]
    }
  ]
}
```

| Field | Meaning and omission rule |
|---|---|
| `schema_version` | Exact `1.2`, used for parser dispatch. |
| `kind` | `task` or `non_actionable`. A theoretical question or cited command has no goal. |
| `goals` | Zero or more explicit requested outcomes; ID unique within request. |
| `goal.id` | Local stable reference, not a provider/job ID. |
| `goal.action` | One of the 11 observed V19 actions (`find`, `book`, `buy`, `compare`, `contact`, `rent`, `monitor`, `order`, `hire`, `cancel`, `notify`). No new taxonomy invented. |
| `goal.target.name/type` | Grounded entity name and optional type hint. `name` required for every current action. |
| `goal.facts` | One fact once; `hard` excludes nonmatching results, `soft` ranks them. The field's relationship is encoded by `op`, not by a third bucket. |
| `fact.field/op/value` | Open snake_case field, V1.1 operator vocabulary, typed literal; `value_to` only for `between`. `value_ref` replaces `value` when a prior goal result supplies it. |
| `fact.unit/currency` | Explicit physical unit/ISO currency for typed numbers. Missing units on required numeric dimensions are semantic errors. |
| `goal.depends_on` | IDs of goals whose results must exist first; optional empty array. No cycles. |
| `goal.condition` | Optional predicate over a named predecessor's result. The condition is user-grounded and creates no new action by itself. |
| `goal.ranking` | Optional one-field sort such as lowest price (`price/asc`); avoids fabricating a numeric threshold for “il più economico”. |
| `goal.missing` | Optional list of known missing fields and the blocked phase (`search`/`action`). It does not claim to list every provider requirement. |
| `goal.policy` | Optional explicit `forbid` and `confirm_before` action lists. Other confirmation rules are enforced downstream. |
| `context` | Optional user situation facts, never silently copied into a target filter. |

## Progressive execution and policy

For “Prenotami un tavolo”, a human-reviewed annotation may record a `book` goal and `missing` date/time/party size that block `action`, while a future Planner may still search. The V19 dataset does not supply those labels, so this case is **requires_review**, not auto-migrated. For “trova un hotel, non prenotare”, the `find` goal records `policy.forbid=["book"]`. For “prenota solo dopo la mia conferma”, `policy.confirm_before=["book"]`. A contract without policy is **not** an execution permit; default system rules still apply. `forbid` and `confirm_before` for the same action are contradictory and invalid.

## Multi-goal, reference and condition

Independent flight and hotel lookups receive `g1` and `g2` with no dependency. If hotel location depends on the arriving flight, `g2.depends_on=["g1"]`, and its location fact uses `value_ref={"goal":"g1","field":"arrival_airport"}`. A conditional hotel lookup can add `condition={"goal":"g1","test":{...}}`; its source must be a dependency. This is a dependency **declaration**, not an executable plan. The Planner later turns it into steps after Discovery binds real capabilities. “Prenota il secondo di quelli mostrati prima” cannot name an ID here without conversation state; it is marked unresolved for review/clarification rather than inventing one.

## Grounding and unknowns

The model may emit literal facts grounded in the user request or structured references. Default values and provider facts are applied downstream with separate provenance (`user`, `context`, `tool`, `default`) and must never masquerade as user-supplied facts. Optional field absence is the sole “not specified” representation. Required input missing is an explicit `missing` entry. An unresolved prior-result reference is represented by a missing reference entry until state resolution. JSON `null` is not used as a universal unknown sentinel. Semantic validation can flag ungrounded literals for review, but lexical grounding alone cannot prove correctness.

| State | V1.2 representation |
|---|---|
| Not applicable | Omit the optional field; no downstream decision depends on it. |
| Optional but unspecified | Also omit it: this distinction has no operational effect until a capability requires it. |
| Required but unspecified | `missing:{field,blocks,reason:"unspecified"}` on the affected goal. |
| Referent unknown/unresolved | `missing:{field,blocks,reason:"unresolved_reference"}`; never invent a conversation/result ID. |

The two omission cases intentionally share syntax because neither blocks a current step. If a future provider makes the difference operational, the Planner attaches a typed requirement before execution. This is more precise than using `null` for every state.

V19 implicit-default audit: when a surface form has no verb, the generator/Guard uses `raw_action="(implicito)"` and `action=find`; a singular object does not create an explicit `quantity=1` fact; absent/null `gate` means no branch. These are representation conventions, **not** defaults for cuisine, budget, time, party size, currency, location or permission to act. V1.2 keeps implicit `find` only when the request genuinely asks Jase to search; theoretical/quoted commands use `kind=non_actionable`. Quantity one remains omitted. Currency without an explicit unit is flagged for review. A provider/system default must be attached downstream with its own provenance and never appear as a user fact.

## Compatibility, alternatives and cost

V1.1 and V1.2 parsers coexist; existing `jase.guard` keeps V1.1 behavior. The V1.2 migration is a new directory. Existing frontend and model runtime remain on V1.1. A future deployment needs version negotiation, a V1.2 Guard, Planner policy/input contract, Discovery registry and benchmarked adapter. A lossy 1.1→1.2 conversion can merge hard attributes/constraints and keep soft preferences; it cannot infer policy, dependencies, context, concrete capabilities or missing booking inputs. Such rows are reviewed.

Rejected alternatives: (a) carrying all three old buckets, which preserves observed ambiguity; (b) target.attributes plus separate constraints, which still makes `cuisine=japanese` ambiguous operationally; (c) model-generated provider ID or full plan, which has no matching registry/runtime in this repository; (d) `can_act`/`can_search` booleans, which depend on provider inputs and live policy; (e) provenance on every token, which increases output cost without a trusted source verifier. Token size is measured after migration, not assumed.
