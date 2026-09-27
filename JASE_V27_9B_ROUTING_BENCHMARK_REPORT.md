# Jase Interpreter 9B: V27 fresh routing benchmark

V27 was frozen **before** the routing Guard changes (commit `513fb27`), and
both models were evaluated on the identical 100 requests. Panel SHA-256:
`b2d622a7c8f5515c4ac4420139ff3ffc7c8b326d35dfcdcdc2902dff8787f050`.
The panel contains 20 capability families in five related contexts, with no
exact or Jaccard ≥0.85 overlap against the recorded training and earlier
panels. It is AI-authored and needs independent human review.

| Measure on V27 | V24 9B final | V25 9B final |
| --- | ---: | ---: |
| Valid JSON | 98/100 | 97/100 |
| Canonical exact, V24 projected 1.2→1.3 | 29/100 | **40/100** |
| Paired exact gain/loss | — | **16 gained / 5 lost** |
| Raw action (before schema projection) | 95/110 | **106/110** |
| Raw target type | 101/110 | 103/110 |
| Raw goal count | 97/100 | 97/100 |
| Fact recall after projection | 68/140 | 73/140 |
| Temporal recall after projection | 28/65 | 29/65 |
| Candidate schema/projection holds | 38/100 | 32/100 |
| Output tokens/s | 18.14 | 18.00 |
| Peak inference VRAM | 8.19 GB | 8.18 GB |

The raw action and goal counts are directly comparable. Projected field scores
are biased against V24 because it emits V1.2 and 38 outputs could not be safely
projected. The 11-point exact gain is a combined model/contract/prompt effect,
not evidence that the additional fine-tuning alone caused it. V25's 40% exact
and 52% projected fact recall remain inadequate for a general action system.

V25 improves find-then-book from 1/5 to 4/5, food order from 0/5 to 4/5,
direct vehicle rent from 0/5 to 4/5, and transport price sorting from 0/5 to
3/5. The largest paired regression is **all five** origin/arrival train
requests: V24 is 5/5 and V25 is 0/5. V25 puts `origin` in an unsupported
`spatial` block instead of `facts`, so these contracts cannot be routed under
GoalContract 1.3. Five older V26 fictional quoted commands became bookings;
V27's more explicit non-action wording succeeds 5/5. Quoted commands are
therefore still a wording-sensitive safety risk, not a solved capability.
Neither model gets any of the five conditional fallbacks, five explicit
no-book hotel searches, five direct-flight cases or five appointment changes
exact. These failures mix aliases, missing policy and temporal-role mistakes.

## Experimental routing Guard

The separate opt-in Guard and field vocabulary are in commit `6cec92e`.
Historical Guard metrics and frontend behavior are unchanged. On the **same
V25 predictions**, the existing Guard accepts 64/100; the routing Guard
accepts 48/100. It holds 16 former ACCEPTs: five missing no-book policies,
four missing result limits, five unproven `result_count` aliases and two dates
in free-form facts. It holds none of the 40 canonical-exact outputs. The
old Guard's 24 accepted exact mismatches fall to eight; this is a triage count,
not a risk rate.

Manual review of all eight new-Guard ACCEPT mismatches finds five clear
operational errors: `v27-019` routes a bicycle rental search as transport and
loses the no-rent policy; `v27-047` and `v27-087` use `search_time` rather than
the arrival deadline; `v27-050` drops the open-pharmacy requirement; and
`v27-055` drops walking access. Three (`v27-065`, `v27-074`, `v27-078`) differ
only by the article in a date string (`il 12 giugno` vs `12 giugno`). Thus the
new Guard's reviewed accepted-material lower bound is **5/48**, with 48%
coverage. Zero newly held canonical-exact cases is an exact-metric false-hold
check; semantically correct non-exact holds still need human adjudication.
The V26 diagnostic 18 labelled accepted risks are all held by this Guard, but
V26 was consulted during rule design and cannot validate generalization.

## Problems and next gate

The V1.3 structural validator permits temporal roles such as
`notification_date` inside free-form `facts`; the new ontology catches this at
handoff without changing historical scoring. `delivery` and `result_count`
aliases are held rather than rewritten because no V1.3 Planner/provider mapping
has been verified. The Gate only permits Planner inspection, never execution.
The Python stack still uses slower fallback CUDA kernels for several Qwen
operations; inference remains about 18 tokens/s.

Do not promote V25 or this Guard to the frontend. Next, freeze a genuinely
separate, linguistically reviewed panel before adjusting rules again. Fix the
five accepted V27 risks and the V25 `spatial`/time-role failures with explicit
Planner mapping and targeted, independent data only if deterministic checks
cannot resolve them. Measure accepted material errors, coverage and false
holds on that successor panel. No new training is justified by V27 exact alone.

Outputs: `benchmarks/outputs/v24-9b-v27-fresh/` and
`benchmarks/outputs/v25-9b-v27-fresh/`. Earlier adapters, 2B and frontend
were untouched.
