# GoalContract 1.3 field ontology (experimental)

The V1.3 schema permits arbitrary `facts[].field` strings. The experimental
`field_ontology_v1_3.py` closes the vocabulary at the **Guard → Planner
inspection** boundary. It does not grant execution permission and is not
connected to the frontend. The live Planner has not been migrated to V1.3, so
these names are proposed handoff slots, not verified provider mappings.

| Contract slot | Meaning | Mapping rule |
| --- | --- | --- |
| `facts.destination`, `facts.origin` | Route endpoints | Preserve direction; never infer a missing origin. |
| `facts.stops = 0` | Direct flight | Required for an explicit direct or nonstop flight. |
| `facts.delivery_location` | Delivery destination | `delivery` is held pending a proven Planner mapping. |
| `facts.message` | Contact or reminder content | Quoted text can be content when the user requests sending it. |
| `facts.selection.value_ref` | Selected search result | Booking after search must reference the prior goal and depend on it. |
| `temporal.*` | Search, action, arrival or notification time | Date and time fields in `facts` are held for typed role review. |
| `modifiers.compare_count` | Number of alternatives compared | `facts.quantity` does not replace this modifier. |
| `modifiers.limit` | Number of results shown | Distinct from a product quantity. |
| `condition.test.results_count` | Number of prior results | `result_count` is held; no silent alias rewrite. |
| `policy.forbid`, `policy.confirm_before` | Negative permission and confirmation | Must bind to the affected action. |

The vocabulary includes the field families found in V25 training plus explicit
critical fields such as `stops` and `open_now`. Unknown fields and known aliases
are held for review by `semantic_guard_v1_3_routing.py`; they are not rewritten.
This deliberately trades coverage for less silent routing risk. A matching
field name alone does not prove the value was copied correctly. The routing
Guard adds narrow checks for quoted fictional commands, explicit no-book/no-buy
policy, nonstop flights, result cardinality, and dependent find-then-book
requests. Its checks are deterministic and opt-in. It cannot verify every
intent, field value, provider capability, or real-world feasibility.

V27 was frozen in commit `513fb27` before this ontology and Guard change. It is
100 AI-authored requests in 20 families across five related contexts, with
no exact or Jaccard ≥0.85 overlap against the listed training and earlier
panels. Its gold remains subject to independent human review. V26 remains a
consulted diagnostic panel; its exact mismatches are not all material errors.
