# V24 semantic audit (draft, no model changes)

The V23 transfer panel has 10 cases where Guard 1.2 said `ACCEPT` while the exact V1.2 metric disagreed with the gold. This is a **manual provisional classification**, not an independent production measure. `ACCEPT` only hands a contract to the Planner; it does not authorize execution.

| Case | Difference | Provisional assessment |
|---|---|---|
| t02 | `amenity contains parcheggio` vs `eq parcheggio` | Uncertain: equivalent for a boolean amenity only if the provider's matching semantics agree. |
| t10 | reminder content under `reminder` vs `message` | Material: an open field name can cause Planner misrouting. |
| t14 | `peruviano` vs `peruviana` cuisine | Likely equivalent Italian inflection; exact metric false negative. |
| t17 | generic `time` vs `arrival_time` | Material: could filter departures instead of arrivals. |
| t23 | generic `visita`/`professional_service` vs dental appointment | Material: cancellation could target the wrong appointment. |
| t25 | omitted `quantity=2` in comparison | Material: requested comparison cardinality is lost. |
| t26 | rented bicycle typed `product` vs `vehicle` | Material routing risk, even though the target name is intact. |
| t27 | `tablet` vs `tablet X` | Material: model identifier is lost. |
| t28 | `flower` vs `product` for tulips | Likely taxonomy alias; exact metric false negative if both route to the same order capability. |
| t30 | station preference promoted to a hard location | Material: hard/soft constraint and geographic scope changed. |

Provisional split: 7 material mismatches, 2 likely equivalent labels, 1 unresolved operator difference. This is **not** a corrected benchmark score. A person should review the labels and the downstream Planner/provider semantics before metric changes. Historical gold and raw generations remain untouched.

## GoalContract 1.3 direction

The contract currently permits arbitrary `fact.field` and `target.type` strings. That makes schema validity weaker than operational correctness. Introduce a versioned vocabulary for critical fields (`arrival_time`, `departure_time`, reminder `message`, quantity and identifiers) and target routing categories. Keep an explicit extension namespace for unknown provider fields. Reject or hold unknown fields at the Planner boundary rather than silently mapping them. Canonicalization may collapse vetted surface aliases such as cuisine gender inflection; it must never collapse arrival/departure, hard/soft, specific/generic target identity, or provider routing types without a documented equivalence.

The new V24 safety panel was frozen at commit `5346f18` before any Guard or contract optimization. It has 20 newly authored, schema-validated cases and zero exact text overlap with V23 training. It is small, feature-targeted and AI-authored, so performance on it cannot establish production reliability. V22/V23 baseline evaluations should be recorded before changing the Guard or dataset. Do not copy panel requests or model outputs into future training examples.
