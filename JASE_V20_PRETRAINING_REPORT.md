# Jase Interpreter V20 / GoalContract 1.2 — pretraining report

**Status: NOT_READY. No training, model download, adapter modification or frontend change occurred in this phase.**

## V19 findings

The completed 9B V19 run is structurally strong but semantically weaker than the historical 2B on the shared 246-case dev panel: 86.18% versus 92.68% complete success, 32 versus 15 accepted-but-wrong contracts. Intent/action was 99.61% for both. V19's three predicate buckets produce observed preference/constraint/attribute swaps; the 9B also omits and invents facts. The 27-case release panel is mixed (9B 8/27 versus 2B 6/27) and too small to justify promotion. Details and category definitions are in `docs/V19_FAILURE_ANALYSIS.md`.

## GoalContract 1.1 limitations and 1.2 decision

V1.1 has no goal IDs/dependencies, context, progressive missing inputs, explicit user policy or result references. Its `attributes` and `constraints` are often operationally identical; `preferences` are operationally different but compete for the same predicate syntax. `gate` has no bound result source and was malformed in the post-training multi-step probe. The full V1.1 field/runtime map is in `docs/JASE_INTERPRETER_V20_ARCHITECTURE.md`.

V1.2 uses one `facts` list with `strength=hard|soft`, local goal IDs, optional goal dependencies and conditions, per-goal known missing inputs, explicit user prohibitions/confirmation requests, optional user context, result value references and ranking. The normative schema is `schema/goal_contract_v1_2.schema.json`, semantic validator `jase/goal_contract_v1_2.py`, rationale `docs/GOAL_CONTRACT_1_2_DESIGN.md`. V1.1 remains available and unchanged. The parser distinguishes explicit `1.1` and `1.2`; it does not silently upgrade a V1.2 contract.

## Interpreter boundary

The Interpreter extracts grounded intent and explicit user policy. Deterministic code validates structure and numeric/dependency invariants. A future Planner/policy engine decides progressive steps, required provider inputs and system confirmation; Discovery binds registered capability/provider IDs; Executor performs permitted side effects; Verifier checks real outcomes. This repository has **no product Planner, Discovery registry, Executor, Verifier or conversation store**. Therefore the model does not generate concrete tool IDs, executable plans, `can_act` booleans or guessed previous-result IDs. The legacy `scripts/serve_mlx.py` still serves the old V1 schema without the semantic Guard and is not a V1.2 endpoint.

## Migration and dataset

`scripts/migrate_v19_to_v20.py` preserves `data/v19_clean` and conservatively migrates only one-goal, read-only `find` contracts passing both old Guard and new validation. From 29,814 source rows: **12,022 auto train, 803 auto validation, 16,989 requires review, zero rejected**. No new training examples were created. Most review cases involve non-`find` actions (11,174), gates (2,371), multi-goal (968), references (833), clarifications (662) or unspecified currency (537); reasons overlap. The 305 near-duplicate validation candidates were quarantined. Source/output hashes and reasons are in `data/v20/manifest.json`; detailed process in `docs/V19_TO_V20_MIGRATION.md`.

The resulting train/validation data are **severely imbalanced**: all 12,825 auto rows have one `find` goal. Train contains 24,733 hard and 1,990 soft facts, but zero context, missing-input, dependency, policy, confirmation, non-actionable or multi-goal labels. Validation has the same structural holes. `data/v20/audit.json` gives full distributions. `benchmarks/v20/hard_dev.jsonl` contains 16 labeled diagnostic cases; `benchmarks/v20/holdout.jsonl` contains 12 distinct sealed cases, both excluded from training. They are small benchmark scaffolds, not substitutes for reviewed training coverage.

## Tokenization and compactness

The exact pinned Qwen3.5-9B tokenizer `c202236235762e1c871ad0ccb60c8ee5ba337b9a` was loaded offline with the actual chat template. Auto train input/target/total mean tokens are **106.51 / 82.08 / 188.59**; total p90/p95/p99/max **222 / 235 / 261 / 298**. Auto validation total mean **190.30**, p99 **266**, max **290**. No auto example exceeds 512. For matched train rows, V19 targets average **98.74** tokens; V1.2 targets average **82.08**, a **16.9% reduction** on this restricted subset. The new prompt is longer and increases total tokens despite the shorter target. This does not predict lengths of reviewed multi-goal examples; retokenize before any future run. Full distributions: `data/v20/token_audit.json`.

The static hard-dev/holdout panels also fit within 512 in this tokenizer (max total **241/254** respectively). They are small authored examples and do not bound future reviewed training examples.

## Leakage and evaluator

Normalized exact and token-set near overlap (Jaccard ≥0.85 under a rare-token candidate heuristic) are zero across auto train/validation and between auto train/validation and the new hard-dev/holdout. The new holdout also has no detected overlap with V19 train, validation, historical dev or historical release. The heuristic is not proof of family independence; V19 does not retain the source-family graph. Audit: `data/v20/audit.json`; holdout hash: `benchmarks/v20/manifest.json`.

`jase/metrics_v1_2.py` and `evaluation/evaluate_v20.py` measure all-case JSON/schema/full success and conditional action/target/fact/hard-soft/context/missing/dependency/reference/explicit-policy/condition/ranking metrics. Static self-evaluation on the 16 hard-dev gold cases gives 16/16 success, establishing scorer and schema compatibility, **not model quality**. No V1.2 model has been run. Capability, plan, runtime confirmation and accepted-but-wrong-after-Guard metrics are unavailable until matching components exist. Attribute accuracy is replaced by hard-fact accuracy because the old attribute bucket was removed. Protocol: `docs/V20_BENCHMARK_PROTOCOL.md`.

An evaluator bug was fixed: `evaluation/compare.py --include-outputs` now persists per-case **dev** rows as advertised. The previously completed 9B dev report remains aggregate-only; it was not rerun. Sealed release reports remain aggregate-only.

## Representative input → GoalContract 1.2 examples

These are authored **hard-dev gold**, never training or holdout inputs. They illustrate the proposed representation, not measured model behavior.

### d01

**Input:** Cerca una trattoria siciliana a Genova, massimo 30 euro, meglio con tavoli fuori; non prenotare.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"trattoria","type":"place"},"facts":[{"field":"cuisine","op":"eq","value":"siciliana","strength":"hard"},{"field":"location","op":"near","value":"Genova","strength":"hard"},{"field":"price","op":"lte","value":30,"strength":"hard","currency":"EUR"},{"field":"seating","op":"eq","value":"tavoli fuori","strength":"soft"}],"policy":{"forbid":["book"]}}]}`

### d02

**Input:** Cerca un treno per Venezia domani e un albergo vicino alla stazione di arrivo.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"treno","type":"transport"},"facts":[{"field":"destination","op":"eq","value":"Venezia","strength":"hard"},{"field":"date","op":"eq","value":"domani","strength":"hard"}]},{"id":"g2","action":"find","target":{"name":"albergo","type":"accommodation"},"facts":[{"field":"location","op":"near","value_ref":{"goal":"g1","field":"arrival_station"},"strength":"hard"}],"depends_on":["g1"]}]}`

### d03

**Input:** Se trovi un volo per Madrid sotto i 250 euro, cerca anche una stanza.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"volo","type":"transport"},"facts":[{"field":"destination","op":"eq","value":"Madrid","strength":"hard"},{"field":"price","op":"lt","value":250,"strength":"hard","currency":"EUR"}]},{"id":"g2","action":"find","target":{"name":"stanza","type":"accommodation"},"facts":[],"depends_on":["g1"],"condition":{"goal":"g1","test":{"field":"result_count","op":"gt","value":0}}}]}`

### d04

**Input:** Invia il messaggio a Giulia soltanto dopo che te lo confermo.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"contact","target":{"name":"Giulia","type":"person"},"facts":[],"missing":[{"field":"message","blocks":"action","reason":"unspecified"}],"policy":{"confirm_before":["contact"]}}]}`

### d05

**Input:** Mostrami quattro pizzerie e prenota la terza domani.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"pizzeria","type":"place"},"facts":[{"field":"quantity","op":"eq","value":4,"strength":"hard"}]},{"id":"g2","action":"book","target":{"name":"pizzeria","type":"place"},"facts":[{"field":"selection","op":"eq","value_ref":{"goal":"g1","field":"result","ordinal":3},"strength":"hard"},{"field":"date","op":"eq","value":"domani","strength":"hard"}],"depends_on":["g1"]}]}`

### d06

**Input:** Cercami ramen; se non ne trovi, va bene anche un ristorante coreano.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"ramen","type":"food"},"facts":[]},{"id":"g2","action":"find","target":{"name":"ristorante","type":"place"},"facts":[{"field":"cuisine","op":"eq","value":"coreana","strength":"hard"}],"depends_on":["g1"],"condition":{"goal":"g1","test":{"field":"result_count","op":"eq","value":0}}}]}`

### d07

**Input:** Cerco un hotel entro 90 euro, possibilmente con piscina.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"hotel","type":"accommodation"},"facts":[{"field":"price","op":"lte","value":90,"strength":"hard","currency":"EUR"},{"field":"amenity","op":"eq","value":"piscina","strength":"soft"}]}]}`

### d08

**Input:** Ho già il biglietto; trovami un hotel vicino al teatro.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"hotel","type":"accommodation"},"facts":[{"field":"location","op":"near","value":"teatro","strength":"hard"}]}],"context":[{"field":"ticket_owned","op":"eq","value":true}]}`

### d09

**Input:** Prenotami un tavolo.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"book","target":{"name":"tavolo","type":"place"},"facts":[],"missing":[{"field":"date","blocks":"action","reason":"unspecified"},{"field":"time","blocks":"action","reason":"unspecified"},{"field":"guests","blocks":"action","reason":"unspecified"}]}]}`

### d10

**Input:** Voglio arrivare a Torino prima delle 13 domani, spendendo il meno possibile.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"trasporto","type":"transport"},"facts":[{"field":"destination","op":"eq","value":"Torino","strength":"hard"},{"field":"arrival_time","op":"before","value":"13:00","strength":"hard"},{"field":"date","op":"eq","value":"domani","strength":"hard"}],"ranking":{"field":"price","direction":"asc"}}]}`

### d11

**Input:** La frase 'prenota un volo' è solo un esempio di comando, non una richiesta.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"non_actionable","goals":[]}`

### d12

**Input:** In teoria, quali dati servono per comprare un biglietto aereo?

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"non_actionable","goals":[]}`

### d13

**Input:** Trova una farmacia aperta oggi, ma non comprare nulla.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"farmacia","type":"place"},"facts":[{"field":"availability","op":"contains","value":"oggi","strength":"hard"}],"policy":{"forbid":["buy"]}}]}`

### d14

**Input:** Cerca un fotografo con almeno cinque anni di esperienza sotto i 400 euro.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"find","target":{"name":"fotografo","type":"professional_service"},"facts":[{"field":"experience","op":"gte","value":5,"strength":"hard","unit":"years"},{"field":"price","op":"lt","value":400,"strength":"hard","currency":"EUR"}]}]}`

### d15

**Input:** Ricordamelo domani.

**GoalContract 1.2:** `{"schema_version":"1.2","kind":"task","goals":[{"id":"g1","action":"notify","target":{"name":"promemoria","type":"reminder"},"facts":[{"field":"date","op":"eq","value":"domani","strength":"hard"}],"missing":[{"field":"referent","blocks":"action","reason":"unresolved_reference"}]}]}`


## Remaining issues and training readiness

1. **NOT_READY:** 16,989 source cases still need semantic review; training only on auto data would teach the 9B almost exclusively single-goal search.
2. Add reviewed train/validation cases for clarification, progressive execution, multi-goal dependencies, anaphora, context, explicit policy, negative/non-actionable requests and side effects. Define annotation guidelines and domain/provider input requirements first.
3. Contradictory/impossible requests currently fail semantic validation but lack a compact valid blocked-contract gold representation. Previous-conversation references need a real state ID service. Concrete capability metrics need a Discovery registry. These are explicit design gaps, not silently guessed labels.
4. Family-level leakage is not proven from V19; the near-duplicate audit is a conservative heuristic. Build final splits by semantic family after review, then retokenize and freeze hashes. The 12-case new holdout is too small for model selection.
5. A future fair V19→V20 model comparison must use identical requests and a version-neutral semantic gold projection, with V1.2-only abilities separately reported. Training loss is not a cross-schema quality metric.

**Training readiness: NOT_READY.** No Qwen 9B V20 training was started.
