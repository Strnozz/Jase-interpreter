# Provenance V1 for dry-run planning

The internal `jase/provenance_v1.py` records evidence for each model-emitted field/value without changing GoalContract 1.3. Sources are `USER_EXPLICIT`, `TRUSTED_CONVERSATION_CONTEXT`, `PROVIDER_RESULT`, `CAPABILITY_RESULT`, `DETERMINISTIC_DERIVATION`, `MODEL_INFERENCE`, and `UNKNOWN`.

An exact value appearing in the user's request gets a real character span. Trusted context and mock provider/capability results must match **both field and value** and carry explicit trust; their values never gain an invented user span. Limited deterministic rules cover open-now wording, non-stop flights and small number words. Everything else emitted by the model remains `MODEL_INFERENCE` unless another trusted source is supplied. A location such as `Como` that appears only in model output is ungrounded.

This layer proves only a limited form of grounding. Literal presence alone does **not** establish the correct semantic role: a date inside message content may be present but must still fail temporal-role validation. Quoted instructions may contain literal values but must still fail actionable-intent validation. Conversely, safe paraphrases or temporal normalizations may be held until a reviewed derivation rule exists. These conservative false holds must be measured in the replay; they must not be hidden by semantic repair.

No source grants permission to execute. The Planner must hold ungrounded execution-sensitive facts, unresolved references, unsupported aliases and missing required slots, and its final `execution_permitted` value stays `false`.
