# Audit preliminare — challenger Granite 4.1 8B

Data: 2026-09-27. Questo audit precede qualsiasi modifica alla pipeline per Granite.

## Identità del challenger

Il modello instruct ufficiale è [`ibm-granite/granite-4.1-8b`](https://huggingface.co/ibm-granite/granite-4.1-8b), anche se l'ID non contiene `instruct`. La [model card IBM](https://huggingface.co/ibm-granite/granite-4.1-8b) lo chiama esplicitamente modello instruct e mostra l'uso di `AutoModelForCausalLM`, `AutoTokenizer` e `apply_chat_template`. La revisione esatta va ancora congelata prima del download dei pesi. Non usare `granite-4.1-8b-base` come challenger.

## Pipeline attuale

| Componente | Implementazione | Specificità / intervento necessario |
| --- | --- | --- |
| Config | `configs/models/*.json`, letti da `jase/multimodel.py` | Già nominative, ma senza strategia chat/loader/decoding/LoRA esplicite. |
| Load train | `training/train.py:load_base` | Solo `AutoModelForMultimodalLM`, NF4/bf16/SDPA, device CUDA 0, pad=eos. Selezione classe da config necessaria. |
| Target LoRA | `select_language_linears` | Regex sui layer `language_model`/`text_model` di Qwen multimodale, ultime 16 layer; Granite richiede ispezione dei nomi reali e policy distinta. |
| Training | `training/train.py` | QLoRA 4-bit, gradient checkpointing, AdamW, cosine con warmup, accumulation 4, batch 1, 512 token, checkpoint atomici/resume, validation su 256 esempi. Log per step. |
| Dati | `data/v26_9b/{train,valid}.jsonl` | 6026/1087 esempi, stessi messaggi semantici per entrambi. SHA train `54c5e512...`, valid `cc748b13...`. Nessun duplicato/cross-split nel precedente audit. |
| Tokenizzazione/loss | `jase/multimodel.py:encode_supervised` | Prefix user/system mascherato con `-100`; target assistant attivo. `enable_thinking=False` è passato al chat template e può essere Qwen-specifico. La formattazione specifica del modello è ammessa, ma non la modifica delle richieste o del GoalContract gold. Nessuna troncatura. |
| Prompt | `V26_SYSTEM_PROMPT.txt` | Deve restare semanticamente identico per il confronto. Il dataset incorpora quel prompt in ogni riga. |
| Inferenza | `evaluation/run_semantic_v1_3_9b.py` | Ricarica solo adapter PEFT da manifest, usa `load_base`, greedy `do_sample=False`, `max_new_tokens=480`, `enable_thinking=False`, decodifica raw senza repair. Occorre supportare anche base vanilla e modello causale Granite con stessa policy. |
| Validazione/metriche | `jase/semantic_contract_v1_3.py`, `jase/canonical_semantics.py`, `jase/metrics_v1_3.py` | Indipendenti dal foundation model; mantenerli congelati nel confronto. Exact, field-level, projection hold e status Guard sono separati. |
| Guard/Planner | `jase/semantic_guard_v1_3*.py`, `jase/planner_handoff_v1_3.py` | Guard V4/V5 sperimentali; handoff mock isolato, `execution_permitted=false`. Nessun registry/capability provider reale. Non cambiare comportamento in questo esperimento. |
| Report | `benchmarks/outputs/*/{cases.jsonl,summary.json}` e report markdown | Runner restartable, pannello hashato, raw JSON e output/s/VRAM. Serve tabella paired e audit manuale del rischio. |

## Baseline e limite causale

Qwen V26 best è `training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c/best_adapter` (step 1200), quantizzazione NF4/bf16, 21.639.168 parametri trainabili, 1500 step di run, seed 2032. **V26 è warm-start da V25**, non un training dalla base Qwen sul solo corpus V26. Il run V26 ha circa un'epoca del corpus V26, ma l'adapter incorpora apprendimento precedente da V25 (e la sua genealogia). Addestrare Granite dalla base solo su V26 con 1500 step produce un confronto fra *candidati operativi con storie di training diverse*, non una stima causale isolata dell'effetto del foundation model. Riportare sempre questa limitazione. Non riscrivere V26 o riaddestrare Qwen solo per nasconderla.

## Refactor minimo previsto

1. Aggiungere nei nuovi config strategia di loader/chat/LoRA/decoding senza cambiare il config V26. Default legacy deve riprodurre Qwen esattamente.
2. Scegliere classe `AutoModelForCausalLM` per Granite; ispezionare i `Linear4bit` effettivi prima di fissare target module e scala parametri.
3. Consentire al tokenizer Granite di usare il suo chat template, mantenendo prefix supervision/inferenza identici e `-100` solo sul target. Audit completo delle lunghezze; non ridurre 512 o troncare dati silenziosamente.
4. Estendere evaluator a base vanilla e adapter, preservando file raw, stesse metriche e stesso prompt semantico.
5. Congelare V33 prima del training Granite; ottenere revisione umana del gold se disponibile. L'assenza di tale revisione va dichiarata, non sostituita con auto-conferma.

Nessun benchmark storico, dataset, adapter, Guard, Planner o frontend deve essere modificato.
