# Jase Interpreter 9B V20 — training e valutazione

**Esito:** training completato; **non promuovere** questo adapter come Interpreter principale. Il modello impara bene i contratti semplici migrati, ma non generalizza alle nuove funzioni di GoalContract 1.2.

## Run e ambiente

| Campo | Valore |
|---|---|
| Modello | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Parametri checkpoint | 9.653.104.368 |
| GPU | RTX 4080 SUPER, 16 GiB; CUDA 12.8; PyTorch 2.9.1+cu128 |
| CPU/RAM | Ryzen 7 7800X3D / 32 GiB |
| Quantizzazione | bitsandbytes 4-bit NF4, double quant, BF16, SDPA |
| Dataset del run | `data/v20_9b`, 19.948 train / 1.411 validation; V19 clean → GoalContract 1.2; seed 2026; max 512 token |
| Hash train / valid | `aa63a1c96ce9f584b2a7f876deb0edfb59d17efcb51a57e44399309e9ca54e89` / `1ec2825767f18dac5db9bbf3cd09f1a920fc4a318bd708f6b8e09813d545de23` |
| Lunghezza massima verificata | train 298, validation 290 token; nessun troncamento |
| LoRA | r=16, alpha=32, dropout=0,05; 124 moduli lineari nei 16 ultimi layer del language model (attention e MLP) |
| Parametri addestrabili | 21.639.168, pari allo 0,2242% del checkpoint |
| Training | batch 1, accumulo 4, effective batch 4; 5.000 step ≈ 1,003 epoche; LR 3e-5, warmup 100, cosine decay, AdamW, weight decay 0,01, grad norm 1 |
| Durata / throughput | 5 h 6 min; 208 token input/s; 1,10 esempi/s |
| Memoria | peak allocato GPU 13,38 GB; peak RAM processo 2,96 GB; nessun OOM |
| Loss | ultima train 0,0101; migliore validation 0,00565 allo step 5.000, su campione fisso di 64 casi |
| Checkpoint | ogni 250 step; adapter finale 106,6 MB; optimizer, scheduler e RNG salvati per resume |

Lo smoke test di 3 step ha completato backward, salvataggio e ricarica dell'adapter in un nuovo processo. Il test post training ha nuovamente caricato base 4-bit e adapter, generando i risultati sotto `benchmarks/outputs/v20-9b-posttrain/`. Nessun peso del 2B o del precedente 9B è stato modificato. Il trainer calcola la loss solo sui token assistant; il prompt e l'input sono mascherati, come documentato in `loss_mask_inspection.json` nel run.

## Risultati

| Pannello | Casi | JSON valido | Schema valido | Interpretazione completa |
|---|---:|---:|---:|---:|
| V20 hard dev | 16 | 16/16 | 14/16 | **1/16** |
| V20 holdout sigillato | 12 | 12/12 | 8/12 | **1/12** |
| Dev storico V19 convertibile in V1.2 | 198 | 198/198 | 198/198 | **181/198 (91,41%)** |
| 2B storico sul medesimo sottoinsieme, proiettato in V1.2 | 198 | non ricostruibile per tutti i raw output | 196/198 | **186/198 (93,94%)** |

Sul sottoinsieme condiviso, il 9B V20 riconosce l'action in 198/198 casi contro 195/196 casi validi per il 2B proiettato. Tuttavia fa più errori nei fatti: precisione/recall **96,05%/95,22%** contro **98,24%/98,24%** del 2B. Guadagna 3 casi che il 2B sbagliava e ne perde 8 che il 2B risolveva, saldo −5. Il 9B V19 precedente otteneva 212/246 (86,18%) e il 2B 228/246 (92,68%) sul pannello V19 completo: quei valori hanno un denominatore e uno schema diversi, quindi non provano un miglioramento del V20 rispetto al 9B V19.

**Metodo del confronto 2B:** l'adapter MLX del 2B non è presente su questo PC. Il report storico `reports/v20/baseline-dev.json` conserva l'output per i fallimenti; per i successi, il contratto gold equivale all'output ai fini della metrica completa. Ho convertito in V1.2 i 198 gold senza gate, clarification o multi-goal e i relativi output falliti quando validi. Questo è un confronto ricostruito, non una nuova inferenza 2B. La proiezione elimina alcune distinzioni V1.1 e non misura le nuove funzioni V1.2.

## Difetti osservati

- `Prenotami un tavolo.` → JSON valido con `book`, ma senza `missing` per data, ora e ospiti. È un'omissione critica per un'azione con effetti esterni.
- `Mostrami quattro pizzerie e prenota la terza domani.` → un solo goal `book`: perde ricerca, dipendenza e selezione ordinale.
- `Invia il messaggio a Giulia soltanto dopo che te lo confermo.` → inventa action `send` (fuori schema) e omette `confirm_before`.
- Una frase citata come esempio di comando viene trattata come richiesta attiva. I casi non actionable falliscono.
- La richiesta di un fotografo con esperienza e prezzo (hard dev) e quella di un tablet con memoria e prezzo (holdout) sono interpretate correttamente. Sono entrambe richieste semplici, non verificano le nuove capacità.

I pannelli V20 mostrano **zero recall** per missing, context, dependencies e soft facts dove misurabili. Il dataset del run contiene dieci azioni ma solo goal singoli: nessun esempio train/validation con missing, policy, dependency, condition, reference, ranking, context o non actionable. La bassa validation loss è quindi coerente con i dati migrati, non con la qualità sui nuovi casi. I pannelli V20 sono piccoli e diagnostici; non bastano a misurare un tasso di errore di produzione.

## Migliorie e problemi tecnici

- Creato `data/v20_9b` separato: conversione conservativa con Guard V1.1 ACCEPT e validazione V1.2, incluse dieci azioni invece del solo `find`. `data/v20` originale e `data/v19_clean` restano invariati.
- Trainer aggiornato per verificare l'hash del system prompt scelto dalla config e la corrispondenza del prompt in ogni riga; smoke e resume sono legati a config, dataset e codice esatti.
- Aggiunti avvio staccato, stato PowerShell, test di ricarica e valutatore V1.2 con risultati salvati. Nessun OOM o crash durante il run.
- I kernel opzionali `causal_conv1d` e `flash-linear-attention` non sono installati: Transformers usa implementazioni PyTorch corrette ma più lente. Non è stato necessario cambiare precisione o lunghezza.

## Decisione

Il 9B V20 **peggiora di 5/198 casi** rispetto al 2B storico sul sottoinsieme condiviso e **non ha appreso** le funzioni centrali dello schema 1.2. Preservare gli adapter attuali; non sostituire il frontend. Prima di un altro training servono etichette revisionate per multi-goal, missing, policy/confirmation, non actionable, references e context, più un benchmark più ampio e una valutazione A/B sullo stesso contratto e sullo stesso host. Non basta prolungare questo run: la validation loss non segnala l'assenza di tali esempi.

Artefatti: `training/runs/qwen35-9b-v20/20260925T190927Z-0468b634/`, `benchmarks/outputs/v20-9b-posttrain/`, `data/v20_9b/manifest.json`.
