# Jase Interpreter — Qwen V26 vs Granite 4.1 8B Challenger

27 settembre 2026. Esperimento **completo**; nessun modello è stato promosso nel frontend e nessuna azione verso provider è stata abilitata.

## Confronto controllato

Il pannello V33 di 100 richieste è stato congelato prima del download e del training Granite (SHA-256 `4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37`). Entrambi hanno usato il medesimo corpus V26 (6026 train, 1087 validation), GoalContract 1.3, system prompt, evaluator, Guard, greedy decoding e massimo 480 nuovi token. I 100 gold V33 sono stati scritti da un agente e **non** revisionati da un umano indipendente; 19 sono già segnalati per revisione. Nessun caso V32/V33 è entrato nel training.

| V33 blind, 100 casi | Qwen3.5-9B V26 best | Granite 4.1 8B finale |
| --- | ---: | ---: |
| JSON valido | 96 | 95 |
| Schema valido | 84 | 70 |
| Contratto canonical exact | **27** | 25 |
| Action corrette | **87/111** | 69/111 |
| Target corretti | **68/111** | 60/111 |
| Fact recall | **99/151** | 88/151 |
| Temporal recall | **43/68** | 32/68 |
| Policy corrette | **75** | 60 |
| Guard ACCEPT | 80 | 65 |
| ACCEPT ma non exact | 54 | 42 |
| Exact trattenuti dal Guard | 1 | 2 |
| Output token/s | 18,06 | **21,15** |
| Picco VRAM inferenza | 7,62 GiB | **4,87 GiB** |

La differenza di exact è piccola: **18** casi corretti da entrambi, **9** solo da Qwen, **7** solo da Granite, **66** da nessuno. Sul V32 storico (30 casi), Qwen è 10 exact e Granite 8; è solo una diagnostica già consultata. Granite base senza adapter produceva JSON formalmente parseable in 98/100 V33 ma **0** contratti validi secondo lo schema. Il fine-tuning ha quindi insegnato il formato, senza raggiungere la robustezza richiesta.

## Rischio operativo e regressioni

Ho esaminato tutti i **54** ACCEPT non exact di Qwen e i **42** di Granite contro richieste e gold. La [revisione per caso](evaluation/reviews/v33_qwen_granite_operational_review_draft.json) conserva richiesta, gold, output e classificazione. È una revisione dell'agente, **non** una validazione umana indipendente né una stima di rischio in produzione. Con criterio conservativo, almeno **27** ACCEPT Qwen e **28** ACCEPT Granite contengono un errore materiale o di policy evidente; rispettivamente 24 e 10 restano da giudicare con semantica del Planner o revisione umana, mentre 3 e 4 paiono equivalenze innocue. La minore copertura del Guard con Granite non equivale a maggiore sicurezza: gli errori chiari accettati non diminuiscono.

Esempi verificabili:

- `v33-003`: Qwen interpreta una frase inglese **citata per traduzione** come richiesta di noleggio e inventa `Como`; Granite non passa il Guard. `v33-012`: Granite inventa una distanza di **1 km** per «a piedi».
- `v33-020`: entrambi omettono il goal di prenotazione condizionato al consenso. `v33-041`: entrambi accettano `hire` senza segnalare il provider ancora da scegliere.
- `v33-065`: Qwen conserva il ruolo `arrival_time` ma interpreta «essere a Lecce alle 09:30» come uguaglianza invece di arrivo **entro** le 09:30; Granite lo mette addirittura in `search_date`.
- `v33-074`: Granite trasforma il giovedì **contenuto nel messaggio** in data dell'azione e perde la conferma prima dell'invio. `v33-090`: Granite inventa un goal `cancel` in una ricerca condizionale di mezzi a noleggio.
- Granite migliora alcuni casi di ordinamento (`v33-059/060`) e riprogrammazione (`v33-033/035`); Qwen conserva meglio treni con vincoli temporali (`v33-062/064`) e più ricerche indipendenti (`v33-078/079`).

I mismatch exact comprendono anche sinonimi e differenze di normalizzazione. Il repository non dispone ancora di un vero registry capability/Planner 1.3 che convalidi ogni alias o mapping; perciò i casi incerti restano tali. Nessun output di benchmark è stato usato per modificare Guard, contratto o gold.

## Costo e provenienza

| Full run, 1500 step nominali | Qwen V26 | Granite V26 challenger |
| --- | ---: | ---: |
| Durata | 119,9 min | **50,2 min** |
| Input token/s | 338,7 | **821,2** |
| Picco VRAM training | 13,10 GiB | **6,56 GiB** |
| Parametri LoRA trainabili | 21.639.168 | 19.791.872 |
| Adapter | 101,7 MiB | **82,4 MiB** |
| Best step per validation loss | 1200 | 1500 |
| Best validation loss | 0,01378 | 0,02733 |

Le validation loss **non** sono direttamente confrontabili tra tokenizer e foundation differenti. Granite best e final sono lo stesso adapter step 1500. Nessun CUDA OOM: smoke, backward, salvataggio e reload adapter sono passati; checkpoint intermedi e finale sono presenti. Il dato di picco VRAM training proviene dal processo PyTorch, non dall'intero sistema.

Qwen: `Qwen/Qwen3.5-9B` revisione `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, run `training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c`. Granite: `ibm-granite/granite-4.1-8b` revisione `1504002f650e656a0a3789d99574df12e3e94ed0`, run `training/runs/granite41-8b-v26-challenger/20260927T180708Z-cd7fcd66`. Seed 2032, NF4 4-bit, BF16, LoRA r16/alpha32/dropout 0,05 sulle ultime 16 layer, batch 1 × accumulation 4, learning rate 8e-6, max length 512. Ambiente: RTX 4080 SUPER 16 GB, CUDA 12.8, PyTorch 2.9.1+cu128, Transformers 5.17.0, PEFT 0.21.0, bitsandbytes 0.50.2.

**Limite causale:** Qwen V26 parte dall'adapter V25, Granite da base instruct. Stesso corpus e budget nominale non isolano l'effetto del solo foundation model. Anche il costo Qwen riportato è del solo run V26, non del curriculum cumulativo V25→V26.

## Decisione

**Qwen V26 resta il candidato corrente.** Granite è molto più leggero e veloce su questo hardware, ma sul blind V33 perde in validità di schema, action, fatti, vincoli temporali e policy; non mostra un vantaggio dimostrato nel rischio operativo. Nessuno dei due è pronto per esecuzione autonoma: gli ACCEPT con errori materiali sono numerosi. Prima di una scelta per produzione servono revisione umana dei gold e degli errori, un mapping Planner/capability verificabile e un pannello realistico indipendente. Su richiesta dell'utente, l'esperimento si ferma qui: **nessun'altra iterazione o training**.
