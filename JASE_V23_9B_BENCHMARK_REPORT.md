# Jase Interpreter 9B V23 — benchmark

**Esito:** V23 migliora il vecchio hard dev, ma **non migliora il nuovo pannello transfer** e perde casi storici rispetto a V22. Non promuovere al frontend.

## Run e metodo

- Base `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; warm start dall'adapter finale V22. QLoRA NF4 4-bit, BF16, LoRA r16/alpha32/dropout 0,05 sugli ultimi 16 layer testuali: 21.639.168 parametri addestrabili (0,2242%).
- Dataset `data/v23_9b`: 4.411 train / 829 validation, seed 2029. Conserva V22 e aggiunge 504 esempi mirati su richieste senza città, azioni concatenate, riferimenti, ranking, contesto, policy e informazioni mancanti. SHA-256 train `e41f542e39ed98a8b67ae46631f6ff1ffe16f45f6906b7e97dad3ca3dca8c9ca`, valid `e1c6b48e50f75e89467fd0cafd18cbc14238d0af97bd36549787b41f378f60f5`; massimo 448/512 token, nessun troncamento, loss solo sui token assistant.
- Training: 1.100 step (circa 1 epoca), batch effettivo 4, LR 8e-6, warmup 50, AdamW/cosine; 65 min 43 s, 403 input token/s, 1,165 esempi/s, peak VRAM allocata 13,91 GB, RAM processo 2,65 GB. Nessun OOM. Best validation loss 0,015083 **allo step finale 1.100**; adapter circa 106,6 MB. I checkpoint per resume sono conservati.
- Il pannello `benchmarks/v23/transfer.jsonl` di 30 casi è stato congelato **prima** del build del dataset V23; SHA-256 `c2aa1f4cef1b9734b91aed4996b7b89b5f3686d8d527cdb4ff9bca481fd9455b`. È mirato alle funzioni critiche, non un campione casuale di richieste di produzione. V22 è stato valutato sullo stesso pannello con base e adapter ricaricati.

## Risultati: interpretazione completa

| Pannello | V20 | V21 finale | V22 finale | V23 finale |
|---|---:|---:|---:|---:|
| Storico convertibile | 181/198 | 169/198 | **178/198** | 172/198 |
| Hard dev già consultato | 1/16 | 6/16 | 7/16 | **10/16** |
| Holdout V20 già consultato | 1/12 | 6/12 | 6/12 | 6/12 |
| Transfer V23 | — | — | **14/30** | **14/30** |

La baseline 2B storica è **186/198** sui convertibili, ricostruita da output V1.1 e non rieseguita. Sullo storico V23 guadagna 4 casi e ne perde 10 rispetto a V22 (saldo −6). L'accuratezza action scende da 97,49% a 96,97%; precisione/recall dei fatti da 96,96%/96,96% a 96,26%/96,05%. Sul transfer V23 guadagna un caso e ne perde uno rispetto a V22. Schema valido: storico 197/198 (V22 198/198), hard dev 15/16 (uguale), holdout 10/12 (V22 12/12), transfer 28/30 (V22 29/30).

La cartella `v23-9b-best` **non contiene un checkpoint diverso**: la validation loss minima è al passo finale, quindi il manifest ha selezionato `adapter`. La pipeline ha valutato due volte lo stesso adapter in `v23-9b-best` e `v23-9b-final`; non sono due repliche indipendenti. Questo spreco di tempo va corretto nella prossima iterazione.

## Errori osservati

- Migliorano tre casi hard dev: catena pizzerie→prenotazione, ranking trasporto generico e contesto del gruppo. Sul nuovo transfer la catena autobus→hotel con riferimento all'arrivo diventa corretta.
- `Prenotami un tavolo.` e `Ricordamelo domani.` restano **senza missing**. Nel holdout `Prenotami un volo per Oslo.` perde i `missing` per origine e data: regressione che può rendere insicura un'azione se il runtime interpreta il contratto come pronto.
- Il transfer fallisce su fatti/entità anche con struttura corretta: `agriturismo ... con parcheggio` usa `contains` invece di `eq`; `pullman ... prima delle 17` usa `time` invece di `arrival_time`; `Monitora il prezzo del tablet X` perde `X`. Sul contesto `Siamo in sei, senza automobile` sposta persone/auto nei filtri del ristorante anziché in `context`.
- Due output transfer sono schema-invalid: condizione inventata con rating 4,5 per contattare una libreria, e un `context` con struttura non consentita. Nel holdout la conferma finisce direttamente in un goal senza `policy`, oppure viene inventato `confirm_before: true`. Sono errori del modello, non limiti di lunghezza o memoria.
- Sul transfer recall `missing` e dipendenze è 100% nei pochi casi misurati, ma il contesto è 0/2. Sul hard dev `missing` resta 1/5. Questi denominatori sono troppo piccoli per stimare affidabilità di produzione.

## Decisione

V23 **non supera V22** come candidato generale. Tre training successivi con template sintetici mostrano oscillazione fra categorie, senza aumento sul transfer congelato. Continuare semplicemente con un altro LoRA simile consumerebbe tempo e rischierebbe di adattarsi ai pannelli già letti. La prossima iterazione deve introdurre un **Guard GoalContract 1.2** con esito `ACCEPT/REVIEW/REJECT` e misurare soprattutto contratti sbagliati accettati, copertura e astensioni su un nuovo pannello separato; poi servono etichette reali revisionate da persone per i casi ambigui e un benchmark più ampio. Lo schema V1.2 e il frontend restano invariati fino a una valutazione convincente.

Artefatti: `training/runs/qwen35-9b-v23/20260926T130732Z-b8c7a29b/`, `benchmarks/outputs/v23-9b-final/`, `benchmarks/outputs/v22-9b-transfer/`, `data/v23_9b/manifest.json`.
