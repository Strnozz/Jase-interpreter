# Jase Interpreter 9B V22 — benchmark

**Esito:** training e valutazione best/final completati senza OOM. V22 recupera parte delle regressioni V21, ma non raggiunge ancora la baseline storica né una qualità adatta all'esecuzione autonoma.

## Run

- Base: `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; warm start dall'adapter **finale V21**. QLoRA NF4 4-bit, BF16, LoRA r16/alpha32/dropout 0,05 su 124 moduli language-model; 21.639.168 parametri addestrabili (0,2242%).
- Dati V22: 3.907 train / 692 validation; seed 2028; 1.979 replay V20 distribuiti sulle dieci azioni, esempi V21 strutturali e template aggiuntivi. SHA-256 train `4cdd12f80147e3b7b5e13546a9d08dc180121bffe5c4600bf4c84c837fd76849`, valid `30387bbd8a896f457414b9555cf1d4ce82854d3d6a126d632e364caff18a28fc`. Massimo reale 498/512 token, nessun troncamento; loss solo sull'assistente.
- Training: 1.000 step, batch effettivo 4, LR 1e-5, warmup 50, cosine decay, AdamW; 74 min 45 s, 363,8 input token/s, peak VRAM allocata 14,09 GB, RAM processo 2,72 GB. Best validation loss 0,012935 allo step 600; finale 0,013329 allo step 1.000. Adapter circa 106,6 MB. Checkpoint resume a 25, 75 e ogni 200 step.

## Interpretazione completa sullo stesso gold V1.2

| Pannello | V20 9B | V21 finale | V22 best (600) | V22 finale (1000) |
|---|---:|---:|---:|---:|
| Hard dev | 1/16 | 6/16 | 7/16 | **7/16** |
| Holdout V20 | 1/12 | 6/12 | 5/12 | **6/12** |
| Storico convertibile | 181/198 | 169/198 | 175/198 | **178/198** |
| Schema valido hard dev | 14/16 | 15/16 | 13/16 | **15/16** |
| Schema valido holdout | 8/12 | 9/12 | 10/12 | **12/12** |
| Schema valido storico | 198/198 | 198/198 | 198/198 | **198/198** |

Il 2B storico proiettato su questi 198 casi è **186/198**; è una ricostruzione da output V1.1, non una nuova inferenza. V22 finale guadagna 20 casi e ne perde 11 rispetto a V21 finale (saldo +9); rispetto a V20 guadagna 7 e ne perde 10 (saldo −3). Sui casi storici l'accuratezza action sale da 93,03% V21 a 97,49% V22, ma resta sotto il 100% V20. Precisione/recall dei fatti V22: 96,96%/96,96%; V21 finale: 97,60%/97,39%. La validation loss preferisce lo step 600, mentre sui pannelli diagnostici il finale recupera più casi: la loss da sola non sceglie l'adapter più utile.

## Errori ancora aperti

- `Prenotami un tavolo.` resta senza `missing` per data, ora e ospiti. `Ricordamelo domani.` resta senza `missing: referent`. Sono omissioni rilevanti per un'azione futura.
- La catena ricerca → prenotazione di un risultato e alcuni fallback cibo → ristorante diventano un solo goal. Una ricerca treno → albergo conserva due goal ma inserisce una condizione inventata invece del riferimento alla stazione di arrivo.
- La richiesta del trasporto meno caro inventa `price < 0`; JSON e schema sono validi, ma il fatto è falso. Il Guard deve intercettare output semanticamente errati anche quando validi.
- Preferenze (`piscina`, vicinanza al centro), contesto del gruppo e divieti espliciti non sono sempre rappresentati correttamente. In un caso la policy viene posta al livello sbagliato del JSON; in un altro il divieto d'acquisto diventa un vincolo sul biglietto.
- Due gold del piccolo pannello V20 richiedono revisione: `s02` annota `contact` senza messaggio ma non marca `missing: message`, benché il contratto di design lo richieda; `d05` annota la prenotazione di pizzeria senza ora/ospiti né relativi `missing`. Non correggere silenziosamente questi gold: la serie storica qui usa gli stessi file. Un pannello nuovo e revisionato deve misurare anche la sicurezza delle informazioni mancanti.

Il corpus V22 contiene ancora molti template e un replay storico limitato. Hard dev e holdout contano solo 16 e 12 casi; sono stati consultati più volte, quindi non sono più un test indipendente per selezionare V23. Il frontend e gli adapter precedenti restano invariati. V23 userà più forme senza città, coppie contrastive per goal/contesto/policy e un nuovo pannello indipendente; il criterio sarà migliorare gli errori critici senza perdere il pannello storico.

Artefatti: `training/runs/qwen35-9b-v22/20260926T105104Z-581359c7/`, `benchmarks/outputs/v22-9b-best/`, `benchmarks/outputs/v22-9b-final/`, `data/v22_9b/manifest.json`.
