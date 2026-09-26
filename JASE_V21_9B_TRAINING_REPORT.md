# Jase Interpreter 9B V21 — training e valutazione

**Esito:** training ed entrambe le valutazioni completati. V21 recupera alcune funzioni di GoalContract 1.2 assenti in V20, ma regredisce sui casi storici. **Non sostituire il modello del frontend.**

## Run

| Voce | Valore |
|---|---|
| Base | `Qwen/Qwen3.5-9B`, revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; 9.653.104.368 parametri del checkpoint |
| Hardware/stack | Ryzen 7 7800X3D, RTX 4080 SUPER 16 GiB, RAM 32 GiB; CUDA 12.8, PyTorch 2.9.1+cu128, Transformers 5.17.0, PEFT 0.21.0, bitsandbytes 0.50.2 |
| Metodo | Warm start dall'adapter V20; QLoRA NF4 4-bit, double quant, BF16, gradient checkpointing; LoRA r16, alpha32, dropout 0,05 su 124 moduli attention/MLP degli ultimi 16 layer |
| Parametri addestrabili | 21.639.168 (0,2242% del checkpoint) |
| Dati | `data/v21_9b`, 2.781 train / 437 validation, seed 2027; prompt e input mascherati dalla loss; max 512 token, lunghezza effettiva massima 432 |
| SHA-256 train / validation | `503285eb68dbae8c85c08cc6059e6c8b39cbdc7da89ee53c071e766643ee1ef0` / `cbb3ce4cdfb65197f9f3c89825abb12b2a76a87a4173be6c4efccb892660e0be` |
| Training | Batch 1, accumulo 4, effective batch 4, AdamW, LR 2e-5, warmup 40, weight decay 0,01, grad norm 1, 850 step (1,223 epoche) |
| Risorse | 67 min 12 s; 307 input token/s, 0,94 esempi/s; peak VRAM allocata 13,86 GB, peak RAM processo 2,73 GB; nessun OOM |
| Loss | Train finale 0,000135; migliore validation 0,012525 allo step 170; validation finale 0,012609 allo step 850 |
| Adapter | `best_adapter` step 170 scelto dalla validation loss; `adapter` finale step 850, 106.605.970 byte. Checkpoint con optimizer, scheduler e RNG per resume |

Smoke test, backward, salvataggio e ricarica dell'adapter sono riusciti. Le valutazioni sono state fatte ricaricando base e adapter. Il primo tentativo completo si è interrotto allo step 14 in coincidenza con un aggiornamento Windows/Codex e un evento driver NVIDIA; la causa precisa non è dimostrata. Il secondo tentativo è terminato. Aggiunti checkpoint precoci agli step 25 e 75 per limitare la perdita di lavoro. I kernel opzionali `causal_conv1d` e `flash-linear-attention` sono assenti: fallback PyTorch corretto ma più lento. Nessun peso V20/2B o frontend è stato modificato.

## Benchmark V1.2

`Interpretazione completa` richiede il contratto intero corretto, non solo JSON valido. V20, V21 best e V21 final hanno usato gli stessi pannelli e gold. Il best è lo step 170; il final lo step 850.

| Pannello | V20 9B | V21 best | V21 final |
|---|---:|---:|---:|
| Hard dev, 16 casi | 1/16 | 5/16 | **6/16** |
| Holdout, 12 casi | 1/12 | 5/12 | **6/12** |
| Storico convertibile, 198 casi | **181/198** | 164/198 | 169/198 |
| Schema valido, hard dev | 14/16 | 14/16 | 15/16 |
| Schema valido, holdout | 8/12 | 9/12 | 9/12 |
| Schema valido, storico | 198/198 | 195/198 | 198/198 |

Il 2B storico ottiene **186/198** sul sottoinsieme convertibile: è una proiezione documentata dagli output V1.1 archiviati, **non** una nuova inferenza sullo stesso host. I 212/246 del 9B V19 e 228/246 del 2B V19 riguardano un pannello e uno schema diversi e non sono direttamente confrontabili.

Sul pannello storico V21 final guadagna 6 casi rispetto a V20, ne perde 18 (saldo **−12**). La precisione/recall dei fatti sale da 96,05%/95,22% a **97,60%/97,39%**, ma l'accuratezza dell'azione scende da 100% a **93,03%**. Il 2B proiettato ha 98,24%/98,24% sui fatti. V21 final genera circa 18,16 token/s nella valutazione, con peak VRAM 8,18 GB; V20 circa 14,68 token/s. La differenza può dipendere da lunghezza degli output e prompt, quindi non misura da sola la velocità intrinseca.

## Casi osservati e difetti

- `Invia il messaggio a Giulia soltanto dopo che te lo confermo.` → V21 final emette `contact`, `missing: message` e `confirm_before: contact`: caso completo, dove V20 perdeva la policy.
- `La frase 'prenota un volo' è solo un esempio...` e una domanda teorica sui dati necessari per un biglietto → `kind: non_actionable`: entrambi corretti.
- `Prenotami un volo per Oslo.` → segnala `missing` per origine e data: corretto, senza inventarli.
- `Prenotami un tavolo.` → ancora nessun `missing` per data, ora e numero di persone. È un errore operativo importante.
- `Fammi vedere tre elettricisti..., poi contatta il primo solo se confermo.` → omette il secondo goal; conferma applicata al goal di ricerca. Perdita di dipendenza e azione.
- `Voglio arrivare a Torino... spendendo il meno possibile.` → inventa un volo e rappresenta un ranking di prezzo come confronto invalido (`lt` con stringa). JSON valido, schema invalido.
- Fallback sushi/thailandese → action `propose` fuori schema. Museo senza acquisto → `policy.reason` fuori schema. Taxi con alloggio già disponibile → inventa un divieto sugli alloggi. Questi sono errori del modello/dati, non richiedono un nuovo schema.
- `Compra il secondo dei libri che mi hai mostrato prima.` → segnala correttamente il referente non risolto, ma V1.2 non conserva l'ordinale né l'identificatore del risultato precedente. Qui esiste un limite reale del contratto.

Nel hard dev il recall di `missing` è **1/5**, delle dipendenze **2/4**, dei riferimenti **0/2**; nel holdout dipendenze **0/1** e riferimenti **0/1**. Campioni molto piccoli: indicano priorità diagnostiche, non tassi di errore di produzione. I dati V21 contengono 1.000 esempi V20 di rehearsal e molti template sintetici; i casi multi-goal storici non templated sono 38. Train loss quasi nulla e validation loss bassa non garantiscono generalizzazione. Lo step finale è migliore del best sui tre pannelli, ma la scelta post hoc su un holdout di 12 casi non costituisce selezione robusta: mantenerli entrambi come candidati sperimentali.

## Decisione e prossimi passi

V21 risolve diversi difetti strutturali di V20, ma **non è un interprete affidabile per ogni caso**: resta sotto V20 e sotto il 2B proiettato sul pannello storico. Tenere V20, V21 best/final e 2B separati. Prima di altro training: correggere etichette e ampliare esempi reali revisionati per `missing`, ordini, dipendenze, divieti e distinzione action/ranking; aggiungere regressioni storiche come gate; creare un holdout più ampio e indipendente, e un Guard V1.2 che rifiuti output validi ma semanticamente rischiosi. La proposta di evoluzione del contratto è in `docs/V22_CONTRACT_PROPOSAL.md`; non cambia schema, prompt o adapter V21.

Artefatti: `training/runs/qwen35-9b-v21/20260926T074448Z-f7d42b08/`, `benchmarks/outputs/v21-9b-posttrain/`, `benchmarks/outputs/v21-9b-final/`, `data/v21_9b/manifest.json`.
