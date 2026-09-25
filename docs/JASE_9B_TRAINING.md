# Jase Interpreter 9B — training v19

## Run completato

- Run ID: `20260925T123205Z-21249614`
- Manifest, log e checkpoint: `training/runs/qwen35-9b/20260925T123205Z-21249614/`
- Stato rapido in PowerShell dalla root del repository: `.\scripts\training_status.ps1`; con `-Watch` aggiorna ogni cinque secondi e mostra `OK`, `RUNNING`, `INTERRUPTED` o `FAILED` per i run completi. `-IncludeSmoke` include anche i test iniziali.
- Modello: [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B), revisione `c202236235762e1c871ad0ccb60c8ee5ba337b9a`.
- Smoke che abilita questo run: `20260925T123008Z-f963a031`; tre update, caso massimo 327 token, loss finale 1,5239, validation loss 1,3083, nessun OOM o NaN, salvataggio e ricarica adapter con inferenza riusciti.

## Dati verificati

I file v19 in questa checkout hanno **27.504 train e 2.310 validation**. Il valore 2.413 nel prompt di avvio non corrisponde al file `data/v19_clean/valid.jsonl`. La validation effettiva contiene 2.143 prompt dopo la rimozione in memoria di sovrapposizioni col train, conflitti ed esempi ripetuti. Durante il training, una selezione deterministica di 64 esempi da questa validation viene usata per misure periodiche. I file sorgente non sono stati modificati.

Gli SHA-256 dei file sono nel manifest del run. Il tokenizer esatto del checkpoint è stato rieseguito su tutti i 27.504 train e sui 2.143 esempi validation effettivi: massimo 327 e 304 token rispettivamente, nessun troncamento con limite configurato a **512**. Il test di release rimane separato e non entra nel training o nella selezione del checkpoint.

## Ricetta

| Parametro | Valore |
|---|---:|
| Parametri originali del checkpoint | 9.653.104.368 |
| Quantizzazione | bitsandbytes NF4 4-bit, double quant |
| Compute | BF16, SDPA, gradient checkpointing |
| LoRA | rank 16, alpha 32, dropout 0,05, ultimi 16 layer testuali |
| Moduli | `down_proj`, `gate_proj`, `in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`, `k_proj`, `o_proj`, `out_proj`, `q_proj`, `up_proj`, `v_proj` |
| Parametri trainabili | 21.639.168 (0,2242% del checkpoint originale) |
| Batch e accumulo | 1 × 4 = 4 esempi per update |
| Durata programmata | 3.250 update, 13.000 esempi esposti, 0,4727 epoche |
| LR e schedule | 3e-5, warmup 100 update, cosine fino al 10% del LR massimo |
| Optimizer | AdamW dei soli parametri LoRA, weight decay 0,01 |
| Clipping | Norma massima 1,0 |
| Validation | 64 casi fissi ogni 250 update e alla fine; minimo della loss |
| Checkpoint | Ogni 100 update, più finale |

La struttura LoRA è stata scelta dopo aver ispezionato i moduli `Linear4bit` reali: 124 moduli negli ultimi 16 layer del language model, senza torre visiva. La profondità, il rank, l'alpha, il dropout, il numero di update e il batch effettivo seguono la baseline 2B dove ha senso; il modello resta quantizzato a 4 bit e viene addestrato solo l'adapter.

La loss è **completion-only**: nell'esempio di ispezione iniziale del run, 77 token di system+user hanno etichetta `-100` e 129 token della risposta assistant contribuiscono alla loss. `loss_mask_inspection.json` contiene anteprime brevi dei due tratti, senza stampare il corpus.

## Affidabilità e ripresa

Ogni checkpoint contiene adapter LoRA e `state.pt` con optimizer, scheduler, RNG, global step, tempo e storia della validation. Viene scritto in una directory temporanea e poi rinominato come `checkpoint-step-NNNNNN`, evitando checkpoint apparentemente completi dopo un crash durante il salvataggio. Non vengono salvate copie del modello base. I checkpoint precedenti non vengono cancellati automaticamente.

Il primo checkpoint reale (`checkpoint-step-000100`) è stato riletto con `torch.load(weights_only=True)`: global step 100, scheduler epoch 100, 248 slot optimizer, adapter da 86.594.112 byte e stato da 173.297.547 byte.

La ripresa è stata provata sul run reale: il worker iniziale è stato fermato dopo aver verificato `checkpoint-step-000600` (global step e scheduler a 600). Un nuovo processo Windows indipendente, avviato con `--resume-from-checkpoint`, ha ricaricato adapter e stato e ha proseguito dallo step 601. Gli output del processo ripreso sono in `resume-600.stdout.log` e `resume-600.stderr.log`.

Per riprendere un run interrotto da un checkpoint valido:

```powershell
$env:HF_HUB_OFFLINE='1'
.\.venv-nvidia\Scripts\python.exe training\train.py --model qwen35-9b --full --resume-from-checkpoint training\runs\qwen35-9b\20260925T123205Z-21249614\checkpoint-step-000100
```

Il resume verifica configurazione, hash dataset, prompt, schema e sorgenti. Dopo un'interruzione, scegliere l'ultimo `checkpoint-step-*` completo. Il log `steps.jsonl` conserva anche eventuali step tentati dopo l'ultimo checkpoint; il riepilogo finale usa l'ultima registrazione di ogni global step.

## Risorse

Ambiente: Windows 11, Ryzen 7 7800X3D, RTX 4080 SUPER (16 GiB), RAM 32 GiB, PyTorch 2.9.1+cu128. Lo smoke finale ha allocato al picco 13.217.885.184 byte e riservato circa 16,53 GB di VRAM PyTorch, senza OOM; throughput attivo circa 197 input token/s. Prima del run erano liberi 113,3 GB sul disco C:. I pesi originali sono già nella cache (19,3 GB); 33 checkpoint da circa 0,3 GB l'uno richiedono stimati meno di 10 GB. Sono presenti 1,39 GB di download parziali duplicati, non eliminati.

## Esito del training

Il run è completo a **3.250/3.250 update** (13.000 esempi, 0,4727 epoche), in 10.548,99 secondi (2 h 55 min 49 s). Loss train finale **0,0101643**; miglior validation loss **0,0106937** al passo 3.250 sui 64 casi fissi. La validation loss è scesa a ogni misura, da 0,04768 al passo 250, senza segni di overfitting in questa finestra. Throughput medio **206,5 input token/s** e **1,2622 esempi/s**. Il monitor GPU ha raccolto 637 campioni: utilizzo medio **97,1%**, picco memoria GPU fisica **15.994 MiB**, picco RAM di sistema **17,8 GiB**. Il picco allocato da PyTorch è stato **13.419.172.864 byte** e il picco riservato **16.538.140.672 byte**. Nessun OOM o NaN registrato.

L'adapter finale è in `training/runs/qwen35-9b/20260925T123205Z-21249614/adapter/` (106.605.970 byte per la directory; 86.594.112 byte per `adapter_model.safetensors`). È anche il best checkpoint selezionato. Sono conservati i checkpoint intermedi e finali con stato optimizer e scheduler; non è stato salvato né sovrascritto il modello base. Alla fine del run lo spazio libero sul disco C: era circa 104,25 GB.

La training loss non misura da sola la qualità Jase. Il benchmark su dev e il test di release valutano schema, intent, parametri e guard oltre alle prestazioni. Il tool/capability ID e un piano eseguibile non sono campi di GoalContract 1.1: queste metriche restano non disponibili finché il contratto non offre un target annotato.

## Dev benchmark, 246 casi v19-exact-unseen

Il test usa lo stesso file e gli stessi 246 casi del report storico 2B; la differenza di hash su Windows deriva da CRLF e gli hash coincidono dopo normalizzazione LF. Anche il testo del system prompt coincide dopo rimozione del newline finale. Il 2B storico girava su MLX/Mac e l'adapter non è presente su questo PC: le metriche sotto sono un confronto indicativo tra due runner, non un A/B rieseguito nello stesso processo.

| Metrica | 2B storico | 9B nuovo |
|---|---:|---:|
| Interpretazione completa | 92,68% (228/246) | 86,18% (212/246) |
| Intent/action corretto | 99,61% | 99,61% |
| Precisione fatti | 99,63% | 98,15% |
| Recall fatti | 98,35% | 97,26% |
| Contratti accettati ma sbagliati | 15 | 32 |
| Preferenze, successo | 79,49% | 64,10% |
| Tre o più fatti, successo | 87,38% | 74,76% |
| JSON / schema validi | non registrati direttamente | 100% / 100% |
| Generazione | non registrata in token/s | 15,26 token/s |
| Picco VRAM inferenza | non comparabile con MLX | 7,56 GiB |

Il 9B conferma la capacità di riconoscere l'action, ma peggiora soprattutto su assegnazione di fatti a preferenze/attributi/constraint e su richieste con molti fatti. La tassonomia registra 10 omissioni di fatti, 8 errori sul tipo del target, 4 allucinazioni e vari scambi tra bucket. Nei 7 probe non copiati dal train: 7/7 JSON validi, 6/7 schema validi; la richiesta multi-step crea un `gate` incompleto e inventa una condizione non richiesta. Una prenotazione senza dettagli essenziali viene accettata senza chiarimento. La richiesta di inviare una mail viene invece fermata dal guard con `CLARIFY`. Gli output grezzi sono in `benchmarks/outputs/jase9b-posttrain-probes.json`; le misure complete dev sono in `benchmarks/outputs/jase9b-v20-dev.json`.

## Release holdout, 27 casi

Il pannello di release è stato eseguito una sola volta dopo il dev, senza modificare l'adapter. Come sopra, i numeri del 2B provengono dal report storico MLX; il 9B usa il runner Windows Transformers. Nel 9B la precisione/recall dei fatti e l'accuratezza action sono calcolate sui **26 casi con schema valido**, perciò il confronto diretto con i 27 casi del 2B ha questa limitazione.

| Metrica | 2B storico | 9B nuovo |
|---|---:|---:|
| Interpretazione completa | 6/27 (22,22%) | 8/27 (29,63%) |
| JSON valido | 27/27 | 27/27 |
| Schema valido | non registrato direttamente | 26/27 |
| Action corretta | 93,10% | 89,29% |
| Precisione fatti | 66,67% | 58,62% |
| Recall fatti | 55,32% | 72,34% |
| Contratti accettati ma sbagliati | 15 | 16 |
| Casi con fatti allucinati | 40,74% | 73,08% sui 26 valutabili |

Il 9B riesce in **due casi in più**, ma su questo campione piccolo mostra anche più invenzioni e un contratto fuori schema. Le richieste con chiarimenti, anafora, più goal e numeri/ruoli restano fragili. Il report è in `benchmarks/outputs/jase9b-release-holdout.json`; il runner conserva aggregate e ID, senza salvare nel report le richieste e le predizioni del pannello sigillato.

## Problemi, fix e decisione

- **Dati:** la validation reale era 2.310 righe, non 2.413. Rimossi solo in memoria 167 esempi sovrapposti, duplicati o conflittuali; file v19 intatti.
- **Stabilità:** nessun OOM o NaN. La ripresa reale dallo step 600 ha verificato optimizer, scheduler e RNG. Non sono stati eliminati checkpoint o modelli 2B.
- **Prestazioni:** training GPU 97,1% medio. In inferenza su Windows mancano i kernel opzionali `causal_conv1d` e `flash-linear-attention`; Transformers usa fallback PyTorch corretti ma lenti. Il dev 9B ha prodotto 15,26 token/s e 7,56 GiB di picco VRAM PyTorch.
- **Qualità:** il 9B peggiora nettamente sul dev e migliora di soli due casi sul holdout, con più allucinazioni. Il `gate` malformato nel probe è un bug dell'output del modello, non del parser. Nessun fix post-test è stato applicato ai pesi o all'evaluatore.

**Raccomandazione:** conservare entrambi gli adapter e non sostituire il modello frontend. Prima di scegliere, serve un A/B con l'adapter 2B originale sullo stesso runner, ulteriori casi sigillati soprattutto per preferenze, multi-step e chiarimenti, e un controllo esplicito dei casi accettati ma semanticamente sbagliati. Il contratto 1.1 non rappresenta tool selezionato e piano eseguibile: richiede uno schema/benchmark dedicato prima di confrontare quelle capacità.
