# Jase Interpreter: fase multi-model

## Ambiente misurato (Windows, 25 settembre 2026)

| Componente | Misura |
|---|---|
| OS | Windows 11, build 26200 |
| CPU | AMD Ryzen 7 7800X3D, rilevato dal registro Windows |
| RAM | 33.514.676.224 byte (31,2 GiB) |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 17.170.956.288 byte (15,99 GiB) VRAM |
| Driver / CUDA indicata dal driver | 596.49 / 13.2 |
| PyTorch / runtime CUDA | 2.9.1+cu128 / 12.8 |
| CUDA reale | `torch.cuda.is_available() = True`, matmul GPU passato, bf16 disponibile, compute capability 8.9 |
| Python / pip / Git | 3.12.14 / 25.0.1 / 2.55.0.windows.5 |
| Disco libero prima del download modello | circa 124 GiB su C: |

`docs/MULTIMODEL_HARDWARE.json` contiene lo snapshot leggibile dalla macchina. L'ambiente Python è dedicato al progetto; non è stato modificato il runtime Python generale.

## Pipeline ereditata

- Il codice è in `jase/`, `scripts/`, `configs/`, `schema/`, `benchmarks/`.
- Il candidato 2B ufficiale di questa fase è `mlx-community/Qwen3.5-2B-4bit` con `adapters/jase-v19-cap16`. Il repository contiene il **riferimento** all'adapter e i report, ma sulla postazione Windows la directory dell'adapter non è presente. Il modello e l'adapter MLX non sono un checkpoint PEFT per PyTorch.
- MLX carica modello e tokenizer con `mlx_lm.load`. Training in `configs/mlx_lora_v19_capacity16.yaml`: LoRA 16 layer, rank 16, scale 32, dropout 0.05, batch 1, accumulo 4, 13.000 microbatch / 3.250 update, AdamW, LR cosine 3e-5 → 3e-6, seed 2026, sequenza massima 512, gradient checkpointing e loss mascherata sul prompt. Sono circa 0,473 epoche sul train v19. Il dataset originale è sintetico e versionato; `scripts/generate_dataset_v2.py` produce conversazioni `system/user/assistant` con GoalContract 1.1 nell'ultimo messaggio.
- Il system prompt è `TRAIN_SYSTEM_PROMPT.txt`. Il tokenizer applica il chat template del checkpoint con `enable_thinking=False`. In `scripts/run_v20_experiment.py` il prefisso usato dalla loss e quello di inferenza sono stati verificati uguali. La nuova pipeline rifà la verifica su ogni esempio prima del training.
- Il target è `schema/goal_contract_v1_1.schema.json`: goal, action, target, attributes, constraints, preferences, clarification e gate. Non contiene ancora un piano eseguibile o un ID di tool. I tool sono scelti più avanti nel Capability Resolver; il benchmark non deve attribuire al modello metriche inesistenti.
- L'inferenza Mac è in `scripts/infer_mlx.py` e `scripts/interpret.py`. Quest'ultimo usa `jase/canon.py` e `jase/guard.py`, con retry solo per guasti strutturali. `scripts/eval_v2.py` confronta output raw e Guard tramite `jase/metrics.py`. `gold_v1` è regressione di sviluppo; `release_holdout_v1` è il test di release, piccolo e sigillato.
- I milestone precedenti sono documentati in `docs/HYPOTHESES.md`, `docs/ITERATIONS.md`, `docs/EXPERIMENT_V20.md` e `INTERPRETER_RELEASE.md`. La v20 è stata valutata, ma non promossa automaticamente: i criteri predefiniti non sono tutti soddisfatti. `reports/BEST.json` non viene modificato.

### Baseline 2B già misurata sul Mac

| Fonte | Risultato storico |
|---|---:|
| `reports/v20/baseline-dev.json`, pannello `v19-exact-unseen` | 246 casi, semantic exact 0,9268, accepted but wrong 15 |
| `reports/v20/baseline-gold.json` | 77 mandatory corretti su 85, accepted but wrong 10 |
| `reports/holdout-v19cap16-s2026.json` | 6 mandatory corretti su 25, semantic exact 0,2222, accepted but wrong 15 su 27, validità JSON 1,0 |

Questi sono report preesistenti, non un nuovo benchmark Windows. L'adapter corrispondente manca in questa copia e va ripristinato per misure A/B controllate.

## Scelta 9B e compatibilità

[`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) è il checkpoint ufficiale post-trained, Apache-2.0, circa 19,3 GB. Per il task attuale, che fornisce conversazioni instruction-to-JSON con chat template e system prompt, il post-trained conserva la condizione iniziale della baseline 2B. [`Qwen/Qwen3.5-9B-Base`](https://huggingface.co/Qwen/Qwen3.5-9B-Base) esiste, ma sarebbe un esperimento separato: non permette di attribuire al numero di parametri il cambiamento di qualità.

Il 9B è multimodale. La nuova ricetta usa `AutoModelForMultimodalLM` e quantizzazione bitsandbytes NF4 a 4 bit, doppia quantizzazione, compute bf16, SDPA, gradient checkpointing e LoRA sui moduli lineari degli ultimi 16 layer del language model, come profondità della baseline. La configurazione del training completo è stata portata a 512 token su richiesta successiva; il corpus verificato arriva a 327 token. Il trainer interrompe il run se non può identificare quei moduli con certezza: non addestra la torre visiva per caso. Il run completo è descritto in `docs/JASE_9B_TRAINING.md`.

## Dati e confronto

La configurazione 9B usa **lo stesso train v19_clean della baseline 2B**, non i dati v20. Il train originale resta immutato. I file hanno 27.504 train (24.489 prompt distinti) e 2.310 validation (2.148 prompt distinti). L'audit ha trovato 4 prompt della validation anche nel train, 160 righe duplicate da filtrare e 1 prompt validation con etichette in conflitto. Il nuovo trainer filtra questi elementi **solo dalla validation**, ottenendo 2.143 esempi effettivi; la baseline già addestrata non viene toccata. Le 3.015 ripetizioni del train storico restano presenti in entrambi gli esperimenti e sono un limite del corpus condiviso. La versione v19 è identificata dagli hash dei file; il comando originale di generazione non è stato conservato, quindi il seed di generazione non è dimostrabile (il default del generatore è 2026). Il seed di training 2026 è esplicito nelle configurazioni.

Il pannello comparativo v20 contiene 256 richieste (253 prompt distinti). Dieci righe, corrispondenti a 8 prompt distinti, sono etichettate `v19-exact-seen` e vengono escluse per impostazione predefinita: 246 richieste per il confronto principale. Il benchmark di release ha 27 richieste, senza overlap testuale esatto con il train v19. Entrambi sono troppo piccoli o vicini al processo di sviluppo per proclamare un vincitore generale. Il manifest con conteggi, distribuzioni e hash è `docs/MULTIMODEL_DATA_MANIFEST.json`; riporta sia l'hash dei byte locali sia quello con newline LF, perché `core.autocrlf=true` modifica i byte della checkout Windows.

## Esecuzione

L'ambiente isolato è `.venv-nvidia` (ignorato da Git). Su Windows con Python 3.12:

```powershell
& 'C:\Users\jerry\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv-nvidia
.\.venv-nvidia\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.\.venv-nvidia\Scripts\python.exe -m pip install -r requirements-multimodel.txt
.\.venv-nvidia\Scripts\python.exe scripts\hardware_probe.py
.\.venv-nvidia\Scripts\python.exe scripts\model_preflight.py --model qwen35-9b
.\.venv-nvidia\Scripts\python.exe scripts\token_preflight.py --model qwen35-9b
.\.venv-nvidia\Scripts\python.exe scripts\download_model.py --model qwen35-9b --http-resume
```

Su un altro host sostituire il percorso dell'interprete nella prima riga con un proprio Python 3.12. La verifica hardware deve confermare `torch.cuda.is_available()` e una moltiplicazione reale di matrici sulla GPU. `model_preflight.py` blocca owner/licenza diversi da quelli attesi e stampa il commit preciso; `download_model.py` usa un solo snapshot nella cache. `--http-resume` riprende i singoli shard via HTTP Range e ne verifica lo SHA-256: su questa postazione il trasferimento Xet non ha completato i pesi.

```powershell
.\.venv-nvidia\Scripts\python.exe scripts\smoke_qwen9b.py
.\.venv-nvidia\Scripts\python.exe training\train.py --model qwen35-9b --full
.\.venv-nvidia\Scripts\python.exe evaluation\compare.py --models qwen35-2b qwen35-9b --adapter qwen35-2b=adapters/jase-v19-cap16 --adapter qwen35-9b=training/runs/qwen35-9b/<run-id>/adapter
.\.venv-nvidia\Scripts\python.exe evaluation\compare.py "Trovami un ristorante giapponese stasera sotto i 40 euro" --models qwen35-2b qwen35-9b --adapter qwen35-9b=training/runs/qwen35-9b/<run-id>/adapter
```

Il primo comando esegue tre update, include il caso più lungo del train, salva un adapter in un percorso con run ID e lo ricarica in un nuovo processo per l'inferenza. `--full` richiede un manifest smoke riuscito sulla stessa configurazione e sugli stessi hash train/validation, con caricamento dell'adapter riuscito. Ogni run salva configurazione, hash dei dati, commit Git, versioni, GPU, revisione risolta, loss, tempo, VRAM, log step e adapter. Per il training completo: 3.250 update, batch 1, accumulo 4, seed 2026 e schedule cosine. La loss di validazione è calcolata ogni 250 update e a fine training su un campione deterministico di 64 casi unici; l'adapter con loss minima viene selezionato, conservando anche quello finale. Il test non viene consultato durante il training.

`evaluation/compare.py` passa lo stesso input ai modelli **in sequenza** per rispettare la VRAM da 16 GB e salva JSON, CSV e Markdown in `benchmarks/outputs/`. Riporta intent/action, fatti, JSON, schema, chiarimenti, successo semantico, errori accettati dal Guard, latenza, token/s e VRAM. Le metriche fini su intent e fatti sono condizionate all'output conforme allo schema; il report indica quanti casi sono effettivamente valutabili e calcola il successo completo su tutti i casi etichettati. `--sample 32 --sample-seed 2026` sceglie un sottoinsieme riproducibile; il report salva gli ID. Tool selection e plan correctness sono `N/A` finché non esiste un target annotato nel contratto. I report storici del 2B sono utili come baseline documentale; per un confronto numerico controllato occorre ripristinare l'adapter MLX e rivalutare entrambi i modelli sugli stessi casi con le stesse impostazioni.

## Smoke e collaudo benchmark su questa macchina

Lo smoke definitivo `20260925T110736Z-f917d27f` ha eseguito tre update su 12 esempi; il microbatch includeva il caso più lungo di tutto il train (327 token). Loss train all'ultimo step 1,5239; validation loss su 8 esempi 1,3083; durata 43,48 s, di cui 16,12 s per caricare il modello; picco allocato PyTorch 13.217.885.184 byte (12,31 GiB). NVIDIA SMI indicava circa 16 GiB occupati durante il training, quindi il margine fisico è ridotto. Nessun OOM o NaN. L'adapter è stato salvato, caricato in un nuovo processo e usato per generare una risposta. La risposta era JSON sintatticamente valido ma **non** conforme a GoalContract 1.1, come prevedibile dopo soli tre update; questo smoke dimostra la fattibilità tecnica, non la qualità del modello. Il manifest salva anche gli SHA-256 dei sorgenti del trainer e della pipeline, e `--full` ne verifica la corrispondenza.

Il collaudo `benchmarks/outputs/phase1-smoke-32` ha scelto 32 casi deterministici dal pannello dev esclusi i 10 marcati come visti. Il 9B dopo soli tre update ha prodotto JSON valido nell'84,375% dei casi e **zero** output conformi allo schema; pertanto le metriche fini su intent e fatti non sono stimabili, mentre il successo completo è 0/32. La velocità aggregata era 17,77 token/s e il picco allocato in inferenza era circa 7,56 GiB. Il 2B compare nel report con errore esplicito per adapter storico assente. Il comando ha quindi stato `incomplete` e non costituisce un confronto numerico 2B/9B. Questi valori sono diagnostici, non risultati di un 9B addestrato completamente.

Il test di release `release_holdout_v1` non è stato eseguito sul 9B smoke: va riservato al checkpoint addestrato e selezionato senza consultare il test. Il training completo è ora configurato e sbloccato dallo smoke, ma non è stato avviato in questa fase; non esistono quindi ancora risultati di qualità del 9B finale. Per completare l'A/B scientifico occorrono anche l'adapter 2B storico e un host MLX compatibile, oppure una conversione PEFT verificata numericamente contro l'originale.

Il prossimo candidato resta [`ibm-granite/granite-4.2-8b`](https://huggingface.co/ibm-granite/granite-4.2-8b). La struttura comune è pronta a ospitare una configurazione separata, ma architettura, template, moduli LoRA e comportamento di tool calling devono superare un preflight e uno smoke propri. Nessun peso Granite è stato scaricato o addestrato in questa fase.

## Storage e sicurezza

`hf-cache/` è la cache unica dei pesi originali. `training/runs/` conserva gli adapter e i log. `benchmarks/outputs/` conserva i report. Sono ignorati da Git insieme a pesi, cache e ambienti virtuali. La pipeline non crea copie merged o GGUF. I segreti e gli adapter esistenti non vengono modificati.

## Limiti della fase

La baseline MLX non si può caricare su Windows senza adapter e un runtime MLX su Mac. Una conversione dell'adapter richiederebbe verifica numerica separata, quindi non è stata simulata. Il confronto storico usa una pipeline Mac con retry sui guasti strutturali, mentre il nuovo benchmark usa un singolo campionamento deterministico; quei numeri non vanno mescolati. Non vengono avviati altri candidati prima di uno smoke e di un confronto 2B/9B credibili.
