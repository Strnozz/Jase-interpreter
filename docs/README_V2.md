# Jase Interpreter — v2

Questo documento spiega com'è fatto il sistema dopo il rifacimento v2 e come
si esegue ogni pezzo. Il v1 resta intatto (`data/`, `adapters/jase-v1-*`,
`scripts/generate_dataset.py`, `scripts/eval_mlx.py`) come baseline storica.

## Perché una v2

Il v1 aveva dimostrato una cosa importante — il fine-tuning insegna il
protocollo Jase a un modello da 2B (schema valido: 0% → 91%) — e ne aveva
nascoste tre:

1. **Nessuna semantica scritta.** Non esisteva un documento che dicesse
   quando un fatto è attributo, vincolo o preferenza. Generatore, valutatore
   e aspettative umane divergevano senza che nessuno se ne accorgesse.
2. **Dataset incoerente.** Il generatore accoppiava target e proprietà a caso
   ("un grafico rosso", "un regalo con 4K"). Scalarlo avrebbe insegnato più
   forte i suoi difetti.
3. **Valutatore cieco.** Misurava `exact_contract_rate` su stringhe JSON:
   una differenza cosmetica pesava quanto un prezzo finito nel bucket
   sbagliato.

La v2 affronta le tre cose nell'ordine giusto: prima la semantica, poi gli
strumenti di misura, poi i dati, poi il modello.

## Mappa dei componenti

```
docs/SEMANTICS.md          la fonte di verità: regole di bucket, operatori,
                           unità, valute, anti-allucinazione, fail-closed
jase/lexicon.py            le tabelle di SEMANTICS.md in forma eseguibile
jase/canon.py              estrazione JSON robusta + canonicalizzazione
jase/guard.py              Semantic Guard deterministico, fail-closed
jase/metrics.py            confronto semantico e metriche v2

benchmarks/gold_v1_source.py   benchmark scritto a mano (99 casi)
benchmarks/build_gold.py       validatore + serializzatore → gold_v1.jsonl
benchmarks/gold_v1.jsonl       il benchmark. MAI usarlo in training.

scripts/generate_dataset_v2.py generatore semantics-first
scripts/coverage_check.py      copertura del benchmark da parte del dataset
scripts/eval_v2.py             valutazione RAW + GUARDED con report JSON
scripts/compare_reports.py     confronto checkpoint, regressioni, BEST
scripts/pick_checkpoint.py     estrae un checkpoint intermedio valutabile
scripts/token_stats.py         lunghezza in token degli esempi
scripts/interpret.py           pipeline di produzione (modello + Guard)
scripts/jase_runner.py         esecutore di job locale (vedi sotto)
scripts/guard_falsepos.py      falsi positivi del Guard, misurati sul gold
scripts/release_check.py       i sei gate di release, in un posto solo

tests/test_guard.py            suite di regressione, non richiede MLX
configs/mlx_lora_v2*.yaml      configurazioni di training
reports/                       un report JSON per versione del modello
```

## La catena della verità

Un solo documento definisce le regole, un solo modulo le implementa, tutto
il resto le importa:

```
docs/SEMANTICS.md
        │
        └── jase/lexicon.py
                 ├── jase/guard.py ────── scripts/interpret.py   (produzione)
                 ├── jase/metrics.py ──── scripts/eval_v2.py     (misura)
                 ├── scripts/generate_dataset_v2.py              (dati)
                 └── benchmarks/build_gold.py                    (benchmark)
```

`build_gold.py` verifica che il Guard accetti ogni contratto gold **senza
riparazioni**. Se il Guard ripara il gold, o il gold viola la semantica o il
Guard è troppo aggressivo: in entrambi i casi è un bug, e il build fallisce.
`generate_dataset_v2.py` fa la stessa verifica su un campione di ciò che
produce, con l'ancoraggio attivo. È così che si scopre subito se generatore e
regole hanno preso strade diverse.

`scripts/guard_falsepos.py` ripete la verifica sul gold con l'ancoraggio e il
rilevatore di omissioni attivi, cioè **come in produzione**. Ogni esito
diverso da `ACCEPT` su un caso gold è un falso positivo del Guard: deve
essere zero, e ogni controllo nuovo lo deve dimostrare *prima* di diventare
bloccante. Un controllo che "trova errori" senza questa misura sta solo
fermando richieste legittime.

## Il Semantic Guard

Il Guard sta fra il modello e il Planner e ha un solo principio: **non far
passare niente di dubbio**.

| Esito | Significato | Cosa succede a valle |
|---|---|---|
| `ACCEPT` | contratto valido così com'è | va al Planner |
| `REPAIRED` | valido dopo riparazioni deterministiche elencate | va al Planner |
| `CLARIFY` | errore semantico: si sa cosa chiedere | si chiede all'utente |
| `REJECT` | output inutilizzabile | retry, poi chiarimento |

Riparazioni ammesse (e solo queste): rimozione di duplicati esatti,
spostamento di un fatto dal bucket sbagliato a quello giusto quando la regola
è deterministica, normalizzazione di valuta e stringhe numeriche,
azzeramento di `value_to` senza `between`, rimozione di campi extra vuoti.
Tutto il resto ferma il contratto. **Non si indovina mai.**

I controlli semantici, tutti nati da errori misurati:

| controllo | cosa intercetta |
|---|---|
| ancoraggio dei valori | `stasera` → `20:00`, o un ingrediente mai nominato |
| ancoraggio della valuta | `EUR` quando il testo non nomina nessuna valuta |
| ancoraggio di `raw_action` | un verbo messo in bocca all'utente |
| strettezza dell'operatore | `sotto i 10` interpretato come `lte` |
| ruolo del numero | `alle 20` che diventa `guests` o `price` |
| parole di cortesia come valori | `ingredient contains senti` |
| coerenza del gate | un gate su `buy`, o la condizione duplicata nei filtri |
| **fatti non rendicontati** | un `senza X` o un `80 euro` che non ha prodotto niente |

L'ultimo è l'unico che guarda il **testo** invece del contratto: è l'unico
modo di vedere un'omissione, perché in ciò che il modello produce non c'è
niente di storto — c'è solo un buco.

## Comandi

Tutto gira dalla radice del repo, con il venv attivo.

### Prima di allenare (niente di tutto questo richiede MLX)
```bash
python tests/test_guard.py          # regressioni
python scripts/guard_falsepos.py    # falsi positivi sul gold: deve dare 0
python scripts/coverage_check.py data/<versione>/train.jsonl
python scripts/token_stats.py data/<versione>/train.jsonl
```

### Benchmark gold
```bash
python benchmarks/build_gold.py
```

### Dataset
```bash
python scripts/generate_dataset_v2.py --n 26000 --check 26000 --out data/v9
python scripts/coverage_check.py data/v9/train.jsonl
```
`coverage_check.py` va eseguito **prima** di allenare. Confronta le
caratteristiche strutturali del dataset con quelle del benchmark — ha un
gate? un operatore di lista? nessun verbo? due numeri con ruoli diversi? — e
segnala le forme che il gold chiede e il dataset non contiene. Per cinque
iterazioni quei buchi li ho scoperti dopo un'ora di training; adesso si
vedono in due secondi.
Lo split è **per famiglia di scenario**: due parafrasi dello stesso scenario
non finiscono mai una in train e una in valid. Lo script lo verifica e
stampa il numero di famiglie sovrapposte, che deve essere 0.

### Training su Mac
```bash
mlx_lm.lora --config configs/mlx_lora_v2.yaml
# se la memoria METAL non regge:
mlx_lm.lora --config configs/mlx_lora_v2_lowmem.yaml
```

### Valutazione
```bash
# baseline: modello nudo, per avere il punto di partenza
python scripts/eval_v2.py --label base --report reports/base.json

# un checkpoint
python scripts/eval_v2.py --adapter adapters/jase-v2 \
    --label jase-v2-iter1500 --report reports/jase-v2-iter1500.json
```
Il report contiene metriche RAW (solo modello) e GUARDED (pipeline completa),
la distribuzione degli esiti del Guard, i gate di release, le metriche per
tag e la tassonomia dei fallimenti con esempi.

### Inferenza di produzione
```bash
python scripts/interpret.py --adapter adapters/jase-v2 \
    "Trovami un panino con la mortadella a Milano sotto i 10 euro"
```

### Training su NVIDIA (RTX 4080 Super, più avanti)
`scripts/train_nvidia.sh` va puntato su `data/v2`. Con 16 GB, per un 2B, LoRA
BF16 a rank 16 su `all-linear` entra comodo e si fonde in modo pulito nel
modello base. Il dataset è nello stesso formato chat, quindi non cambia nulla
a monte.

## Il job runner

L'agente che sviluppa Jase lavora in una VM Linux con questa cartella
montata: legge e scrive i file, ma non può eseguire MLX, che vuole macOS
nativo. Il runner colma il vuoto.

```bash
.venv/bin/python scripts/jase_runner.py
```

Lasciandolo aperto in un Terminale, l'agente può depositare job in
`runs/queue/` e rileggere i risultati da `runs/done/`. Il runner esegue
**solo** gli eseguibili in `ALLOWED` (il python del venv e gli entrypoint
`mlx_lm`), solo con working directory dentro il repo, e senza shell: niente
pipe, niente redirect, niente comandi di cancellazione. Ogni comando viene
stampato prima di partire.

## Gate di release

Una versione non è "pronta" perché le metriche sono salite. Deve superare
tutti questi punti, sul benchmark gold:

- validità JSON: 100%
- validità di schema: 100%
- casi `mandatory` corretti: 100% (81 su 99 casi gold sono mandatory)
- fatti duplicati: 0
- campi non previsti dallo schema: 0
- loop o troncamenti non rilevati: 0
- **`accepted_but_wrong` = 0** — nessun contratto semanticamente sbagliato
  supera il Guard. È l'invariante più importante: un contratto fermato si
  recupera con una domanda, uno sbagliato che passa fa agire Jase nel mondo.

`scripts/release_check.py` li verifica tutti insieme e stampa il verdetto,
così non dipende da chi legge il report.

`reports/` conserva un report per versione, così una regressione si vede
subito. Il `BEST` si sceglie sulle metriche semantiche del gold, mai sulla
validation loss: nel v1 la iter400 aveva loss peggiore della iter300 e
generalizzava meglio.

## Cosa NON si fa

- non si allena sul benchmark gold;
- non si mettono le risposte nel system prompt (il prompt lungo esiste solo
  come diagnostica per confronti col modello base);
- non si allentano validatore o schema per far passare contratti sbagliati;
- non si tolgono i casi difficili dal benchmark;
- non si scrivono regole su misura per singoli prodotti, città o entità.

L'obiettivo è la generalizzazione semantica vera. Tutto il resto è un numero
che sale mentre il sistema resta rotto.
