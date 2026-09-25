# H14 — qualità dei dati a budget di training fisso

Protocollo scritto il 24 settembre 2026, prima di generare predizioni v20.

Domanda: correggere la perdita di significato nelle rese telegrafiche,
eliminare i duplicati e separare le famiglie derivate riduce i contratti
errati accettati, senza perdere fatti o aumentare i chiarimenti inutili?

## Dati e modifiche

- `generate_dataset_v2.py`: quando il verbo è eliminato, azione `find` come
  previsto dalla semantica; gli operatori temporali e quantitativi restano
  nel testo; arricchimenti e contrasti conservano la famiglia sorgente.
- `prepare_v20.py`: seed dati 20260924, 30000 esempi candidati. Rimozione dei
  duplicati del gold; nessuna corrispondenza esatta con il release holdout.
- 3074 ripetizioni eliminate, nessun conflitto residuo e nessuna quarantena.
- Split per componenti di famiglie con unione delle famiglie che generano
  lo stesso testo, prima della deduplicazione. Stratificazione per richieste
  di chiarimento, multi-goal e gate. Metadata `source_scenario_id` conservati.
- 24560 train / 2191 valid; 118 / 30 chiarimenti. Nessuna sovrapposizione
  testuale o di famiglia, nessun tag strutturale mancante in validation.
- Controllo del Guard su tutte le 26751 righe: nessun errore o riparazione.

È un intervento sul pacchetto dati, non un'ablation che separa l'effetto di
ogni singola correzione. Il nuovo seed dei dati e la deduplicazione cambiano
anche frequenze e numero di righe. Il vecchio dataset v19 resta intatto.
In particolare la frequenza dei chiarimenti scende: il loro apprendimento
va verificato, non dato per acquisito.

## Training

Base `mlx-community/Qwen3.5-2B-4bit`, LoRA 16 layer, rank 16, scale 32,
dropout 0.05, AdamW, LR 3e-5 → 3e-6, warmup 100, batch 1, accumulo 4,
13000 iterazioni, seed modello 2026, max_seq 512. Stessa ricetta v19-cap16
salvo dati e frequenza/ampiezza della misurazione della validation loss.
Loss su tutta la validation a inizio, 6500 e 13000; nessuna selezione
automatica basata sulla loss. Checkpoint candidato: finale 13000, prefissato.

**Correzione del conteggio storico delle epoche:** il trainer MLX installato
conta microbatch, non aggiornamenti dell'ottimizzatore. L'accumulo non
moltiplica il numero degli esempi consumati:

- v19: 13000 / 27504 = 0.473 epoche;
- v20: 13000 / 24560 = 0.529 epoche;
- entrambi: 3250 aggiornamenti dell'ottimizzatore.

I conteggi in `ITERATIONS.md` che moltiplicano per l'accumulo sono errati.
Il budget qui è fisso in microbatch, non in token né in ore. Non si avviano
ancora tre training da 0.5/1/2 epoche: prima si misura questo intervento.

## Valutazione e criterio predefinito

`data/v20_quality/dev.jsonl` contiene 256 esempi fissi della nuova validation:
246 testi assenti dal training v19 (`v19-exact-unseen`) e 10 chiarimenti
già presenti (`v19-exact-seen`). Non esistono chiarimenti nuovi rispetto a
v19 fra quelli di questa validation: i dieci casi sono solo diagnostica.
Ogni esempio del pannello è escluso dal training v20 con tutta la famiglia.

Il confronto principale usa soltanto i 246 testi `v19-exact-unseen`.
Si rivalutano v19-cap16 e v20 finale con lo stesso evaluator, Guard,
pipeline, seed 2026, max_tokens 420 e prompt. Gold v1: sola regressione.

Un risultato è **promettente**, non dimostrato né pronto per la produzione,
solo se tutte queste condizioni sono soddisfatte:

1. Diminuiscono gli `accepted_but_wrong` sul pannello principale.
2. Semantic exact e recall dei fatti sullo stesso pannello non peggiorano.
3. La precisione dei fatti perde al massimo un punto percentuale.
4. I chiarimenti inutili sul pannello completo non aumentano.
5. I mandatory corretti sul gold diminuiscono al massimo di due casi.

Queste soglie sono operative, non un test di significatività statistica.
Un solo seed e un generatore condiviso non dimostrano generalizzazione OOD;
assenza di duplicati esatti non implica indipendenza strutturale.
Il release holdout non viene eseguito e nessun adapter viene promosso
automaticamente. `reports/BEST.json` resta invariato.

## Esecuzione e risultati

`runs/v20_experiment.json` congela gli hash di dati, codice e baseline.
`run_v20_experiment.py` verifica Metal, lunghezze e corrispondenza fra
maschera del training e prefisso d'inferenza. Poi esegue training e quattro
valutazioni in sequenza, interrompendosi al primo errore.

- Stato: `reports/v20/status.json`.
- Lunghezze, versioni delle librerie e revisione del modello: `reports/v20/preflight.json`.
- Log training: `reports/v20/train.log`.
- Report: `reports/v20/{baseline,candidate}-{dev,gold}.json`.
- Risultato finale: `reports/v20/comparison.json` e `reports/v20/SUMMARY.md`.

Durata indicativa 6–8 ore sul Mac, a seconda della velocità delle valutazioni.
Il runner nativo esegue la sequenza anche dopo la fine della conversazione,
finché il Mac resta attivo. Non invia dati o messaggi a servizi esterni.

## Passi successivi separati

Raccolta di richieste umane con annotazione, ampliamento dei fraseggi,
teacher con controllo dei campi letterali, inventario dei fatti H12 e
confronti multi-seed restano esperimenti successivi. Questo test non spaccia
nuovi esempi sintetici per dati di utenti reali e non usa il teacher.
