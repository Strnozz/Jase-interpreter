# Jase 9B V26 — audit prima del training

V29 (40 richieste, SHA `4aac94d42333ee02812f0da81fef2189508c5f3a3f3c7b36e5a21622cc0b80e6`) è stato congelato prima di creare V26. È un pannello mirato, scritto da agenti e non revisionato indipendentemente. Non è un campione di produzione. Le baseline già calcolate sullo stesso pannello sono V24 12/40 e V25 12/40 canonical exact; V25 ha 27/43 action corretti e V24 28/43 dopo proiezione 1.2→1.3. Non si è usato il gold V29 per generare V26.

## Diagnosi e intervento

V25 aveva 1870 train, molti pattern ripetuti e 170 esempi con articolo grammaticalmente errato fra train e valid. Alcune famiglie critiche (origine+arrivo, comando citato) erano poco rappresentate. V26 riparte dall'adapter V25, reinserisce il replay V24 proiettato validamente a GoalContract 1.3, seleziona V25 scartando quelle 170 frasi e aggiunge contrasti nuovi per origine/destinazione, arrivo/partenza, volo diretto con divieto di prenotazione, orario contenuto in messaggio citato, luoghi aperti ora/raggiungibili a piedi, reminder e comando citato non azionabile. I nuovi esempi sono generati da un codice indipendente dai pannelli di benchmark. Il prompt specifica la separazione dei blocchi facts, temporal e context, soprattutto per origin/destination e per l'ora interna a un messaggio.

## Integrità dei dati

| Split | Totale | V24 convertito | V25 selezionato | Nuovo |
| --- | ---: | ---: | ---: | ---: |
| Train | 6026 | 4341 | 1457 | 228 |
| Valid | 1087 | 830 | 217 | 40 |

Seed 2032. Gold validato per ogni riga; duplicati testuali e overlap Jaccard ≥0,85 con tutti i pannelli fino a V29 esclusi; valid vicino al train escluso. 427 train/44 valid V24 sono stati tenuti fuori perché la proiezione 1.2→1.3 non era sicura. Sono stati rimossi 150 train/20 valid V25 per la grammatica. Manifest, hash delle fonti, pannelli esclusi, generatore, prompt e split in `data/v26_9b/manifest.json`.

Audit dell'intero corpus con il tokenizer della revisione pinned `c202236235762e1c871ad0ccb60c8ee5ba337b9a`: train massimo 491 token, valid massimo 491, nessuna riga oltre 512. Il collator ha mascherato i token non-target con `-100` e mantenuto la completion assistant come target su ogni riga. Split 6026/1087 verificato. Il corpus non è stato troncato.

## Training pianificato

Qwen3.5-9B ufficiale, NF4 4-bit con double quant, BF16, LoRA r16/alpha32/dropout0,05 sugli ultimi 16 layer come V25, sequence 512, batch1/accumulation4, LR 8e-6, 1500 step (~1 passaggio del train), warmup60, validazione/checkpoint ogni 300 step e checkpoint iniziali a 25/75. La pipeline esegue smoke, reload, training riprendibile da checkpoint, quindi valuta adapter best/final su V29. Conserva baseline, adapter e frontend precedenti. Spazio libero all'audit: circa 93,4 GiB.

**Limiti:** dati sintetici, V24/V25 ancora templated, solo 228 nuovi train dopo deduplica, V29 piccolo e non human-reviewed. Il benchmark deve misurare anche errori operativi/coverage del Guard e regressioni appaiate, non solo exact.
