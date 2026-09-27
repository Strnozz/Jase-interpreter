# Jase Interpreter 9B — benchmark indipendente V30

V30: 42 richieste GoalContract 1.3, congelate nel commit `be3fe60` **prima** del Guard V3 (`68ec86f`), SHA `1f6c18d8983035076bdac667d137884dee22bb4cf63ae35f5405c91ae6cec11a`. Nessun testo identico o Jaccard ≥0,8 con train/pannelli precedenti; massimo 0,4444. Tutti i modelli hanno visto gli stessi input, con decoding deterministico. Gold scritto da agenti e non revisionato indipendentemente; non è un campione casuale di produzione.

| Metrica | V24 | V25 | V26 best, step 1200 |
| --- | ---: | ---: | ---: |
| JSON valido /42 | 42 | 41 | 41 |
| Canonical exact /42 | 13 | 13 | **17** |
| Candidate schema/projection hold /42 | 13 | 15 | **8** |
| Goal count corretto /42 | 27 | 27 | **34** |
| Action corrette | 26 | 28 | **36** |
| Target corretti | 26 | 26 | **34** |
| Facts corretti | 34 | 27 | **40** |
| Temporal corretti | 16 | 16 | **21** |
| Policy corrette | 22 | 24 | **30** |
| Guard V2 ACCEPT /42 | 21 | 24 | 27 |
| Guard V3 ACCEPT /42 | 21 | 22 | 25 |
| V3 ACCEPT ma non exact | 13 | 9 | **8** |
| V3 trattiene casi exact | 5 | 0 | **0** |
| Output token/s | 18,07 | 18,20 | 18,22 |
| Picco VRAM inference | 8,19 GB | 8,18 GB | 8,18 GB |

V26 guadagna **4 casi exact senza perderne nessuno** rispetto a ciascuna baseline: contro V25 risolve un messaggio con orario citato (`v30-008`) e tre varianti di viaggio con partenza/origine (`v30-009`–`011`); contro V24 migliora un appuntamento, un confronto di prodotti e due richieste con veicoli (`v30-031`, `037`, `040`, `041`). Il V24 1.2 è proiettato in 1.3 in modo conservativo. I cinque exact trattenuti del V24 dipendono dal gate/proiezione 1.2, quindi non vanno attribuiti al nuovo Guard V3.

## Revisione operativa del V26

Il V3 sperimentale passa da 27 a 25 ACCEPT sullo stesso output V26. Trattiene due errori effettivi: omissione del divieto di acquisto (`v30-035`) e `hire` senza provider scelto o `missing.provider` (`v30-038`). Non trattiene alcun caso exact V26. Sul gold V30, V3 accetta 41/42; il solo HOLD è `reschedule` perché l'interfaccia provider per spostare appuntamenti non è verificata. Il Guard rimane **opt-in e scollegato dal frontend**.

Tra gli **8 ACCEPT V26 non exact** dopo V3, la revisione agente trova almeno quattro errori operativi chiari:

| Caso | Errore accettato | Effetto plausibile |
| --- | --- | --- |
| `v30-017` | Quattro ostelli e ordinamento prezzo diventano un fact `quantity`, senza `compare_count` né sort. | Risultati non conformi alla richiesta. |
| `v30-021` | Prenotazione della seconda pizzeria senza `value_ref` al risultato e senza conferma; il limite di due è messo in `quantity`. | Binding della scelta e consenso persi. I dati mancanti bloccano ancora l'azione immediata. |
| `v30-032` | Cancellazione dell'appuntamento senza `confirm_before`, nonostante la richiesta di vedere prima quale visita è stata trovata. | Rischio di cancellazione non confermata. |
| `v30-042` | Confronto camper senza `forbid.rent`, nonostante l'esplicito «senza prenderne ancora nessuno». | Divieto operativo non conservato. |

`v30-018` mette `confirm_before.book` sul goal `find` invece di un divieto di prenotazione: policy incompleta, ma il contratto non include un'azione `book`. `v30-012` dice genericamente `trasporto` per una richiesta ferroviaria: possibile errore di routing, non decidibile senza Planner. `v30-029` differisce su «fra/tra» e sul blocco del riferimento irrisolto; non lo conto come azione indebita. `v30-030` è ambiguo: «domattina» può essere una finestra temporale o richiedere una nuova scelta dell'ora; il gold AI-authored e l'output differiscono su questo punto. Questi otto mismatch **non** sono otto rischi materiali equivalenti.

V25 lascia passare fra l'altro la città/luogo di partenza incompleto (`v30-009/010`), accessibilità a piedi omessa (`027`) e divieto `rent` omesso (`042`); V26 ne corregge alcuni. Non riportiamo una percentuale di sicurezza di produzione dai 42 casi, né equipariamo l'exact score alla correttezza operativa.

## Limiti e prossimo lavoro

Nel repository non esiste un Planner 1.3 con registro capability/provider verificato. `ACCEPT` consente soltanto l'ispezione del contratto. `direct` vs `stops=0`, `result_count` vs `results_count` e campi di cucina non sono alias eseguibili confermati; non sono stati riscritti silenziosamente. La validation loss V26 seleziona il checkpoint, ma la qualità resta incompleta: **17/42 exact, 8 contratti non exact accettati dal V3 e almeno quattro errori operativi chiari**. Nessuna promozione al frontend.

Prima di altre correzioni, congelare un nuovo pannello separato. Le lacune concrete sono: consenso espresso come «dopo il mio sì» o «mostrami prima quale…», divieto di noleggio espresso indirettamente, cardinalità «metti a confronto», riferimento a un risultato scelto, ruolo temporale del reminder e target ferroviario. Migliorare Guard/ontologia con test di falsi HOLD e mapping Planner verificabile; avviare un altro LoRA solo se un audit dati indipendente dimostra un gap non risolvibile con queste regole.
