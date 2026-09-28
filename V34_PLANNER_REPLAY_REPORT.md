# Jase Planner 1.3 — replay V34

## Verdetto

Il nuovo pannello V34 è stato congelato prima dei fix (commit `861b144`, 40 richieste, SHA `b4a86432b691eb64220b64a24be29d29ee8dbd9b7b59d2d0fd75e2e900ff7734`). Nessuna richiesta coincide esattamente con corpus o pannelli precedenti; Jaccard massimo 0,52381. I gold sono scritti da un agente e non revisionati indipendentemente.

Qwen V26 best ottiene **9/40 exact**, **40/40 JSON valido**, **28/40 schema valido**, **26/40 Guard ACCEPT**. Il Planner V1 aveva 6 `READY_FOR_DRY_RUN` più 3 `NO_ACTION`; il Planner aggiornato ne ha ancora 6 più 3. La copertura resta **9/40**. Il miglioramento è nella composizione dei casi ammessi: il promemoria con contenuto troncato viene trattenuto e una ricerca autobus exact passa. Tra gli output exact di sola lettura, i falsi HOLD scendono da **1 a 0**. Tra le ammissioni, l'errore materiale chiaro etichettato provvisoriamente scende da **1 a 0**. Un contratto ammesso nonexact (`v34-037`) ha gold discutibile e richiede revisione umana. Questi numeri non dimostrano sicurezza per produzione.

| Misura, stessi 40 output Qwen | Prima | Dopo |
| --- | ---: | ---: |
| READY_FOR_DRY_RUN | 6 | 6 |
| NO_ACTION | 3 | 3 |
| HOLD_GUARD | 21 | 21 |
| HOLD Planner | 10 | 10 |
| Exact read-only trattenuti erroneamente | 1 | 0 |
| Errori materiali chiari ammessi, minimo revisionato | 1 | 0 |
| Nonexact ammessi con gold ambiguo | 1 | 1 |
| `execution_permitted` | sempre false | sempre false |

## Cosa è cambiato

- Registry mock V2 con elenco esplicito di luoghi locali e trasporto autobus. Si estende dal V1 con hash vincolato; nessun wildcard di scrittura o chiamata provider. Risolve l'esatto `v34-012`, prima fermato solo per assenza di interfaccia.
- Il Planner verifica che il testo di un promemoria introdotto da “ricordami di” sia preservato integralmente quando la clausola è delimitabile. Ferma `v34-021`: il modello aveva perso “dire a Paolo che”.
- Il divieto di acquisto vale anche per negozi locali; le frasi di scelta rinviata richiedono un divieto di prenotazione nel contratto. `v34-008` e `v34-007` restano in HOLD. Il passaggio intermedio, preservato nel replay `after`, mostra che la sola espansione del Registry avrebbe ammesso erroneamente `v34-007`.
- I 22 test mirati sul Planner/Registry passano. Baseline e finale registrano hash del pannello, degli output, del Planner, del Registry e del Guard. La valutazione Qwen ha prodotto 18,38 token/s e picco VRAM 8,18 GB.

## Errori e limiti rimasti

La maggioranza dei casi rimane in HOLD: 21 al Guard e 10 nel Planner. Nove output del modello violano direttamente lo schema. Il Registry V2 è solo mock e non copre ancora tutti i domini. I gold stessi portano 22 HOLD e un HOLD_GUARD con il Planner finale: parte della copertura bassa dipende quindi dal contratto e dalla disponibilità delle capability, oltre che dagli errori del modello.

`v34-037` è l'unico output ammesso nonexact nel replay finale. Il testo chiede un idraulico senza aver scelto chi assumere; la ricerca prodotta da Qwen può essere sensata, mentre il gold agent-authored descrive un'assunzione bloccata. Non lo contiamo come errore materiale chiaro finché un revisore umano non decide l'intento. `v34-022` è exact ma resta correttamente in HOLD: è un invio con conferma e binding mancanti.

Il benchmark è piccolo, scritto da agenti e usato per questa iterazione; non è una stima affidabile del tasso di errore in produzione. Mancano revisione umana dei gold, un capability/provider registry reale, verifica dei risultati esterni, prove end-to-end e valutazione su richieste reali con distribuzione rappresentativa. Nessun frontend, adapter, dato di training o provider è stato modificato. Serve un nuovo pannello separato prima di un'altra ottimizzazione.

Artefatti: `benchmarks/v34/manifest.json`, `benchmarks/outputs/qwen-v26-v34-blind/summary.json`, `evaluation/replays/v34_qwen_v26_planner_before_summary.json`, `evaluation/replays/v34_qwen_v26_planner_after_v2_summary.json`, `evaluation/reviews/v34_planner_acceptance_review_draft.json`.
