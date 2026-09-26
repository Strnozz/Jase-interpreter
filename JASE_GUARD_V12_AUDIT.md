# GoalContract 1.2 Guard — primo audit offline

È stato aggiunto `jase/guard_v1_2.py` come gate separato, **non collegato al frontend**. Il Guard non ripara output e non autorizza azioni: `ACCEPT` consente solo il passaggio al Planner, `REVIEW` trattiene richieste incomplete o sospette, `REJECT` scarta output malformati/contraddittori. Un contratto accettato richiede comunque policy, provider e conferme a valle.

Controlli iniziali: JSON e schema V1.2, valori monetari impossibili, campi noti mancanti per prenotazione di tavolo/volo e contatto, riferimenti non risolti, conferma esplicita, divieti, ranking richiesto, secondo step o fallback omessi e condizioni senza appiglio nel testo. Nessun valore viene inventato o corretto in automatico. Sei test di sicurezza sono passati eseguendo direttamente le funzioni (il venv non contiene `pytest`).

## Audit sui raw output congelati

`Wrong accepted` significa **diverso dal gold esatto V1.2**, non "azione sicuramente pericolosa": alcuni casi sono sinonimi/normalizzazioni plausibili. `Correct held` include contratti rappresentati bene ma deliberatamente fermati perché richiedono dettagli o conferma.

| Modello/pannello | ACCEPT | REVIEW | REJECT | Corretti accettati | Diversi dal gold accettati |
|---|---:|---:|---:|---:|---:|
| V22 transfer 30 | 19 | 10 | 1 | 11 | 8/19 |
| V23 transfer 30 | 21 | 7 | 2 | 11 | 10/21 |
| V22 storico 198 | 181 | 17 | 0 | 164 | 17/181 |
| V23 storico 198 | 181 | 16 | 1 | 159 | 22/181 |
| V23 hard dev 16 | 10 | 5 | 1 | 8 | 2/10 |
| V23 holdout 12 | 8 | 2 | 2 | 5 | 3/8 |

Il gate ferma correttamente i contratti `book` di tavolo/volo privi di dati necessari, il ranking con `price < 0`, JSON fuori schema e alcune policy/dipendenze sospette. Però lascia passare molti contratti semanticamente diversi dal gold, soprattutto per entità, campo del fatto e contesto. Sul transfer V23, per esempio, l'output per il gruppo senza auto sposta i dati dell'utente nei filtri del ristorante; altri errori sono `time` invece di `arrival_time` e perdita del modello `X` del tablet. Le euristiche testuali non dimostrano che un contratto sia corretto.

Questo è un **gate sperimentale fail-closed sui problemi noti**, non un Guard sufficiente per produzione. I pannelli storici sono stati consultati ripetutamente e il transfer di 30 casi è mirato; nessuno misura affidabilità reale. Prima di collegare il Guard al Planner servono etichette riviste da persone, un vocabolario/canonicalizzazione esplicita per campi e tipi, un pannello indipendente più ampio e una misura di falsi negativi operativi. La discrepanza tra `contact` senza messaggio e due gold V20 già segnalati va risolta con una versione del benchmark, senza riscrivere la serie storica.

Artefatti: `benchmarks/outputs/guard-v12-v22/`, `benchmarks/outputs/guard-v12-v23/`; valutatore `evaluation/evaluate_guard_v1_2.py`.
