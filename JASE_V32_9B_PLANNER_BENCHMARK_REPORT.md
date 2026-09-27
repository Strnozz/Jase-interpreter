# Jase Interpreter 9B — benchmark V32 e handoff Planner isolato

Il pannello V32 (30 casi) è stato congelato **prima** dei fix: commit `2ad669b`, SHA `af76398ca906a2a794dd8f682739290c54daf3ea68b0e984ae929fc6743c3097`. Audit: zero richieste identiche a train/pannelli precedenti e Jaccard massimo 0,5. Gold scritti dall'agente, senza revisione umana indipendente; campione piccolo e mirato, non rappresentativo della produzione. V24, V25 e V26 best step 1200 hanno ricevuto gli stessi input. Nessun adapter, dataset, gold o frontend è stato modificato.

| Metrica V32 | V24 | V25 | V26 best |
| --- | ---: | ---: | ---: |
| JSON valido /30 | **29** | 28 | 27 |
| Canonical exact /30 | 7 | 7 | **10** |
| Candidate schema/projection hold /30 | 14 | 13 | **8** |
| Goal count corretto /30 | 16 | 17 | **22** |
| Action corrette /33 | 16 | 17 | **23** |
| Target corretti /33 | 14 | 15 | **20** |
| Facts corretti /48 | 17 | 19 | **27** |
| Temporal corretti /22 | 14 | 12 | **18** |
| Context corretto /30 | 15 | 17 | **22** |
| Policy corrette | 13 | 12 | **19** |
| Guard V4 ACCEPT /30 | 14 | 14 | **16** |
| Guard V5 ACCEPT /30 | 14 | 14 | **16** |
| V5 ACCEPT ma non exact | 9 | 7 | **6** |
| Output token/s | 18,01 | 18,06 | 18,05 |
| Peak VRAM inference | 8,19 GB | 8,18 GB | 8,18 GB |

V26 guadagna **3 casi senza perderne** rispetto a ciascuno di V24 e V25, appaiati sul V32. Rispetto a V24 recupera `v32-004/015/026`; rispetto a V25 recupera `005/015/021`. Il vantaggio nelle action, nei facts e nei ruoli temporali è più ampio del +3 exact. Tuttavia la validità JSON **peggiora**: V26 produce tre JSON malformati (`012`, `020`, `022`), V25 due, V24 uno. In `020` usa anche `result_count` dove il contratto richiede `results_count`; il Planner non può assumere che siano alias. In `022` il JSON ha parentesi fuori posto. La diminuzione dei projection hold non elimina gli errori di sintassi.

## Guard e rischi operativi

Guard V5 è opt-in (`36c4158`), copre frasi esplicite di consenso rinviato e divieto di prenotare; 28 test Guard/handoff passano. **Sul V32 non cambia alcuna decisione rispetto a V4**. Tutti i 30 gold sono ACCEPT per V4/V5; questo controlla solo falsi HOLD sul pannello. I due V24 canonical-exact trattenuti sono contratti 1.2 con conferma esplicitamente pendente, quindi non sono falsi HOLD di sicurezza.

Dei 16 output V26 ACCEPT, sei non sono exact. Revisione caso per caso:

| Caso | Valutazione |
| --- | --- |
| `v32-007` | **Rischio materiale:** inventa `search_time=11:00` dall'orario di un appuntamento esterno, pur conservando `arrival_time<10:30`. |
| `v32-016` | **Rischio materiale:** vuole prenotare il secondo hotel, ma omette che manca la data. Conserva la conferma. |
| `v32-024` | **Rischio materiale:** perde «biblioteca di Foggia» come luogo unitario, aggiunge `destination=Foggia`; `access=walking` non ha mapping Planner verificato. |
| `v32-029` | **Rischio materiale:** `hire` di un falegname senza segnalare che il provider va ancora scelto. Conserva il consenso. |
| `v32-009` | `treno` al posto di `treno regionale`: tempi e città corretti, ma l'equivalenza di routing non è verificata. |
| `v32-027` | `bici` al posto di `bicicletta`: probabile sinonimo; consenso, luogo e data corretti. |

Quindi V5 accetta **almeno 4 errori operativi chiari su 16 ACCEPT** V26. Non trattiamo tutti i non-exact come errori materiali. V5 non ha falsi HOLD sui 10 V26 canonical-exact, ma questo pannello è troppo piccolo per stimare affidabilità generale.

## Handoff Planner 1.3

Il prototipo separato `jase/planner_handoff_v1_3.py` (`25a7645`, poi controllo required-slot) richiede capability mock esplicite per action/target, binding di fatti, ruoli temporali, modifier, policy, dipendenze, riferimenti ordinali e condizioni. Senza registry restituisce `HOLD_NO_CAPABILITY`; con mapping assente o ambiguo restituisce HOLD. Una capability può dichiarare slot obbligatori anche se il modello dimentica `missing`: sul `v32-016` V26 il mock `book` richiede `temporal.action_date` e restituisce `HOLD_MISSING_INFORMATION`. `execution_permitted` resta sempre `false`. Non esiste nel repository un registry/Planner reale né un binding provider verificato: il prototipo non dimostra che le azioni siano eseguibili o sicure in produzione.

## Decisione

V26 è il candidato più forte sul V32 per interpretazione, ma **10/30 exact, 27/30 JSON e quattro errori materiali accettati** sono insufficienti per promuoverlo. V5 non migliora la copertura V32. La prossima verifica utile è definire capability reali e obblighi di input con un contratto di interfaccia revisionato, aggiungere revisione umana dei gold e misurare il rischio su richieste naturali indipendenti. Un nuovo QLoRA con dati simili non è giustificato da questi risultati.
