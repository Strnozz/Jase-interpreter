# Planner 1.3 — conclusione del milestone

## Evidenza

I modelli sono rimasti congelati: Qwen3.5-9B V26 best è il champion, Granite 4.1 8B V26 è archiviato. Nessuna QLoRA, integrazione frontend o chiamata provider è stata eseguita. In ogni replay e nell'handoff mock `execution_permitted` è `false`.

| Pannello congelato | Interpretazioni Qwen exact | Planner finale: READY + NO_ACTION | Errori materiali chiari ammessi, limite inferiore provvisorio | Falsi HOLD exact di lettura/promemoria |
| --- | ---: | ---: | ---: | ---: |
| V33 (100) | 27 | 18 | 5, contro 27 prima del Planner | 6 |
| V34 (40) | 9 | 9 | 0, contro 1 nella baseline V34 | 0, contro 1 |
| V35 (60) | 9 | 9 | 0, contro 3 nella baseline V35 | 0, contro 2 |
| V36 (42) | 11 | 10 | 0, contro 1 nella baseline V36 | 0, contro 1 |

I pannelli sono diversi e non vanno sommati come se fossero un campione casuale di produzione. V33 usa la prima versione del Planner; V34–V36 sono trasferimenti congelati prima dei rispettivi fix. Tutti i gold e le review di rischio sono agent-authored; mancano adjudication indipendente e dati di traffico reale. Anche gli “zero” sono limiti inferiori delle sole ammissioni revisionate, non garanzie di assenza di rischio.

## Cosa è stato realizzato

- GoalContract 1.3 con validazione strutturale e semantica, Guard opt-in e Planner dry-run fail-closed.
- Capability Registry versionato V1–V3, con hash del parent, slot richiesti autorevoli, mapping esplicito e precedenza dei target exact sugli alias.
- Binding di fatti e contesto con provenance, check di riferimenti, dipendenze, operatori temporali, conferma e divieti per azione.
- Handoff mock tipizzato e atomico: sul V36 finale emette **4 soli draft di lettura** e 5 NO_ACTION; tutte le scritture, gli slot/modificatori non mappati o non fidati e gli altri 33 casi restano in HOLD. Il mock non contiene endpoint né codice di esecuzione.
- Replay con hash di pannello, output, Planner, Registry e Guard; 36 test mirati passano.

## Limiti decisivi

La qualità dell'Interpreter è il collo di bottiglia principale: nel V36 solo 11/42 contratti sono exact, 30/42 conformi allo schema e 27/62 fatti gold vengono recuperati. Il Planner può rifiutare un piano sbagliato ma non ricostruire con sicurezza un fatto omesso. Un'altra QLoRA **può diventare giustificata** per gli errori puramente interpretativi, ma è esclusa da questo milestone e non dovrebbe partire su benchmark agent-authored consultati ripetutamente.

Nessuna capability è sufficientemente verificata per collegare un provider reale. Le quattro query mock mostrano solo che il passaggio interno è tipizzato. Mancano contratti API reali, semantica degli alias e operatori, identità dei risultati, autenticazione, consenso, gestione dei fallimenti e test end-to-end. Le azioni con effetti esterni restano fuori dall'handoff.

## Decisione necessaria per il prossimo milestone

Continuare a produrre piccoli pannelli sintetici e nuove regex non fornirà una prova credibile di qualità per produzione. Prima di un'altra iterazione servono:

1. **Revisione umana indipendente** dei 19 gold V33 flaggati nel [pacchetto di review](V33_REVIEW_QUEUE.md), e di un campione di richieste reali non usate nei benchmark.
2. **Una capability pilota di sola lettura** con contratto provider reale documentato (per esempio ricerca treni oppure hotel): campi, operatori, alias, risultati e casi di errore. L'implementazione può iniziare in sandbox con esecuzione ancora disabilitata.
3. Solo dopo la revisione, una decisione esplicita su dati versionati e un eventuale nuovo training dell'Interpreter. V26 e Granite restano intatti.

Fino a queste decisioni, il risultato utilizzabile è un **prototipo di interpretazione e pianificazione in dry run**, con astensione conservativa e tracce ispezionabili. Non è un sistema di esecuzione pronto per produzione.
