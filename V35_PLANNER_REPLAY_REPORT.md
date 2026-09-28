# Jase Planner 1.3 — replay V35

Il pannello V35 di 60 richieste è stato congelato prima dei fix (commit `3626f22`, SHA `c0e4d08208406cc4d1198ee84b432d6ee8fc5eaf678ad641dbcbf3074fb41ac7`). Audit: nessun overlap esatto con corpus e pannelli precedenti, Jaccard massimo 0,411765. Gold e review operativa sono scritti da agenti, senza verifica umana indipendente.

**Il sistema non è pronto per la produzione.** Qwen V26 best produce 58/60 JSON validi, 41/60 contratti validi e solo **9/60 interpretazioni canonical exact**. Il Guard accetta 39/60; il Planner finale ne rende presentabili in dry run solo **5 più 4 NO_ACTION**, cioè 9/60. `execution_permitted` resta sempre `false`.

| Stessi 60 output Qwen | Planner prima | Planner dopo |
| --- | ---: | ---: |
| READY_FOR_DRY_RUN | 6 | 5 |
| NO_ACTION | 4 | 4 |
| HOLD_GUARD | 28 | 28 |
| HOLD Planner | 22 | 23 |
| Errori materiali chiari ammessi, limite inferiore revisionato | 3 | 0 |
| Falsi HOLD su output exact di lettura/promemoria | 2 | 0 |
| Ammessi nonexact con equivalenza non verificata | 1 | 1 |

I tre errori chiari ammessi dalla baseline sono `v35-008` (visitabilità richiesta persa), `v35-020` (città e divieto di prenotazione persi) e `v35-057` (mattino interpretato come data). Il Planner finale li ferma. Il caso `v35-009` resta ammesso: `contains` invece di `eq` sulla disponibilità potrebbe essere equivalente, ma il comportamento di un provider reale non è noto. Gli output exact `v35-010` e `v35-030`, prima trattenuti perché “16” e “18” non coincidevano letteralmente con `16:00`/`18:00`, ora sono pronti in dry run; la derivazione viene annotata in provenance e si applica solo quando nel testo compare un unico orario di quel tipo. L'unico exact ancora in HOLD è `v35-037`, una cancellazione che richiede correttamente binding e conferma.

I fix sono regole generali isolate nel Planner/provenance: controllo della completezza della disponibilità esplicita, conservazione della città quando la location prodotta è solo un landmark generico, scelta rinviata come vincolo di non prenotazione, distinzione fra parte del giorno e data, normalizzazione prudente di un'ora intera. Nessun adapter, Guard storico, gold, frontend o provider è stato cambiato. I 27 test mirati Planner/Registry passano. Eval Qwen: 18,22 token/s, picco VRAM 8,18 GB.

La copertura resta troppo bassa. Anche i gold V35 portano 28 HOLD e 6 HOLD_GUARD nel Planner finale: il Registry mock e il contratto richiedono ulteriore lavoro. Le 19 candidate projection holds e la recall dei fatti (40/89) mostrano che molti errori nascono già nell'Interpreter; un Planner deterministico può fermarli, ma non recuperarne con certezza il contenuto. Non si possono inferire tassi di rischio in produzione da un pannello agent-authored di 60 casi. Servono gold revisionati da persone, binding provider verificato e dati rappresentativi di richieste reali.

Artefatti: `benchmarks/v35/manifest.json`, `benchmarks/outputs/qwen-v26-v35-blind/summary.json`, `evaluation/replays/v35_qwen_v26_planner_before_summary.json`, `evaluation/replays/v35_qwen_v26_planner_after_summary.json`, `evaluation/reviews/v35_planner_acceptance_review_draft.json`.
