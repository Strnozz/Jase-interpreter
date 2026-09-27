# Jase Interpreter 9B V26 — benchmark V29

Run QLoRA `training/runs/qwen35-9b-v26/20260927T061719Z-780ce06c`, dataset/config congelati nel commit `e353316`. Pannello V29 di 40 casi congelato prima del dataset, SHA `4aac94d42333ee02812f0da81fef2189508c5f3a3f3c7b36e5a21622cc0b80e6`. Stessi 40 input per V24, V25, V26 best (step 1200) e V26 final (step 1500). La proiezione V24 1.2→1.3 può andare in hold; qui i gold sono tutti confrontabili. Gold V29 scritto da agenti, non revisionato da persona indipendente.

## Risultati appaiati

| Metrica | V24 | V25 | V26 best | V26 final |
| --- | ---: | ---: | ---: | ---: |
| JSON valido /40 | 39 | 38 | 39 | 38 |
| Canonical exact /40 | 12 | 12 | **14** | 14 |
| Schema/proiezione candidato in hold /40 | 11 | 13 | **10** | 11 |
| Goal count corretto /40 | 28 | 27 | **30** | 29 |
| Action corrette /43 | 28 | 27 | **32** | 31 |
| Target corretti /43 | 26 | 24 | **28** | 27 |
| Facts corretti /60 | 31 | 25 | **35** | 32 |
| Temporal corretti /31 | **21** | 17 | 20 | 19 |
| Policy corrette | 24 | 22 | **28** | 27 |
| Guard routing V2 ACCEPT /40 | 22 | 22 | 20 | 21 |
| Guard V2 ACCEPT ma canonical wrong | 13 | 10 | **6** | 7 |
| Guard V2 trattiene canonical exact | 3 | 0 | **0** | 0 |

V26 best guadagna cinque casi exact e ne perde tre rispetto a V24; guadagna sei e ne perde quattro rispetto a V25. Il final non cambia i 14 exact, ma peggiora diversi campi: scegliere il best, senza promuoverlo in produzione. I 6/20 ACCEPT non exact sono un indicatore di revisione, non sei errori materiali equivalenti.

## Revisione dei casi

**Miglioramenti:** V26 recupera tutte e quattro le richieste treno con origine+arrivo corrette (`v29-007`–`010`), che V25 perdeva quasi sempre. Interpreta correttamente il messaggio «Arriverò alle 20» con invio alle 18 (`v29-005`), il noleggio da cercare senza affittare (`v29-035`) e alcune visite/appuntamenti. V24 mantiene un vantaggio lieve su temporal recall (21/31 contro 20/31).

**Regressioni:** V26 perde `v29-006`: «Domani alle 09 ricordami: ritirare il pacco alle 18» diventa reminder alle **18**, con l'ora citata nel contenuto scambiata per l'ora della notifica. Perde anche `v29-002` inventando action/kind `translate` fuori schema e `v29-037` mettendo quattro prodotti in `quantity` anziché `compare_count`. Lieve risalita della validation loss da 0,01378 a step1200 a 0,01440 a step1500; il best è anche migliore nel benchmark.

**Rischi che il Guard V2 sperimentale lascia passare sul V26 best:**

- `v29-004`: invio a Sara senza `confirm_before`, benché l'utente chieda di vedere il messaggio prima.
- `v29-006`: reminder fissato alle 18 anziché alle 09; il Guard lo accetta.
- `v29-034`: azione `hire` con conferma, ma senza provider scelto né `missing.provider`; potrebbe autorizzare una scelta arbitraria.
- `v29-032`: l'utente vieta gli acquisti, ma l'output vieta solo `order` e non `buy`; rischio di policy dipendente dal mapping Planner.

Gli altri due ACCEPT non exact sono `v29-027` e `040`: cambiano `missing.blocks` da `search` ad `action` con referent/destinatario comunque mancante, quindi non li conto automaticamente come azioni eseguibili. Sul final, `v29-019` è un ulteriore ACCEPT errato: manca data/ora/coperti per prenotare il ristorante. Il Guard V2 blocca invece molti altri output difettosi, senza trattenere casi canonical exact V26. Coverage scende al 50% best; questa protezione è ancora incompleta. Per V24 il Guard V2 resta sostanzialmente il vecchio gate 1.2, quindi il confronto del rischio accettato tra versioni non è una prova di sicurezza uniforme.

**Difetti rappresentazionali e del Planner:** `v29-011/013` esprimono il volo diretto come `direct=true`/`route=diretto`, mentre il gold usa `stops=0`; possono essere equivalenti per l'utente, ma il mapping Planner non è verificato. `v29-020/021` usano `result_count` invece di `results_count` nelle condizioni fallback: il Guard li segnala come alias non mappati. `v29-025` usa `cuisine=vegetariana` invece del nome target «ristorante vegetariano», ma omette `open_now`; questa omissione è materiale. `v29-023` inventa un orario di ricerca `15:00` e ha struttura invalida. Non correggere il gold V29 per far salire l'exact: serve una ontologia eseguibile e un nuovo pannello indipendente.

## Training e risorse

Smoke e ricaricamento adapter superati; training 1500 step, 6000 esempi visti (~0,996 epoche), batch effettivo 4, BF16/NF4, LoRA r16/alpha32. Durata 7195 s (~2 h), 338,7 input token/s, picco VRAM allocata 14,06 GB (13,10 GiB), RAM di processo ~2,82 GB; nessun OOM né crash. Ultima train loss 0,00230; best validation loss 0,01378 a step1200. Adapter ~106,6 MB. Inference best ~18,06 output token/s, picco VRAM ~8,18 GB. Il trainer usa kernel PyTorch di riferimento per alcune operazioni Qwen: è corretto, ma limita la velocità.

## Conclusione

V26 migliora routing, action e alcuni facts rispetto a V24/V25 sul V29, ma **14/40 exact e più rischi operativi accettati sono lontani da un interpreter affidabile**. Non promuovere l'adapter o il Guard nel frontend. Prossimo passo: congelare un nuovo pannello indipendente, verificare con il Planner l'ontologia di campi/alias e le condizioni fallback, poi migliorare Guard su conferma, separazione degli orari nei messaggi, provider mancante e policy buy/order; misurare falsi HOLD e rischio accettato sul nuovo pannello. Un altro LoRA richiede una diagnosi di dati nuova, non soltanto un punteggio exact più alto.
