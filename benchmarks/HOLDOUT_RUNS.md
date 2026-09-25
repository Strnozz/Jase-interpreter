# Esecuzioni del release holdout

Una riga per esecuzione. Se il sigillo SHA-256 cambia, la ragione va scritta
qui, altrimenti i numeri prima e dopo non sono confrontabili.

| data | modello / checkpoint | sigillo | casi | mandatory | accepted_but_wrong | note |
|---|---|---|---|---|---|---|
| 2026-09-23 | — (costruzione) | `cf0e2460…` | 27 | 25 | — | prima stesura |
| 2026-09-23 | — (correzione) | `4477fa1c…` | 27 | 25 | — | h-026 riscritto: la frase "fammi sapere" compariva 30 volte nel training set, trovata dal gate anti-leakage. Un caso già nei dati misura la memoria, non la generalizzazione. |
| 2026-09-24 | jase-v19-cap16 seed 2026 iter13000 (candidato B) | `4477fa1c…` | 27 | **0.240** (6/25) | **15** | prima e unica esecuzione. DEV sullo stesso modello: 0.906. Eseguito con `--no-failures`: i casi falliti NON sono nel report. |
