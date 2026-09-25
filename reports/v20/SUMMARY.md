# Esperimento v20 — risultato

I criteri predefiniti non sono tutti soddisfatti. Analizzare il DEV prima di estendere il training.

| Metrica | v19-cap16 | v20-quality |
|---|---:|---:|
| Casi nel pannello principale | 246 | 246 |
| Semantic exact | 0.9268 | 0.9024 |
| Accepted but wrong | 15 | 23 |
| Recall fatti | 0.9835 | 0.989 |
| Precision fatti | 0.9963 | 0.9927 |
| Chiarimenti inutili, pannello completo | 3 | 1 |
| Mandatory gold superati | 77 | 80 |

Confronto principale sui testi assenti dal training v19. I dieci chiarimenti già visti da v19 sono diagnostica separata. Dati sintetici dello stesso generatore, un solo seed: non è una misura di generalizzazione su utenti reali. Holdout di release non eseguito; candidato non promosso automaticamente.
