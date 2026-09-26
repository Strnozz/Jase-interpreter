# Proposta separata: GoalContract V22

Questa proposta non modifica GoalContract 1.2, il run V21 o gli adapter. Le regressioni V21 su missing, dipendenze, policy e action vanno corrette nei dati e nella validazione: cambiare schema da solo non le risolve.

## Limiti del contratto osservati

1. **Riferimenti a risultati precedenti.** In `s09`, “Compra il secondo dei libri che mi hai mostrato prima” diventa solo `missing: referent`. La rappresentazione è prudente, ma perde `secondo` e la fonte “libri mostrati prima”. Proporre un riferimento strutturato con `source_result_set_id` opzionale, `ordinal` e stato `resolved/unresolved`. Se lo stato conversazionale non fornisce un ID, il contratto deve conservare l'ordinale e mantenere l'azione bloccata.
2. **Richieste contraddittorie o bloccate.** V1.2 dispone di `task`/`non_actionable`, missing e policy, ma non di una diagnosi esplicita per istruzioni incompatibili o un'azione impossibile rispetto a vincoli dichiarati. Proporre un campo `blockers` con motivo enumerato e riferimenti ai goal/fatti in conflitto. Il Guard deve impedirne l'esecuzione; l'Interpreter non deve risolvere il conflitto inventando preferenze.

## Prima di adottare V22

- Definire JSON Schema, esempi gold positivi/negativi e regole di migrazione V1.2 senza cambiare i dataset storici.
- Valutare con annotatori umani se i nuovi campi conservano informazione utile al Planner e al Guard; includere test con e senza stato conversazionale reale.
- Eseguire un benchmark parallelo V1.2/V22 e un gate di regressione sui 198 casi storici. Solo poi decidere un training dedicato.
