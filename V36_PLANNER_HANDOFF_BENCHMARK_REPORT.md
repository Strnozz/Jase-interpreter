# Jase Planner 1.3 — V36 e handoff mock

V36 è stato congelato prima dei fix (commit `c152c09`, 42 richieste, SHA `af40145327eb755a2496344df858f5ec26264a881a72b099ab73ee7640f5078d`). Nessun overlap esatto con corpus e pannelli precedenti; Jaccard massimo 0,5. Gold e review operativa sono agent-authored, senza adjudication umana.

Qwen V26 best ottiene **11/42 canonical exact**, 41/42 JSON validi, 30/42 conformi allo schema e 28/42 Guard ACCEPT. Le candidate projection holds sono 12. Eval: 18,32 token/s, picco VRAM 8,18 GB. La qualità dell'Interpreter è ancora insufficiente per produzione.

| Stessi 42 output | Planner prima | Planner dopo |
| --- | ---: | ---: |
| READY_FOR_DRY_RUN | 5 | 5 |
| NO_ACTION | 5 | 5 |
| HOLD_GUARD | 18 | 18 |
| HOLD Planner | 14 | 14 |
| Errori materiali chiari ammessi, minimo revisionato | 1 | 0 |
| Falsi HOLD su exact di sola lettura | 1 | 0 |
| `execution_permitted` | sempre false | sempre false |

Il contratto `v36-006` era ammesso nonostante il modello avesse sostituito il divieto di **incaricare** con un divieto di **contattare**: il Planner aggiornato lo ferma. `v36-020`, ricerca esatta di un veterinario, non aveva una capability mock registrata: ora diventa READY. `v36-041` resta fermo perché il modello rappresenta il fine settimana come data di ricerca e perde la disponibilità richiesta della camera. L'unico output nonexact ammesso (`v36-003`) usa l'alias mock registrato “bici” per “bicicletta”; l'equivalenza con provider reali non è verificata. Gli exact `v36-015` e `v36-016` sono messaggi con conferma/binding mancanti e restano correttamente in HOLD.

È stato aggiunto un handoff **mock tipizzato e atomico**, isolato da frontend e provider. Sul replay finale produce 4 draft di lettura e 5 NO_ACTION; mantiene in HOLD 33/42, compreso il promemoria `v36-014` che il Planner segnava READY per il solo dry run. Nessuna chiamata provider è possibile. Vengono preservati slot, provenance, policy, dipendenze e contesto; capability write/destructive, valori non fidati, mapping modificatori non verificati e versioni discordanti non generano draft. Il Registry V3 estende il V2 senza modificare le versioni storiche; gli exact target prevalgono sugli alias. **36 test** mirati passano.

Il benchmark conferma il collo di bottiglia: 27/62 fatti gold sono recuperati e solo 11/42 interpretazioni sono exact. Il Planner può bloccare errori, ma non recuperare con certezza i fatti omessi dal modello. Anche il gold stesso ha molte richieste non mappabili nelle capability mock. Il pannello è piccolo e scritto da agenti; le etichette di rischio sono un limite inferiore provvisorio, non un tasso di rischio in produzione. Prima di qualsiasi promozione servono revisione umana indipendente dei gold, contratto di un provider reale, binding verificato, consenso, controlli end-to-end e traffico rappresentativo. Nessun adapter, frontend, provider o gold storico è stato modificato.

Artefatti: `benchmarks/v36/manifest.json`, `benchmarks/outputs/qwen-v26-v36-blind/summary.json`, `evaluation/replays/v36_qwen_v26_planner_before_summary.json`, `evaluation/replays/v36_qwen_v26_planner_after_v2_summary.json`, `evaluation/reviews/v36_planner_acceptance_review_draft.json`.
