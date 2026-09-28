# V33_REVIEW_PENDING_2

Il pannello `V33_RAW` resta invariato (SHA-256 `4818c5ea0b35aa0c5b6ff467ca1656b4e7313ed87d431975db18e2cb8b566f37`). I 19 casi segnalati sono nel pacchetto separato `benchmarks/v33_review/review_pending_v2.jsonl` (SHA-256 `dec2f810ad6476e52fdf42c4cca98f3659fc83c0e8992ebc6faaee1df148e6e7`). Nessun gold è stato corretto e **V33_REVIEWED non esiste**.

Ogni riga contiene richiesta, gold originale, output grezzi Qwen V26 e Granite, risultato Planner del replay V33, ragione del flag, quattro dimensioni di possibile ambiguità e le tre interpretazioni sorgente affiancate. La proposta finale, i campi da cambiare e l'impatto su exact scoring e Planner sono `PENDING_INDEPENDENT_HUMAN_REVIEW`; scegliere automaticamente tra gold e modelli falserebbe l'adjudication.

Procedura per il revisore: leggere richiesta e tre candidati senza considerarne uno autorevole; compilare interpretazione proposta, campi esatti da cambiare, motivazione, impatto su canonical exact e Planner, identità del revisore e data. Registrare le decisioni in una nuova versione, preservando `V33_RAW` e questo pacchetto. Pubblicare eventuali nuovi punteggi come `V33_REVIEWED` separati dallo storico.

Lo script `scripts/prepare_v33_review_v2.py` verifica l'hash del pannello e rifiuta la sovrascrittura dell'artefatto. I 19 gold, e più in generale i gold agent-authored V33, non sono stati revisionati indipendentemente da un umano.
