# Jase Conversational Interpreter V27 — benchmark finale

**Esito:** training e valutazioni completati; **V27 non è pronto a sostituire V26**. Il nuovo protocollo riduce alcune false azioni su V37, ma perde affidabilità semantica, soprattutto nei turni contestuali, e consente almeno sei `READY_FOR_DRY_RUN` materialmente incompleti o errati. Nessuna esecuzione esterna conseguenziale è stata abilitata.

## Provenienza

- Base `Qwen/Qwen3.5-9B`, revisione `c202236235762e1c871ad0ccb60c8ee5ba337b9a`; V27 warm-start dall'adapter V26 congelato. Granite, V26, frontend e benchmark storici invariati.
- V37 congelato prima dei dati mirati: commit `90b3eef`, SHA-256 `1f44fa9512dc1c2b8b5947da406f86083f48d98f3fc499998d450d98794e55e1`, 120 scenari/150 turni. Nessun overlap testuale esatto con dati/pannelli precedenti nell'audit; gold scritti da agenti, **non revisionati indipendentemente da una persona**.
- Corpus V27: 3.780 train/655 validation, SHA train `48b3a708b43df5e9949a0a3b36e8da2b7e106fafc2e53a43a2e7d2558c5ce8ba`, valid `00ac5f34767f7be0fdc1430ae76d9d1702e596e043462097e1acee8b9080b24a`; massimo 432 token, limite 512, loss solo sui token del target verificata sull'intero corpus. Architettura, dataset e config: commit `d586d89`.
- Pilot indipendente dal V37: 60 step warm-start V26, validation loss 0,54995; 60 step dal base, 1,08638. Scelto warm-start con regola prefissata; nessuna selezione tramite V37.
- Full QLoRA NF4/BF16, LoRA r16/alpha32/dropout 0,05, 21.639.168 parametri addestrabili (0,2242% del checkpoint), batch 1 × accumulazione 4, LR `6e-6`, 900 step, seed 2041. Adapter best allo step 600 (validation loss 0,01896; finale 0,01907); train loss finale 0,000355. Durata 3.149 s, throughput input 449 token/s, picco VRAM allocata 13,82 GB (13,82 miliardi di byte), riservata 16,53 GB, nessun OOM/NaN registrato. Adapter best circa 106,6 MB. Il campione validation fisso è di 256/655 righe: la loss molto bassa non dimostra generalizzazione.
- Output: `training/runs/qwen35-9b-v27/20260928T114238Z-d2316823/best_adapter`; manifest completo nella stessa directory. Valutatore V37 commit `ff3d67c`; replay legacy draft commit `e3fddba`.

## V37 nuovo: V26 storico e V27

I due protocolli sono diversi. V26 produce un GoalContract a colpo singolo e riceve i turni precedenti come testo; V27 riceve stato strutturato e propone modifiche a un GoalDraft. Confrontiamo rilevamento dell'azione e semantica osservabile; **non** confrontiamo `next_correct` né GoalContract exact tra protocolli.

| Misura su 150 turni | V26 one-shot | V27 conversazionale |
|---|---:|---:|
| JSON valido | 145 | 137 |
| Output valido per il rispettivo protocollo | 145 | 130 |
| Actionable corretto | 133 | 136 |
| Falsa azione | 12 | 1 |
| Falso non-action/invalid | 2 | 12 |
| Prima azione corretta su 98 turni actionable | 93 | 90 nello stato draft¹ |
| Primo target corretto su 98 turni actionable | 69 | 67 nello stato draft¹ |
| Token generati/s | 17,91 | 17,72 |
| Picco VRAM inference | 8,18 GB | 8,20 GB |

¹ Un aggiornamento contestuale V27 non deve ripetere `action` e `target` nel JSON. I valori 90/98 e 67/98 sono ricalcolati dallo **stato dopo il turno**; i conteggi grezzi `action_correct=113` e `target_correct=96` del valutatore includono i non-action e penalizzano aggiornamenti che omettono correttamente quei campi.

Su 30 scenari a due turni, il secondo turno ha un output accettato dal protocollo e dallo state machine in **20/30** casi; il campo aggiornato corrisponde testualmente al gold in **7/26** aggiornamenti attesi. Questa misura è severa verso sinonimi/normalizzazione, ma gli esempi mostrano difetti reali: `v37-096` tratta «Anzi, parto da Vicenza» come cancellazione; `v37-099` apre un promemoria invece di applicare un vincolo di arrivo al treno; `v37-113` interpreta «Il secondo» come nuovo target. Alcuni casi contestuali non forniscono reali risultati provider, quindi la risoluzione completa del riferimento non è verificabile.

Su 20 richieste inizialmente incomplete, solo 7 producono il prossimo passo `ASK_USER` atteso nel gold; molte altre sono invalide o bloccate. Non esiste ancora una misura affidabile di **domande superflue**: V37 usa spesso `CHECK_REQUIREMENTS` come gold astratto e non contiene sempre risultati provider o dati di disponibilità con cui distinguere una domanda necessaria da una evitabile. Non trasformiamo `next_correct=60/150` in un indicatore di qualità conversazionale.

### Rischio operativo dopo l'interprete

Il runtime ha finalizzato 26 draft e il Planner ne ha marcati **19 `READY_FOR_DRY_RUN`**, 5 `HOLD_GUARD`, 2 `HOLD`. La revisione operativa agent-authored, salvata in `evaluation/reviews/v27_v37_operational_review_draft.json`, identifica **almeno 6/19 READY materialmente sbagliati**:

- `v37-051`: omesso «mattina» dalla ricerca ferroviaria;
- `v37-052`: omesso il vincolo «in centro»;
- `v37-068`: omesso il tipo «regionale»;
- `v37-085`: «bagno» scambiato per posizione geografica, poi ricerca idraulico marcata ready;
- `v37-104`: omessa la domenica della ricerca camera;
- `v37-118`: richiesta hotel **e** treno, ma il runtime passa al Planner solo il primo draft e lo marca ready.

Il contatore automatico `material_false_ready_candidate=1` sottostima dunque il rischio: segnala solo i conflitti con alcuni valori `expected_next`. Il Planner resta fail-closed rispetto all'esecuzione esterna (`execution_permitted=false`), ma `READY_FOR_DRY_RUN` non equivale a obiettivo semanticamente corretto. Non è stata effettuata alcuna chiamata provider conseguenziale.

## Regressione V33–V36 congelata

V27 è stato eseguito **una volta** sui 242 input storici, con proiezione dichiarata dal GoalContract gold a indicatori del draft. I gold non sono stati cambiati. Risultati V27: JSON 238/242; protocollo valido 202/242; actionable corretto 203/242; 39 falsi non-action/invalid; 0 false azioni validate; 164 prime azioni e 130 primi target esatti sui 242 input (257 goal gold complessivi); 210/506 slot gold corrispondono esattamente a uno slot proposto. Il registry blocca 58 draft per capability sconosciuta e il runtime indica 67 candidati ready. Fra questi 67, 9 hanno un numero di goal diverso dal gold e 53 non corrispondono testualmente a tutti gli slot del primo goal. Sono **proxy di triage**, non 53 errori materiali dimostrati: alias, operatori e normalizzazione possono differire.

Esempi confermati dal replay: `v33-011` perde `open_now`; `v33-066` perde «senza scali» e la policy di sola consultazione; `v35-045` separa due goal ma il primo draft è hotel anziché volo e omette il divieto di prenotare; `v36-020` non esprime correttamente «senza fissare la visita». Il Planner non può ricostruire in modo affidabile vincoli linguistici omessi.

I GoalContract V26 storici restano un confronto di contesto: canonical exact 27/100 V33, 9/40 V34, 9/60 V35, 11/42 V36. V27 non emette GoalContract one-shot e un nuovo exact ricavato forzando draft incompleti sarebbe fuorviante. Il replay V27 registra proposta, stato/plan e indicatori comuni senza riscrivere gli output V26.

## Diagnosi e decisione

**Miglioramento osservato:** le false azioni su chat/meta/quoted V37 scendono da 12 a 1; il protocollo consente davvero GoalDraft, risposte brevi e selezione del curriculum tramite validation interna. Esempi corretti includono «Da Ferrara», «Verso Bologna», «Per lunedì prossimo» e «Siamo cinque», che aggiornano un draft invece di partire da zero.

**Regressioni e difetti:** JSON/protocollo meno robusti; più falsi non-action/invalid; aggiornamenti contestuali spesso errati; omissioni di vincoli che passano in READY; multi-goal finalizzato solo in parte. La loss validation bassa convive con questi errori, quindi non deve guidare una promozione. Il protocollo addestrato è compatto ma mostra slot inventati e enum fuori schema; la validation sintetica è troppo vicina al curriculum per certificarne la robustezza.

La verbalizzazione Stage B ha esempi nel corpus e un renderer deterministico di fallback, ma questo benchmark misura Stage A e la progressione deterministica del draft: **non misura** qualità linguistica delle domande, soddisfazione utente o una conversazione con provider reale.

**Verdetto:** mantenere **Qwen V26 come baseline storica congelata**, V27 come esperimento opt-in. Non abilitare attuazione o promuovere il frontend. La prossima correzione prioritaria è una barriera deterministica di completezza semantica dell'intera conversazione, in particolare per goal multipli, seguita da un dataset conversazionale più indipendente e revisionato da umani. Servono inoltre casi con risultati provider reali o registrati per misurare riferimenti e domande superflue. Qualsiasi V28 dovrebbe avere un nuovo pannello cieco congelato prima di generare dati, senza riusare V37 per tuning. Questa milestone V27 termina qui: un'altra QLoRA non è giustificata dai numeri attuali.

## File di evidenza

- `benchmarks/outputs/v27-conversational-v37/{cases.jsonl,summary.json}`
- `benchmarks/outputs/v26-legacy-v37/{cases.jsonl,summary.json}`
- `benchmarks/outputs/v27-v33-v36-draft-regression/{cases.jsonl,summary.json}`
- `training/runs/qwen35-9b-v27/20260928T114238Z-d2316823/manifest.json`
- `training/v27_pilot_comparison.json`
- `evaluation/reviews/v27_v37_operational_review_draft.json`
- `evaluation/reviews/v27_conversational_benchmark_metrics.json` (riepilogo machine-readable versionato)

Gli output sotto `benchmarks/outputs` e gli adapter sono artefatti locali ignorati da Git; gli script, i gold frozen, l'audit e questo report sono versionabili. Nessun dato benchmark è stato inserito nel train.
