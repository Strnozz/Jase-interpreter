# Jase train search: valutazione end-to-end V1

**Stato:** milestone sperimentale read-only completato, **non pronto per produzione**. Nessun modello è stato addestrato. Qwen3.5-9B V26 best, Granite, V33_RAW, V34–V36, output storici e frontend sono rimasti congelati. Baseline in `TRAIN_VERTICAL_SLICE_BASELINE.md`. I 19 casi V33 sono in `V33_REVIEW_PACKAGE_V2.md`, ancora **pending independent human review**.

## Fonte, contratto e isolamento

Il slice legge un vero [feed GTFS pubblico di Regione Lombardia per Trenord/Malpensa Express](https://www.dati.lombardia.it/Mobilit-e-trasporti/Orario-Ferroviario-Regionale-Gtfs/3z4k-mxz9) con licenza CC BY 4.0. ZIP verificato SHA-256 `b4296f145b42ccb35c26085470ff4b3fd5dffe533251c0aab312312a73820ad6`, calendario 26/07–12/12/2026, 566 fermate e 6.745 corse rail. `source_kind=REAL_PROVIDER` significa **snapshot ufficiale verificato**: non è una API live di vendita, né disponibilità posti o prezzo. I bus sostitutivi sono esclusi. Contratto e limiti: `TRAIN_SEARCH_CAPABILITY_V1.md`, `TRAIN_ONTOLOGY_DECISIONS.md`, `TRAIN_PROVIDER_INTEGRATION.md`; tipi in `jase/train_provider_v1.py`, schemi in `schema/train_search_request_v1.schema.json` e `schema/train_search_result_v1.schema.json`.

La catena opt-in è GoalContract → Guard/Planner 1.3 → Registry train-v1 → verifica slot/provenance/policy → mapping stazioni e data → lettura feed → filtro/ranking → risultati normalizzati. `execution_permitted=false` e `external_side_effect_permitted=false` in **tutti** i casi; `provider_read_permitted=true` solo per i quattro handoff validati. Il Registry storico e il frontend restano invariati nel loro uso predefinito. Nessuna chiamata a booking, payment, account o provider write.

## Pannello e misure

`TRAIN_SLICE_V1` è stato congelato **prima** del contratto/provider, commit `2372b41`, 61 richieste, SHA-256 `339aad6389e9076e4a2f08436dbe44fcc9f920c3fa58f8f79e204df83ae96783`. Zero richieste identiche a dati/pannelli precedenti, massimo Jaccard token 0,473684. Gold agent-authored, senza review umana indipendente. Output Qwen V26 congelato SHA-256 `77bf9aa4596b4c2584ca737c3c86dfede1ff2690fcddca5df2fe9b14eca73f70`. Data di riferimento per «domani»: 28/09/2026 12:00 Europe/Rome.

| Livello | Misura | Risultato |
| --- | --- | ---: |
| Interpreter | JSON valido | 60/61 |
| Interpreter | Schema/proiezione validi | 40/61 |
| Interpreter | Exact canonico storico | 3/61 |
| Interpreter | Exact ignorando solo array opzionali vuoti nel gold | 10/61 |
| Interpreter | Action corretto | 38/57 goal gold |
| Interpreter | Target corretto | 17/57 |
| Interpreter | Fact recall / temporal recall | 72/116; 40/70 |
| Sistema | Planner/Guard → GET del feed | 4/61 |
| Sistema | READY_RESULTS dopo GET | 3/61 |
| Sistema | NO_ACTION | 2/61 |
| Sistema | HOLD prima della lettura / HOLD per copertura incompleta dopo lettura | 55/61; 1/61 |
| Sistema | GET su sei frasi non actionable | 0/6 |
| Sistema | GET su richieste write/booking | 0/2 |
| Sistema | Risultati normalizzati sui quattro GET | 22, riferimenti unici 22/22 |
| Sistema | Richieste provider con campi inventati/non supportati tra i quattro GET | 0 |
| Sistema | Material false READY chiari, review agent | 0/3 READY; quarto GET termina in HOLD (limite inferiore) |
| Sistema | Falsi HOLD contro gold agent-authored eseguibile | 17/21 |

Il gold congelato contiene spesso `missing: []` mentre Qwen omette l'array opzionale: l'exact storico 3/61 penalizza questa differenza rappresentazionale. La seconda misura 10/61 elimina **solo** campi opzionali vuoti per il confronto, senza cambiare il gold o usare il contratto normalizzato in esecuzione. Anche così la qualità semantica rimane bassa.

## Casi reali e sicurezza

- `train-v1-001`: Lecco → Monza, 14 ottobre; 10 orari diretti da GTFS, primo 05:06–05:51. Le città, data e stop ID hanno provenance, nessun prezzo inventato.
- `train-v1-005`: Varese → Saronno, domani; il gruppo città registrato include Varese e Varese Nord; 10 risultati da **Varese Nord**, stazione mostrata esplicitamente. Prima del fix il solo match esatto «Varese» dava zero fuorviante.
- `train-v1-006`: Milano Porta Garibaldi → Novara, dopodomani; due risultati programmati 06:19–07:16 e 06:49–07:45, con trip/stop ID e ref stabili.
- `train-v1-004`: Pavia → Lodi; zero **soluzioni dirette nel feed**. Stato finale `HOLD_INCOMPLETE_COVERAGE`, poiché non possiamo escludere itinerari con cambio.
- `train-v1-018`: «Domani mattina» era interpretato senza `search_time`. Il replay iniziale eseguiva una ricerca senza fascia, errore materiale. La regola generale di completezza ora produce `HOLD_OMITTED_DAYPART`, senza aggiungere un valore al contratto Qwen. GET 5→4; chiari false READY 1→0. Una verifica successiva dell'ordinamento ha reso non eseguibile anche un gold dal sort semanticamente ambiguo: il denominatore gold eseguibile passa 22→21, senza attribuire ciò a un miglioramento del modello.
- `train-v1-012`: scadenza di arrivo prima/entro 09:30 correttamente rappresentata nel goal, ma Qwen aggiunge contesto fuori schema. Il Planner tiene in HOLD. Un replay **del gold**, non del modello, dimostra `arrival_time before 09:30` con bound inclusivo per «entro».

Le quattro letture provider sono state revisionate singolarmente in `evaluation/train_slice_v1_after/operational_review.json`: origine, destinazione, data, vincoli effettivi e risultati sono coerenti con le richieste. Tre producono risultati `READY_RESULTS`; il quarto termina in HOLD per copertura incompleta. Tutti i 22 risultati rispettano schema, limite, ordinamento cronologico e ref univoci. I test verificano inoltre filtro su partenza/arrivo senza collasso del ruolo, secondo/terzo risultato, risultato delle 17:30, HOLD per prezzo non disponibile e cambio dell'ora legale. **Nessuna richiesta Qwen con vincolo temporale è arrivata al provider nel replay finale**: il comportamento end-to-end per questa famiglia resta non dimostrato nonostante i test tipizzati e i replay dei gold.

## Errori, tassonomia e qualità del gold

I 17 falsi HOLD rispetto al gold eseguibile non sono 17 errori puramente del modello. Cinque sono Interpreter chiari; un sesto caso mostra `ACTION_ERROR` («ordina» interpretato come acquisto invece di sort), ma il gold di quel caso ora resta in HOLD perché «più vicino» non determina un sort univoco. Tutti e sei sono documentati in `evaluation/train_slice_v1_after/V27_INTERPRETER_FAILURE_CANDIDATES.json`: `TEMPORAL_ROLE_ERROR` (soglia di imbarco come `search_time`), `FACT_HALLUCINATION`/schema (contesto inventato), omissione della fascia mattutina, omissione di «senza cambi» con JSON malformato, `POLICY_ERROR` (no-book omesso), e l'`ACTION_ERROR` citato. Il Planner non può ricostruire questi campi in sicurezza. Il file è **candidato di review, non dataset di training**.

Undici altri falsi HOLD riguardano gold che assumono «treno» benché la richiesta autonoma dica soltanto «trasporto», «corsa», «partenza» o non specifichi il mezzo (`010, 013, 014, 015, 017, 022, 031, 034, 037, 053, 054`). Non li contiamo come errori certi di Qwen: serve adjudication umana, mantenendo il pannello congelato. Il caso `002` («andare in ferrovia») è un gap di ontologia/alias potenziale, non un target-error certo. Il numero **conservativo** di errori non riparabili deterministicamente è quindi **6** sul pannello, non 18.

Limiti Planner/provider: Registry V1 legacy non richiedeva la data, corretto solo nel Registry train-v1 opt-in; supporto città/stazioni circoscritto; GTFS statico copre soltanto il perimetro Trenord e il prototipo ricostruisce tratte dirette, senza coincidenze. Prezzo, classe, brand fuori feed, disponibilità posti e real-time restano `HOLD_UNSUPPORTED_CONSTRAINT`. Un giorno di cambio DST è tenuto in HOLD. L'inclusività «entro/non oltre» è registrata nel provider request; la semantica degli altri operatori e i casi di orario citato richiedono test con output Interpreter validi. L'eventuale `HOLD` per città fuori feed non prova un errore del modello.

## Verifica e decisione

Schema JSON verificato su 4 request e 22 result; 12 test nuovi del slice/provider e 42 test di regressione Planner/Guard/Registry passano. Replay e summary prima/dopo sono machine-readable in `evaluation/train_slice_v1/` e `evaluation/train_slice_v1_after/` e non cambiano i benchmark precedenti.

**V27:** un esperimento mirato è giustificabile per le sei famiglie Interpreter chiare e per il recall insufficiente, ma **non va avviato ora**. Servono revisione indipendente dei gold ambigui, esempi nuovi non derivati dai pannelli e una valutazione separata dei vincoli temporali/di policy. Target specifici: distinzione find/sort da order, ruolo departure/arrival, completezza di fascia/diretto/no-book e schema valido. I sei casi del benchmark restano materiale diagnostico, non esempi train.

Per una seconda capability read-only servono registry e contratto tipizzati specifici, provider documentato con licenza/coverage, mapping e provenance di ogni campo, identità risultati, pannello nuovo congelato, e verifica di false READY/HOLD. Per ogni azione consequenziale mancano ancora binding provider reale, identità/autorizzazione utente, consenso verificabile, gestione errori e review di sicurezza indipendente: nessuna di queste azioni è stata abilitata.
