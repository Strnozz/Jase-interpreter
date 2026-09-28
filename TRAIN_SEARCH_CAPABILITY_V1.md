# Capability `transport.train.search` V1

Contratto definito dopo il freeze del pannello `TRAIN_SLICE_V1` (SHA-256 `339aad6389e9076e4a2f08436dbe44fcc9f920c3fa58f8f79e204df83ae96783`) e prima dell'adapter. La capability restituisce **orari ferroviari programmati**, non disponibilità di posti né biglietti acquistabili. È un'operazione di sola lettura. Acquisto, prenotazione, riserva, autenticazione, invio di dati personali e qualunque write restano esclusi; `execution_permitted=false` per azioni consequenziali.

## Input e condizioni di handoff

`origin`, `destination` e `travel_date` sono obbligatori. Una città o stazione deve risolversi senza ambiguità in identificativi del feed tramite mapping registrato e tracciato. Riferimenti come «da qui» senza contesto verificato producono `HOLD_UNRESOLVED_REFERENCE`; slot assenti `HOLD_MISSING_INFORMATION`. `travel_date` relativa è derivata deterministicamente dalla data della richiesta e da `Europe/Rome`, con input e derivazione registrati. Input inventati o provenance `UNKNOWN`/inferenza modello non ancorata producono HOLD.

Ogni vincolo hard nel GoalContract deve essere tradotto in una restrizione provider oppure applicato con post-processing deterministico ai risultati completi. Se non è verificabile, `HOLD_UNSUPPORTED_CONSTRAINT`. Nessun vincolo viene cancellato per ottenere una query. Un GoalContract omesso o semanticamente errato resta visibile come errore Interpreter: l'adapter non lo ripara consultando la richiesta originale.

## Semantica temporale

| Vincolo | Interpretazione |
| --- | --- |
| `departure_time after 15:00` | partenza strettamente dopo 15:00; «non prima delle 15» usa soglia inclusiva |
| `departure_time before 18:00` | partenza strettamente prima delle 18:00; «entro le 18» è inclusivo |
| `arrival_time before 10:00` | arrivo strettamente prima delle 10:00; «entro le 10» è inclusivo |
| `arrival_time after 19:00` | arrivo strettamente dopo le 19:00 |
| `between 08:00,10:00` | intervallo chiuso sul ruolo specificato |
| `search_time=mattina/pomeriggio/sera` | fascia di partenza configurata e dichiarata nel risultato; HOLD se ruolo non determinabile |

La differenza inclusivo/esclusivo nelle parole naturali non è rappresentata in modo completo dagli operatori GoalContract 1.3 (`before`, `after`, `between`, `eq`). V1 registra un flag `inclusive` quando «entro», «non oltre» o «non prima» compare nella richiesta; gli altri `before/after` sono stretti. Non trasforma mai un termine di arrivo in uguaglianza di partenza. Combinazioni temporalmente impossibili restano in HOLD. Orari GTFS oltre 24:00 sono relativi al giorno di servizio; conversione in datetime locale esplicita. Nei giorni di cambio ora legale/solare V1 tiene in HOLD, poiché un orario locale ripetuto o inesistente richiede una disambiguazione ulteriore.

## Filtri, ordinamento e supporto effettivo

| Richiesta | V1 con GTFS statico regionale |
| --- | --- |
| Treni diretti / senza cambi | Sì, `number_of_changes=0` sui viaggi ricostruiti |
| Categoria regionale | Sì solo se mappatura della riga `routes.txt` a categoria è verificata; altrimenti HOLD |
| Alta velocità, Frecciarossa, Intercity | HOLD se la fonte non copre quel servizio; non equiparare a «treno» generico |
| Prezzo massimo o sort per prezzo | HOLD; il feed selezionato non fornisce tariffe acquistabili per itinerario |
| Prima/seconda classe, disponibilità posti | HOLD; non presenti nel GTFS statico |
| Durata massima | Sì se partenza e arrivo sono presenti |
| Partenza minima / arrivo massimo | Sì, filtro deterministico su datetime normalizzati |
| Limit e sort per partenza, arrivo, durata | Sì, dopo filtri e con ordine stabile; `limit` non limita la raccolta iniziale |
| Stato in tempo reale | Non dichiarato; il feed contiene orario programmato |

V1 può iniziare con tratte dirette nel perimetro del feed. Se la ricerca non calcola coincidenze, il risultato deve dichiarare `coverage=direct_only`; zero risultati significa «nessuna soluzione diretta nel feed», non «nessun treno esistente». Una richiesta che esige confronto esaustivo di soluzioni con cambio va in HOLD finché il routing non è implementato e verificato.

## Flusso e permessi

GoalContract → Guard → Planner 1.3 → capability read-only → validazione slot, policy e provenance → mapping stazioni/data/operatori → GET del feed o lettura snapshot verificato → normalizzazione, filtri, ranking e `jase_result_ref`. Solo un piano singolo, read-only, completamente vincolato può avere `provider_read_permitted=true`; `execution_permitted` resta sempre `false`. Multi-goal non atomico e qualsiasi goal write producono HOLD per questo slice. Le ragioni di HOLD devono essere tracciabili a campo e goal.

## Contratto di risultato

Ogni viaggio ha `jase_result_ref` deterministico nella sessione, `provider_result_ref` (trip/stop/ora del feed), `provider`, `query_ref`, stazioni e datetime di partenza/arrivo, durata, `number_of_changes`, segmenti/categoria solo se disponibili, e metadata della fonte. Prezzo, valuta, disponibilità e stato non sono presenti salvo dato verificato: non sono riempiti con valori null inventati. Riferimenti successivi («il secondo», «quello delle 17:30», «il diretto», «il più economico») richiedono un insieme di risultati identificato; l'ultimo resta in HOLD senza prezzi per tutti i risultati. Nessuna identità si basa sul testo renderizzato.
