# Fonte dati ferroviaria: indagine e integrazione V1

## Fonte candidata verificata

La [Regione Lombardia pubblica un GTFS ufficiale per Trenord e Malpensa Express](https://www.dati.lombardia.it/Mobilit-e-trasporti/Orario-Ferroviario-Regionale-Gtfs/3z4k-mxz9). La [FAQ del portale](https://www.dati.lombardia.it/faq) spiega il riuso gratuito. La metadata API ufficiale `https://www.dati.lombardia.it/api/views/3z4k-mxz9.json`, verificata il 28/09/2026, espone asset file `trenord_gtfs.zip`, 1.660.043 byte, licenza **CC BY 4.0**, data di modifica dichiarata 26/09/2026. HEAD su `https://www.dati.lombardia.it/download/3z4k-mxz9` ha restituito HTTP 200 e la stessa lunghezza. Il file deve essere scaricato e controllato (hash, date di servizio, tabelle GTFS) prima di etichettare un run `REAL_PROVIDER`.

Il feed è stato poi scaricato nella cache ignorata `data/provider_cache/trenord_gtfs_2026-09-26.zip`; SHA-256 `b4296f145b42ccb35c26085470ff4b3fd5dffe533251c0aab312312a73820ad6`. Contiene `agency`, `stops`, `routes`, `trips`, `stop_times`, `calendar_dates`, `feed_info`; `agency_timezone=Europe/Rome`, 566 fermate, 6.745 corse con `route_type=2`. Il calendario contiene servizi fino a dicembre 2026. Il dataset include anche una linea bus sostitutiva: l'adapter esclude qualsiasi route diversa da GTFS rail `2`. La metadata API pubblica una licenza CC BY 4.0 specifica per questo asset, più precisa della licenza generale citata nella FAQ.

La [specifica GTFS Schedule](https://gtfs.org/documentation/schedule/reference/) definisce orari, fermate, corse e calendari; tariffe e disponibilità non sono garantite. Un GTFS statico offre orari programmati, non ritardi o posti liberi. Va attribuita la fonte Regione Lombardia/Trenord e mostrata data/versione del feed.

È stata considerata la [API OJP svizzera ufficiale](https://opentransportdata.swiss/en/cookbook/open-journey-planner-ojp-landing-page/), che richiede un token. Non sarà usata senza credenziali già disponibili/autorizzate. Non usare scraping HTML o API di vendita non documentate.

## Etichette obbligatorie

- `REAL_PROVIDER`: lettura del dataset ufficiale corrente, con data, URL, hash e limiti di copertura. Se si usa una cache di quel feed verificato, è un'istantanea reale, non live.
- `RECORDED_FIXTURE`: estratto deterministico registrato da una fonte reale con provenienza documentata, utile ai test; non è una ricerca corrente.
- `SANDBOX_PROVIDER`: implementazione deterministica che simula il contratto, senza pretesa di dati reali.
- `MOCK`: solo draft/risposta inventata per unit test.

La metadata API e il GET dello ZIP sono le sole chiamate esterne proposte. Nessun endpoint di vendita, autenticazione, pagamento o write. Se la copertura/validità del feed non è confermata, il slice deve restare su fixture e dichiararlo nel report.
