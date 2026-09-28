# Ontologia dei treni e mapping provider V1

Il target canonico GoalContract resta `{"name":"treno","type":"transport"}`. `treno regionale` non è un nuovo tipo di target: è `treno` più vincolo hard di categoria `regional`. L'alias legacy nel Registry V1 risolve la capability, ma non è autorizzato a perdere la categoria. Se l'Interpreter emette solo il target «treno regionale» senza campo categoria, il binding deve conservare esplicitamente il qualificatore o tenere in HOLD. Questo è un limite da misurare, non da correggere invisibilmente.

| Espressione utente | Significato Jase | Mapping provider |
| --- | --- | --- |
| regionale / regional train | `train_category=regional` hard | classi del feed esplicitamente registrate |
| alta velocità | `train_category=high_speed` hard | solo se il feed include e identifica il servizio |
| Frecciarossa | `train_service=Frecciarossa` hard | brand-specific nell'adapter, mai alias globale a high-speed |
| Intercity | `train_service=Intercity` hard | identificatore di servizio verificato nell'adapter |
| diretto / senza cambi | `stops=0` nel lessico corrente, normalizzato a `number_of_changes=0` | segmenti dello stesso viaggio senza cambio; fermate intermedie non contano come cambi |
| prima / seconda classe | `travel_class=first/second` hard | HOLD sul feed statico senza classe verificabile |

`origin` e `destination` sono fatti distinti con valore di stazione/città; `search_date` è il giorno del viaggio; `departure_time` e `arrival_time` mantengono ruolo e operatore. Un `search_time` di fascia è un filtro di partenza soltanto se la formulazione lo supporta; non diventa scadenza d'arrivo. `price` e `duration` sono vincoli numerici, con EUR/minuti dichiarati. `modifiers.sort` e `limit` sono istruzioni sul result set, non fatti del viaggio.

L'ontologia è indipendente da Trenord, RFI o altri provider. Il mapping da `regional` a linee/categorie del feed, dai nomi a stop ID e dai brand a route ID vive solo nell'adapter versionato. Alias non verificati o stazioni ambigue producono HOLD. Questa decisione non cambia retroattivamente il GoalContract o i gold congelati.
