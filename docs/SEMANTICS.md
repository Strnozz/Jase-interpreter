# Jase GoalContract — Semantica Canonica v1

> **Questo documento è l'unica fonte di verità.**
> Generatore di dataset, Semantic Guard, evaluator e benchmark gold DEVONO
> implementare esattamente queste regole. Se una regola cambia qui, cambia
> ovunque, e il benchmark va rigenerato/rivisto.

Versione: `semantics-1.1`
Schema di riferimento: `schema/goal_contract_v1_1.schema.json`
(lo schema v1 resta in `schema/goal_contract_v1.schema.json` come baseline storica)

---

## 0. Principio guida

L'Interpreter descrive **COSA** vuole l'utente, mai **COME** ottenerlo.

Due invarianti sopra ogni altra cosa:

1. **Non inventare mai** un fatto che l'utente non ha espresso.
   Un fatto mancante è un errore recuperabile (si può chiedere chiarimento).
   Un fatto inventato fa agire Jase nel mondo reale in modo sbagliato.
2. **Un fatto, un ruolo.** Lo stesso fatto dell'utente non compare in due
   bucket (`attributes` / `constraints` / `preferences`) né due volte nello
   stesso bucket.

---

## 1. Vocabolario aperto

`target.name`, `target.type`, `property` e `value` sono **aperti**.
Non esiste un'enumerazione chiusa di prodotti, razze, marchi, città, cibi.

Se l'utente dice "una bicicletta color Blorch", il contratto corretto è
`property=color, operator=eq, value=Blorch`, anche se "Blorch" non esiste.
Il modello deve dedurre il **ruolo semantico** dal contesto linguistico,
non dalla conoscenza del mondo.

Per i `property` esiste invece un **lessico preferito** (§4): non è una
chiusura, è una normalizzazione. Se nessun nome preferito calza, si usa un
nome descrittivo in inglese, snake_case, singolare.

---

## 2. Regola di assegnazione al bucket (ordinata, first match wins)

Ogni fatto estratto dall'utente passa questa cascata **nell'ordine**.
La prima regola che scatta decide il bucket. Non si valuta oltre.

| # | Condizione | Bucket | Note |
|---|-----------|--------|------|
| R1 | È presente un **marcatore di desiderabilità debole** che governa quel fatto | `preferences` | vedi §2.1 |
| R2 | Il fatto è una **negazione / esclusione** | `constraints` | operatore `excludes`, `neq` o `not_in` |
| R3 | È presente un **marcatore di obbligo esplicito** | `constraints` | vedi §2.3 |
| R4 | La proprietà appartiene alla **classe di selezione** (§3) | `constraints` | prezzo, luogo, tempo, distanza, disponibilità… |
| R5 | Il fatto porta un **operatore comparativo / di soglia / di intervallo** | `constraints` | `lt lte gt gte between before after within` |
| R6 | altrimenti — descrizione positiva semplice | `attributes` | `eq`, `contains`, `in` |

**Il Guard applica questa cascata come riparazione deterministica**, ma solo
quando nel testo dell'utente non compare né un marcatore debole né un
obbligo: in quel caso R1 e R3 sono fuori gioco e il bucket dipende soltanto
dall'operatore e dalla classe della proprietà, che il Guard conosce. Se un
marcatore c'è, il Guard non tocca niente — non sa a quale fatto si
riferisca, e una riparazione sbagliata è peggio di una mancata. Verso
`attributes` si scende inoltre solo su proprietà che §3.2 dichiara
descrittive: il vocabolario è aperto, e di una proprietà inventata non si sa
se selezioni o descriva.

### 2.1 Marcatori di desiderabilità debole (R1 → `preferences`)

IT: `possibilmente`, `preferibilmente`, `preferirei`, `mi piacerebbe`,
`se possibile`, `magari`, `idealmente`, `meglio se`, `gradirei`,
`non è indispensabile ma`, `se c'è`, `se si può`, `eventualmente`,
`opzionale`, `male che vada anche senza`.

EN: `preferably`, `if possible`, `ideally`, `I'd prefer`, `nice to have`,
`would like`, `optionally`.

Lo scope del marcatore è il **sintagma che governa**, non l'intera frase.
`"un panino possibilmente con mortadella, ma sotto i 10 euro"`
→ mortadella in `preferences`, prezzo in `constraints`.

Un marcatore debole **vince anche su una negazione**:
`"preferibilmente senza glutine"` → `preferences: {diet, excludes, glutine}`.

### 2.2 Negazione / esclusione (R2 → `constraints`)

IT: `senza`, `non`, `niente`, `no`, `escluso`, `tranne`, `eccetto`,
`evita`, `mai`, `a parte`.

Mappatura operatore:

| Forma dell'utente | operator |
|---|---|
| esclusione di un componente/feature contenibile (`senza glutine`, `senza scalo`) | `excludes` |
| negazione di un valore scalare identificante (`non rosso`, `non Ryanair`) | `neq` |
| esclusione di un insieme (`né Ryanair né Wizzair`) | `not_in` + `value` = lista serializzata (vedi §6.4) |

`"senza glutine"` è un vincolo **duro**, non un attributo: la forma negativa
esprime sempre un requisito, mai una descrizione.

### 2.3 Marcatori di obbligo (R3 → `constraints`)

IT: `deve`, `dev'essere`, `devono`, `obbligatorio`, `tassativo`,
`assolutamente`, `necessariamente`, `mi serve per forza`, `solo`, `soltanto`,
`unicamente`, `purché`, `a patto che`, `indispensabile`, `richiesto`,
`proprio` (in `"cento euro proprio massimo"`).

EN: `must`, `has to`, `required`, `mandatory`, `only`, `strictly`.

`"un panino con la mortadella"` → attributo (R6).
`"un panino che deve avere la mortadella"` → vincolo (R3).

### 2.4 Comparativi e soglie (R5)

Vedi §5 per la mappatura completa degli operatori.

---

## 3. Classi di proprietà

### 3.1 Classe di selezione — SEMPRE `constraints` (salvo R1/R2 che vengono prima)

Sono criteri di scelta esterni all'oggetto, non proprietà intrinseche:

```
price          budget        location      distance
time           date          deadline      departure_time
arrival_time   check_in      check_out     availability
delivery_time  delivery      radius
destination    origin        route
```

Razionale: "a Milano" non descrive *che cosa* è un panino; restringe
*quale* panino accettiamo. `"un hotel a Roma"` → `constraints: {location, near, Roma}`.

`destination` e `origin` stanno qui per la stessa ragione: *"un volo per
Londra"* non descrive com'è fatto il volo, dice quale volo accettiamo. Ed è
`destination`, non `location`: un mezzo di trasporto non sta in un posto, ci
va. (Il lessico eseguibile le dava per descrittive mentre il benchmark le
chiedeva fra i vincoli. Nessuno dei due se n'era accorto finché
`scripts/guard_falsepos.py` non ha passato il Guard su tutto il gold.)

### 3.2 Classe descrittiva — `attributes` salvo R1–R5

Proprietà intrinseche dell'oggetto desiderato:

```
ingredient  cuisine   diet      allergen    portion    spiciness
color       material  size      condition   model      brand
ram         storage   display   gpu         cpu        battery
breed       sex       age       coat        weight
stars       room_type guests    amenity
class       carrier   stops     baggage
specialization  experience  language  certification  rating
rooms  floor  feature  format  level  quantity
```

Se una proprietà descrittiva compare con comparativo (`almeno 3 stelle`,
`minimo 32 GB`, `sotto i 6 mesi`) scatta R5 → `constraints`.
Se compare in forma piana (`4 stelle`, `32 GB`, `rosso`) resta `attributes`.

Esempi decisivi:

| Frase | property | operator | value | bucket | regola |
|---|---|---|---|---|---|
| `un hotel a 4 stelle` | stars | eq | 4 | attributes | R6 |
| `un hotel da almeno 3 stelle` | stars | gte | 3 | constraints | R5 |
| `un MacBook con 32 GB di RAM` | ram | eq | 32 (unit GB) | attributes | R6 |
| `un MacBook con almeno 32 GB di RAM` | ram | gte | 32 (unit GB) | constraints | R5 |
| `uno Sphynx femmina` | sex + breed | eq | female / Sphynx | attributes | R6 |
| `sotto i sei mesi` | age | lt | 6 (unit months) | constraints | R5 |

---

## 4. Lessico preferito dei `property`

Nomi in **inglese, snake_case, singolare**. Elenco non chiuso: preferire
questi quando calzano, altrimenti coniare un nome descrittivo coerente.

| Concetto | property canonica | unità tipica |
|---|---|---|
| prezzo / budget | `price` | — (usa `currency`) |
| luogo | `location` | — |
| distanza / raggio | `distance` | `km`, `m`, `min` |
| momento / orario | `time` | — |
| data | `date` | — |
| ingrediente | `ingredient` | — |
| regime alimentare | `diet` | — |
| allergene | `allergen` | — |
| colore | `color` | — |
| condizione (nuovo/usato) | `condition` | — |
| memoria RAM | `ram` | `GB` |
| archiviazione | `storage` | `GB`, `TB` |
| schermo | `display` | `inch` |
| razza animale | `breed` | — |
| sesso | `sex` | — |
| età | `age` | `months`, `years` |
| stelle struttura | `stars` | — |
| ospiti | `guests` | — |
| servizio/dotazione | `amenity` | — |
| compagnia/vettore | `carrier` | — |
| scali | `stops` | — |
| bagaglio | `baggage` | — |
| classe di viaggio | `travel_class` | — |
| partenza | `origin` | — |
| arrivo | `destination` | — |
| specializzazione | `specialization` | — |
| valutazione | `rating` | — |
| caratteristica generica | `feature` | — |

`feature` è la valvola di sfogo: si usa solo quando nessun nome più
specifico esiste. Un contratto pieno di `feature` è un contratto pigro.

---

## 5. Mappatura canonica degli operatori

### 5.1 Numerici e quantità

| Espressione italiana | operator | note |
|---|---|---|
| `sotto i X`, `meno di X`, `sotto X`, `inferiore a X` | `lt` | **stretto** |
| `massimo X`, `al massimo X`, `non più di X`, `fino a X`, `entro X` (quantità), `al più X`, `X max` | `lte` | inclusivo |
| `più di X`, `oltre X`, `superiore a X`, `sopra X` | `gt` | stretto |
| `almeno X`, `minimo X`, `a partire da X`, `da X in su`, `X o più` | `gte` | inclusivo |
| `X` senza comparativo | `eq` | |
| `tra X e Y`, `da X a Y`, `fra X e Y` | `between` | `value`=X, `value_to`=Y |
| `entro X km/minuti` | `within` | distanza/tempo di viaggio, con `unit` |

> **`sotto i 10 euro` è `lt 10`, non `lte 10`.**
> Questa distinzione è vincolante: è uno degli errori storici del v1.

### 5.2 Temporali

| Espressione | property | operator | value |
|---|---|---|---|
| `alle 18`, `alle 20:30` | `time` | `eq` | `"18:00"`, `"20:30"` |
| `entro le 18`, `prima delle 18` | `time` | `before` | `"18:00"` |
| `dopo le 18`, `dalle 18 in poi` | `time` | `after` | `"18:00"` |
| `stasera`, `domani`, `sabato`, `questa settimana` | `date` | `eq` | `"stasera"`, `"domani"`, `"sabato"`, `"questa settimana"` |
| `entro venerdì` | `date` | `before` | `"venerdì"` |
| `dal X al Y` | `date` | `between` | X / `value_to` Y |

**Le espressioni temporali relative NON si risolvono qui.**
`stasera` resta `"stasera"`. Trasformarlo in `20:00` o in una data ISO è
un'allucinazione: la risoluzione deterministica avviene a valle
(Capability Resolver), che conosce fuso orario e ora corrente.

### 5.3 Spaziali

| Espressione | property | operator | value | reference |
|---|---|---|---|---|
| `a Milano`, `in zona Milano`, `qui a Milano` | `location` | `near` | `Milano` | — |
| `entro 100 km da Milano` | `distance` | `within` | `100` (unit `km`) | `Milano` |
| `tra Milano e Torino` | `location` | `between` | `Milano` / `value_to` `Torino` | — |

Policy: per una menzione di città si usa `near`, non `eq`. Motivo: una
ricerca reale accetta risultati nell'area urbana, non solo nel centroide
esatto. `eq` è riservato a un luogo puntuale nominato come tale
(`"proprio in Piazza Duomo"`).

### 5.4 Insiemistici

| Espressione | operator |
|---|---|
| `con X` (componente presente) | `contains` |
| `senza X` | `excludes` |
| `o X o Y` (alternative accettate) | `in` |
| `né X né Y` | `not_in` |
| `senza X e senza Y` | due predicati `excludes`/`neq` separati |

Le ultime due righe non sono la stessa cosa. *"né Ryanair né Wizzair"* è **un**
insieme chiuso di alternative escluse: un fatto solo, con `not_in`. *"senza
glutine e senza lattosio"* ripete la negazione su ciascun elemento: sono
**due** esclusioni indipendenti, e restano due predicati. La forma
linguistica lo dice; appiattirle entrambe su `not_in` perde l'informazione
che le due esclusioni possono essere valutate separatamente.

---

## 6. Campi del predicato

```json
{
  "property": "price",
  "operator": "lt",
  "value": 10,
  "value_to": null,
  "unit": null,
  "currency": "EUR",
  "reference": null
}
```

### 6.1 `value`
- Numerico quando il concetto è numerico: `10`, non `"10"`.
- Numerali scritti a lettere si normalizzano: `dieci` → `10`. È normalizzazione
  sicura, non invenzione.
- Stringhe nella lingua dell'utente, minuscole, senza articoli:
  `"mortadella"`, non `"la mortadella"`.
  Eccezione: nomi propri, marchi e modelli mantengono la capitalizzazione
  originale: `"Milano"`, `"Ryanair"`, `"Zorba X7"`, `"Sphynx"`.

### 6.2 `unit`
Obbligatorio quando il numero non è denaro e ha un'unità:
`km`, `m`, `GB`, `TB`, `inch`, `months`, `years`, `kg`, `min`, `h`, `night`, `person`.
`null` altrimenti.

### 6.3 `currency`
Solo su proprietà monetarie. Codice ISO-4217: `EUR`, `USD`, `GBP`.

**La valuta va ancorata al testo, esattamente come ogni altro valore.**
Vale come ancora: un simbolo (`€`, `$`, `£`), una parola di valuta (`euro`,
`dollari`, `sterline`), oppure un termine esplicitamente monetario vicino al
numero (`spendere`, `spesa`, `costare`, `costo`, `prezzo`, `pagare`,
`budget`, `tariffa`).

Se non c'è **nessuna** di queste ancore, `currency` resta `null`.
`"Vorrei restare sotto i dieci"` da solo non autorizza `EUR`: la valuta si
deduce dalla localizzazione dell'utente, che è informazione di contesto e
appartiene al Capability Resolver a valle, esattamente come `stasera`.

Questa regola è più severa della v1 e ha un motivo misurato: senza di essa
il modello produceva `price eq 20 EUR` per `"Prenotami un tavolo alle 20"`,
e nulla lo fermava.

### 6.4 `reference`
Il punto di riferimento di una relazione: `"Milano"` in `entro 100 km da Milano`,
`"centro"` in `vicino al centro`. `null` altrimenti.

### 6.5 `value_to`
Solo con `between`. `null` altrimenti.

### 6.6 Liste
Per `in` / `not_in` il `value` è una **stringa con valori separati da `|`**
(lo schema v1 ammette solo scalari):
`{"property":"carrier","operator":"not_in","value":"Ryanair|Wizzair"}`.
Il Guard verifica che `in`/`not_in` abbiano almeno un `|` oppure un solo
elemento coerente.

---

## 7. `action` e `raw_action`

`action` è un verbo canonico in inglese, minuscolo, vocabolario aperto ma
con preferenza per:

```
find      buy       book      hire      contact   compare
rent      sell      schedule  send      create    monitor
notify    cancel    renew     order     subscribe
```

Regola: **l'azione è quella che l'utente chiede a Jase**, non quella che
l'utente farà. `"Cercami un elettricista"` → `find` (non `hire`), a meno che
l'utente chieda esplicitamente di ingaggiare/prenotare.

`"Mi serve un elettricista"` → `find`: esprime un bisogno di reperimento.
`"Prenotami…"` → `book`. `"Compra…"` → `buy`. `"Contatta…"` → `contact`.

`raw_action` è la porzione **letterale** del testo utente che esprime
l'azione, minuscola, senza punteggiatura: `"trovami"`, `"mi serve"`,
`"prenotami"`, `"se la trovi comprala"` → `"comprala"`.

> **Policy `raw_action` implicito:** lo schema richiede `minLength 1`, quindi
> quando il testo non contiene alcun verbo d'azione
> (`"panino mortadella milano <10€"`) si usa la costante letterale
> `"(implicito)"` e `action` è inferita dal contesto (default `find`).

---

## 8. `target`

- `target.name`: il nome **essenziale** della cosa richiesta, minuscolo,
  singolare, senza articoli e senza modificatori.
  `"un panino con la mortadella"` → `name = "panino"`.
  Nomi propri/modelli mantengono la forma originale: `"Zorba X7"`.

  **Regola del livello di specificità (vocabolario aperto).**
  `target.name` è l'entità **più specifica che l'utente ha nominato**, senza
  risalire a una categoria superiore. Risalire richiederebbe conoscenza del
  mondo — sapere che uno Sphynx è un gatto — e il vocabolario aperto è
  proprio l'assunto che questa conoscenza non ci sia (§2).

  | Testo | `target.name` | predicati |
  |---|---|---|
  | `uno Sphynx femmina` | `Sphynx` | `sex eq female` |
  | `un gatto Sphynx` | `gatto` | `breed eq Sphynx`, … |
  | `un Blorch X` | `Blorch X` | — |
  | `un allevatore di Zwergschnauzer` | `allevatore` | `breed eq Zwergschnauzer` |

  Nella prima riga `breed` **non** compare: sarebbe una duplicazione del
  target (§12, un fatto un ruolo). Nella seconda compare, perché l'utente ha
  nominato sia il genere sia la razza. Nella quarta il target è l'allevatore,
  non il cane: la razza è una proprietà del servizio richiesto.
- `target.type`: vocabolario aperto. Preferiti:
  `food`, `drink`, `product`, `electronics`, `service`, `professional_service`,
  `animal`, `place`, `accommodation`, `transport`, `real_estate`, `media`,
  `education`, `event`, `ticket`, `job`, `information`.
  In caso di dubbio reale → `null`. Mai inventare un tipo che non si sa.
- `target.raw`: il sintagma originale così come compare nel testo utente,
  articoli inclusi: `"un panino con la mortadella"`.

I modificatori che finiscono in `target.raw` **non vanno duplicati**: se
`mortadella` è già un predicato `ingredient`, non diventa parte di
`target.name`.

---

## 9. `clarification`

`clarification.required = true` **solo** quando manca l'informazione che
impedisce di capire *quale esito* l'utente desidera. Tipicamente:

- il target è assente o irrecuperabile (`"Prenotamelo per domani"`)
- l'azione è incompatibile con il target in modo irrisolvibile
- il testo è troppo vago per produrre un goal (`"aiutami"`)

**Non** si chiede chiarimento perché mancano prezzo, data, luogo o dettagli
di esecuzione: l'assenza di un vincolo è un'informazione legittima
("nessun vincolo di prezzo"), non un buco.

Quando `required = true`:
- `reason` è un codice snake_case: `target_missing`, `action_ambiguous`,
  `goal_unintelligible`, `conflicting_constraints`.
- `missing_fields` elenca i campi mancanti: `["target"]`, `["action"]`.
- `goals` deve comunque contenere almeno un goal (vincolo di schema): si usa
  il goal parziale con `target.name = "unknown"`, `type = null`,
  `raw = <frammento utente>`.

Quando `required = false`: `reason = null`, `missing_fields = []`.

---

## 10. Multi-goal

Si creano più elementi in `goals` quando l'utente esprime **obiettivi
indipendenti**, ciascuno con un proprio target.

`"Trovami un hotel a Roma e prenotami un treno per sabato"` → 2 goals.

**Non** sono multi-goal:
- più vincoli sullo stesso target
- un'azione composta sullo stesso oggetto (`"trova e prenota un hotel"`
  → un solo goal, `action = "book"`)

I vincoli espressi una sola volta ma validi per entrambi i goal
(`"a Roma"` in `"un hotel e un ristorante a Roma"`) si **ripetono** in
entrambi i goal: la ripetizione tra goal diversi non è una duplicazione.

---

## 11. Condizionali — `gate` (GoalContract v1.1)

### 11.1 Perché lo schema è stato evoluto

GoalContract v1 non sapeva rappresentare una struttura umana comunissima:

```
"Se la trovi sotto i 600 comprala, altrimenti dimmi solo dove l'hai vista."
"Comprami la scheda video solo se costa meno di 400 euro."
"Trova un volo per Madrid, e se ne trovi uno diretto prenotalo subito."
```

La policy di ripiego era `degrade_to_clarification`: serializzare il ramo
osservativo sicuro e chiedere conferma. Era difendibile come fail-closed, ma
su quattro iterazioni di training (v3, v4, v5, v6) il modello **non l'ha mai
imparata**: 0 su 3 ogni volta. E l'errore che produceva è proprio quello che
la policy voleva evitare — su *"comprala solo se costa meno di 400"*
generava `action: "buy"`, cioè l'azione irreversibile, senza condizione.

Un ripiego che il modello non impara e che, sbagliando, produce l'azione
pericolosa, non è un ripiego: è un buco. Da qui la v1.1.

### 11.2 Il campo `gate`

Un goal può portare un `gate` **opzionale** (assente o `null` per default):

```json
{
  "action": "find",
  "target": { "name": "scheda video", "type": "product", "raw": "la scheda video" },
  "attributes": [], "constraints": [], "preferences": [],
  "gate": {
    "when": { "property": "price", "operator": "lt", "value": 400, "currency": "EUR" },
    "then": "buy",
    "otherwise": null
  }
}
```

- `when` — un predicato, valutato **sul risultato trovato**, non sulla ricerca
- `then` — l'azione da eseguire se la condizione è soddisfatta
- `otherwise` — l'azione se non lo è; `null` significa "non fare nulla"

### 11.3 Regole

1. **`action` resta l'azione sempre eseguita**, ed è osservativa:
   `find`, `compare`, `monitor`. Un gate su un'azione già irreversibile
   (`buy`, `book`) non ha senso: se l'acquisto è già l'azione, non c'è più
   niente da condizionare. Il Guard lo rifiuta.
2. **La condizione non è un filtro di ricerca.** In *"comprala solo se costa
   meno di 400"* non si cercano solo le schede sotto i 400: si cerca la
   scheda, si guarda il prezzo, si decide. Quindi `price` va nel `gate`,
   **non** in `constraints` — e per §12 non può stare in entrambi.
3. **Se l'unica conseguenza è essere informati, non è un gate.** *"Avvisami
   se il prezzo scende sotto i 500"* si scrive `action: monitor` con
   `price lt 500` fra i `constraints`: un gate con `then: "notify"` su
   un'azione osservativa è una tautologia, perché `monitor` già significa
   "guarda e dimmelo". La condizione lì è il criterio di osservazione, non
   un bivio. Il bivio c'è quando il ramo vero fa qualcos'altro — comprare,
   prenotare, contattare.
4. **Un gate per goal.** Condizioni annidate o concatenate non sono
   rappresentabili: restano `clarification.required = true` con
   `reason = "conditional_not_representable"`.
5. `clarification` resta indipendente. *"Se la trovi sotto i 600 comprala"*
   ha un gate valido **e** un target mancante: il gate copre il condizionale,
   la clarification copre il target. Sono due problemi distinti e il
   contratto li dichiara separatamente.

### 11.4 Cosa NON è stato aggiunto, e perché

`depends_on` (un goal che ne attende un altro), `priority`, condizioni
composte con `and`/`or`. Nessuno dei tre ha oggi un caso nel benchmark che li
richieda, e §14 dice di aggiungere solo la semantica necessaria. Si
aggiungeranno quando un fallimento misurato lo imporrà, non prima.

### 11.5 Migrazione da v1

Un contratto v1 è un contratto v1.1 valido senza `gate`. Il Guard applica
`schema_version: "1.0"` → `"1.1"` come riparazione deterministica. Lo schema
v1 resta in `schema/goal_contract_v1.schema.json` per i test storici e per i
report già prodotti.

## 12. Anti-allucinazione — normalizzazioni ammesse

| Ammesso (normalizzazione) | Vietato (invenzione) |
|---|---|
| `dieci euro` → `10` + `EUR` | `stasera` → `20:00` |
| `€` → `EUR` | aggiungere `guests: 2` non detto |
| `32GB` → `32` + unit `GB` | aggiungere `rating gte 4` non detto |
| `femmina` → `female` | aggiungere email / telefono / nome |
| `rossa` → `rosso` (lemma) | aggiungere `date` perché "di solito serve" |
| `Milano` → `Milano` | espandere `Milano` in `Milano, Italia` |
| `<10€` → `lt 10 EUR` | dedurre `currency EUR` senza indizi |

Ogni `value` prodotto deve essere **tracciabile** a un frammento del testo
utente o a una normalizzazione di questa tabella. Il Guard applica una
verifica di ancoraggio (grounding) su questa base.

---

## 13. Invarianti verificati dal Semantic Guard (fail-closed)

1. JSON singolo, ben formato, non troncato, senza degenerazione ripetitiva.
2. Valido contro lo schema; nessun campo extra.
3. Nessun predicato duplicato dopo canonicalizzazione, né dentro un bucket
   né tra bucket diversi dello stesso goal.
4. Nessun conflitto diretto (`price lt 10` + `price gt 20` nello stesso goal).
5. Operatore compatibile col tipo del valore
   (`lt/lte/gt/gte` richiedono `value` numerico o temporale riconoscibile).
6. `currency` solo su proprietà monetarie; `value_to` solo con `between`.
7. `in`/`not_in` con valore serializzato coerente.
8. Ogni `value` stringa non banale ancorato al testo utente o alla §12.
9. `clarification` coerente (`required=false` ⇒ `reason=null` e
   `missing_fields=[]`).
10. `target.name` non vuoto e non uguale a un articolo.

Esito del Guard: `ACCEPT` | `REPAIRED` | `CLARIFY` | `REJECT`.
Nessun contratto in stato `CLARIFY` o `REJECT` prosegue verso il Planner.

Riparazioni **deterministiche e sicure** (e solo queste):
- rimozione di un duplicato esatto (si tiene l'occorrenza nel bucket a
  priorità più alta: `constraints` > `attributes` > `preferences`);
- normalizzazione di `currency` (`€`/`euro` → `EUR`);
- normalizzazione di stringhe numeriche (`"10"` → `10`) quando la proprietà
  è numerica;
- azzeramento di `value_to` quando l'operatore non è `between`;
- riempimento dei campi opzionali mancanti con `null`;
- rimozione di campi extra sconosciuti **solo** se vuoti o `null`.

Tutto il resto → `CLARIFY` o `REJECT`. Non si "indovina".
