# Registro delle iterazioni

Un'iterazione per sezione. Ogni sezione dice cosa si è cambiato, **perché**
(quale numero lo chiedeva) e cosa è successo. Le decisioni prese su intuito
invece che su evidenza vanno marcate come tali.

Tutti i numeri sono sul benchmark gold (99 casi, 81 mandatory), pipeline
GUARDED, `semantic_exact` come metrica di sintesi.

---

## Baseline — 20 settembre 2026

| modello | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | prec | bloccati |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3.5-2B nudo | 0.000 | 0 | 0.000 | 0.798 | 0.000 | – | – | 0.000 | – | 1.000 |
| jase-v1-iter400 | 0.062 | 66 | 0.070 | 0.960 | 0.051 | 0.900 | 0.650 | 0.328 | 0.550 | 0.283 |
| jase-v2-iter1500 | 0.272 | 62 | 0.271 | 0.980 | 0.232 | 0.821 | 0.932 | 0.639 | 0.696 | 0.141 |
| jase-v2-lowmem | 0.198 | 71 | 0.184 | 0.980 | 0.162 | 0.788 | 0.970 | 0.541 | 0.728 | 0.121 |

`acc_prec` = fra i contratti che superano il Guard, la frazione corretta.
È la metrica onesta dell'invariante fail-closed: il conteggio grezzo di
`accepted_but_wrong` premia chi non produce niente, e infatti alla prima
stesura il comparatore dichiarava "migliore" il modello base, che vince solo
perché il Guard gli blocca il 100% degli output. Bug corretto.

**Cosa dice la baseline.** Il dataset v2 e la semantica scritta valgono più
del fine-tuning da solo: rispetto al v1, recall dei fatti 0.33 → 0.64,
accuratezza degli operatori 0.65 → 0.93, contratti bloccati 28% → 14%.
Ma `mandatory` al 27% e `acc_wrong` a 62 dicono che siamo lontanissimi dal
gate di release.

**Nota sulla `bucket_accuracy` che "peggiora"** (0.900 → 0.821): non è un
peggioramento reale. Il v1 la calcolava su pochissimi fatti riconosciuti
(recall 0.33); il v2 ne riconosce il doppio e quindi si espone di più. Una
media su denominatori diversi non è confrontabile: è il tipo di numero che
fa prendere decisioni sbagliate se letto da solo.

### Tassonomia degli errori (jase-v2-iter1500)

```
 37  missing_fact                  fatti dell'utente persi
 30  hallucinated_fact             fatti non richiesti
 18  target_name                   nome del target diverso dall'atteso
 15  target_type                   tipo del target diverso
 13  action                        azione diversa
 12  value_error                   proprietà giusta, valore sbagliato
  9  bucket:constraints->attributes
  5  operator:gte->eq              comparativo appiattito
  5  bucket:constraints->preferences
```

---

## Iterazione 1 → v3

### Errori del benchmark, non del modello

Due casi gold erano sbagliati, e il modello aveva ragione.

1. **Specificità del target.** `nat-002` e `nat-012` pretendevano
   `target.name = "gatto"` da *"uno Sphynx femmina"*. Astrarre Sphynx in
   gatto richiede conoscenza del mondo, cioè esattamente l'assunto che il
   vocabolario aperto nega (§2). Con `"un Blorch X"` la stessa astrazione
   sarebbe impossibile. Nuova regola in §8: `target.name` è l'entità più
   specifica che l'utente ha nominato, e in quel caso `breed` **non** si
   ripete perché sarebbe un duplicato del target.

2. **Valuta inventata.** `nat-008` e `unit-005` pretendevano `EUR` da testi
   che non nominano nessuna valuta. Ma allora `"Prenotami un tavolo alle 20"`
   → `price eq 20 EUR` è coerente con le stesse regole, e infatti il modello
   lo produceva. Nuova regola in §6.3: **la valuta va ancorata al testo**
   (simbolo o parola di valuta); altrimenti resta `null` e la risolve il
   Capability Resolver dalla localizzazione, come fa con `stasera`.

### Un bug del generatore che *insegnava* l'errore

`f_price` sceglieva a caso il suffisso fra `"€"`, `" euro"` e `""`, ma
scriveva sempre `currency: "EUR"` nel contratto. Il dataset insegnava
letteralmente ad allucinare la valuta, e poi la misuravamo come errore del
modello. Ora la valuta è dichiarata solo se compare nella frase, e anche le
varianti contrastive la ricalcolano dalla propria frase invece di ereditarla.

Altri due difetti dello stesso tipo:
- `"mi serve"` stava sia fra i verbi di `find` sia fra quelli di `hire`:
  il dataset insegnava che la stessa frase ha due azioni diverse;
- la lambda che generava il sesso degli animali estraeva a caso **due volte**,
  una per il valore e una per la parola: `"maschio"` poteva finire con
  `sex eq female`.

### Un bug del Guard: il controllo che non scattava mai

Il rilevatore di strettezza dell'operatore era attivo, ma saltava i predicati
con operatore `eq` — cioè esattamente il caso più frequente
(`operator:gte->eq`, 5 occorrenze). Su 7 errori di operatore ne intercettava
**zero**. Corretto e promosso da `warn` a `error`: un contratto con
`stars eq 3` su *"da almeno 3 stelle"* ora viene fermato.

Nel farlo sono emersi due falsi positivi sul gold, entrambi da collisione di
numeri: in *"sotto i 300 euro e confrontami tre amplificatori"* il marcatore
`sotto i` veniva attribuito anche al `3`. Regola aggiunta: **un marcatore
governa il primo numero che lo segue**; se fra i due ce n'è un altro, è già
consumato. E se lo stesso numero compare più volte con indicazioni discordi,
il rilevatore tace — preferisce non segnalare che bloccare a caso.

### Riparazione aggiunta al Guard

Negazione inghiottita nel valore: `color eq "non rossa"` → `color neq "rossa"`,
e il predicato si sposta in `constraints`. È deterministico e reversibile,
quindi è una riparazione legittima, non un indovinello.

### Effetto delle sole correzioni al gold + valuta ancorata

| | mandatory | acc_wrong | acc_prec | bloccati |
|---|---|---|---|---|
| jase-v2-iter1500 | 0.272 | 62 | 0.271 | 0.141 |
| stesso modello, gold corretto | 0.296 | **52** | **0.325** | 0.222 |

Dieci contratti sono passati da "sbagliati e accettati" a "fermati". Il
modello non è cambiato di una virgola: è cambiato ciò che lasciamo passare.
È esattamente l'effetto voluto dal fail-closed, e si paga con più domande di
chiarimento (bloccati 14% → 22%).

### Dati correttivi (data/v3, 16.000 esempi)

Generati dalla distribuzione degli errori, non a intuito:

- **blocco numero-ruolo** — lo stesso numero in frasi quasi identiche come
  orario, prezzo, distanza, ospiti, durata. Più le ore scritte a lettere
  (*"alle dieci"*), che il v2 non aveva mai visto.
- **blocco comparativo** — forma piana contro forma comparativa sulla stessa
  proprietà: *"da 4 stelle"* (`eq`, attributo) contro *"da almeno 4 stelle"*
  (`gte`, vincolo), con valori plausibili per ogni proprietà.
- **blocco forza** — lo stesso fatto in sei forze: descrizione, obbligo,
  rinforzo, preferenza, esclusione, esclusione preferita.
- **blocco specificità** — *"un gatto Sphynx"* contro *"uno Sphynx"*.
- domini mancanti: media (il gold chiede `libro → media` e il v2 non aveva
  nessun libro) ed education.
- razze per specie: il v2 generava *"un pappagallo Labrador"*.

Configurazione: `iters` 1500 → 3000, perché con 16k esempi, batch 1 e
accumulo 4, tremila step vedono circa un'epoca; 1500 ne vedevano mezza.

### Risultato v3

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | bloccati |
|---|---|---|---|---|---|---|---|---|---|
| jase-v2-iter1500 | 0.296 | 47 | 0.347 | 0.980 | 0.253 | 0.817 | 0.933 | 0.663 | 0.273 |
| jase-v3-iter2000 | 0.420 | 48 | 0.422 | 0.980 | 0.353 | 0.894 | 0.947 | 0.624 | 0.162 |
| jase-v3-iter2500 | 0.370 | 41 | 0.422 | 1.000 | 0.303 | 0.863 | 0.944 | 0.685 | 0.283 |
| **jase-v3-iter3000** | **0.444** | 51 | **0.427** | **1.000** | **0.384** | 0.869 | **0.967** | 0.674 | **0.101** |

Validità JSON al 100%, operatori al 96.7%, contratti bloccati scesi al 10%.
I checkpoint **non** migliorano in modo monotono: iter2500 ha meno contratti
sbagliati accettati ma ne blocca il triplo, iter2000 ha bucket migliore e
recall peggiore. Sceglierli sulla validation loss sarebbe stato un sorteggio.

---

## Iterazione 2 → v4

### Il buco più grosso: nessuna richiesta senza verbo

Quattordici casi gold su 99 sono richieste senza verbo (*"Un hotel a 4 stelle
a Bologna"*, *"panino mortadella milano <10€"*). Il dataset non ne conteneva
**nemmeno una**. Il modello, non avendo un verbo da copiare, ne inventava uno:

```
testo   "Un hotel a 4 stelle a Bologna"
modello raw_action = "avrei bisogno di"
```

È l'allucinazione peggiore di tutte, perché mette in bocca all'utente parole
che non ha mai detto. Due interventi:

- **Guard**: `raw_action` è per definizione testo letterale dell'utente (§7),
  quindi deve comparire nel testo o valere la costante `"(implicito)"`.
  Controllo deterministico, zero falsi positivi sul gold.
- **Dataset**: il 9% degli esempi ora non ha verbo. Ha richiesto di rendere
  il contratto dipendente dalla SUPERFICIE e non solo dallo scenario: lo stile
  telegrafico toglie il verbo dal testo, quindi quella resa deve dichiarare
  `"(implicito)"` anche se lo scenario aveva un verbo.

### Sovra-correzione dell'iterazione precedente

Il blocco "forza" generava 1 variante-attributo contro 3 vincoli e 2
preferenze, e il modello ha imparato a spingere tutto in `constraints`:
`bucket:attributes->constraints` è passato da 1 a 10 casi. Riequilibrato.
È il rischio strutturale dei dati correttivi: curano un errore e ne creano
l'opposto, e senza le metriche per bucket non te ne accorgi.

### Un sinonimo che insegnavo per sbaglio

Il generatore aveva target come `("bici", ["una bici", "una bicicletta"])`:
la frase *"una bicicletta"* produceva `target.name = "bici"`. Insegnava a
normalizzare il lessico, che è l'opposto di §8 (il nome viene dal testo). Ora
ogni forma ha il proprio target.

### Il generatore come fuzzer del Guard

Promuovere il rilevatore di strettezza da avviso a blocco ha significato
cercarne i falsi positivi, e il generatore li ha trovati a decine di migliaia
di esempi. Ognuno era un modo diverso di attribuire un marcatore al numero
sbagliato:

| frase | cosa andava storto |
|---|---|
| `Buongiorno, ... sotto i 500€` | la finestra tagliava "Buongior-no" e quel `no` sembrava una negazione |
| `sabato meno di 250` | `o meno` annidato dentro *sabatO MENO* |
| `24 pollici max 200` | il `max` è del 200, non del 24 |
| `il Talin Max sotto i 700` | `Max` è un nome proprio |
| `32 GB di RAM, ... Zorba Max` | marcatore in un'altra proposizione |
| `entro lunedì, 1000€` | l'`entro` è del lunedì |
| `entro dopodomani da 6 mesi` | idem, senza virgola |
| `niente gatti magari meno di 120` | il `niente` governa i gatti, non la soglia |
| `1500 euro proprio massimo 2 GB` | marcatore stretto fra due numeri |
| `Trovami un panino sotto i 10` | `un` trattato come numerale invece che articolo |

Regole finali: un marcatore prefisso deve essere **incollato** al suo numero
(nemmeno una parola in mezzo); un marcatore suffisso deve **chiudere** la
frase e non stare oltre una virgola; un marcatore stretto fra due numeri
**tace**. Si perde qualche rilevazione — `meno di ben 600` non scatta più —
ma questo controllo blocca contratti, quindi la precisione vale più della
copertura.

### Ambiguità dichiarata invece che arbitrata

Il gold pretendeva `storage` da *"32gb"* in un caso e `ram` da *"16 GB"* in un
altro: stessa forma linguistica, risposta opposta. Era incoerente, e per di
più nessun essere umano saprebbe scegliere senza sapere che cos'è un
"Zorba X7". I casi ora dichiarano `equivalences: [["ram", "storage"]]` e
l'evaluator le rispetta. Non è un modo per far passare il modello: è il modo
di ammettere che la domanda non ha una risposta canonica unica, invece di
premiarne una a caso.

### Refusi che cambiavano il senso

La rete di sicurezza sui refusi controllava i valori ma non il resto. Sono
passati `"on più di 20"` (da `non più di`, che ribalta il vincolo da tetto a
minimo) e `"100k msiena"` (da `100km siena`, che scollegava il riferimento
spaziale). Ora un refuso viene annullato se rompe l'ancoraggio di valore,
valuta, riferimento, `raw_action` **o** il senso del comparativo.

Dataset v4: 16.000 esempi, 16.000 su 16.000 accettati dal Guard con
ancoraggio attivo, zero famiglie condivise fra train e valid.

### Risultato v4 (Guard finale, numeri confrontabili)

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | prec | bloccati |
|---|---|---|---|---|---|---|---|---|---|---|
| jase-v2-iter1500 | 0.296 | 47 | 0.347 | 0.980 | 0.253 | 0.817 | 0.933 | 0.663 | 0.714 | 0.273 |
| jase-v3-iter3000 | 0.420 | 39 | 0.480 | 1.000 | 0.364 | 0.871 | 0.968 | 0.685 | 0.770 | 0.242 |
| jase-v4-iter2000 | 0.494 | 46 | 0.483 | 1.000 | 0.434 | **0.914** | **0.983** | 0.641 | 0.758 | 0.101 |
| **jase-v4-iter2500** | **0.518** | 49 | 0.473 | 0.990 | **0.444** | 0.874 | 0.953 | **0.702** | 0.799 | 0.061 |
| jase-v4-iter3000 | 0.481 | 55 | 0.433 | 0.980 | 0.424 | 0.914 | 0.957 | 0.641 | **0.817** | 0.020 |

**BEST = jase-v4-iter2500.** Progressione dei casi mandatory:
v1 6% → v2 30% → v3 42% → v4 52%.

Un andamento che vale la pena notare: più iterazioni, meno contratti bloccati
(10% → 6% → 2%) ma più contratti sbagliati accettati (46 → 49 → 55). Il
modello diventa più sicuro di sé, e il Guard trova meno appigli per fermarlo.
Più addestramento non è gratis: sposta errori da "fermati" a "silenziosi".

---

## Iterazione 3 → v5

Tre buchi netti nel dataset, tutti trovati guardando i tag con punteggio zero.

### Preferenze: 0 su 7

Il generatore metteva il marcatore debole **sempre prima** del fatto
("possibilmente con X"). Il gold lo ha spesso dopo o come coda:

```
"oggi pomeriggio se possibile"                 → il v4 lo mette in constraints
"magari che parli inglese, ma non è indispensabile"  → il v4 legge `non` come
                                                        negazione e produce neq
"se possibile con parcheggio e colazione"      → il v4 ne mette una in
                                                 attributes e una in constraints
```

Aggiunti: marcatori posticipati (`se riesci`, `ma non è indispensabile`,
`se capita`) e un blocco in cui **un solo marcatore governa due fatti
coordinati**, con il contrasto dello scope stretto (`con A, e magari con B`).

### Operatori di lista: 0 su 3

Il dataset non conteneva **nessun** `in` / `not_in`. Su
*"né con Ryanair né con Wizzair"* il modello produceva due attributi separati
`carrier eq Ryanair` e `carrier eq Wizzair` — che semanticamente vuol dire
l'opposto. Aggiunto un blocco con `o X o Y`, `X oppure Y`, `né X né Y`,
`senza X e senza Y`, più il contrasto del valore singolo.

### Inglese: 0 esempi

Il dataset era interamente in italiano; il gold ha 3 casi inglesi e il v4 ne
prendeva 1. Aggiunto un blocco inglese al 6.5%.

Prima stesura sbagliata e rifatta: era combinatoria e produceva
*"a plumber with at least 4 rooms"*, *"a restaurant red"* — esattamente il
difetto del dataset v1 che questa riscrittura doveva eliminare. Ogni target
inglese ora porta solo le proprietà che gli competono.

### Ancora il generatore come fuzzer

Altri quattro falsi positivi del rilevatore di soglie, tutti su casi che un
umano scioglierebbe senza pensarci:

| frase | cosa andava storto |
|---|---|
| `15 pollici sarebbe il massimo` | `il massimo` è un modo di dire, non una soglia |
| `ma non è un problema se no, sotto 800` | `se no` è "altrimenti", non una negazione |
| `senza superare i 80€. Poi ... Farnex Max` | `len(str(80.0))` conta 4 caratteri e il punto finiva fuori dal controllo |

L'ultimo è il più istruttivo: un baco di tipo, non di linguistica. Misurare
la lunghezza di un numero sulla sua rappresentazione float invece che sul
testo spostava di due caratteri la finestra di controllo della punteggiatura.

Dataset v5: 20.000 esempi, 20.000 su 20.000 accettati dal Guard.

### Risultato v5

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | prec | bloccati |
|---|---|---|---|---|---|---|---|---|---|---|
| jase-v4-iter2500 | 0.518 | 49 | 0.473 | 0.990 | 0.444 | 0.874 | 0.953 | 0.702 | 0.799 | 0.061 |
| jase-v5-iter2500 | 0.531 | 45 | 0.505 | 1.000 | 0.465 | 0.884 | 0.946 | **0.713** | 0.817 | 0.081 |
| **jase-v5-iter3000** | **0.531** | **45** | **0.505** | 1.000 | **0.465** | **0.905** | **0.984** | 0.696 | 0.787 | 0.081 |
| jase-v5-iter3500 | 0.481 | 54 | 0.438 | 1.000 | 0.424 | 0.875 | 0.967 | 0.663 | **0.863** | 0.030 |

I tag che avevo preso di mira si sono mossi tutti:

| tag | v4 | v5 |
|---|---|---|
| implicit-action (n=14) | 0.29 | **0.57** |
| preference (n=7) | 0.00 | **0.29** |
| number-role (n=8) | 0.38 | **0.50** |
| open-vocab (n=8) | 0.12 | **0.25** |
| natural (n=20) | 0.10 | **0.20** |
| list-operator (n=3) | 0.00 | **0.67** |

Ma il totale è salito poco (52% → 53%): ogni tag riparato vale pochi casi, e
nel frattempo qualcosa si muove in negativo altrove. È il momento in cui i
guadagni facili sono finiti.

---

## Iterazione 4 → v6

### L'errore più ostinato

Dopo due tornate di dati correttivi, `Prenotami un tavolo alle 20` produce
ancora `guests eq 20`, e `alle dieci` diventa `price eq 10`. Il blocco
numero-ruolo esisteva già: evidentemente non bastava vedere il contrasto,
serve vederlo **molto più spesso e in forma più difficile**.

- blocchi correttivi dal 20% al 28% del dataset, con il numero-ruolo a farne
  la fetta maggiore;
- nuovo blocco a **due numeri nella stessa frase con ruoli diversi**
  (`un tavolo per 4 alle 21`, `un idraulico entro 20 km sotto i 100 euro`):
  il caso a un numero solo non insegna a tenerne due separati, ed è lì che il
  modello collassava.

### Due controlli nuovi nel Guard

Il caso `alle 20 → guests eq 20` passava indenne: il 20 nel testo c'è davvero,
quindi l'ancoraggio non ha niente da dire. Serviva un controllo diverso.

**Ruolo del numero** (`number_role_mismatch`). La preposizione che precede un
numero, o l'unità che lo segue, ne fissano la classe: `alle 20` è un orario,
`20 euro` è denaro, `20 km` una distanza, `20 persone` una quantità. Se la
proprietà del predicato appartiene a un'altra classe, il contratto si ferma.
Stessa disciplina delle soglie: se lo stesso numero compare due volte con
indicazioni discordi, il controllo tace.

**Parole di cortesia come valori** (`filler_as_value`). Da
*"ciao! senti, mi servirebbe una pizza…"* il modello estraeva
`ingredient contains senti`. Anche qui l'ancoraggio era soddisfatto — "senti"
è nel testo. Una lista chiusa di aperture e formule di cortesia risolve il
caso in modo deterministico.

### Ancora falsi positivi, e uno era un baco di confini

| frase | cosa andava storto |
|---|---|
| `Talin Max 13 pollici` | `Max` maiuscolo a metà frase è un nome proprio: ora si guarda il testo ORIGINALE, non la versione minuscola |
| `min 50€ 50km verbania` | `\b50\b` non trova il 50 dentro `50km` (la `k` è un carattere di parola): il rilevatore vedeva una sola occorrenza invece di due e perdeva l'ambiguità che doveva farlo tacere |

Il secondo è il più insidioso di tutta la serie: non produceva un errore
visibile, rendeva solo il controllo *troppo sicuro di sé*. Sostituiti i
confini di parola con confini numerici `(?<!\d)…(?!\d)`.

E una fonte di ambiguità che fabbricavo io: i nomi inventati finivano in
`Max`, quindi il dataset conteneva `prax max 16gb`, indistinguibile in
minuscolo da "al massimo 16 GB". Tolto dal generatore.

Dataset v6: 22.000 esempi, 21.999 su 22.000 accettati dal Guard.

### Risultato v6 — nessun guadagno netto

Confronto a parità di Guard:

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | bloccati |
|---|---|---|---|---|---|---|---|---|---|
| **jase-v5-iter3000** | **0.531** | 42 | 0.523 | 1.000 | **0.465** | 0.905 | 0.984 | 0.696 | 0.111 |
| jase-v6-iter2500 | 0.494 | 47 | 0.472 | 1.000 | 0.424 | 0.875 | 0.938 | **0.707** | 0.101 |
| jase-v6-iter3000 | 0.506 | **34** | **0.564** | 0.929 | 0.444 | 0.906 | 0.966 | 0.646 | 0.212 |
| jase-v6-iter3500 | 0.531 | 52 | 0.464 | 1.000 | 0.455 | **0.912** | 0.968 | 0.691 | 0.020 |

La correzione mirata ha funzionato dove puntava — `number-role` 0.50 → 0.625 —
ma il totale non si è mosso. **Il v5 resta il BEST.** Aggiungere altri dati
correttivi sugli stessi assi ha smesso di pagare.

### Due ipotesi strutturali, testate e falsificate

Prima di aggiungere altri dati, valeva la pena verificare due sospetti che
avrei potuto coltivare per settimane.

**Gli esempi vengono troncati da `max_seq_length: 512`?** Sarebbe stata una
spiegazione elegante di `missing_fact`: un esempio tagliato perde i fatti
finali e insegna a perderli. Misurato: l'esempio più lungo del dataset è di
**473 token**, il 100% sta dentro 448. Ipotesi morta.

**I 5 output "troncati" in valutazione sono un artefatto del limite di token?**
Rieseguita la valutazione con `--max-tokens 760` invece di 420: metriche
**identiche**, cifra per cifra. Non sono troncamenti da limite, sono
degenerazioni vere — il modello produce JSON che non chiude mai (§18 della
specifica). Il Guard li rifiuta, che è il comportamento giusto.

Due ipotesi eliminate in venti minuti, prima di spendere un'ora di training
su ciascuna.

---

## Iterazione 5 → GoalContract v1.1 e capacità

Restano due leve, e si tirano in parallelo perché toccano variabili diverse.

### Leva 1 — lo schema: il campo `gate`

Promessa mantenuta: i condizionali non si sono mossi in quattro iterazioni
(0 su 3 in v3, v4, v5, v6), quindi la policy di ripiego va sostituita, non
rinforzata. Vedi §11 di SEMANTICS.md per il disegno completo.

```json
"gate": {
  "when": { "property": "price", "operator": "lt", "value": 400 },
  "then": "buy",
  "otherwise": null
}
```

Tre regole non negoziabili, tutte verificate dal Guard:

1. `action` resta **osservativa** (`find`, `compare`, `monitor`). Un gate su
   `buy` non condiziona niente, perché l'acquisto è già l'azione.
2. La condizione **non è un filtro di ricerca**. In *"comprala solo se costa
   meno di 400"* non si cercano le schede sotto i 400: si cerca la scheda, si
   guarda il prezzo, si decide. Se lo stesso fatto compare nel gate e in
   `constraints`, il contratto si ferma (§12, un fatto un ruolo).
3. La condizione è soggetta all'ancoraggio come ogni altro valore.

`clarification` resta indipendente: *"Se la trovi sotto i 600 comprala"* ha
un gate valido **e** un target mancante, e il contratto dichiara le due cose
separatamente. Che i due problemi si scompongano senza interferire è la
prova migliore che il disegno regge.

**Non** sono stati aggiunti `depends_on`, `priority`, condizioni composte:
nessuno ha oggi un caso nel benchmark che li richieda (§14).

Migrazione: un contratto v1 è un contratto v1.1 senza gate, e il Guard
applica `"1.0"` → `"1.1"` come riparazione deterministica. Lo schema v1 resta
su disco per i report storici.

Il benchmark passa da 99 a 102 casi (85 mandatory): i tre condizionali
esistenti riscritti con il gate, più tre nuovi — due rami, `monitor`,
e un `contact` condizionato.

### Leva 2 — la capacità

Esperimento pulito: **stessi identici dati del v6**, solo più capacità
(8 → 16 strati, rank 16 → 32, da 5.6M a 22.4M parametri allenabili). Se il
plateau dipende dai dati, non si muoverà nulla; se dipende dalla capacità,
si vedrà. Tenere le due variabili separate è l'unico modo per sapere quale
delle due stavo esaurendo.

Nessun OOM con sequenze da 512 e grad checkpoint attivo, circa due ore.

### Risultati: capacità e schema v1.1

Tutti sul gold a 102 casi, stesso Guard, quindi confrontabili fra loro.

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall |
|---|---|---|---|---|---|---|---|---|
| jase-v5-iter3000 (schema 1.0) | 0.506 | 45 | 0.505 | 1.000 | 0.451 | 0.905 | **0.984** | 0.706 |
| jase-v7cap-iter3000 (4× capacità) | 0.412 | **33** | 0.522 | 0.912 | 0.353 | **0.955** | 0.964 | 0.622 |
| jase-v8-iter2500 (schema 1.1) | 0.506 | 42 | **0.528** | 1.000 | 0.461 | 0.926 | 0.956 | **0.750** |
| **jase-v8-iter3500** (schema 1.1) | **0.518** | 43 | 0.522 | 0.990 | **0.461** | 0.888 | 0.955 | 0.744 |

#### La capacità peggiora le cose

Quattro volte i parametri (5.6M → 22.4M), stessi identici dati: casi
mandatory da 0.506 a **0.412**, contratti esatti da 0.451 a **0.353**,
validità JSON da 1.000 a 0.912. L'unica metrica migliorata è
`bucket_accuracy` (0.905 → 0.955): con più capacità il modello impara meglio
il ruolo dei fatti, ma ne perde di più e degenera più spesso.

Il plateau **non è capacità**. È quasi certamente sovradattamento: 22M
parametri su 22k esempi ripetitivi imparano il generatore, non la lingua.
L'esperimento è costato due ore e ha chiuso una domanda che altrimenti
sarebbe rimasta aperta a ogni iterazione successiva.

#### Lo schema paga, ma poco

La recall dei fatti sale a **0.750**, il massimo mai misurato, e i contratti
esatti salgono di un punto. I condizionali passano da 0 su 6 a 1 su 6, con
`gate_accuracy` a 0.57: **la struttura viene imparata**, i dettagli no.

**Nuovo BEST: jase-v8-iter3500.**

### Perché i gate sbagliano ancora — tre lacune del generatore

Guardando i sei casi uno per uno:

| caso | cosa produce il modello |
|---|---|
| `cond-002` "Comprami X **solo se** costa meno di 400" | gate perfetto, ma `action: buy`. **Il Guard lo ferma** — fail-closed che funziona — però il modello continua a mettere l'azione condizionata dove non va |
| `cond-003` "se ne trovi uno **diretto** prenotalo" | nessun gate: `diretto` finisce in `preferences` |
| `cond-006` "**se è libero oggi** contattalo" | nessun gate: la condizione diventa un vincolo di data |
| `cond-001` "**Se la trovi** sotto i 600 comprala" | gate perfetto, ma `target.name = "se"` e nessuna richiesta di chiarimento |

Il filo comune è il generatore, non il modello:

1. **Generava solo gate sul prezzo.** I due casi con condizione diversa
   (`route`, `availability`) fallivano entrambi perché il modello non aveva
   mai visto un gate che non parlasse di soldi.
2. **Non generava mai l'azione condizionata in apertura.** Tutti i template
   iniziavano con `Cercami`/`Trovami`; nessuno con `Comprami X solo se…`,
   che è proprio la forma su cui il modello sbaglia.
3. **Non generava target anaforici.** "Se **la** trovi sotto i 600" non ha un
   target nominato: serve un gate valido *più* una richiesta di chiarimento,
   e le due cose vanno tenute separate.

Tutte e tre chiuse nel v9. È il quinto giro in cui l'errore del modello si
rivela un buco del dataset, non un limite del modello.

### Risultato v9 — i gate si sbloccano

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | bloccati |
|---|---|---|---|---|---|---|---|---|---|
| jase-v5-iter3000 (schema 1.0) | 0.506 | 45 | 0.505 | 1.000 | 0.451 | 0.905 | 0.984 | 0.706 | 0.108 |
| jase-v8-iter3500 (schema 1.1) | 0.518 | 43 | 0.522 | 0.990 | 0.461 | 0.888 | 0.955 | 0.744 | 0.118 |
| **jase-v9-iter2500** | **0.541** | 44 | **0.532** | **1.000** | **0.490** | 0.920 | 0.944 | 0.694 | **0.078** |
| jase-v9-iter3000 | 0.482 | 56 | 0.434 | 1.000 | 0.422 | 0.843 | 0.967 | 0.672 | 0.029 |
| jase-v9-iter3500 | 0.471 | 54 | 0.443 | 1.000 | 0.422 | 0.919 | 0.944 | 0.689 | 0.049 |

Le tre lacune chiuse hanno pagato dove dovevano:

| | v8 | v9 |
|---|---|---|
| condizionali (n=6) | 0.17 | **0.50** |
| gate a due rami (n=2) | 0.00 | **0.50** |
| accuratezza del gate | 0.33 | **0.57** |

E si aggiunge un risultato che non avevo cercato: **zero REJECT**. Nessun
output degenerato, nessun JSON troncato, validità al 100%. La degenerazione
che il v1 mostrava e che riaffiorava a ogni iterazione è sparita.

**Nuovo BEST: jase-v9-iter2500.** Casi mandatory 6% → 54%.

---

## Il controllo che mancava: copertura del benchmark

Cinque volte su sei, quello che sembrava un limite del modello era un buco
del dataset — e ogni volta l'ho scoperto **dopo** un'ora di training,
leggendo i fallimenti uno per uno. `scripts/coverage_check.py` fa la stessa
scoperta in due secondi e **prima** di allenare.

Estrae da ogni esempio le caratteristiche **strutturali** — non il dominio,
non il lessico, ma la forma semantica: ha un gate? un operatore di lista? una
negazione? due numeri con ruoli diversi? nessun verbo? — e confronta la
distribuzione del dataset con quella del benchmark. Una forma che il gold
chiede e il dataset non contiene è un buco garantito, non un rischio.

Sul dataset v9:

```
Nessun buco strutturale: ogni forma richiesta dal benchmark
ha una controparte nel dataset.
```

Applicato retroattivamente avrebbe segnalato in due secondi, e prima di
spendere ore di training:

| iterazione | cosa avrebbe detto |
|---|---|
| v3 → v4 | `implicit-action`: il gold la chiede al 13.7%, il dataset non ce l'ha |
| v4 → v5 | `list-operator`, `english`: il gold le chiede, il dataset non le ha |
| v8 → v9 | `gate-non-price`: il gold la chiede, il dataset non ce l'ha |

È lo strumento che avrei dovuto scrivere per primo. Ora esiste, e chiude
l'era dei buchi strutturali: gli errori che restano non sono categorie
assenti, sono imprecisioni dentro categorie già coperte. La prossima leva
dovrà essere un'altra.

---

## Il controllo delle omissioni

Dei 44 contratti sbagliati che superavano il Guard, **30 erano omissioni**:
il modello non sbaglia un fatto, ne perde uno. Nessun controllo poteva
vederlo, perché in ciò che produce non c'è niente di storto — c'è solo un
buco. Tutti i controlli fino a qui guardavano il contratto; questo guarda il
**testo** e cerca frammenti che devono per forza aver prodotto un predicato.

È volutamente conservativo, due sole forme ad altissima precisione:

- **`senza X` / `niente X`** — una negazione esplicita deve produrre un
  predicato negativo. Se manca, il modello ha ignorato una richiesta.
- **un numero seguito da valuta o unità** (`80 euro`, `20 km`, `32 GB`,
  `3 stelle`) — è un fatto, non rumore. Se non compare in nessun predicato
  né nel gate, manca.

Due eccezioni, entrambe trovate misurando:

- un numero dentro il **nome del target** non è un vincolo (`iPhone 15`,
  `Fiat 500`, `RTX 6080`);
- **`senza` davanti a un infinito** è una soglia, non un'esclusione:
  *"senza arrivare a spendere 10€"*, *"senza superare i 500"*. Era l'unico
  falso positivo su 102 casi gold.

Risultato: **0 falsi positivi su 102 casi gold**, e tre contratti sbagliati
in meno che raggiungono il Planner.

| | acc_wrong | acc_prec | mandatory | bloccati |
|---|---|---|---|---|
| jase-v9-iter2500 | 44 | 0.532 | 0.541 | 0.078 |
| **stesso modello, Guard con omissioni** | **41** | **0.549** | 0.541 | 0.108 |

Il modello non è cambiato: è cambiato ciò che lasciamo passare. Tre contratti
si spostano da "sbagliati e accettati" a "fermati", e nessun contratto
corretto viene toccato. È esattamente la forma di guadagno che il fail-closed
chiede.

---

## Iterazione 7 → v10: i buchi LESSICALI

`coverage_check.py` diceva "nessun buco", ma guardava solo la **forma** della
richiesta. Esteso al **lessico** — proprietà, tipi di target, azioni — il
verdetto è cambiato:

```
MAI GENERATE: arrival_time, quantity, section, year, mileage, seller,
              shipping_cost, delivery, certification,
              type:ticket, type:service, action:cancel
```

Dodici cose che il benchmark chiede e che il generatore non ha mai prodotto
in 25.000 esempi. Il caso più netto: *"devo arrivare prima delle 18"* vuole
`arrival_time`, e il modello rispondeva `time` — la proprietà generica —
semplicemente perché l'altra non esisteva nei suoi dati. Non era un errore di
comprensione: era una parola che non aveva mai visto.

Più cinque pattern dall'analisi dei 39 mandatory falliti:

| pattern | cosa faceva il modello |
|---|---|
| gate preceduto da un vincolo | metteva nel gate la prima cosa che vedeva: *"hotel a Lisbona e se costa meno di 80"* → `gate{location near Lisbona}` |
| luogo in prima posizione | *"A Milano vorrei un panino…"* → location persa |
| multi-goal con modificatore condiviso | *"un idraulico e un elettricista a Torino"* → il secondo goal spariva |
| azione composta | *"trova e prenota"* → `find` invece di `book` |
| `"ricondizionato va benissimo"` | letto come marcatore debole → preferences |

### Risultato v10

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall | prec |
|---|---|---|---|---|---|---|---|---|---|
| jase-v9-iter2500 | 0.541 | 41 | 0.549 | 1.000 | 0.490 | 0.920 | 0.944 | 0.694 | 0.781 |
| jase-v10-iter2500 | 0.506 | **35** | 0.568 | 1.000 | 0.451 | 0.871 | 0.971 | **0.778** | 0.714 |
| jase-v10-iter3000 | 0.588 | 45 | 0.541 | 1.000 | 0.520 | 0.900 | 0.962 | 0.722 | **0.878** |
| **jase-v10-iter3500** | **0.600** | 40 | **0.579** | 1.000 | **0.539** | **0.925** | 0.925 | 0.744 | 0.827 |

**Casi mandatory dal 54% al 60%.** I buchi lessicali erano il freno.

---

## Iterazione 8 → v11: il parlato

Con il lessico a posto, il blocco più grosso diventa il tag `natural`:
20 casi al 10%. Sono le frasi lunghe e disordinate, cioè quelle vere.
Analizzati tutti e diciotto, emergono cinque famiglie.

### I modi di dire del rifiuto

Tre casi su venti sbagliavano la stessa cosa:

```
"dieci euro per un panino proprio non li spendo"  → il modello: price eq 10
"Dieci euro però non li spendo."                  → il modello: price eq 10
"senza arrivare a spendere 10€"                   → il modello: price lte 10
```

In italiano parlato il tetto di spesa quasi mai si dice con *"sotto i"*: si
dice rifiutando la cifra. Il generatore aveva tre di queste forme su
venticinque; ora sono dieci e pesano il doppio. E il rilevatore di soglie del
Guard le riconosce, quindi l'errore viene anche **fermato** quando capita.

Tre vincoli di precisione, tutti trovati generando: l'idioma non vale oltre
una virgola (*"con 1 posti, a Trieste, venerdì, non voglio spendere 150€"*
parla dei 150, non del posto), non vale se fra il numero e l'idioma c'è un
altro numero, e la finestra di ricerca deve essere di 90 caratteri — a 40 il
caso più lungo veniva tagliato a metà.

### Le altre quattro

- **due frasi, un goal solo**: *"…qui a Milano? Vorrei restare sotto i dieci."*
  → il modello creava due goal
- **`"dal 12 al 14"`** è un intervallo di date → il modello produceva
  `time eq 12:00` e `time eq 14:00`
- **`"aperto la domenica"`** è disponibilità, non la data della richiesta
- **la negazione si attacca a UN elemento**: in *"una stampante 3d usata, che
  non sia cinese"* il modello propagava la negazione all'indietro e produceva
  `condition neq usata`

### Un'altra incoerenza del benchmark, corretta

`nat-003` pretendeva `target = "elettricista"` da *"Serve qualcuno per
sistemare una presa"*. Ma sapere che le prese sono lavoro da elettricisti è
conoscenza del mondo — **esattamente l'astrazione che §8 vieta**, la stessa
che avevo già tolto per "Sphynx" sei iterazioni fa. L'utente ha detto
"qualcuno"; mappare `specialization: presa` su un elettricista è lavoro del
Capability Resolver.

È la terza volta che una regola scritta nel documento non era applicata in
tutto il benchmark. Vale la pena, prima o poi, verificarlo meccanicamente.

### Risultati del v11

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | operator | recall |
|---|---|---|---|---|---|---|---|---|
| jase-v10-iter3500 | 0.600 | 37 | 0.598 | 1.000 | 0.539 | 0.925 | 0.925 | 0.744 |
| jase-v11-iter3000 | 0.518 | 43 | 0.522 | 1.000 | 0.461 | 0.932 | 0.932 | 0.650 |
| jase-v11-iter3500 | 0.529 | 46 | 0.505 | 1.000 | 0.461 | 0.916 | 0.947 | 0.728 |
| **jase-v11-iter4000** | **0.624** | 39 | 0.585 | 1.000 | 0.539 | **0.945** | **0.953** | 0.706 |

Il tag `natural` sale da 0.100 a 0.250: i modi di dire del rifiuto sono stati
imparati. Ma `accepted_but_wrong` non si muove (37 → 39), e il gate di
release chiede zero. Vale la pena fermarsi a guardare *perché*.

---

## Iterazione 9 → v12: l'omissione

Il conto dei fallimenti del v11 dice una cosa sola, e la dice forte:

```
missing_fact        35
hallucinated_fact   15
value_error          8
target_name          7
```

Il modello non sbaglia i fatti. **Ne scrive uno e si ferma.** Invece di
guardare altri diciotto casi uno per uno, per la prima volta ho misurato la
forma del dataset e l'ho confrontata con quella del benchmark:

```
fatti per goal     0      1      2      3      4
training (v11)   0.109  0.572  0.117  0.097  0.061
gold             0.105  0.390  0.238  0.219  0.048
```

Il 57% degli esempi di training aveva **un fatto solo**, contro il 39% del
gold; due o tre fatti erano il 21% del training contro il 46% del gold. La
causa è nei blocchi correttivi: insegnano una cosa per volta, e per farlo
hanno frasi con un fatto solo. Iterazione dopo iterazione erano diventati
**metà del dataset**, e il modello ha imparato benissimo anche la lezione che
non volevo dargli.

Nessun difetto di capacità, nessun limite del 2B: una distribuzione storta
che nessuno aveva mai guardato.

### Cosa cambia nei dati

- il numero di fatti per frase si decide **una volta sola**, con una
  distribuzione dichiarata e confrontabile col gold, invece di emergere da
  venti tiri di dado indipendenti (che producevano o un fatto o otto);
- i blocchi correttivi ricevono un **contorno** coerente col tipo di target,
  identico per tutte le varianti di un blocco perché il contrasto resti di
  un elemento solo. Nel 45% dei casi il contorno è una proprietà
  *descrittiva*: è quella che il modello lasciava per strada;
- i blocchi correttivi scendono dal 28% al 17%: il lavoro l'hanno fatto
  (bucket 0.945, operatori 0.953), continuare a pesarli tanto costa solo
  esempi poveri.

### Quattro contraddizioni fra generatore, lessico e benchmark

Tutte trovate dagli strumenti, non leggendo i casi:

1. **`destination` era descrittiva nel lessico e vincolo nel gold.** Il
   generatore, per giunta, attaccava `location near Roma` a un volo nel 23%
   dei goal `transport`: su sei casi gold il modello scriveva `location` al
   posto di `destination`, come gli era stato insegnato.
2. **`"senza A e senza B"` erano due fatti nel gold e una lista nel
   generatore.** Il generatore insegnava l'errore che poi misuravo.
3. **Tre casi gold portavano `currency: EUR` senza nessuna valuta nel
   testo**, in violazione del §6.3 che avevo scritto io. Il modello faceva
   la cosa giusta e prendeva un errore.
4. **`date` e `location` finivano fra gli `attributes`** in 5 esempi su 100,
   perché il generatore tirava il ruolo a caso senza passare dalla cascata.

### Il Guard impara tre controlli nuovi

Valgono in produzione, non solo sul benchmark:

- **la cascata §2 come riparazione deterministica**, applicata solo quando
  nel testo non c'è né un marcatore debole né un obbligo — cioè solo quando
  R1 e R3 non possono entrare in gioco e il bucket è funzione di operatore e
  proprietà;
- **polarità**: *"no rosso"* non può diventare `color eq rosso`;
- **ruolo del numero anche dopo la normalizzazione oraria**: da *"un
  idraulico entro 20 euro"* non passa più `time before 20:00`, che finora
  arrivava al Planner indisturbato.

### Lo strumento che mancava

`scripts/guard_falsepos.py` passa il Guard su tutti i 102 contratti gold, che
sono corretti per costruzione: **ogni errore o riparazione è un falso
positivo**. È il test che ogni controllo nuovo deve superare prima di essere
promosso a bloccante — senza, un controllo che "trova errori" sta solo
fermando richieste legittime.

Ha trovato subito i punti 1 e 3 qui sopra, e tre falsi positivi miei:

- *"un pani**no** possibilmente con la mortadella"*: la finestra della
  polarità partiva a 24 caratteri fissi, cadeva in mezzo a una parola, e
  `\b` crede sempre che l'inizio di una stringa sia un confine. È lo stesso
  errore di "Buongior-no", la terza volta che lo faccio;
- *"con la bufala **senza cipolla** dopodomani"*: la negazione aveva già il
  suo bersaglio. Ora fra negazione e valore possono stare solo parole vuote;
- *"300 euro però non li spendo, **superiore a 4 stelle**"*: il verbo
  d'intenzione vale solo dentro la stessa clausola.

Alla fine: **0 falsi positivi su 102 casi gold, 0 errori e 0 riparazioni su
6000 esempi generati.** Generatore, regole scritte, Guard e benchmark dicono
per la prima volta tutti la stessa cosa.

### Risultati del v12

| | mandatory | acc_wrong | acc_prec | exact | bucket | operator | recall | prec |
|---|---|---|---|---|---|---|---|---|
| jase-v11-iter4000 (Guard v11) | 0.624 | 39 | 0.585 | 0.539 | 0.945 | 0.953 | 0.706 | 0.830 |
| jase-v11-iter4000 (Guard v12) | 0.682 | **24** | **0.714** | 0.588 | 0.985 | 0.954 | 0.722 | 0.850 |
| jase-v12-iter3500 | 0.647 | 27 | 0.686 | 0.578 | 0.921 | 0.964 | 0.778 | 0.886 |
| jase-v12-iter4000 | 0.682 | 29 | 0.678 | 0.598 | 0.971 | 0.979 | 0.778 | 0.870 |
| **jase-v12-iter4500** | **0.729** | 31 | 0.677 | **0.637** | 0.967 | 0.953 | **0.833** | 0.877 |

La riga più utile è la seconda: **lo stesso modello v11, con il solo Guard
nuovo**. Separa il merito dei dati da quello delle regole.

- il **Guard** da solo: `accepted_but_wrong` 39 → 24, bucket 0.945 → 0.985;
- i **dati** da soli: recall dei fatti 0.706 → 0.833, `missing_fact` 35 → 20.

`accepted_but_wrong` del v12 (31) è più alto di quello del v11 col Guard
nuovo (24), e non è un peggioramento: il v12 viene fermato molto meno
(5.9% contro 17.6%) perché produce contratti più completi. Fra i contratti
che passano, la frazione corretta è quasi identica.

**Zero REJECT, quattro REPAIRED**: la cascata §2 lavora in produzione.

### La curva non è finita

```
iterazioni   3500    4000    4500
mandatory    0.647   0.682   0.729
```

Monotona, e ripida, fino all'ultimo checkpoint. Il conto lo spiega: 4500
iterazioni × batch 1 × accumulo 4 = 18 000 esempi su 27 500, cioè **0,65
epoche**. Il modello non ha ancora visto una volta sola il dataset.

`configs/mlx_lora_v12long.yaml` cambia **solo** il numero di iterazioni
(12 000 = 1,74 epoche). Il batch 2 è stato provato e scartato: regge la
memoria (4,24 GB di picco) ma va più piano — 57 token/s contro 70-90 —
perché MLX riempie il batch fino alla sequenza più lunga, e con frasi da
160 token l'imbottitura costa più di quanto il parallelismo renda.

---

## Iterazione 10 → v13: la lingua dei valori

Dei 23 casi mandatory ancora falliti, un quarto non sono errori di
comprensione ma di **forma del valore**:

```
"Preferirei giovane"        → age eq "young"      (gold: "giovane")
"qualcosa da mangiare"      → target "food"       (gold: "cibo")
"idealmente in pelle"       → material "in pelle" (gold: "pelle")
"preferably with a pool"    → amenity "a pool"    (gold: "pool")
"sotto i sei mesi"          → unit "mesi"         (gold: "months")
```

Le prime due sono traduzioni: il modello porta in inglese un valore che
l'utente ha detto in italiano. Le altre due sono ritagli: il frammento si
porta dietro la preposizione. L'ultima è **un'incoerenza mia**: §6.2 dichiara
che il vocabolario delle unità è chiuso e inglese, il benchmark chiedeva
`months`, e il generatore scriveva `mesi`. Per nove iterazioni.

- preposizione e articolo iniziali cadono in canonicalizzazione, come già
  facevano gli articoli: `"in pelle"` → `pelle`, `"a pool"` → `pool`;
- le unità hanno una tabella di alias e **una sola grafia** nel dataset;
- `hallucinated_target` passa da avviso a **errore**: un nome di target che
  nel testo non c'è non arriva più al Planner. Misurato: 0 falsi positivi
  sul gold — l'unico era un errore del gold.

### Due casi gold corretti (di nuovo per coerenza interna)

- **`cond-005`** chiedeva un `gate` con `then: notify` su `monitor`, mentre
  `act-005` — la stessa frase scritta in altro modo — chiedeva `monitor` con
  la soglia fra i vincoli. §11.3 dice che se l'unica conseguenza è essere
  informati non c'è nessun bivio da rappresentare. Il modello faceva la cosa
  giusta e prendeva un errore.
- **`amb-004`** chiedeva `target = "cibo"` da *"qualcosa da mangiare"*.
  "Cibo" nel testo non c'è: è la stessa astrazione che §8 vieta e che avevo
  già tolto per "Sphynx" e per "elettricista". Ora il target è quello che
  l'utente ha detto.

### Un sinonimo rientrato di soppiatto

`"un albergo"` produceva `target.name: "hotel"`. I sinonimi nel target erano
stati tolti nel v4; questo era tornato dentro con il dominio
`accommodation` e nessuno se n'era accorto, perché il controllo di
ancoraggio del target era solo un avviso. Promuoverlo a errore l'ha trovato
in un secondo, su 23 esempi di 6000.

### Buchi lessicali chiusi

`brand` dei locali (*"da Frobnicato"* non è un ingrediente) · `route`
(*"un volo diretto"*, anche come condizione di un gate) · `cuisine` espressa
come nome (*"di pesce"*) · `availability` contro `date` in un blocco
contrastivo dedicato (*"aperto la domenica"* è una caratteristica del posto,
*"domenica"* è quando ci vuoi andare) · *"un panino **alla** mortadella"*,
che è la forma più comune in italiano e il generatore non ne produceva
nemmeno una.

### Risultati della corsa lunga (v12long)

Stessi dati del v12, stessa capacità, stessi iperparametri: cambia solo il
numero di iterazioni.

| | mandatory | acc_wrong | acc_prec | exact | bucket | operator | recall | prec |
|---|---|---|---|---|---|---|---|---|
| jase-v12-iter4500 | 0.765 | 27 | 0.716 | 0.667 | 0.967 | 0.954 | 0.845 | 0.895 |
| jase-v12long-iter6000 | 0.635 | 28 | 0.674 | 0.569 | 0.962 | 0.985 | 0.735 | 0.893 |
| jase-v12long-iter9000 | 0.824 | **18** | 0.808 | 0.745 | 0.987 | 0.987 | 0.834 | 0.899 |
| jase-v12long-iter10500 | 0.765 | **18** | 0.798 | 0.696 | 0.981 | 0.974 | 0.851 | 0.933 |
| **jase-v12long-iter12000** | **0.847** | 19 | **0.804** | **0.765** | **0.994** | 0.975 | **0.873** | **0.935** |

(la riga del v12-iter4500 è rimisurata sul gold corretto, per essere
confrontabile)

**Casi mandatory dal 76% all'85%, contratti sbagliati accettati da 27 a 19,
accuratezza dei bucket al 99.4%, e nessun REJECT.** Nessuna riga di codice
nuova: solo iterazioni.

L'iter6000 crolla a 0.635 e poi risale: un checkpoint intermedio non è una
tendenza, ed è la ragione per cui si salvano cinque punti e non uno.

L'ipotesi "il plateau è un limite del modello da 2B" era falsa per la
seconda volta — la prima con la capacità dell'adattatore, ora col numero di
passi. In entrambi i casi la risposta stava in un conto che nessuno aveva
fatto: quanti parametri, quante epoche.

Restano due gate su sei: casi mandatory al 100% (siamo all'85%) e zero
contratti sbagliati accettati (siamo a 19). La causa numero uno resta
l'omissione (16 su 36 errori), ma ora pesa la metà di tre iterazioni fa.

### Risultati del v13 — e il massimo della curva

| | mandatory | acc_wrong | acc_prec | json | exact | bucket | recall | prec | halluc |
|---|---|---|---|---|---|---|---|---|---|
| jase-v12long-iter12000 | 0.847 | 19 | 0.804 | 1.000 | 0.765 | 0.994 | 0.873 | 0.935 | 0.078 |
| jase-v13-iter12000 | 0.788 | 21 | 0.772 | 0.980 | 0.696 | 0.980 | 0.829 | 0.904 | 0.098 |
| jase-v13-iter14000 | 0.824 | 19 | 0.800 | 1.000 | 0.745 | 0.988 | **0.895** | 0.920 | 0.108 |
| **jase-v13-iter16000** | **0.847** | **15** | **0.839** | 0.990 | 0.765 | 0.987 | 0.873 | **0.946** | **0.059** |
| jase-v13-iter18000 | 0.776 | 25 | 0.740 | 0.990 | 0.696 | 0.986 | 0.801 | 0.890 | 0.118 |

**A 18 000 iterazioni peggiora su quindici tag su venti.** Il massimo della
curva è intorno a 16 000: da qui in poi non serve allungare, serve scegliere.

Il v13 pareggia il v12long sui casi mandatory ma vince dove conta per
l'invariante fail-closed: **contratti sbagliati accettati da 19 a 15**,
precisione dei fatti 0.946, allucinazioni dimezzate (0.118 → 0.059).

Un caso — `multi-003`, *"Cercami una chitarra usata sotto i 300 euro e
confrontami tre amplificatori"* — produce JSON rotto, degenerato o troncato
a seconda del checkpoint. Il Guard lo ferma sempre (fail-closed funziona) e
in produzione `interpret.py` riprova con temperatura diversa, ma il gate
"validità JSON 100%" si misura al primo colpo, e lì cade.

**BEST: jase-v13-iter16000.** Preferito al v12long anche se perde il gate
JSON: un contratto fermato si recupera con una domanda, quattro contratti
sbagliati in più agiscono nel mondo.

---

## Iterazione 12 → v14: il dataset che insegnava l'omissione

Dei 13 casi mandatory falliti, **sei erano già fermati dal Guard**. I sette
che passano si spiegano quasi tutti con una svista nel generatore:

**`"due biglietti"` era una frase del target.** Il dominio `ticket` aveva
`["un biglietto", "due biglietti", "dei biglietti"]` fra le forme del
target, e il contratto corrispondente non conteneva nessun `quantity`. Il
dataset insegnava, nero su bianco, a leggere "due" e non scriverlo da
nessuna parte — cioè **esattamente l'errore che misuravo** su nat-020. È la
quarta volta che il generatore insegna l'errore che poi conto.

Insieme: i numeri piccoli ora si scrivono anche in lettere ("due biglietti",
"tre amplificatori"), come si fa parlando, e §12 dichiara già la
normalizzazione.

Le altre:
- *"un allevatore di Zwergschnauzer"*: la razza è `breed`, non la
  specializzazione del professionista. Mancava il target "allevatore" nel
  dominio animale;
- **target vaghi tenuti parola per parola**: "qualcosa da mangiare" non
  diventa `food`;
- **blocco nuovo, tempo contro soglia**: in *"un elettricista stasera entro
  100 euro"* l'`entro` governa i cento euro, non la sera, e il modello
  produceva `date before stasera`;
- `monitor` dal 25% al 40% del ramo condizionale: è l'unico posto dove
  quell'azione si impara, e su act-005 il modello scriveva ancora `find`.

### Due definizioni fantasma

Cercando il dominio `ticket` ne ho trovate due: una in `DOMAINS`, l'altra
incollata dentro `DOMAIN_ACTION_PHRASES`, dove un dizionario di frasi
d'azione conteneva `targets`, `actions` e `facts`. Non faceva danno — la
lettura restituiva `None` e tutto proseguiva — ma per mezz'ora ho modificato
la copia morta chiedendomi perché il dataset non cambiasse.

### Risultati del v14 — e la scoperta del rumore

| | mandatory | acc_wrong | acc_prec | json | exact | recall | prec | halluc |
|---|---|---|---|---|---|---|---|---|
| jase-v13-iter16000 | 0.847 | 15 | 0.839 | 0.990 | 0.765 | 0.873 | 0.946 | 0.059 |
| jase-v14-iter12000 | 0.824 | **13** | 0.854 | 0.990 | 0.745 | 0.840 | **0.956** | **0.039** |
| jase-v14-iter14000 | 0.812 | 20 | 0.792 | 1.000 | 0.745 | 0.912 | 0.907 | 0.137 |
| jase-v14-iter16000 | 0.788 | 20 | 0.785 | 1.000 | 0.716 | 0.851 | 0.880 | 0.147 |
| jase-v14-iter17000 | 0.859 | 15 | 0.840 | 1.000 | 0.774 | 0.889 | 0.915 | 0.118 |

Le ultime tre righe sono la stessa corsa, gli stessi dati, mille iterazioni
di distanza: **da 0.788 a 0.859**. Sono tre casi su ottantacinque. Non è
apprendimento, è rumore — e significa che tutti i confronti fatti finora fra
due versioni su un singolo checkpoint misuravano in buona parte il caso.

Il segnale pulito sono le correzioni mirate, perché si leggono su tutti e
quattro i checkpoint: `amb-004` (target vago tenuto parola per parola) e
`nat-013` (`entro` che governava la data invece del prezzo) sono **corretti
ovunque**; `act-005` e `multi-001` in tre casi su quattro; `nat-020` — la
quantità "due biglietti" — **in nessuno**.

---

## Iterazione 13: curare il rumore invece dei casi

### La media dei pesi

`scripts/soup_adapters.py` fa la media aritmetica, tensore per tensore, di
più checkpoint della stessa corsa. Costa trenta secondi e produce un
adattatore vero, che si spedisce come qualsiasi altro.

| | mandatory | acc_wrong | acc_prec | exact | recall | prec | halluc |
|---|---|---|---|---|---|---|---|
| miglior checkpoint singolo (v14-17000) | 0.859 | 15 | 0.840 | 0.774 | 0.889 | 0.915 | 0.118 |
| media di 9 checkpoint (dal 2000) | 0.871 | 16 | 0.833 | 0.784 | 0.895 | 0.953 | 0.049 |
| media della coda (14k, 16k, 17k) | 0.882 | 16 | 0.835 | 0.794 | 0.912 | 0.954 | 0.059 |
| **media dei 4 maturi (12k–17k)** | **0.906** | **11** | **0.883** | 0.814 | 0.906 | **0.959** | 0.049 |
| **media incrociata v13+v14 (5 ckpt)** | **0.918** | 12 | 0.875 | **0.824** | **0.917** | 0.949 | 0.059 |

**Dal 86% al 92% dei casi obbligatori, e i contratti sbagliati accettati da
15 a 11, senza riallenare niente.** Includere i checkpoint acerbi peggiora,
come previsto: la media aiuta solo fra punti già maturi.

**Una cautela onesta**: ho provato quattro combinazioni e tenuto la
migliore, su un benchmark da 102 casi dove un caso vale un punto. La
differenza fra le prime due (0.906 e 0.918) è dentro il rumore, e sceglierla
guardando il gold è una forma blanda di adattamento al benchmark. Il
risultato robusto — quello vero — è che **tutte e tre le medie di
checkpoint maturi battono ogni singolo checkpoint** su precisione,
allucinazioni e contratti sbagliati accettati. Quale delle tre sia la
migliore, il benchmark non lo sa dire.

### Cosa resta, e cosa NON è più colpa del modello

Sette casi mandatory falliti, e **tre li ferma già il Guard**. I quattro che
passano:

- `nat-020` — la quantità "due biglietti", sbagliata da cinque versioni.
  Guardando finalmente cosa produce il generatore: *"Un biglietto, 2
  biglietti, con mostra"*. Il sintagma del target diceva "un biglietto" e il
  fatto quantità ne aggiungeva "2 biglietti" a parte, in una frase che
  nessuno direbbe mai. E `quantity` compariva nell'1,1% degli esempi. Nuovo
  blocco dedicato: il numero sta **dentro** il sintagma ("compra due
  biglietti"), col singolare come contrasto — un oggetto solo è il caso di
  default, non un fatto;
- `multi-001` — il secondo goal ereditava il luogo del primo. Nei multi-goal
  generati i due luoghi potevano coincidere per caso: ora sono diversi per
  costruzione;
- `trap-002` — *"Un volo il 3 per 2 persone sotto i 3 scali"*: tre numeri,
  tre ruoli. Resta aperto;
- `neg-003` — *"usata ma non incidentata"*: perde l'aggettivo negato quando
  c'è anche un terzo fatto.

### Il learning rate

La cura a monte dello stesso male: con il passo costante a 3·10⁻⁵
l'ottimizzatore rimbalza attorno al minimo, e ogni checkpoint è un punto a
caso in quella nuvola. Il v15 usa un coseno che scende a 3·10⁻⁶, e alla fine
se ne fa comunque la media: le due cure si sommano invece di escludersi.

### Il programma del learning rate conta gli step, non le iterazioni

La sonda da 60 iterazioni l'aveva già detto e l'avevo letta male. A
iterazione 60 il log dava un passo di 2,983·10⁻⁵ — quasi il valore
iniziale — e avevo concluso che MLX ignorasse la configurazione. Poi, a
iterazione 150 della corsa vera, il passo era 2,16·10⁻⁶: né il valore
iniziale né quello che mi aspettavo dalla rampa.

Il conto torna solo in un modo. Con `grad_accumulation_steps: 4`
l'ottimizzatore fa **uno** step ogni quattro iterazioni, e il programma
segue quelli: a iterazione 150 siamo allo step 37, e 37/500 della rampa
lineare fa esattamente 2,16·10⁻⁶.

Quindi `decay_steps: 17000` non descrive la corsa: 17 000 iterazioni sono
**4250 step**, e il coseno si sarebbe fermato a un quarto della discesa,
arrivando in fondo con il passo ancora a 2,6·10⁻⁵ — praticamente invariato.
L'esperimento sarebbe stato un esperimento su niente.

Il v15 è già partito e lo lascio correre: con il passo quasi costante
misura le **sole** modifiche ai dati, ed è il confronto pulito col v14. Il
v16 ha la stessa configurazione con `decay_steps: 4250` e `warmup: 100`, e
isola l'effetto del programma. Una variabile per volta, anche quando la
variabile è un mio errore di lettura.

### Risultati del v16 — il decadimento funziona, la media smette di servire

| | mandatory | acc_wrong | acc_prec | exact | recall | prec | halluc |
|---|---|---|---|---|---|---|---|
| jase-v14-soup-4ckpt | 0.906 | 11 | 0.883 | 0.814 | 0.906 | 0.959 | 0.049 |
| **jase-cross-soup-v13v14** | **0.918** | 12 | 0.875 | **0.824** | **0.917** | 0.949 | 0.059 |
| jase-v16-iter12000 | 0.882 | 15 | 0.842 | 0.784 | 0.889 | 0.947 | 0.078 |
| jase-v16-iter13000 | 0.906 | **11** | 0.883 | 0.814 | 0.895 | 0.942 | 0.069 |
| jase-v16-soup-4ckpt | 0.894 | 14 | 0.853 | 0.794 | 0.901 | 0.942 | 0.078 |
| jase-cross-soup-v14v16 | 0.906 | 12 | 0.874 | 0.814 | 0.901 | 0.959 | **0.039** |

Due risposte pulite:

- **il decadimento riduce l'oscillazione.** Fra gli ultimi due checkpoint il
  v14 saltava di 0.071 (0.788 → 0.859), il v16 di 0.024 (0.882 → 0.906).
  L'ottimizzatore atterra invece di rimbalzare;
- **la media dei pesi smette di aiutare quando il passo decade**: 0.894
  contro 0.906 del checkpoint finale. È coerente con il perché funzionava:
  se il punto finale è già stabile, mediarlo con punti presi mentre ancora
  rimbalzava lo tira indietro. Le due cure non si sommano — **sono la stessa
  cura**, una a monte e una a valle.

### Il tetto

Tre configurazioni diverse e cinque modelli atterrano tutti fra 0.89 e 0.92.
**Sei casi mandatory falliscono in tutti e cinque**, e di quei sei **tre li
ferma comunque il Guard**: l'invariante fail-closed regge anche dove il
modello non arriva.

I tre che passano, guardati uno per uno:

- **`neg-003`** — *"usata ma non incidentata"*: due predicati sulla **stessa
  proprietà**, uno positivo e uno negativo. Il generatore non poteva
  produrli: `_sample_facts` tiene un fatto per proprietà, quindi in 30 000
  esempi quella forma non esisteva **nemmeno una volta**. Non era un limite
  del modello, era una riga di codice scritta due settimane fa per evitare i
  duplicati che escludeva anche i non-duplicati;
- **`nat-020`** — il modello produceva `event contains concerto` da *"per il
  concerto di sabato"* e prendeva un errore, perché **il gold lo
  dimenticava**. Corretto: è un fatto dichiarato dall'utente quanto la data,
  ed è lo stesso principio su cui è costruito il rilevatore di omissioni;
- **`trap-002`** — *"Un volo il 3 per 2 persone sotto i 3 scali"*: tre
  numeri, tre ruoli, nessuna unità. Resta aperto, e onestamente è difficile
  anche per una persona distratta.

Aggiunto anche il modo di dire *"non voglio farmi più di 80 km"* (nat-002),
che nel generatore non c'era.

### Il v17 è andato indietro, ed è colpa di due dati incoerenti

| | mandatory | acc_wrong | exact | bucket | negation |
|---|---|---|---|---|---|
| jase-cross-soup-v13v14 (gold nuovo) | **0.918** | 12 | **0.824** | **0.988** | **0.73** |
| jase-v17-iter12000 | 0.847 | 13 | 0.765 | 0.956 | 0.55 |
| jase-v17-iter13000 | 0.835 | 16 | 0.755 | 0.957 | 0.55 |
| jase-v17-soup-3ckpt | 0.835 | 13 | 0.755 | 0.962 | 0.55 |

Sei casi in meno, e **nessuno risolto**: il tag `negation`, che era l'unica
cosa che volevo migliorare, è sceso da 0.73 a 0.55. Andando a vedere cosa
era finito nel dataset:

1. **"una moto restaurata, modificata".** Le coppie sulla stessa proprietà
   che avevo aggiunto generavano anche la variante *senza* negazione: due
   valori diversi, entrambi dati per certi, sulla stessa proprietà. Non è
   una richiesta, è una contraddizione, e il modello ci ha imparato sopra.
2. **"un volo per Amsterdam per Atene".** `enrich_block` poteva pescare la
   destinazione due volte, una dal dominio e una dal ramo dedicato. Erano
   61 esempi su 28 000 — e c'erano **in ogni dataset dalla nona iterazione
   in poi**, il che spiega perché il tag `transport` non saliva mai.

Nessuno dei due era visibile ai controlli: il dato era coerente con sé
stesso, il Guard lo accettava, il generatore lo validava. Solo il benchmark
protestava, tre ore dopo.

### L'invariante che mancava

```
due predicati sulla stessa proprietà, operatori `eq` o `near`,
valori diversi  →  conflicting_predicates
```

`eq` e `near` designano **un** valore. `contains` no — un hotel può avere
piscina e palestra — e nemmeno le soglie, perché un prezzo ha un minimo e un
massimo. Zero casi gold lo violano; il generatore ne produceva sessantuno.

È il quarto difetto di coerenza del generatore in quattordici iterazioni
("un idraulico con 4 stanze", "se trovi un fabbro sotto i 1000 compralo",
"due biglietti senza quantità", e ora questo). I primi tre li ho trovati
leggendo esempi a caso; per il quarto, finalmente, c'è una regola che li
ferma tutti. **La lezione l'avevo già scritta tre volte nel registro senza
mai implementarla**: rendere l'incoerenza strutturalmente impossibile, non
cercarla a occhio.

### Risultati del v18, e la fine della corsa ai numeri

| | mandatory | acc_wrong | exact | bucket | negation | transport |
|---|---|---|---|---|---|---|
| cross-soup v13+v14 (gold nuovo) | 0.918 | 12 | 0.824 | 0.988 | 0.73 | 1.00 |
| jase-v17-iter13000 | 0.835 | 16 | 0.755 | 0.957 | 0.64 | 0.50 |
| jase-v18-iter13000 | 0.882 | 13 | 0.784 | 0.976 | 0.55 | **1.00** |
| **media v13+v14+v18 (6 ckpt)** | **0.929** | 12 | **0.833** | 0.982 | 0.73 | 1.00 |

I difetti puntuali sono stati corretti — `neg-003` risolto dal v17,
`nat-020` dal v18, `transport` tornato a 1.00 appena "un volo per Amsterdam
per Atene" è sparito dai dati — ma il **tag `negation` nel v18 scende a
0.55**: avendo aggiunto cinque coppie `condition` a una lista che ne aveva
sei, quel blocco ha smesso di insegnare la varietà che insegnava prima.
Correggere un caso spostando la distribuzione ne rompe altri tre.

La media di sei checkpoint su tre corse riporta tutto a posto (0.929) e
lava via anche quella regressione. Ma a quel punto avevo provato **sette**
combinazioni di media tenendo la migliore, su un benchmark dove un caso
vale un punto: il valore onesto di quel 0.929 è "da qualche parte fra 0.89
e 0.93", e la differenza fra le combinazioni il benchmark non la sa dire.

### Misurare la pipeline vera, e sbagliare la misura

I gate parlano di ciò che arriva al Planner, e al Planner arriva
`interpret.py` — con i suoi tentativi — non una generazione secca. Aggiunto
`--pipeline` a `eval_v2.py`, il primo risultato sembrava un disastro: i
contratti fermati scendevano a **zero** e quelli sbagliati accettati
salivano da 12 a 16.

Stavo per concludere che il retry indebolisce il fail-closed. Era invece la
misura a essere rotta: la pipeline ha **quattro** esiti e io ne prendevo
solo il contratto, così una richiesta di chiarimento veniva contata come un
contratto accettato e sbagliato. La quota di bloccati usciva a zero per
costruzione.

Con la misura corretta la pipeline dà **gli stessi identici numeri**
dell'eval a colpo singolo (0.929 e 12), con accuratezza degli operatori a
1.000 e allucinazioni allo 0.020. Nessun disastro, e nessun merito
nascosto.

Resta però una cosa vera trovata per sbaglio: **il retry non deve valere
per gli errori semantici.** Se il Guard dice che il contratto contraddice il
testo, rigenerare a temperatura più alta non lo rende più giusto, lo rende
più fortunato — è campionare finché il controllo non si distrae. Ora
`interpret.py` ritenta solo sui guasti strutturali (JSON rotto, output
degenerato) e su un errore semantico chiede, punto.

### Dove siamo, in una riga

```
casi mandatory        79 su 85   (0.929)
contratti sbagliati accettati    12
validità JSON                  100%
fatti duplicati                   0
valori non ancorati accettati     0
output degenerati                 0
```

Sei casi mandatory falliti, **tre dei quali il Guard li ferma**: l'invariante
fail-closed regge anche dove il modello non arriva. I tre che passano sono
`nat-020` (una quantità dentro una frase lunga), `trap-002` (tre numeri con
tre ruoli e nessuna unità) e `neg-003` (due predicati sulla stessa
proprietà).

**Il tetto è questo.** Quindici iterazioni, cinque configurazioni, e tutto
atterra fra 0.89 e 0.93. Le leve rimaste non sono più nei dati.

---

## v19: il benchmark decontaminato e la capacità

Due esperimenti controllati, una variabile ciascuno, con la domanda scritta
prima in `docs/HYPOTHESES.md`.

### H10 — la contaminazione non stava reggendo il punteggio

Il vecchio dataset conteneva 186 duplicati esatti del `gold_v1`, e la
revisione tecnica concludeva che il punteggio fosse gonfiato. La misura dice
altro. Confronto appaiato (unica differenza il dataset): **0.882 contro
0.847**, tre casi su ottantacinque. E il conto caso per caso è più netto
ancora: i 186 duplicati erano **7 prompt distinti**, e sui quei sette il
modello contaminato faceva 7/7 contro 6/7 del pulito. Sugli altri
novantacinque: 0.768 contro 0.758.

**La contaminazione valeva un caso.** Il salto apparente da 0.929 a 0.847
nella tabella generale è un artefatto del confronto: media di sei checkpoint
contro checkpoint singolo, differenza che già sapevamo valere 4-7 punti.

La decontaminazione resta giusta — un benchmark che contiene frasi del
training non misura la generalizzazione — ma la diagnosi era sbagliata. Il
vecchio 0.929 era gonfiato semmai dall'**aver scelto la migliore fra sette
medie provate**, che è un problema diverso e che solo un holdout
indipendente risolve.

### H11 — la capacità dell'adattatore ERA un collo di bottiglia

| | 8 layer | 16 layer |
|---|---|---|
| `missing_fact` | 17 | **10** |
| `facts.recall` | 0.802 | **0.857** |
| `hallucinated_fact` | 2 | 4 |
| azione / bucket | 0.981 / 0.993 | **1.000 / 1.000** |
| `accepted_but_wrong` | 16 | **10** |
| casi mandatory | 0.847 | **0.906** |

Omissioni giù del 40%, recall su di cinque punti e mezzo, contratti
sbagliati accettati da sedici a dieci, al prezzo di due allucinazioni in
più — uno scambio che conviene, perché un fatto inventato il Guard lo vede
e uno mancante no.

**E ribalta una conclusione che avevo scritto sei iterazioni fa.** Nel v7
la capacità sembrava non servire; ma quel test girava su dati che avevano
ancora il 57% di esempi a fatto singolo. Una conclusione negativa vale solo
nelle condizioni in cui è stata misurata — e quelle erano cambiate sotto il
naso senza che rimettessi in discussione la conclusione.

### Multi-seed, e il candidato di release

Tre seed a 16 layer, configurazione identica: **0.906 · 0.918 · 0.894**,
media 0.906, scarto due casi. Due seed a 8 layer: 0.847 e 0.812. Gli
intervalli non si toccano — H11 regge al rumore fra seed.

**La media dei tre seed crolla a 0.259**, con il 58% delle richieste
bloccate. Non è un peggioramento: è un modello rotto, e spiega
retroattivamente perché le medie dell'iterazione 13 funzionavano. Quelle
corse — v13, v14, v18 — usavano **tutte il seed 2026**: stessa
inizializzazione, stesso bacino. Seed diversi finiscono in bacini diversi, e
il punto medio fra due bacini non è un modello. La regola scritta prima di
misurare ha scartato quel candidato senza lasciarmi spazio per
razionalizzarlo.

Candidato scelto dalla regola: **seed mediano (2026), non il migliore**.

### Il numero che conta, e non è quello del gold

| | DEV (`gold_v1`) | RELEASE HOLDOUT |
|---|---|---|
| casi mandatory | 0.906 | **0.240** |
| semantic_exact | 0.843 | 0.222 |
| `accepted_but_wrong` | 10 su 102 | **15 su 27** |
| `target_name` | 0.962 | 0.690 |
| fatti, recall | 0.857 | 0.553 |

Quindici iterazioni di sviluppo guidate dal `gold_v1` lo hanno saturato: non
misura la capacità, misura ciò che ho già corretto. La revisione tecnica
sospettava la contaminazione testuale, ma quella valeva un caso solo (H10):
il problema non erano i 186 duplicati, era **aver usato lo stesso metro
quindici volte**.

Il dato più serio non è 0.240 — l'holdout è anche più difficile per
costruzione — ma **`accepted_but_wrong` a 15 su 27**: su richieste mai viste
più della metà dei contratti sbagliati supera il Guard. L'invariante
fail-closed è tarata sul DEV, e fuori da quello tiene molto meno.

`INTERPRETER_RELEASE.md` ha il rapporto completo.
