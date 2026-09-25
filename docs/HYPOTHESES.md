# Registro delle ipotesi

## Esperimento corrente — H14 (24 settembre, prima del training)

Correzione delle etichette telegrafiche, deduplicazione e split per famiglie
con chiarimenti in validation, a budget fisso di 13000 microbatch. Protocollo,
criteri e limiti: [EXPERIMENT_V20.md](EXPERIMENT_V20.md). Il conteggio storico
delle epoche era errato: nel trainer installato `iters` conta microbatch e
l'accumulo non moltiplica gli esempi consumati. V19 finale equivale a 0.473
epoche, v20 a 0.529. H5/H6 vanno interpretate tenendo conto di questa correzione.

Esito H14: esperimento avviato dal runner nativo il 24 settembre, dopo
preflight superato (Metal disponibile, nessun troncamento, prefissi e
maschera coerenti). Risultato persistente previsto in
`reports/v20/SUMMARY.md`; nessun uso del release holdout in questo esperimento.

Ogni training deve rispondere a una domanda scritta PRIMA di lanciarlo.
Senza domanda, il risultato è un numero che non si sa come leggere.

Formato: ipotesi · come si misura · esito · cosa ne segue.

---

## Chiuse

| # | ipotesi | esito |
|---|---|---|
| 1 | Gli esempi vengono troncati da `max_seq_length` | **falsa** — il più lungo è 323 token |
| 2 | Il plateau dipende dalla capacità dell'adattatore | **falsa nel v7** — 4× i parametri peggiorava tutto tranne il bucket. **Da rifare**: quell'esperimento aveva dati di sei iterazioni fa |
| 3 | I condizionali si risolvono con una policy di ripiego | **falsa** — serviva il campo `gate` (schema 1.1) |
| 4 | Le omissioni sono un limite del modello | **falsa** — era la distribuzione dei fatti nel dataset (57% a fatto singolo contro 39% del gold) |
| 5 | Il plateau è un limite del 2B | **falsa** — allenavo su 0,65 epoche |
| 6 | Allungare ancora il training aiuta | **falsa oltre ~16 000 iterazioni** |
| 7 | `batch_size: 2` dimezza il tempo | **falsa** — MLX imbottisce fino alla sequenza più lunga |
| 8 | Media dei pesi e decadimento del LR si sommano | **falsa** — sono la stessa cura, a valle e a monte |
| 9 | Il retry della pipeline indebolisce il fail-closed | **falsa** — era rotta la misura (quattro esiti appiattiti su uno) |

## Aperte

### H10 — La decontaminazione abbassa il punteggio DEV, e di quanto?

Il vecchio dataset conteneva 186 duplicati esatti del `gold_v1`. Il punteggio
0.929 del BEST è quindi in parte memoria.

*Misura*: `jase-v19-clean` (dati decontaminati) contro `jase-soup-tris`
rimisurato con lo stesso Guard, sullo stesso `gold_v1`.

*Come si legge*: un calo **è il risultato atteso e sano** — è la parte di
punteggio che non era generalizzazione. Un calo grosso (>5 punti) dice che
il vecchio numero era gonfio; un calo nullo dice che i 186 duplicati erano
frasi che il modello avrebbe risolto comunque.

*Esito* (23 settembre): **la contaminazione non stava reggendo il punteggio.**

Confronto appaiato, unica differenza il dataset (v18 su `data/v17`
contaminato, v19-clean su `data/v19_clean`, stessa ricetta, stesso seed):

| | contaminati | puliti |
|---|---|---|
| mandatory | 0.882 | 0.847 |
| exact | 0.784 | 0.765 |
| accepted_but_wrong | 13 | 16 |
| facts recall | 0.901 | 0.802 |
| facts precision | 0.932 | 0.967 |
| missing_fact | 13 | 17 |
| hallucinated_fact | 8 | 2 |

Tre casi su ottantacinque, cioè dentro il rumore che già conosciamo.

Il conto **caso per caso** è più informativo della media. I 186 duplicati
erano in realtà **7 prompt gold distinti**, ripetuti molte volte:

| | modello coi dati contaminati | modello coi dati puliti |
|---|---|---|
| i 7 casi che erano nei dati | 1.000 | 0.857 |
| gli altri 95 | 0.768 | 0.758 |

La contaminazione valeva **un caso** sui sette che toccava, e **un caso**
sugli altri novantacinque: rumore. Il salto apparente da 0.929 a 0.847 che
si legge nella tabella generale **non è contaminazione**: è la differenza
fra una media di sei checkpoint e un checkpoint singolo, che già sapevamo
valere 4-7 punti.

**Cosa ne segue.** La decontaminazione andava fatta lo stesso — un
benchmark che contiene frasi del training non misura la generalizzazione,
qualunque sia l'effetto numerico — ma la tesi "il punteggio era gonfiato
dalla contaminazione" **non regge la misura**. Il vecchio 0.929 era gonfiato
semmai dalla selezione della media migliore fra sette provate, che è un
problema diverso e che il holdout indipendente risolve davvero.

Nota a margine confermata per la terza volta: con il learning rate che
decade, **la media dei pesi peggiora** (0.812 contro 0.847 del checkpoint
finale). Non è una coincidenza: è la stessa cura applicata due volte.

### H11 — Più profondità di adattamento riduce le omissioni?

`missing_fact` è l'errore dominante delle versioni mature.

*Misura*: `jase-v19-cap16` (16 layer LoRA) contro `jase-v19-clean` (8), a
parità di dataset, learning rate, scheduler, rank, seed e prompt.

*Come si legge*: la metrica decisiva non è `mandatory_rate` ma
**`missing_fact` e `facts.recall`**. Se scendono le omissioni senza che
salgano le allucinazioni, la capacità era un collo di bottiglia vero.

*Esito* (24 settembre): **sì, e in modo coerente su tutta la tassonomia.**

| | 8 layer | 16 layer |
|---|---|---|
| `missing_fact` | 17 | **10** |
| `facts.recall` | 0.802 | **0.857** |
| `hallucinated_fact` | 2 | 4 |
| `facts.precision` | 0.967 | 0.946 |
| accuratezza dell'azione | 0.981 | **1.000** |
| accuratezza dei bucket | 0.993 | **1.000** |
| `accepted_but_wrong` | 16 | **10** |
| casi mandatory | 0.847 | **0.906** |

Le omissioni calano del 40%, la recall sale di cinque punti e mezzo, e i
contratti sbagliati che superano il Guard scendono da sedici a dieci. Il
prezzo sono **due casi** di allucinazione in più: uno scambio che conviene,
perché un fatto inventato il Guard lo vede e un fatto mancante no.

Non è un singolo numero che si muove: si muovono insieme, e nella direzione
che l'ipotesi prevedeva. È il tipo di evidenza che un caso di differenza non
può produrre per caso.

**Questo ribalta l'ipotesi 2.** Nel v7 avevo concluso che la capacità
dell'adattatore non fosse il collo di bottiglia — ma quell'esperimento girava
su dati di sei iterazioni prima, quando il dataset aveva ancora il 57% di
esempi a fatto singolo e mezza dozzina di incoerenze. **Con dati maturi la
profondità serve.** Una conclusione negativa vale solo nelle condizioni in
cui è stata misurata, e quelle condizioni erano cambiate sotto il naso.

Conferma numero quattro: con il learning rate che decade, **la media dei
pesi non aggiunge niente** (0.906 sia per il checkpoint finale sia per la
media).

*Cosa ne segue*: la forma del candidato di release diventa 16 layer. Servono
altri due seed su quella forma prima di dichiarare qualsiasi cosa — un solo
seed non distingue cinque casi di merito da cinque casi di fortuna.

**Verifica multi-seed** (due seed per forma, il terzo in corso):

| | 8 layer | 16 layer |
|---|---|---|
| mandatory, media | 0.829 | **0.912** |
| mandatory, intervallo | 0.812 – 0.847 | 0.906 – 0.918 |
| `missing_fact`, media | 16.5 | **9.5** |
| `accepted_but_wrong`, media | 16.5 | **11.5** |
| recall dei fatti, media | 0.794 | **0.868** |

**I due intervalli non si toccano**: il seed peggiore a 16 layer (0.906)
batte il seed migliore a 8 layer (0.847) di cinque casi. Lo scarto fra seed
è di 1-3 casi, la differenza fra le due forme è di sette. Non è fortuna.

### H12 — Un Fact Inventory esplicito riduce le omissioni?

Ipotesi architetturale: `testo → inventario dei fatti → assemblaggio del
contratto → verifica`, invece di un solo salto dal testo al contratto.

*Misura*: da progettare come esperimento piccolo e controllato — stesso
modello in due passaggi, prima di valutare adattatori specializzati.

*Come si legge*: deve ridurre `missing_fact` **senza** aumentare
`hallucinated_fact`. Un inventario che elenca fatti inventati sposta il
problema invece di risolverlo.

*Esito*: non ancora avviata.

---

## Regola di scelta del candidato, dichiarata PRIMA di misurare

Scritta il 24 settembre, con il terzo seed ancora in addestramento e i
numeri del candidato non ancora esistenti.

Il motivo di scriverla adesso: nell'iterazione 13 ho provato **sette**
combinazioni di media dei pesi e tenuto la migliore, su un benchmark dove un
caso vale un punto. Quel 0.929 non era un risultato, era il massimo di sette
estrazioni. Una regola scelta dopo aver visto i numeri non è una regola: è
una giustificazione.

Due candidati, entrambi costruiti solo da corse su **dati puliti**, 16 layer,
stessa configurazione, seed 2026 / 1337 / 42:

- **A** — media dei pesi dei tre checkpoint finali (`soup_adapters.py`);
- **B** — il checkpoint finale del seed **mediano** per casi mandatory (non
  il migliore: il mediano).

**La regola.** Si sceglie **A** se il suo punteggio DEV cade dentro
l'intervallo dei tre seed singoli, perché a parità di risultato un modello
che non dipende dalla fortuna del seme è preferibile. Si sceglie **B** se A
cade **sotto** il seed peggiore, perché vorrebbe dire che mediare pesi di
bacini diversi fa danno — cosa già osservata quattro volte quando il
learning rate decade.

Nessun terzo tentativo. Se nessuno dei due convince, si dichiara che non c'è
un candidato e si riparte dall'analisi degli errori, non da un'altra media.

**L'holdout si esegue una volta sola**, sul candidato scelto da questa
regola, con `--no-failures`. Il suo numero non può cambiare la scelta: la
scelta è già fatta qui sopra. Serve a sapere quanto vale il candidato su
frasi mai viste, non a selezionarlo.

---

## H13 — La media dei pesi funziona solo a parità di inizializzazione

Non era un'ipotesi: era una scoperta dell'esperimento multi-seed, e spiega
in modo meccanico un risultato che avevo interpretato male.

Mediando i pesi dei **tre seed** a 16 layer, il modello **crolla a 0.259** di
casi mandatory, con il 58% delle richieste bloccate dal Guard. Non è un
peggioramento: è un modello rotto.

La ragione è semplice e me l'ero fatta sfuggire. Le medie che avevano
funzionato nell'iterazione 13 — v13, v14, v18 — usavano **tutte lo stesso
seed, 2026**: stessa inizializzazione dell'adattatore, quindi pesi che
vivono nello stesso bacino e che si possono mediare. Seed diversi partono da
punti diversi e finiscono in bacini diversi: il punto medio fra due bacini
non è un modello migliore, è un punto in mezzo al nulla.

**Cosa ne segue.** "La media dei pesi aiuta" era una regola scritta male. La
regola giusta è: *si possono mediare checkpoint che condividono
l'inizializzazione*, cioè checkpoint della stessa corsa o di corse che
partono dallo stesso seed. Fuori da quella condizione la media non è
conservativa, è distruttiva — e la regola di scelta del candidato, scritta
prima di misurare, ha fatto esattamente il suo lavoro: ha scartato A senza
che io potessi raccontarmi che 0.259 era "un caso sfortunato".
