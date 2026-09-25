# release_holdout_v1 — protocollo d'uso

Questo benchmark ha valore **solo** finché resta non contaminato. Un holdout
guardato spesso diventa un altro development set, e smette di misurare la
generalizzazione: misura quanto bene ci si è adattati a lui.

## Le regole

1. **Non si allena su questi casi.** Mai, in nessuna forma, nemmeno
   parafrasati. Il gate `scripts/leakage_check.py` lo verifica prima di ogni
   training e deve dare `exact_duplicate=0`.
2. **Non si corregge il sistema guardando questi casi.** Né il generatore,
   né il Guard, né il lessico, né lo schema. Se un caso fallisce, la
   correzione si progetta sul `gold_v1` (che è il banco di sviluppo) o su
   esempi nuovi scritti apposta — mai su questo file.
3. **Si esegue di rado**: solo su un candidato di release, non a ogni
   iterazione. Ogni esecuzione va annotata in `benchmarks/HOLDOUT_RUNS.md`
   con data, modello, commit dei dati e risultato.
4. **Si leggono i numeri aggregati e la tassonomia**, non i singoli casi.
   Il report dell'holdout viene generato con `--no-failures`: i testi dei
   casi falliti non entrano nel report proprio per non poterli leggere.
5. **Se una regola viene violata, l'holdout è bruciato.** Si archivia, si
   annota perché, e se ne scrive un altro. Non si fa finta di niente: un
   holdout contaminato dà numeri più alti e più falsi di nessun numero.

## Il sigillo

`build_holdout.py` scrive un SHA-256 del file serializzato in
`benchmarks/release_holdout_v1.sha256`. Se cambia senza che sia stata
aggiunta una riga in `HOLDOUT_RUNS.md` che lo spieghi, qualcuno ha ritoccato
il benchmark: i numeri prima e dopo non sono confrontabili.

## Limite dichiarato di questa prima versione

I casi di partenza **li ho scritti io**, la stessa "mano" che ha scritto il
generatore, il Guard e la semantica. Questo significa che il holdout è
garantito *non contaminato* (nessun caso è mai stato usato per correggere
qualcosa) ma **non è indipendente dai miei punti ciechi**: se c'è una forma
linguistica che non mi viene in mente di scrivere, non mi verrà in mente
nemmeno qui.

È un limite reale e non si risolve scrivendone altri cento nello stesso
modo. Si risolve solo con casi scritti da **persone diverse**, meglio se non
coinvolte nel progetto: chiedere a qualcuno "come lo chiederesti a voce?" e
trascrivere, senza aggiustare. Finché non ci sono, questo file va letto per
quello che è — una misura pulita, non una misura indipendente — e i suoi
numeri non vanno chiamati "accuratezza in produzione".

`scripts/collect_holdout.py` prepara i moduli vuoti da far compilare a chi
contribuisce, e `build_holdout.py` li annota secondo SEMANTICS.md.
