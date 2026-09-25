# Jase Interpreter — rapporto di stato, 24 settembre 2026

**Verdetto: NON pronto per la produzione.** Il candidato migliore prende
0.906 sul benchmark di sviluppo e **0.240 su quello di release**. Il primo
numero non descrive la capacità del sistema: descrive quanto bene è stato
adattato a quel benchmark in quindici iterazioni.

---

## Il candidato

| | |
|---|---|
| modello base | `mlx-community/Qwen3.5-2B-4bit` |
| adattatore | `adapters/jase-v19-cap16`, checkpoint finale (iter 13000) |
| dataset | `data/v19_clean` — 27 504 train / 2 310 valid, 0 duplicati esatti del gold |
| semantica / schema | `semantics-1.1` / GoalContract `1.1` |
| Guard | `jase/guard.py` del 23 settembre (cascata §2, polarità, ancoraggio del target, conflitto su `eq` doppio, quattro forme di omissione) |
| prompt | `TRAIN_SYSTEM_PROMPT.txt`, invariato dalla v2. Thinking disabilitato |
| LoRA | 16 layer, rank 16, scale 32, dropout 0.05 |
| training | 13 000 iterazioni, batch 1, accumulo 4, AdamW, cosine decay 3·10⁻⁵ → 3·10⁻⁶ su 3250 step, warmup 100, max_seq 512 |
| seed | 2026 (**seed mediano** dei tre provati, non il migliore) |
| pipeline | `scripts/interpret.py` — retry solo sui guasti strutturali, chiarimento sugli errori semantici |

### Perché questo e non un altro

La regola di scelta era scritta in `docs/HYPOTHESES.md` **prima** di
misurare i candidati: media dei tre seed se cade dentro l'intervallo dei
seed singoli, altrimenti il checkpoint del seed **mediano**. La media è
crollata a 0.259 (vedi H13: mediare pesi di seed diversi è distruttivo), e
la regola ha selezionato il seed mediano — non il migliore (0.918), che
sarebbe stato il massimo di tre estrazioni.

---

## Metriche

### DEV — `gold_v1`, 102 casi, tre seed

| | media | intervallo |
|---|---|---|
| casi mandatory | 0.906 | 0.894 – 0.918 |
| semantic_exact | 0.833 | 0.824 – 0.843 |
| `accepted_but_wrong` | 12.3 | 10 – 14 |
| fatti, precision / recall | 0.948 / 0.874 | |
| bucket / operator | 0.998 / 0.983 | |
| `missing_fact` | 10.7 | 9 – 12 |
| validità JSON | 1.000 | |

**Questo è un punteggio di sviluppo.** Il `gold_v1` è stato usato per
guidare quindici iterazioni: ogni famiglia di errore che ho corretto,
l'ho corretta guardando lui. Misura ciò che ho già sistemato.

### RELEASE HOLDOUT — `release_holdout_v1`, 27 casi, una sola esecuzione

| | |
|---|---|
| casi mandatory | **0.240** (6 su 25) |
| semantic_exact | 0.222 |
| `accepted_but_wrong` | **15 su 27** |
| fatti, precision / recall | 0.667 / 0.553 |
| target_name / target_type | 0.690 / — |
| bucket / operator | 1.000 / 0.846 |
| validità JSON | 1.000 |
| esiti del Guard | ACCEPT 21 · CLARIFY 6 · REJECT 0 |

Tassonomia aggregata: `missing_fact` 13 · `target_name` 8 ·
`hallucinated_fact` 8 · `target_type` 7. I testi dei casi falliti **non
sono nel report** (`--no-failures`): leggerli e correggerli brucerebbe il
benchmark.

---

## Cosa dicono i due numeri, messi insieme

Il divario 0.906 → 0.240 ha due componenti e non sono separabili con i dati
attuali:

1. **Il DEV è saturato.** Quindici iterazioni di correzioni guidate da quel
   file. La revisione tecnica sospettava la contaminazione testuale (186
   duplicati): misurata, valeva **un caso** (H10). Il problema non erano i
   duplicati, era l'uso ripetuto.
2. **L'holdout è più difficile per costruzione.** L'ho scritto con refusi,
   slang, pronomi, ripensamenti, frasi monche, e in domini che il generatore
   non tocca. Parte del divario è "casi più duri", non solo "DEV gonfio" — ma
   è il punto: gli utenti scrivono così.

Il dato più grave non è 0.240: è **`accepted_but_wrong` = 15 su 27**. Su
richieste mai viste, più della metà dei contratti sbagliati **supera il
Guard** e arriverebbe al Planner. Sul DEV erano 10 su 102. L'invariante
fail-closed regge molto meno di quanto il DEV facesse credere: la quota di
richieste bloccate passa da 6% a 22%, ma non basta a compensare.

`target_name` a 0.690 contro 0.962 sul DEV dice dove si rompe per primo: su
fraseggi fuori distribuzione il modello non individua nemmeno l'oggetto
della richiesta.

---

## Limiti noti

- **L'holdout l'ho scritto io**, la stessa mano che ha scritto generatore,
  Guard e semantica. È garantito non contaminato, **non indipendente dai miei
  punti ciechi**. E 27 casi hanno un margine di rumore largo (±9 punti sul
  tasso). Il numero 0.240 va letto come "molto peggio del DEV", non come
  "24,0%".
- Il `gold_v1` non è più un test di generalizzazione. Resta utile come suite
  di regressione: se peggiora, qualcosa si è rotto.
- Nessuna misura su traffico reale. Nessuna misura di latenza e memoria in
  condizioni di produzione (il training gira a 3,7 GB di picco; l'inferenza
  non è stata profilata).
- Sei casi mandatory del DEV falliscono in ogni configurazione provata; tre
  li ferma il Guard.
- La media dei pesi funziona **solo fra checkpoint che condividono il seed**
  (H13). Documentato dopo aver rotto un modello per scoprirlo.

## Stato della suite di regressione

- `tests/test_guard.py` — 170 test, tutti verdi
- `scripts/guard_falsepos.py` — 0 falsi positivi sui 102 casi gold
- `scripts/leakage_check.py` — 0 duplicati esatti: training vs gold, training vs holdout
- `scripts/validate_dataset.py` — 29 814 righe valide contro lo schema 1.1
- `scripts/coverage_check.py` — nessun buco strutturale né lessicale
- generazione di 6000 esempi passata al Guard: 0 errori, 0 riparazioni

## Sicurezza di release

L'architettura è già compatibile con un confine di conferma: `action` resta
osservativa (`find`, `compare`, `monitor`) e le azioni irreversibili
(`buy`, `book`, `cancel`) compaiono solo nel ramo `then` di un `gate`, che
è una decisione esplicita e non una probabilità. **Con `accepted_but_wrong`
a 15 su 27 fuori distribuzione, nessuna azione irreversibile deve partire
senza conferma umana.** Le azioni reversibili possono avere più autonomia.

## Cosa serve prima di riparlare di produzione

1. Un holdout **scritto da persone diverse da me** — è il limite che nessuna
   altra misura aggira.
2. Ridurre `accepted_but_wrong` fuori distribuzione: il Guard non vede gli
   errori su fraseggi che non ha mai visto, e le quattro forme del
   rilevatore di omissioni sono tarate sul DEV.
3. L'ipotesi H12 (inventario dei fatti esplicito prima dell'assemblaggio del
   contratto) attacca `missing_fact`, che domina entrambi i benchmark.
4. Ampliare la **superficie linguistica** del generatore — non la semantica:
   i fraseggi. `target_name` a 0.690 è un problema di forma, non di regole.
