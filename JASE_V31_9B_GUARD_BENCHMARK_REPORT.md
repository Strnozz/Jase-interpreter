# Jase Interpreter 9B — benchmark V31

V31 è stato congelato **prima** dei nuovi controlli Guard V4: commit `172cf0b`, SHA `d3816e6eece2446861d0ef70b4cb01679c8c3ce7dc96a0f08c4a8f6de4e63c1f`. Sono 36 richieste, nessun overlap esatto con train/pannelli precedenti e Jaccard massimo 0,52381. Gold AI-authored, senza revisione umana indipendente. Guard V4 è separato e opt-in (`7716787`), con 19 test di regressione passati; frontend e adapter invariati. Gli stessi 36 input sono stati valutati sequenzialmente su V24, V25 e V26 best step1200.

| Metrica | V24 | V25 | V26 best |
| --- | ---: | ---: | ---: |
| JSON valido /36 | **36** | 34 | 35 |
| Canonical exact /36 | 9 | 9 | **13** |
| Candidate schema/projection hold /36 | 15 | 15 | **9** |
| Goal count corretto /36 | 21 | 21 | **27** |
| Action corrette | 20 | 22 | **29** |
| Target corretti | 16 | 20 | **26** |
| Facts corretti | 21 | 24 | **32** |
| Temporal corretti | 10 | 11 | **15** |
| Policy corrette | 13 | 17 | **22** |
| Guard V2 ACCEPT /36 | 19 | 15 | 21 |
| Guard V3 ACCEPT /36 | 19 | 14 | 20 |
| Guard V4 ACCEPT /36 | 19 | 12 | 19 |
| V4 ACCEPT ma non exact | 13 | 3 | 6 |
| V4 exact trattenuti | 3 | 0 | **0** |
| Output token/s | 18,06 | 18,01 | 17,94 |
| Picco VRAM inference | 8,19 GB | 8,19 GB | 8,18 GB |

V26 guadagna **6 casi e ne perde 2** rispetto a V24; guadagna **5 e ne perde 1** rispetto a V25. I miglioramenti includono arrivo ferroviario da origine esplicita (`v31-008`), reminder con orario fuori dal testo (`006`), prenotazione volo (`011`), servizio senza assunzione (`031`), dipendenza contatto da ricerca (`033`) e no-rent su ricerca veicolo (`034`). Le regressioni rispetto a V24 riguardano i messaggi a Roberta/Karim (`004/005`): V26 in un caso li scambia per una ricerca di messaggio, nell'altro aggiunge un ruolo temporale estraneo. Rispetto a V25 perde `005`. V26 produce anche un JSON invalido dove V24 ne aveva zero. Il vantaggio appaiato è reale su questo pannello, ma 13/36 non è qualità sufficiente per uso generale.

## Guard e revisione dei contratti accettati

V4 aggiunge HOLD a un output V26 non exact (`v31-036`, divieto implicito di prendere a noleggio il furgone), senza nuovi HOLD di casi exact. Sul gold V31 accetta 35/36: trattiene soltanto `reschedule`, per il quale non esiste una capability provider verificata. Questo controlla regressioni sul pannello, non dimostra assenza di falsi HOLD in produzione.

Fra i **6 output V26 ACCEPT ma non exact** dopo V4, la revisione agente trova **cinque errori operativi chiari**:

| Caso | Differenza materiale che passa il Guard |
| --- | --- |
| `v31-013` | Dimentica il divieto di prenotare l'alloggio e inventa un ordinamento per distanza. |
| `v31-022` | Perde la città e il contesto «non guido», scrive `access=walking` senza mapping verificato e inventa `search_date=non guido`. |
| `v31-023` | Omette «aperto adesso» e il contesto di sei persone; la cucina come campo alternativo resta da mappare. |
| `v31-028` | «Quattro offerte» finisce in `quantity` invece di `modifiers.limit=4`. |
| `v31-035` | `rent` di canoa senza `confirm_before`, nonostante «dopo che ti avrò detto di procedere»: rischio di azione senza consenso. |

`v31-007` usa target `regionale` invece di `treno regionale`, mantenendo origine, destinazione e ambedue gli orari: probabilmente equivalente, ma il Planner non è verificato. Dunque i sei non-exact non sono tutti errori materiali. V4 non generalizza ancora alla frase di consenso di `v31-035`; la protezione forte osservata in V30 non basta per dichiarare sicurezza.

## Conclusione

Il 9B V26 migliora stabilmente le metriche semantiche rispetto a V24/V25 anche su questo pannello fresco, ma **restano omissioni operative che il Guard lascia passare**, in particolare conferma di azioni, policy e cardinalità. Non promuovere V26/Guard V4 nel frontend. Un altro fine-tuning simile non è giustificato finché non esiste un handoff Planner 1.3 verificabile e una revisione indipendente dei gold. Prossimo passo: congelare un nuovo pannello prima di correggere i pattern di consenso/policy e prototipare un Planner handoff isolato che richieda binding esplicito dei campi, senza eseguire provider.
