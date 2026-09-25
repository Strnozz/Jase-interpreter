"""release_holdout_v1 — benchmark di RELEASE, indipendente dallo sviluppo.

LEGGERE benchmarks/HOLDOUT_PROTOCOL.md PRIMA DI TOCCARE QUESTO FILE.

In breve: non si allena su questi casi, non si corregge il sistema
guardandoli, si esegue di rado e si leggono solo i numeri aggregati. Ogni
caso è scritto a mano applicando docs/SEMANTICS.md, in italiano parlato —
con refusi, frasi monche, pronomi, ripensamenti — perché è così che la gente
chiede le cose, non come le scrive un generatore.

I domini sono scelti apposta fuori da quelli del generatore: barbiere,
tatuatore, padel, sgombero, bombole, ricambi, ceramica, fisioterapia.
"""
from __future__ import annotations

CASES: list[dict] = []


def p(prop, op, value, unit=None, currency=None, reference=None, value_to=None):
    d = {"property": prop, "operator": op, "value": value}
    if value_to is not None:
        d["value_to"] = value_to
    if unit is not None:
        d["unit"] = unit
    if currency is not None:
        d["currency"] = currency
    if reference is not None:
        d["reference"] = reference
    return d


def g(action, raw_action, name, ttype, raw, A=None, C=None, P=None, gate=None):
    d = {"action": action, "raw_action": raw_action,
         "target": {"name": name, "type": ttype, "raw": raw},
         "attributes": A or [], "constraints": C or [], "preferences": P or []}
    if gate is not None:
        d["gate"] = gate
    return d


def gate(when, then, otherwise=None):
    return {"when": when, "then": then, "otherwise": otherwise}


def case(cid, text, goals, tags, clar=None, mandatory=False, note="",
         equivalences=None):
    CASES.append({
        "id": cid, "text": text, "tags": tags, "mandatory": mandatory,
        "note": note, "equivalences": equivalences or [],
        "contract": {
            "schema_version": "1.1",
            "goals": goals,
            "clarification": clar or {"required": False, "reason": None,
                                      "missing_fields": []},
        },
    })


IMPL = "(implicito)"

# ===================================================== 1. PARLATO SPONTANEO

case("h-001", "ciao, mi servirebbe un barbiere qui vicino a Navigli che sia aperto di lunedì",
     [g("find", "mi servirebbe", "barbiere", "professional_service", "un barbiere",
        C=[p("location", "near", "Navigli"),
           p("availability", "contains", "lunedì")])],
     ["spontaneo", "availability"], mandatory=True)

case("h-002", "senti volevo tagliarmi i capelli sabato mattina, sotto i 25 euro se si può",
     [g("find", "volevo", "taglio di capelli", "service", "tagliarmi i capelli",
        C=[p("date", "eq", "sabato mattina")],
        P=[p("price", "lt", 25, currency="EUR")])],
     ["spontaneo", "preferenza", "soglia"], mandatory=True,
     note="'se si può' è un marcatore debole e governa il prezzo, non la data")

case("h-003", "un tatuatore bravo con il blackwork, zona Trastevere. non mi interessa il prezzo",
     [g("find", "(implicito)", "tatuatore", "professional_service", "un tatuatore",
        A=[p("style", "eq", "blackwork")],
        C=[p("location", "near", "Trastevere")])],
     ["azione-implicita", "vocabolario-aperto"], mandatory=True,
     note="'non mi interessa il prezzo' non è un vincolo sul prezzo: è "
          "l'assenza di vincolo. Non produce predicato.")

case("h-004", "mi trovi un campo da padel libero giovedì sera? in due",
     [g("find", "mi trovi", "campo da padel", "place", "un campo da padel",
        A=[p("guests", "eq", 2)],
        C=[p("availability", "contains", "giovedì sera")])],
     ["spontaneo", "quantita", "ellissi"], mandatory=True,
     note="'in due' è una quantità di persone")

case("h-005", "cerco qualcuno che mi sgomberi la cantina, possibilmente entro il mese",
     [g("find", "cerco", "qualcuno", "professional_service", "qualcuno che mi sgomberi la cantina",
        A=[p("specialization", "contains", "sgomberi la cantina")],
        P=[p("date", "lte", "il mese")])],
     ["target-vago", "preferenza"], mandatory=True,
     note="§8: 'qualcuno' resta 'qualcuno'. Dedurre 'ditta di sgomberi' è "
          "conoscenza del mondo, lavoro del Capability Resolver.")

# ===================================================== 2. REFUSI E FRETTA

case("h-006", "trovmai una bombola del gas da 15 kg a domicilio a Bari",
     [g("find", "trovmai", "bombola del gas", "product", "una bombola del gas",
        A=[p("weight", "eq", 15, unit="kg")],
        C=[p("delivery", "contains", "domicilio"),
           p("location", "near", "Bari")])],
     ["refuso", "unita"], mandatory=True)

case("h-007", "hoo bisogno di un fisioterapista x la spalla, prima possibile",
     [g("find", "hoo bisogno di", "fisioterapista", "professional_service", "un fisioterapista",
        A=[p("specialization", "contains", "spalla")],
        C=[p("date", "eq", "prima possibile")])],
     ["refuso", "slang"], mandatory=True,
     note="'x' per 'per' è abbreviazione da messaggio, non un valore")

case("h-008", "ricambi x fiat panda 2015, il filtro dell'aria. nn originale va bene lo stesso",
     [g("find", IMPL, "filtro dell'aria", "product", "il filtro dell'aria",
        A=[p("model", "eq", "fiat panda 2015")],
        C=[p("brand", "neq", "originale")])],
     ["refuso", "slang", "negazione"], mandatory=True,
     note="'nn originale' è una negazione scritta in abbreviato")

# ===================================================== 3. RIPENSAMENTI

case("h-009", "prenotami un tavolo per stasera alle 21. anzi facciamo alle 20:30",
     [g("book", "prenotami", "tavolo", "place", "un tavolo",
        C=[p("date", "eq", "stasera"),
           p("time", "eq", "20:30")])],
     ["ripensamento", "multi-frase"], mandatory=True,
     note="il ripensamento SOSTITUISCE: vale l'ultimo valore detto, non "
          "entrambi. Due orari sullo stesso goal sarebbero un conflitto.")

case("h-010", "cercami un trapano. no aspetta, un avvitatore, quello a batteria",
     [g("find", "cercami", "avvitatore", "product", "un avvitatore",
        A=[p("feature", "contains", "a batteria")])],
     ["ripensamento"], mandatory=True,
     note="il target corretto è quello finale: il trapano è stato ritirato")

case("h-011", "vorrei un corso di ceramica, la sera. anche il sabato mattina in realtà",
     [g("find", "vorrei", "corso di ceramica", "education", "un corso di ceramica",
        C=[p("availability", "contains", "la sera|il sabato mattina")])],
     ["ripensamento", "lista"], mandatory=True,
     note="qui il ripensamento AGGIUNGE invece di sostituire ('anche'): due "
          "alternative accettate, quindi un insieme, non due fatti")

# ===================================================== 4. PRONOMI E CONTESTO

case("h-012", "me lo prendi tu? quello da 500 giga",
     [g("buy", "me lo prendi", "unknown", None, "quello da 500 giga",
        A=[p("storage", "eq", 500, unit="GB")])],
     ["pronome", "chiarimento"], mandatory=True,
     clar={"required": True, "reason": "target_missing",
           "missing_fields": ["target"]},
     note="la proprietà è chiara, l'oggetto no: si chiede il target ma non "
          "si butta via il fatto già capito")

case("h-013", "quello di ieri però più economico",
     [g("find", IMPL, "unknown", None, "quello di ieri")],
     ["pronome", "chiarimento", "anafora"], mandatory=True,
     clar={"required": True, "reason": "goal_unintelligible",
           "missing_fields": ["target"]},
     note="'più economico' è un comparativo senza termine di paragone: "
          "senza il riferimento di 'ieri' non è rappresentabile")

# ===================================================== 5. NUMERI CON RUOLI

case("h-014", "noleggio furgone per 3 giorni, massimo 200 euro in tutto, a Prato",
     [g("rent", "noleggio", "furgone", "transport", "furgone",
        A=[p("duration", "eq", 3, unit="days")],
        C=[p("price", "lte", 200, currency="EUR"),
           p("location", "near", "Prato")])],
     ["numeri-ruoli", "unita"], mandatory=True)

case("h-015", "un servizio fotografico il 12, per 4 persone, sui 300",
     [g("find", IMPL, "servizio fotografico", "service", "un servizio fotografico",
        A=[p("guests", "eq", 4)],
        C=[p("date", "eq", 12),
           p("price", "eq", 300)])],
     ["numeri-ruoli", "azione-implicita"], mandatory=True,
     note="tre numeri, tre ruoli: data, persone, prezzo. 'sui 300' è un "
          "prezzo approssimativo senza valuta nominata → currency null")

case("h-016", "lezioni di chitarra, 2 a settimana, non più di 30 euro l'una",
     [g("find", IMPL, "lezioni di chitarra", "education", "lezioni di chitarra",
        A=[p("quantity", "eq", 2)],
        C=[p("price", "lte", 30, currency="EUR")])],
     ["numeri-ruoli", "quantita"], mandatory=True)

# ===================================================== 6. NEGAZIONI

case("h-017", "una pizzeria a Salerno, ma non quella con la fila fuori",
     [g("find", IMPL, "pizzeria", "place", "una pizzeria",
        C=[p("location", "near", "Salerno")])],
     ["negazione", "non-rappresentabile"], mandatory=False,
     note="'quella con la fila fuori' è un riferimento a un'esperienza "
          "condivisa, non una proprietà del mondo: non produce predicato")

case("h-018", "cerco un divano, né in pelle né finta pelle, sotto gli 800",
     [g("find", "cerco", "divano", "product", "un divano",
        C=[p("material", "not_in", "pelle|finta pelle"),
           p("price", "lt", 800)])],
     ["negazione", "lista"], mandatory=True,
     note="'né A né B' è un insieme chiuso escluso → not_in. Nessuna valuta "
          "nominata → currency null (§6.3)")

case("h-019", "un appartamento in affitto a Padova, senza spese condominiali e senza cauzione",
     [g("find", IMPL, "appartamento", "real_estate", "un appartamento",
        C=[p("extra_costs", "excludes", "spese condominiali"),
           p("extra_costs", "excludes", "cauzione"),
           p("location", "near", "Padova")])],
     ["negazione"], mandatory=True,
     note="'senza A e senza B': la negazione si ripete, quindi sono DUE "
          "fatti separati, non una lista (§5.4). La proprietà NON è `fee`: "
          "il lessico la dichiara monetaria e vuole un numero, mentre qui si "
          "escludono due voci di spesa, non due importi. Il vocabolario è "
          "aperto: `extra_costs`.")

# ===================================================== 7. PREFERENZE vs VINCOLI

case("h-020", "un hotel a Riccione con parcheggio, magari anche con la spa ma non è fondamentale",
     [g("find", IMPL, "hotel", "accommodation", "un hotel",
        A=[p("amenity", "contains", "parcheggio")],
        C=[p("location", "near", "Riccione")],
        P=[p("amenity", "contains", "spa")])],
     ["preferenza", "contrasto"], mandatory=True,
     note="due dotazioni, due forze diverse nella stessa frase")

case("h-021", "mi serve una stampante, deve stampare fronte retro, il wifi se c'è meglio",
     [g("find", "mi serve", "stampante", "product", "una stampante",
        C=[p("feature", "contains", "fronte retro")],
        P=[p("feature", "contains", "wifi")])],
     ["preferenza", "obbligo"], mandatory=True,
     note="'deve' è obbligo → constraints; 'se c'è meglio' è debole → preferences")

# ===================================================== 8. MULTI-GOAL

case("h-022", "prenotami il dentista per martedì e ricordami di comprare il collutorio",
     [g("book", "prenotami", "dentista", "professional_service", "il dentista",
        C=[p("date", "eq", "martedì")]),
      g("buy", "ricordami di comprare", "collutorio", "product", "il collutorio")],
     ["multi-goal"], mandatory=True,
     note="due goal, e il martedì vale SOLO per il primo")

case("h-023", "cerca un idraulico e un elettricista, tutti e due a Monza, entro venerdì",
     [g("find", "cerca", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Monza"), p("date", "lte", "venerdì")]),
      g("find", "cerca", "elettricista", "professional_service", "un elettricista",
        C=[p("location", "near", "Monza"), p("date", "lte", "venerdì")])],
     ["multi-goal", "vincolo-condiviso"], mandatory=True,
     note="'tutti e due' distribuisce entrambi i vincoli su entrambi i goal")

# ===================================================== 9. CONDIZIONALI

case("h-024", "guarda se c'è ancora il monitor che ti ho detto, se sta sotto i 200 prendilo",
     [g("find", "guarda", "monitor", "product", "il monitor",
        gate=gate(p("price", "lt", 200), "buy"))],
     ["condizionale", "gate"], mandatory=True)

case("h-025", "controlla il prezzo del volo per Siviglia e fammi sapere se cala",
     [g("monitor", "controlla", "volo", "transport", "il volo",
        C=[p("destination", "eq", "Siviglia")])],
     ["condizionale", "monitor"], mandatory=True,
     note="§11.3: se l'unica conseguenza è essere informati non c'è bivio, "
          "è `monitor`. 'se cala' non ha una soglia: nessun predicato.")

# ===================================================== 10. AMBIGUE

# La prima versione di questo caso era la frase "fammi sapere", che il gate
# anti-leakage ha trovato 30 volte nel training set: il generatore la usa
# come coda condizionale. Un caso di holdout che sta già nei dati misura la
# memoria, non la generalizzazione — riscritto.
case("h-026", "boh, vedi tu",
     [g("find", "vedi tu", "unknown", None, "vedi tu")],
     ["ambigua", "chiarimento"], mandatory=True,
     clar={"required": True, "reason": "goal_unintelligible",
           "missing_fields": ["target", "action"]})

case("h-027", "qualcosa per il mal di schiena",
     [g("find", IMPL, "qualcosa per il mal di schiena", "unknown", "qualcosa per il mal di schiena")],
     ["ambigua", "target-vago"], mandatory=False,
     note="ambiguo fra un prodotto e un professionista: il target resta "
          "quello detto, il tipo no. Caso non mandatory proprio perché "
          "un essere umano chiederebbe anche lui un chiarimento.")
