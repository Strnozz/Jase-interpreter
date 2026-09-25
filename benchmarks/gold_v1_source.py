"""Benchmark GOLD v1 — sorgente leggibile, scritto a mano.

REGOLE D'USO
  • Questo file NON è generato da un template e NON deve MAI finire in training.
  • Ogni caso applica docs/SEMANTICS.md alla lettera. Se un caso e il documento
    divergono, uno dei due è sbagliato: si discute, non si ritocca il gold per
    far passare il modello.
  • `mandatory=True` marca i casi che sono gate di release: devono essere
    corretti al 100% perché una versione sia dichiarata production-ready.

`build_gold.py` valida questo file contro lo schema e lo serializza in
benchmarks/gold_v1.jsonl.
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
    d = {
        "action": action, "raw_action": raw_action,
        "target": {"name": name, "type": ttype, "raw": raw},
        "attributes": A or [], "constraints": C or [], "preferences": P or [],
    }
    if gate is not None:
        d["gate"] = gate
    return d


def gate(when, then, otherwise=None):
    """Condizione valutata sul risultato, con le azioni dei due rami (§11)."""
    return {"when": when, "then": then, "otherwise": otherwise}


def case(cid, text, goals, tags, clar=None, mandatory=False, note="",
         equivalences=None):
    """`equivalences`: gruppi di property intercambiabili PER QUESTO caso.

    Si usa solo dove non esiste una risposta canonica unica e nessun essere
    umano saprebbe scegliere. Non è un modo per far passare il modello: è il
    modo di ammettere che la domanda è ambigua invece di premiare a caso una
    delle risposte."""
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


# ============================================================ 1. REGRESSIONI
# Casi storicamente falliti: sono gate di release permanenti.

case("reg-001",
     "Trovami un panino con la mortadella a Milano sotto i 10 euro",
     [g("find", "trovami", "panino", "food", "un panino con la mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10, currency="EUR")])],
     ["regression", "food", "negation:no", "strictness:lt"], mandatory=True,
     note="iter400 perdeva 'mortadella', duplicava il prezzo e metteva la location in attributes")

case("reg-002",
     "Prenotami un hotel da almeno 3 stelle a Roma sabato per favore",
     [g("book", "prenotami", "hotel", "accommodation", "un hotel da almeno 3 stelle",
        C=[p("stars", "gte", 3),
           p("location", "near", "Roma"),
           p("date", "eq", "sabato")])],
     ["regression", "accommodation", "strictness:gte", "temporal"], mandatory=True,
     note="produceva predicati malformati tipo {'property':'Roma','type':'location'}")

case("reg-003",
     "Cercami un hotel a Roma.",
     [g("find", "cercami", "hotel", "accommodation", "un hotel",
        C=[p("location", "near", "Roma")])],
     ["regression", "sparse", "anti-hallucination"], mandatory=True,
     note="il modello inventava stelle, prezzo, date, ospiti, email")

case("reg-004",
     "Mi serve un elettricista.",
     [g("find", "mi serve", "elettricista", "professional_service", "un elettricista")],
     ["regression", "sparse", "anti-hallucination"], mandatory=True,
     note="contratto completamente vuoto è la risposta giusta")

case("reg-005",
     "Trovami un panino a Milano per dieci euro",
     [g("find", "trovami", "panino", "food", "un panino",
        C=[p("location", "near", "Milano"),
           p("price", "eq", 10, currency="EUR")])],
     ["regression", "number-role", "food"], mandatory=True,
     note="'dieci euro' non deve diventare time=10:00")

case("reg-006",
     "Prenotami un tavolo alle dieci",
     [g("book", "prenotami", "tavolo", "place", "un tavolo",
        C=[p("time", "eq", "10:00")])],
     ["regression", "number-role", "temporal"], mandatory=True,
     note="contrasto con reg-005: qui il dieci È un orario")


# ====================================================== 2. CONTRASTI MINIMI
# Terne con/senza/possibilmente e soglie. Testano la cascata R1–R6.

case("con-001", "Vorrei un panino con la mortadella",
     [g("find", "vorrei", "panino", "food", "un panino con la mortadella",
        A=[p("ingredient", "contains", "mortadella")])],
     ["contrast", "bucket:attribute"], mandatory=True)

case("con-002", "Vorrei un panino senza mortadella",
     [g("find", "vorrei", "panino", "food", "un panino senza mortadella",
        C=[p("ingredient", "excludes", "mortadella")])],
     ["contrast", "bucket:constraint", "negation"], mandatory=True)

case("con-003", "Vorrei un panino possibilmente con la mortadella",
     [g("find", "vorrei", "panino", "food", "un panino",
        P=[p("ingredient", "contains", "mortadella")])],
     ["contrast", "bucket:preference"], mandatory=True)

case("con-004", "Vorrei un panino che deve avere la mortadella",
     [g("find", "vorrei", "panino", "food", "un panino",
        C=[p("ingredient", "contains", "mortadella")])],
     ["contrast", "bucket:constraint", "obligation"], mandatory=True)

case("con-010", "Cercami una bici sotto i 500 euro",
     [g("find", "cercami", "bici", "product", "una bici",
        C=[p("price", "lt", 500, currency="EUR")])],
     ["contrast", "strictness:lt"], mandatory=True)

case("con-011", "Cercami una bici massimo 500 euro",
     [g("find", "cercami", "bici", "product", "una bici",
        C=[p("price", "lte", 500, currency="EUR")])],
     ["contrast", "strictness:lte"], mandatory=True)

case("con-012", "Cercami una bici da più di 500 euro",
     [g("find", "cercami", "bici", "product", "una bici",
        C=[p("price", "gt", 500, currency="EUR")])],
     ["contrast", "strictness:gt"], mandatory=True)

case("con-013", "Cercami una bici da almeno 500 euro",
     [g("find", "cercami", "bici", "product", "una bici",
        C=[p("price", "gte", 500, currency="EUR")])],
     ["contrast", "strictness:gte"], mandatory=True)

case("con-014", "Cercami una bici tra i 300 e i 500 euro",
     [g("find", "cercami", "bici", "product", "una bici",
        C=[p("price", "between", 300, currency="EUR", value_to=500)])],
     ["contrast", "strictness:between"], mandatory=True)

case("con-020", "Prenotami un tavolo alle 20",
     [g("book", "prenotami", "tavolo", "place", "un tavolo",
        C=[p("time", "eq", "20:00")])],
     ["contrast", "temporal"], mandatory=True)

case("con-021", "Prenotami un tavolo entro le 20",
     [g("book", "prenotami", "tavolo", "place", "un tavolo",
        C=[p("time", "before", "20:00")])],
     ["contrast", "temporal"], mandatory=True)

case("con-022", "Prenotami un tavolo dopo le 20",
     [g("book", "prenotami", "tavolo", "place", "un tavolo",
        C=[p("time", "after", "20:00")])],
     ["contrast", "temporal"], mandatory=True)

case("con-023", "Cercami un idraulico entro 20 km",
     [g("find", "cercami", "idraulico", "professional_service", "un idraulico",
        C=[p("distance", "within", 20, unit="km")])],
     ["contrast", "spatial"], mandatory=True)

case("con-024", "Cercami un idraulico entro 20 euro",
     [g("find", "cercami", "idraulico", "professional_service", "un idraulico",
        C=[p("price", "lte", 20, currency="EUR")])],
     ["contrast", "number-role"], mandatory=True)

case("con-030", "Trovami un'auto rossa",
     [g("find", "trovami", "auto", "product", "un'auto rossa",
        A=[p("color", "eq", "rossa")])],
     ["contrast", "bucket:attribute"])

case("con-031", "Trovami un'auto non rossa",
     [g("find", "trovami", "auto", "product", "un'auto",
        C=[p("color", "neq", "rossa")])],
     ["contrast", "negation"], mandatory=True)

case("con-032", "Trovami un'auto preferibilmente rossa",
     [g("find", "trovami", "auto", "product", "un'auto",
        P=[p("color", "eq", "rossa")])],
     ["contrast", "bucket:preference"], mandatory=True)

case("con-033", "Trovami un'auto preferibilmente non rossa",
     [g("find", "trovami", "auto", "product", "un'auto",
        P=[p("color", "neq", "rossa")])],
     ["contrast", "negation", "bucket:preference"], mandatory=True,
     note="il marcatore debole (R1) vince sulla negazione (R2)")

case("con-040", "Un hotel a 4 stelle a Bologna",
     [g("find", "(implicito)", "hotel", "accommodation", "un hotel a 4 stelle",
        A=[p("stars", "eq", 4)],
        C=[p("location", "near", "Bologna")])],
     ["contrast", "bucket:attribute", "implicit-action"])

case("con-041", "Un hotel da almeno 4 stelle a Bologna",
     [g("find", "(implicito)", "hotel", "accommodation", "un hotel da almeno 4 stelle",
        C=[p("stars", "gte", 4), p("location", "near", "Bologna")])],
     ["contrast", "bucket:constraint", "strictness:gte", "implicit-action"], mandatory=True)


# ================================================ 3. ANTI-ALLUCINAZIONE (sparse)

case("sparse-001", "Trovami un ristorante",
     [g("find", "trovami", "ristorante", "place", "un ristorante")],
     ["sparse", "anti-hallucination"], mandatory=True)

case("sparse-002", "Mi serve un idraulico a Torino",
     [g("find", "mi serve", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Torino")])],
     ["sparse", "anti-hallucination"], mandatory=True)

case("sparse-003", "Comprami un libro",
     [g("buy", "comprami", "libro", "media", "un libro")],
     ["sparse", "anti-hallucination"], mandatory=True)

case("sparse-004", "Cerco un volo per Londra",
     [g("find", "cerco", "volo", "transport", "un volo",
        C=[p("destination", "eq", "Londra")])],
     ["sparse", "anti-hallucination", "transport"], mandatory=True,
     note="nessuna data, nessun prezzo, nessuna compagnia inventata")

case("sparse-005", "Vorrei adottare un gatto",
     [g("find", "vorrei adottare", "gatto", "animal", "un gatto")],
     ["sparse", "anti-hallucination", "animal"], mandatory=True)

case("sparse-006", "Prenotami un hotel",
     [g("book", "prenotami", "hotel", "accommodation", "un hotel")],
     ["sparse", "anti-hallucination"], mandatory=True)

case("sparse-007", "Cercami un corso di inglese online",
     [g("find", "cercami", "corso", "education", "un corso di inglese",
        A=[p("subject", "eq", "inglese"), p("format", "eq", "online")])],
     ["sparse", "education"],
     note="'di inglese' è la MATERIA del corso; `language` sarebbe la lingua "
          "in cui il corso è tenuto, che può essere un'altra cosa")

case("sparse-008", "Trovami un fotografo",
     [g("find", "trovami", "fotografo", "professional_service", "un fotografo")],
     ["sparse", "anti-hallucination"], mandatory=True)


# ==================================================== 4. LINGUAGGIO NATURALE
# Frasi lunghe, informali, con rumore conversazionale.

case("nat-001",
     "Ho fame e sono a Milano, mi andrebbe qualcosa con la mortadella ma dieci euro per un panino proprio non li spendo.",
     [g("find", "mi andrebbe", "panino", "food", "un panino",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10, currency="EUR")])],
     ["natural", "noisy", "food"], mandatory=True,
     note="'dieci euro proprio non li spendo' = lt 10, non lte")

case("nat-002",
     "Mi trovi uno Sphynx femmina? Preferirei giovane, però non voglio farmi più di 80 km.",
     [g("find", "mi trovi", "Sphynx", "animal", "uno Sphynx femmina",
        A=[p("sex", "eq", "female")],
        C=[p("distance", "within", 80, unit="km")],
        P=[p("age", "eq", "giovane")])],
     ["natural", "animal", "preference", "spatial", "target-specificity"],
     mandatory=True,
     note="'uno Sphynx' non si astrae in 'gatto': servirebbe conoscenza del "
          "mondo, che il vocabolario aperto non assume (§8). E 'breed' non "
          "si ripete, sarebbe un duplicato del target.")

case("nat-003",
     "Serve qualcuno per sistemare una presa oggi, sono a Verbania, cento euro proprio massimo.",
     [g("find", "serve", "qualcuno", "professional_service", "qualcuno per sistemare una presa",
        A=[p("specialization", "contains", "presa")],
        C=[p("date", "eq", "oggi"),
           p("location", "near", "Verbania"),
           p("price", "lte", 100, currency="EUR")])],
     ["natural", "noisy", "service", "implicit-target", "target-specificity"],
     note="CORRETTO: pretendeva target='elettricista', che richiede di sapere "
          "che le prese sono lavoro da elettricisti. È esattamente "
          "l'astrazione per conoscenza del mondo che §8 vieta — la stessa "
          "che avevo già tolto per 'Sphynx'. L'utente ha detto 'qualcuno': "
          "mappare 'specialization: presa' su un elettricista è lavoro del "
          "Capability Resolver, non dell'Interpreter.")

case("nat-004",
     "cerco zorba x7 32gb palermo no rosso max 500",
     [g("find", "cerco", "Zorba X7", "product", "zorba x7",
        A=[p("storage", "eq", 32, unit="GB")],
        C=[p("location", "near", "Palermo"),
           p("color", "neq", "rosso"),
           # "max 500": nessun simbolo né parola di valuta. §6.3 dice
           # che la valuta va ancorata: dedurre EUR è compito del
           # Capability Resolver, che conosce il locale, non qui.
           p("price", "lte", 500)])],
     ["natural", "telegraphic", "open-vocab", "negation", "ambiguous-property"],
     mandatory=True, equivalences=[["ram", "storage"]],
     note="'32gb' su un dispositivo inventato non dice se sia RAM o "
          "archiviazione: nessun umano saprebbe sceglierlo, quindi valgono "
          "entrambe (vedi `equivalences`)")

case("nat-005",
     "panino mortadella milano <10€",
     [g("find", "(implicito)", "panino", "food", "panino mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10, currency="EUR")])],
     ["natural", "telegraphic", "implicit-action"], mandatory=True)

case("nat-006",
     "A Milano vorrei un panino con la mortadella, senza arrivare a spendere 10€.",
     [g("find", "vorrei", "panino", "food", "un panino con la mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10, currency="EUR")])],
     ["natural", "paraphrase"], mandatory=True)

case("nat-007",
     "Sono a Milano e ho voglia di un panino alla mortadella. Dieci euro però non li spendo.",
     [g("find", "ho voglia", "panino", "food", "un panino alla mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10, currency="EUR")])],
     ["natural", "paraphrase", "multi-sentence"], mandatory=True)

case("nat-008",
     "Mi trovi qualcosa tipo un panino con la mortadella qui a Milano? Vorrei restare sotto i dieci.",
     [g("find", "mi trovi", "panino", "food", "un panino con la mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milano"),
           p("price", "lt", 10)])],
     ["natural", "paraphrase", "elliptical-currency"], mandatory=True,
     note="'sotto i dieci' senza simbolo, senza parola di valuta e senza verbo "
          "monetario: currency resta null (§6.3). La valuta la risolve il "
          "Capability Resolver dalla localizzazione, come fa con 'stasera'.")

case("nat-009",
     "Buongiorno, avrei bisogno di un idraulico abbastanza urgentemente, oggi pomeriggio se possibile, zona Genova. Grazie mille!",
     [g("find", "avrei bisogno", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Genova")],
        P=[p("date", "eq", "oggi pomeriggio")])],
     ["natural", "polite-noise", "preference"], mandatory=True)

case("nat-010",
     "vorrei un macbook con almeno 32 gb di ram, ricondizionato va benissimo, sotto i 1500",
     [g("find", "vorrei", "MacBook", "product", "un macbook",
        A=[p("condition", "eq", "ricondizionato")],
        C=[p("ram", "gte", 32, unit="GB"),
           p("price", "lt", 1500)])],  # §6.3: "sotto i 1500", valuta non nominata
     ["natural", "electronics", "strictness:gte"], mandatory=True)

case("nat-011",
     "Trovami un volo per Londra sotto i 90€, non Ryanair, devo arrivare prima delle 18.",
     [g("find", "trovami", "volo", "transport", "un volo",
        C=[p("destination", "eq", "Londra"),
           p("price", "lt", 90, currency="EUR"),
           p("carrier", "neq", "Ryanair"),
           p("arrival_time", "before", "18:00")])],
     ["natural", "transport", "negation", "temporal"], mandatory=True)

case("nat-012",
     "Vorrei uno Sphynx femmina sotto i sei mesi entro 100 km da Milano.",
     [g("find", "vorrei", "Sphynx", "animal", "uno Sphynx femmina",
        A=[p("sex", "eq", "female")],
        C=[p("age", "lt", 6, unit="months"),
           p("distance", "within", 100, unit="km", reference="Milano")])],
     ["natural", "animal", "spatial", "strictness:lt"], mandatory=True)

case("nat-013",
     "Mi serve un elettricista stasera entro 100 euro",
     [g("find", "mi serve", "elettricista", "professional_service", "un elettricista",
        C=[p("date", "eq", "stasera"),
           p("price", "lte", 100, currency="EUR")])],
     ["natural", "temporal", "anti-hallucination"], mandatory=True,
     note="'stasera' NON va risolto in un orario")

case("nat-014",
     "Cercami un MacBook con almeno 32 GB di RAM.",
     [g("find", "cercami", "MacBook", "product", "un MacBook",
        C=[p("ram", "gte", 32, unit="GB")])],
     ["natural", "electronics", "strictness:gte"], mandatory=True)

case("nat-015",
     "ciao! senti, mi servirebbe una pizza senza glutine dalle parti di Bologna, e possibilmente che consegnino",
     [g("find", "mi servirebbe", "pizza", "food", "una pizza senza glutine",
        C=[p("diet", "excludes", "glutine"),
           p("location", "near", "Bologna")],
        P=[p("delivery", "eq", "consegna")])],
     ["natural", "negation", "preference", "food"])

case("nat-016",
     "Ho bisogno di un traduttore dall'inglese all'italiano, madrelingua, entro venerdì.",
     [g("find", "ho bisogno", "traduttore", "professional_service", "un traduttore",
        A=[p("language", "contains", "inglese"),
           p("language", "contains", "italiano"),
           p("certification", "eq", "madrelingua")],
        C=[p("date", "before", "venerdì")])],
     ["natural", "service", "temporal"])

case("nat-017",
     "una stampante 3d usata, che non sia cinese, qui in zona Firenze",
     [g("find", "(implicito)", "stampante 3D", "product", "una stampante 3d usata",
        A=[p("condition", "eq", "usata")],
        C=[p("brand", "neq", "cinese"),
           p("location", "near", "Firenze")])],
     ["natural", "negation", "implicit-action", "open-vocab"])

case("nat-018",
     "Prenotami due notti in un bed and breakfast a Napoli dal 12 al 14",
     [g("book", "prenotami", "bed and breakfast", "accommodation", "un bed and breakfast",
        A=[p("duration", "eq", 2, unit="nights")],
        C=[p("location", "near", "Napoli"),
           p("date", "between", 12, value_to=14)])],
     ["natural", "accommodation", "temporal", "range"])

case("nat-019",
     "Mi cerchi un veterinario aperto la domenica? Sono a Bari.",
     [g("find", "mi cerchi", "veterinario", "professional_service", "un veterinario",
        C=[p("availability", "contains", "domenica"),
           p("location", "near", "Bari")])],
     ["natural", "service"], mandatory=True,
     note="availability è proprietà di selezione (§3.1) → constraints, non attributes")

case("nat-020",
     "compra due biglietti per il concerto di sabato, in platea, non più di 60 euro l'uno",
     [g("buy", "compra", "biglietto", "ticket", "due biglietti per il concerto",
        A=[p("quantity", "eq", 2), p("section", "eq", "platea"),
           # "per il CONCERTO di sabato": l'evento è un fatto dichiarato
           # dall'utente quanto la data, e il gold lo dimenticava. Il modello
           # lo produceva e si prendeva un errore per averlo fatto: lo stesso
           # principio su cui è costruito il rilevatore di omissioni.
           p("event", "contains", "concerto")],
        C=[p("date", "eq", "sabato"),
           p("price", "lte", 60, currency="EUR")])],
     ["natural", "ticket", "strictness:lte"], mandatory=True)


# ============================================================== 5. MULTI-GOAL

case("multi-001",
     "Trovami un hotel a Roma e prenotami un treno per sabato",
     [g("find", "trovami", "hotel", "accommodation", "un hotel",
        C=[p("location", "near", "Roma")]),
      g("book", "prenotami", "treno", "transport", "un treno",
        C=[p("date", "eq", "sabato")])],
     ["multi-goal"], mandatory=True)

case("multi-002",
     "Mi servono un idraulico e un elettricista a Torino",
     [g("find", "mi servono", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Torino")]),
      g("find", "mi servono", "elettricista", "professional_service", "un elettricista",
        C=[p("location", "near", "Torino")])],
     ["multi-goal", "shared-constraint"], mandatory=True,
     note="il vincolo condiviso si ripete in entrambi i goal (§10)")

case("multi-003",
     "Cercami una chitarra usata sotto i 300 euro e confrontami tre amplificatori",
     [g("find", "cercami", "chitarra", "product", "una chitarra usata",
        A=[p("condition", "eq", "usata")],
        C=[p("price", "lt", 300, currency="EUR")]),
      g("compare", "confrontami", "amplificatore", "product", "tre amplificatori",
        A=[p("quantity", "eq", 3)])],
     ["multi-goal"])

case("multi-004",
     "Trova e prenota un hotel a Lisbona",
     [g("book", "trova e prenota", "hotel", "accommodation", "un hotel",
        C=[p("location", "near", "Lisbona")])],
     ["multi-goal:no", "composite-action"], mandatory=True,
     note="azione composta sullo stesso oggetto = UN goal (§10)")


# ============================================================ 6. CONDIZIONALI
# GoalContract v1 non li rappresenta: policy degrade_to_clarification (§11).

case("cond-001",
     "Se la trovi sotto i 600 comprala, altrimenti dimmi solo dove l'hai vista.",
     [g("find", "se la trovi", "unknown", None, "la",
        gate=gate(p("price", "lt", 600), "buy", "notify"))],
     ["conditional", "gate", "two-branch"],
     clar={"required": True, "reason": "target_missing",
           "missing_fields": ["target"]}, mandatory=True,
     note="due problemi distinti dichiarati separatamente: il gate copre il "
          "condizionale, la clarification copre il target mancante ('la'). "
          "'sotto i 600' non nomina la valuta → currency null (§6.3).")

case("cond-002",
     "Comprami la scheda video solo se costa meno di 400 euro",
     [g("find", "comprami", "scheda video", "product", "la scheda video",
        gate=gate(p("price", "lt", 400, currency="EUR"), "buy"))],
     ["conditional", "gate"], mandatory=True,
     note="il prezzo NON è un filtro di ricerca: si cerca la scheda, si "
          "guarda il prezzo, si decide (§11.3.2). Quindi sta nel gate e non "
          "in constraints.")

case("cond-003",
     "Trova un volo per Madrid, e se ne trovi uno diretto prenotalo subito",
     [g("find", "trova", "volo", "transport", "un volo",
        C=[p("destination", "eq", "Madrid")],
        gate=gate(p("route", "eq", "diretto"), "book"))],
     ["conditional", "gate"], mandatory=True,
     note="'diretto' è la condizione della prenotazione, non un attributo "
          "della ricerca: si cerca un volo per Madrid, e SE è diretto lo si "
          "prenota")

case("cond-004",
     "Cercami un hotel a Lisbona e se costa meno di 80 euro prenotalo",
     [g("find", "cercami", "hotel", "accommodation", "un hotel",
        C=[p("location", "near", "Lisbona")],
        gate=gate(p("price", "lt", 80, currency="EUR"), "book"))],
     ["conditional", "gate"], mandatory=True)

case("cond-005",
     "Controlla il prezzo della RTX 6080 e avvisami se scende sotto i 700",
     [g("monitor", "controlla", "RTX 6080", "product", "la RTX 6080",
        C=[p("price", "lt", 700)])],
     ["conditional", "action:monitor"], mandatory=True,
     note="§11.3: se l'unica conseguenza è essere informati non c'è un bivio "
          "da rappresentare — `monitor` significa già 'guarda e dimmelo', e "
          "la soglia è il criterio di osservazione. Il gold diceva `monitor` "
          "senza gate su act-005 e `monitor` CON gate qui: era la stessa "
          "frase scritta in due modi. Nessuna valuta nel testo → currency "
          "null (§6.3)")

case("cond-006",
     "Trovami un idraulico a Torino, se è libero oggi contattalo, altrimenti dimmi chi c'è",
     [g("find", "trovami", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Torino")],
        gate=gate(p("availability", "eq", "oggi"), "contact", "notify"))],
     ["conditional", "gate", "two-branch"], mandatory=True)


# ============================================================= 7. AMBIGUITÀ

case("amb-001", "Prenotamelo per domani.",
     [g("book", "prenotamelo", "unknown", None, "lo",
        C=[p("date", "eq", "domani")])],
     ["ambiguity", "clarification"],
     clar={"required": True, "reason": "target_missing",
           "missing_fields": ["target"]}, mandatory=True)

case("amb-002", "Aiutami.",
     [g("find", "aiutami", "unknown", None, "aiutami")],
     ["ambiguity", "clarification"],
     clar={"required": True, "reason": "goal_unintelligible",
           "missing_fields": ["target", "action"]}, mandatory=True)

case("amb-003", "Trova quello più vicino.",
     [g("find", "trova", "unknown", None, "quello più vicino")],
     ["ambiguity", "clarification"],
     clar={"required": True, "reason": "target_missing",
           "missing_fields": ["target"]}, mandatory=True)

case("amb-004", "Cercami qualcosa da mangiare a Roma sotto i 15 euro",
     [g("find", "cercami", "qualcosa da mangiare", "food", "qualcosa da mangiare",
        C=[p("location", "near", "Roma"),
           p("price", "lt", 15, currency="EUR")])],
     ["ambiguity:no", "vague-target"], mandatory=True,
     note="target vago ma comprensibile: NON serve chiarimento. Il nome è "
          "quello che l'utente ha detto: 'cibo' era una mia astrazione, non "
          "sta nel testo, e §8 vieta di astrarre — come per 'qualcuno' in "
          "nat-003. Il tipo `food` resta, perché i tipi sono un vocabolario "
          "chiuso e dichiarato.")


# ========================================================= 8. VOCABOLARIO APERTO

case("open-001", "Voglio una bicicletta color Blorch",
     [g("find", "voglio", "bicicletta", "product", "una bicicletta color Blorch",
        A=[p("color", "eq", "Blorch")])],
     ["open-vocab"], mandatory=True)

case("open-002", "Cercami un NebulaDrive 12 usato sotto i 200 euro",
     [g("find", "cercami", "NebulaDrive 12", "product", "un NebulaDrive 12 usato",
        A=[p("condition", "eq", "usato")],
        C=[p("price", "lt", 200, currency="EUR")])],
     ["open-vocab"], mandatory=True)

case("open-003", "Mi serve un Kiro Q con almeno 16 GB",
     [g("find", "mi serve", "Kiro Q", "product", "un Kiro Q",
        C=[p("ram", "gte", 16, unit="GB")])],
     ["open-vocab", "strictness:gte", "ambiguous-property"], mandatory=True,
     equivalences=[["ram", "storage"]],
     note="stessa ambiguità di nat-004: il gold prima chiedeva storage lì e "
          "ram qui, sulla stessa forma linguistica. Era incoerente.")

case("open-004", "Trovami un allevatore di Zwergschnauzer pepe e sale vicino a Udine",
     [g("find", "trovami", "allevatore", "service", "un allevatore di Zwergschnauzer",
        A=[p("breed", "eq", "Zwergschnauzer"), p("coat", "eq", "pepe e sale")],
        C=[p("location", "near", "Udine")])],
     ["open-vocab", "animal"], mandatory=True)

case("open-005", "cerco una RTX 6080 nuova, massimo 900 euro",
     [g("find", "cerco", "RTX 6080", "product", "una RTX 6080 nuova",
        A=[p("condition", "eq", "nuova")],
        C=[p("price", "lte", 900, currency="EUR")])],
     ["open-vocab", "strictness:lte"], mandatory=True)

case("open-006", "Prenotami un tavolo da Frobnicato a Lecce per stasera",
     [g("book", "prenotami", "tavolo", "place", "un tavolo da Frobnicato",
        A=[p("brand", "eq", "Frobnicato")],
        C=[p("location", "near", "Lecce"), p("date", "eq", "stasera")])],
     ["open-vocab"], mandatory=True)


# ============================================================ 9. INGLESE / MISTO

case("en-001", "Find me a sandwich with mortadella in Milan under 10 euros",
     [g("find", "find me", "sandwich", "food", "a sandwich with mortadella",
        A=[p("ingredient", "contains", "mortadella")],
        C=[p("location", "near", "Milan"),
           p("price", "lt", 10, currency="EUR")])],
     ["english"], mandatory=True)

case("en-002", "Book a hotel in Rome, at least 4 stars, preferably with a pool",
     [g("book", "book", "hotel", "accommodation", "a hotel",
        C=[p("location", "near", "Rome"), p("stars", "gte", 4)],
        P=[p("amenity", "contains", "pool")])],
     ["english", "preference", "strictness:gte"], mandatory=True)

case("en-003", "I need an electrician tonight, budget 100 euros max",
     [g("find", "i need", "electrician", "professional_service", "an electrician",
        C=[p("date", "eq", "tonight"),
           p("price", "lte", 100, currency="EUR")])],
     ["english", "strictness:lte"], mandatory=True)


# ====================================================== 10. AZIONI NON-"FIND"

case("act-001", "Comprami un tostapane su Amazon",
     [g("buy", "comprami", "tostapane", "product", "un tostapane",
        A=[p("seller", "eq", "Amazon")])],
     ["action:buy"],
     note="'su Amazon' è il canale di acquisto: property aperta 'seller', "
          "descrizione positiva semplice → attributes (R6)")

case("act-002", "Contatta il ristorante La Pergola e chiedi se hanno posto sabato",
     [g("contact", "contatta", "ristorante", "place", "il ristorante La Pergola",
        A=[p("brand", "eq", "La Pergola")],
        C=[p("date", "eq", "sabato")])],
     ["action:contact"])

case("act-003", "Confrontami tre assicurazioni auto per una Panda del 2015",
     [g("compare", "confrontami", "assicurazione auto", "service", "tre assicurazioni auto",
        A=[p("quantity", "eq", 3), p("model", "eq", "Panda"), p("year", "eq", 2015)])],
     ["action:compare"])

case("act-004", "Affittami un trapano per il weekend a Padova",
     [g("rent", "affittami", "trapano", "product", "un trapano",
        C=[p("date", "eq", "weekend"), p("location", "near", "Padova")])],
     ["action:rent"])

case("act-005", "Avvisami se il prezzo del volo per Tokyo scende sotto i 500 euro",
     [g("monitor", "avvisami", "volo", "transport", "il volo per Tokyo",
        C=[p("destination", "eq", "Tokyo"),
           p("price", "lt", 500, currency="EUR")])],
     ["action:monitor"], mandatory=True)

case("act-006", "Disdici la prenotazione dell'hotel di venerdì",
     [g("cancel", "disdici", "prenotazione", "accommodation",
        "la prenotazione dell'hotel",
        C=[p("date", "eq", "venerdì")])],
     ["action:cancel"])


# ========================================================= 11. VALUTE E UNITÀ

case("unit-001", "Cercami un laptop sotto i 900 dollari",
     [g("find", "cercami", "laptop", "product", "un laptop",
        C=[p("price", "lt", 900, currency="USD")])],
     ["currency"], mandatory=True)

case("unit-002", "Un appartamento di almeno 80 mq a Verona",
     [g("find", "(implicito)", "appartamento", "real_estate", "un appartamento",
        C=[p("size", "gte", 80, unit="mq"), p("location", "near", "Verona")])],
     ["unit", "implicit-action"], mandatory=True)

case("unit-003", "Un hard disk da 2 TB",
     [g("find", "(implicito)", "hard disk", "product", "un hard disk da 2 TB",
        A=[p("storage", "eq", 2, unit="TB")])],
     ["unit", "implicit-action"], mandatory=True)

case("unit-004", "Trovami una corsa in taxi sotto i 15 minuti",
     [g("find", "trovami", "taxi", "transport", "una corsa in taxi",
        C=[p("duration", "lt", 15, unit="min")])],
     ["unit"])

case("unit-005", "Cercami un monolocale a 700 al mese a Milano",
     [g("find", "cercami", "monolocale", "real_estate", "un monolocale",
        C=[p("price", "eq", 700, unit="month"),
           p("location", "near", "Milano")])],
     ["currency", "elliptical", "unit"],
     note="'700 al mese': unit='month' per la periodicità; currency null perché "
          "nel testo non c'è né simbolo né parola di valuta né verbo monetario")


# ================================================== 12. TRAPPOLE NUMERO/RUOLO

case("trap-001", "Prenotami un tavolo per 4 alle 21 sotto i 30 euro a testa",
     [g("book", "prenotami", "tavolo", "place", "un tavolo per 4",
        A=[p("guests", "eq", 4)],
        C=[p("time", "eq", "21:00"),
           p("price", "lt", 30, currency="EUR")])],
     ["number-role"], mandatory=True,
     note="tre numeri, tre ruoli diversi")

case("trap-002", "Un volo il 3 per 2 persone sotto i 3 scali",
     [g("find", "(implicito)", "volo", "transport", "un volo",
        A=[p("guests", "eq", 2)],
        C=[p("date", "eq", 3), p("stops", "lt", 3)])],
     ["number-role", "implicit-action"], mandatory=True)

case("trap-003", "Cerco un iPhone 15 sotto i 15 euro di spedizione",
     [g("find", "cerco", "iPhone 15", "product", "un iPhone 15",
        C=[p("shipping_cost", "lt", 15, currency="EUR")])],
     ["number-role", "trap"], mandatory=True,
     note="il 15 del modello NON è un vincolo numerico")

case("trap-004", "Un appartamento con 3 stanze al 3 piano",
     [g("find", "(implicito)", "appartamento", "real_estate", "un appartamento con 3 stanze",
        A=[p("rooms", "eq", 3), p("floor", "eq", 3)])],
     ["number-role", "implicit-action"])

case("trap-005", "Voglio una Fiat 500 del 2020 sotto i 12000",
     [g("find", "voglio", "Fiat 500", "product", "una Fiat 500 del 2020",
        A=[p("year", "eq", 2020)],
        C=[p("price", "lt", 12000)])],  # §6.3: "sotto i 12000", valuta non nominata
     ["number-role", "trap"], mandatory=True)


# ================================================= 13. PREFERENZE SFUMATE

case("pref-001", "Un hotel a Siena, se possibile con parcheggio e colazione inclusa",
     [g("find", "(implicito)", "hotel", "accommodation", "un hotel",
        C=[p("location", "near", "Siena")],
        P=[p("amenity", "contains", "parcheggio"),
           p("amenity", "contains", "colazione")])],
     ["preference", "implicit-action"], mandatory=True)

case("pref-002", "Un idraulico a Pisa, magari che parli inglese, ma non è indispensabile",
     [g("find", "(implicito)", "idraulico", "professional_service", "un idraulico",
        C=[p("location", "near", "Pisa")],
        P=[p("language", "eq", "inglese")])],
     ["preference", "implicit-action"], mandatory=True)

case("pref-003", "Cercami un divano, idealmente in pelle, assolutamente sotto i 800 euro",
     [g("find", "cercami", "divano", "product", "un divano",
        C=[p("price", "lt", 800, currency="EUR")],
        P=[p("material", "eq", "pelle")])],
     ["preference", "obligation"], mandatory=True,
     note="stessa frase, due forze diverse: preferenza + vincolo rafforzato")


# ====================================================== 14. NEGAZIONI COMPOSTE

case("neg-001", "Un volo per Barcellona né con Ryanair né con Wizzair",
     [g("find", "(implicito)", "volo", "transport", "un volo",
        C=[p("destination", "eq", "Barcellona"),
           p("carrier", "not_in", "Ryanair|Wizzair")])],
     ["negation", "list-operator", "implicit-action"], mandatory=True)

case("neg-002", "Una pizza senza glutine e senza lattosio a Modena",
     [g("find", "(implicito)", "pizza", "food", "una pizza senza glutine e senza lattosio",
        C=[p("diet", "excludes", "glutine"),
           p("diet", "excludes", "lattosio"),
           p("location", "near", "Modena")])],
     ["negation", "implicit-action"], mandatory=True)

case("neg-003", "Un'auto usata ma non incidentata, non oltre i 90000 km",
     [g("find", "(implicito)", "auto", "product", "un'auto usata",
        A=[p("condition", "eq", "usata")],
        C=[p("condition", "neq", "incidentata"),
           p("mileage", "lte", 90000, unit="km")])],
     ["negation", "implicit-action"], mandatory=True)

case("neg-004", "Trovami un ristorante a Trieste che non sia di pesce",
     [g("find", "trovami", "ristorante", "place", "un ristorante",
        C=[p("location", "near", "Trieste"),
           p("cuisine", "neq", "pesce")])],
     ["negation"], mandatory=True)


# ================================================= 15. ALTERNATIVE (operatore in)

case("alt-001", "Un volo per Londra o Manchester sotto i 100 euro",
     [g("find", "(implicito)", "volo", "transport", "un volo",
        C=[p("destination", "in", "Londra|Manchester"),
           p("price", "lt", 100, currency="EUR")])],
     ["list-operator", "implicit-action"], mandatory=True)

case("alt-002", "Cerco un gatto Siamese o Sphynx a Roma",
     [g("find", "cerco", "gatto", "animal", "un gatto Siamese o Sphynx",
        A=[p("breed", "in", "Siamese|Sphynx")],
        C=[p("location", "near", "Roma")])],
     ["list-operator"], mandatory=True)
