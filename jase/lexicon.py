"""Lessico canonico condiviso.

Unica implementazione delle tabelle definite in docs/SEMANTICS.md.
Generatore, Guard, evaluator e benchmark importano da qui: se una regola
cambia, cambia in un solo posto.
"""
from __future__ import annotations

SEMANTICS_VERSION = "semantics-1.1"
SCHEMA_VERSION = "1.1"

# ---------------------------------------------------------------- operatori
OPERATORS = (
    "eq", "neq", "lt", "lte", "gt", "gte",
    "contains", "excludes", "before", "after",
    "near", "within", "between", "in", "not_in",
)

#: operatori che richiedono un valore ordinabile (numero o tempo)
ORDERING_OPERATORS = frozenset({"lt", "lte", "gt", "gte", "between", "within"})
#: operatori temporali
TEMPORAL_OPERATORS = frozenset({"before", "after"})
#: operatori insiemistici con valore serializzato "a|b|c"
LIST_OPERATORS = frozenset({"in", "not_in"})
#: operatori che esprimono negazione/esclusione
NEGATIVE_OPERATORS = frozenset({"neq", "excludes", "not_in"})

# ------------------------------------------------------- classi di proprietà
#: §3.1 — sempre `constraints` (salvo marcatore debole o negazione, che vengono prima)
SELECTION_PROPERTIES = frozenset({
    "price", "budget", "cost",
    "location", "distance", "radius", "area",
    "time", "date", "deadline",
    "destination", "origin", "route",
    "departure_time", "arrival_time", "departure_date", "return_date",
    "check_in", "check_out",
    "availability", "delivery", "delivery_time",
})

#: §3.2 — descrittive: `attributes` salvo comparativo/negazione/obbligo
DESCRIPTIVE_PROPERTIES = frozenset({
    "ingredient", "cuisine", "diet", "allergen", "portion", "spiciness",
    "temperature",
    "color", "material", "size", "condition", "model", "brand", "feature",
    "ram", "storage", "display", "gpu", "cpu", "battery", "resolution",
    "breed", "sex", "age", "coat", "weight", "vaccination",
    "stars", "room_type", "guests", "amenity", "rooms", "floor",
    "travel_class", "carrier", "stops", "baggage",
    "specialization", "experience", "language", "certification", "rating",
    "format", "level", "quantity", "duration", "warranty",
    "year", "mileage", "seller", "section", "style", "cuisine",
})

#: proprietà su cui `currency` è ammessa
MONETARY_PROPERTIES = frozenset({
    "price", "budget", "cost", "deposit", "fee", "shipping_cost",
})

#: proprietà numeriche note — il Guard vi richiede valori numerici
NUMERIC_PROPERTIES = frozenset({
    "price", "budget", "cost", "distance", "radius", "stars", "rooms",
    "guests", "quantity", "ram", "storage", "display", "rating", "weight",
    "floor", "stops", "duration", "experience", "warranty", "resolution",
    "year", "mileage", "shipping_cost", "size",
})

#: proprietà temporali
TEMPORAL_PROPERTIES = frozenset({
    "time", "date", "deadline", "departure_time", "arrival_time",
    "departure_date", "return_date", "check_in", "check_out",
})

# --------------------------------------------------------------- marcatori
#: R1 — desiderabilità debole → preferences
SOFT_MARKERS = (
    "possibilmente", "preferibilmente", "preferirei", "preferirebbe",
    "mi piacerebbe", "se possibile", "se si può", "se si puo",
    "magari", "idealmente", "meglio se", "gradirei", "gradirebbe",
    "non è indispensabile", "non e indispensabile", "se c'è", "se c'e",
    "eventualmente", "opzionale", "opzionalmente", "non obbligatorio",
    "preferably", "if possible", "ideally", "nice to have", "optionally",
    "i'd prefer", "would prefer", "would like",
    # code posticipate: il gold le ha spesso in fondo alla frase, e il
    # generatore le produceva già mentre il lessico non le conosceva. La
    # divergenza si vedeva solo contando le riparazioni del Guard sui dati
    # generati: 5% degli esempi insegnava una preferenza che, per le regole
    # scritte, preferenza non era.
    "se riesci", "se ci riesci", "se si riesce", "se capita",
    "ma non è indispensabile", "ma non e indispensabile",
    "ma non è obbligatorio", "ma non e obbligatorio",
    "sarebbe l'ideale", "sarebbe l ideale", "sarebbe il massimo",
    "quando puoi", "se ce la fai",
    "if you can", "but it's not essential", "but it is not essential",
    "would be nice", "no rush",
)

#: R2 — negazione / esclusione → constraints
NEGATION_MARKERS = (
    "senza", "non ", "niente", "nessun", "nessuna", "escluso", "esclusa",
    "tranne", "eccetto", "evita", "evitare", "mai", "a parte", "no ",
    "né", "ne' ",
    "without", "no ", "not ", "except", "avoid", "exclude",
)

#: R3 — obbligo esplicito → constraints
OBLIGATION_MARKERS = (
    "deve", "dev'essere", "devono", "obbligatorio", "obbligatoria",
    "tassativo", "tassativa", "tassativamente", "assolutamente",
    "necessariamente", "per forza", "indispensabile", "richiesto",
    "soltanto", "solo ", "unicamente", "purché", "purche", "a patto che",
    "must", "has to", "have to", "required", "mandatory", "strictly",
)

# ------------------------------------------------------- mappatura operatori
#: frasi → operatore, ordinate dalla più lunga alla più corta in fase di match
COMPARATIVE_PHRASES = {
    # lt — stretto
    "sotto i": "lt", "sotto le": "lt", "sotto ai": "lt", "sotto la": "lt",
    "sotto": "lt", "meno di": "lt", "inferiore a": "lt", "inferiori a": "lt",
    "sotto i quali": "lt", "al di sotto di": "lt", "under": "lt",
    "less than": "lt", "below": "lt",
    # lte — inclusivo
    "massimo": "lte", "al massimo": "lte", "al piu": "lte", "al più": "lte",
    "non più di": "lte", "non piu di": "lte", "non oltre": "lte",
    "fino a": "lte", "entro": "lte", "max": "lte",
    "at most": "lte", "up to": "lte", "no more than": "lte",
    # gt — stretto
    "più di": "gt", "piu di": "gt", "oltre": "gt", "superiore a": "gt",
    "superiori a": "gt", "sopra i": "gt", "sopra": "gt",
    "more than": "gt", "over": "gt", "above": "gt",
    # gte — inclusivo
    "almeno": "gte", "minimo": "gte", "come minimo": "gte",
    "a partire da": "gte", "da ... in su": "gte", "o più": "gte",
    "o piu": "gte", "at least": "gte", "minimum": "gte", "from": "gte",
    # between
    "tra": "between", "fra": "between", "da ... a": "between",
    "between": "between",
}

#: numerali italiani/inglesi → intero (normalizzazione ammessa, §12)
NUMBER_WORDS = {
    "zero": 0, "uno": 1, "una": 1, "un": 1, "due": 2, "tre": 3,
    "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8,
    "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13,
    "quattordici": 14, "quindici": 15, "sedici": 16, "diciassette": 17,
    "diciotto": 18, "diciannove": 19, "venti": 20, "trenta": 30,
    "quaranta": 40, "cinquanta": 50, "sessanta": 60, "settanta": 70,
    "ottanta": 80, "novanta": 90, "cento": 100, "duecento": 200,
    "trecento": 300, "quattrocento": 400, "cinquecento": 500,
    "mille": 1000, "duemila": 2000,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "twenty": 20,
    "fifty": 50, "hundred": 100, "thousand": 1000,
}

#: simboli/parole valuta → ISO-4217
CURRENCY_MAP = {
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euri": "EUR", "eurs": "EUR",
    "$": "USD", "usd": "USD", "dollari": "USD", "dollaro": "USD",
    "dollars": "USD", "dollar": "USD",
    "£": "GBP", "gbp": "GBP", "sterline": "GBP", "pounds": "GBP",
    "chf": "CHF", "franchi": "CHF",
}

#: unità in italiano → forma canonica di §6.2. Il vocabolario delle unità è
#: chiuso e inglese: "sei mesi" produce `value 6, unit "months"`, non
#: `unit "mesi"`. Il generatore scriveva l'italiano e il benchmark chiedeva
#: l'inglese; nessuno dei due se n'era accorto per nove iterazioni.
UNIT_ALIASES = {
    "mesi": "months", "mese": "months", "anni": "years", "anno": "years",
    "giorni": "days", "giorno": "days", "settimane": "weeks",
    "settimana": "weeks", "ore": "hours", "ora": "hours",
    "minuti": "min", "minuto": "min", "notti": "nights", "notte": "nights",
    "persone": "person", "persona": "person", "pollici": "inch",
    "pollice": "inch", "chilometri": "km", "metri": "m",
    "inches": "inch", "month": "months", "year": "years", "night": "nights",
    "persons": "person", "people": "person", "miles": "mi", "hour": "hours",
}

#: come si SCRIVE un'unità canonica (§6.2). `norm_unit` minuscola tutto per
#: poter confrontare, ma il dataset deve insegnare una grafia sola.
UNIT_DISPLAY = {
    "gb": "GB", "tb": "TB", "mb": "MB", "km": "km", "m": "m", "cm": "cm",
    "mi": "mi", "mq": "mq", "sqm": "sqm", "sqft": "sqft", "inch": "inch",
    "months": "months", "years": "years", "days": "days", "weeks": "weeks",
    "hours": "hours", "min": "min", "h": "h", "nights": "nights",
    "person": "person", "kg": "kg", "g": "g", "l": "l",
}

#: unità riconosciute
UNITS = frozenset({
    "km", "m", "mi", "GB", "TB", "MB", "inch", "cm",
    "months", "years", "days", "weeks", "hours", "min", "h", "month",
    "mq", "sqm", "sqft",
    "kg", "g", "l", "night", "nights", "person", "persons", "pieces",
})

#: azioni canoniche preferite
ACTIONS = frozenset({
    "find", "buy", "book", "hire", "contact", "compare", "rent", "sell",
    "schedule", "send", "create", "monitor", "notify", "cancel", "renew",
    "order", "subscribe", "repair", "deliver", "learn", "plan",
})

#: tipi target preferiti (vocabolario comunque aperto)
TARGET_TYPES = frozenset({
    "food", "drink", "product", "electronics", "service",
    "professional_service", "animal", "place", "accommodation", "transport",
    "real_estate", "media", "education", "event", "ticket", "job",
    "information",
})

#: azioni osservative: sono le uniche su cui un `gate` ha senso (§11.3).
#: Un gate su `buy` non condiziona niente, perché l'acquisto è già l'azione.
OBSERVATIONAL_ACTIONS = frozenset({"find", "compare", "monitor", "search",
                                   "check", "watch"})

#: codici ammessi per clarification.reason
CLARIFICATION_REASONS = frozenset({
    "target_missing", "action_ambiguous", "goal_unintelligible",
    "conflicting_constraints", "conditional_not_representable",
})

#: marcatori di richiesta condizionale (§11) — v1 non li rappresenta
CONDITIONAL_MARKERS = (
    "se la trovi", "se lo trovi", "se li trovi", "se le trovi",
    "altrimenti", "in alternativa", "se no", "sennò", "senno",
    "solo se", "a meno che", "nel caso in cui", "qualora",
    "otherwise", "if you find", "only if", "unless", "else",
)

#: preposizioni che possono restare incollate davanti a un valore quando il
#: modello ritaglia il frammento invece della parola: "idealmente **in**
#: pelle" → `material eq "in pelle"`, "with **a** pool" → `amenity contains
#: "a pool"`. Il valore è lo stesso, cambia solo il ritaglio: si tolgono in
#: canonicalizzazione, come già si fa con gli articoli, e solo se dopo
#: resta qualcosa.
LEADING_PREPOSITIONS = (
    "in", "di", "da", "con", "a", "su", "per", "of", "with", "from", "at",
)

#: articoli/determinanti da rimuovere da target.name e dai value
ARTICLES = (
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "un'",
    "del", "dello", "della", "dei", "degli", "delle",
    "the", "a", "an",
)
