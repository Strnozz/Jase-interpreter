#!/usr/bin/env python3
"""Generatore Dataset v2 — semantics-first.

Differenza sostanziale rispetto al v1: il v1 mescolava target e proprietà a
caso e poi ci scriveva sopra una frase, producendo cose come "un grafico
rosso". Qui l'ordine è invertito:

    1. si costruisce uno SCENARIO semantico coerente per dominio
       (dominio → target → fatti ammessi per quel target)
    2. dallo scenario si deriva il GoalContract (deterministico)
    3. dallo stesso scenario si generano MOLTE frasi diverse
       (parafrasi, registri, rumore, ellissi, refusi)

Il contratto non viene mai "letto" dalla frase: entrambi discendono dallo
scenario, quindi sono coerenti per costruzione.

Ogni fatto porta con sé il proprio *ruolo* (attribute / constraint /
preference) e le realizzazioni linguistiche che lo esprimono con quella
forza: è così che si insegnano i contrasti minimi
(con / senza / possibilmente, sotto / massimo / almeno).
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.guard import (  # noqa: E402
    ACCEPT, REPAIRED, check_contract, currency_is_grounded,
    expected_strictness, raw_action_is_grounded, value_is_grounded,
)
from jase.canon import display_unit, norm_unit  # noqa: E402
from jase.guard import bucket_by_cascade  # noqa: E402
from jase.lexicon import SELECTION_PROPERTIES  # noqa: E402
from jase.leakage import decontaminate_exact, load_examples  # noqa: E402

SYSTEM_PROMPT = (ROOT / "TRAIN_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()

# =========================================================== strutture base


@dataclass
class Fact:
    """Un fatto dell'utente: predicato + ruolo + come lo si dice."""
    prop: str
    role: str                    # attributes | constraints | preferences
    operator: str
    value: Any
    phrase: str                  # frammento di testo che lo esprime
    unit: str | None = None
    currency: str | None = None
    reference: str | None = None
    value_to: Any = None
    slot: str = "tail"           # dove va nella frase: head | tail
    family: str = ""             # firma per lo split senza leakage

    def predicate(self) -> dict[str, Any]:
        d = {"property": self.prop, "operator": self.operator, "value": self.value}
        if self.value_to is not None:
            d["value_to"] = self.value_to
        if self.unit is not None:
            # §6.2: il vocabolario delle unità è chiuso e inglese. La FRASE
            # resta in italiano ("sei mesi"), l'unità no.
            d["unit"] = display_unit(self.unit)
        if self.currency is not None:
            d["currency"] = self.currency
        if self.reference is not None:
            d["reference"] = self.reference
        return d


@dataclass
class Scenario:
    action: str
    raw_action: str
    action_phrase: str
    target_name: str
    target_type: str | None
    target_phrase: str           # sintagma nominale usato nella frase
    facts: list[Fact] = field(default_factory=list)
    clarification: dict[str, Any] | None = None
    family: str = ""


# =============================================================== vocabolari

CITIES = ["Milano", "Torino", "Roma", "Verbania", "Bologna", "Firenze",
          "Napoli", "Genova", "Palermo", "Bari", "Padova", "Verona", "Trieste",
          "Lecce", "Cagliari", "Perugia", "Ancona", "Udine", "Como", "Rimini",
          "Siena", "Modena", "Parma", "Brescia", "Catania", "Pescara"]

DATES = ["stasera", "domani", "sabato", "domenica", "lunedì", "venerdì",
         "questa settimana", "il weekend", "oggi", "dopodomani",
         "il prossimo fine settimana", "stamattina", "giovedì"]

HOURS = ["8", "9", "10", "12", "14", "16", "18", "19", "20", "21", "22"]

SOFT = ["possibilmente", "preferibilmente", "se possibile", "magari",
        "idealmente", "preferirei", "meglio se"]

OBLIGE = ["deve essere", "dev'essere", "assolutamente", "tassativamente",
          "per forza"]

#: formule che SEMBRANO deboli ma non lo sono: "ricondizionato va benissimo"
#: descrive il prodotto voluto, non una preferenza cedibile. Il modello le
#: leggeva come marcatori deboli e spostava il fatto in preferences (nat-010).
ACCEPTANCE_TAILS = ["va bene", "va benissimo", "mi sta bene", "per me ok",
                    "ci sta"]


#: ordini di grandezza plausibili per dominio (§7: coerenza del dominio)
PRICE_BY_DOMAIN = {
    "food": [5, 8, 10, 12, 15, 20, 25, 30, 40, 50],
    "electronics": [150, 200, 300, 400, 500, 600, 800, 900, 1000, 1200, 1500, 2000],
    "professional_service": [30, 50, 80, 100, 120, 150, 200, 300, 500],
    "animal": [100, 200, 300, 500, 800, 1000, 1500],
    "accommodation": [40, 60, 80, 100, 120, 150, 200, 250, 300],
    "transport": [20, 30, 50, 80, 90, 100, 150, 200, 300, 500],
    "real_estate": [500, 700, 900, 1200, 1500, 2000, 120000, 200000, 300000],
    "product": [20, 50, 80, 100, 150, 200, 300, 500, 700, 900, 1200],
    "ticket": [15, 20, 30, 40, 50, 60, 80, 100, 150],
    "service": [50, 100, 200, 300, 500, 800, 1200],
    "media": [8, 10, 12, 15, 20, 25, 30],
    "education": [50, 100, 200, 300, 500, 800],
}


def money(rng, domain: str | None = None) -> int:
    pool = PRICE_BY_DOMAIN.get(domain or "", None)
    if pool is None:
        pool = [10, 20, 50, 100, 200, 300, 500, 800, 1000, 1500]
    return rng.choice(pool)


# ------------------------------------------------------------- accordo
#: l'accordo si deduce dall'articolo, senza dover annotare ogni target
_AGR_BY_ARTICLE = {
    "una ": "fs", "un'": "fs", "la ": "fs", "delle ": "fp", "le ": "fp",
    "dei ": "mp", "gli ": "mp", "i ": "mp", "del ": "ms", "un ": "ms",
    "il ": "ms", "lo ": "ms", "uno ": "ms",
}


def agreement_of(phrase: str) -> str:
    low = phrase.lower()
    for art, agr in _AGR_BY_ARTICLE.items():
        if low.startswith(art):
            return agr
    return "ms"


def agree(value: str, agr: str) -> str:
    """Accorda un aggettivo italiano in -o/-e. Lascia stare tutto il resto.

    Il valore del contratto è la forma ACCORDATA, cioè quella che compare nel
    testo: così resta ancorato (§12) e il confronto col gold è esatto."""
    if not isinstance(value, str) or " " in value.strip():
        return value
    if agr == "ms":
        return value
    if value.endswith("o"):
        return value[:-1] + {"fs": "a", "mp": "i", "fp": "e"}[agr]
    if value.endswith("e"):
        return value[:-1] + "i" if agr in ("mp", "fp") else value
    return value


# ============================================================ realizzazioni
# Ogni funzione ritorna un Fact già completo di frammento testuale.

def f_contains(rng, prop, value, role, agr="ms"):
    """Proprietà "contenibile": ingrediente, dotazione, caratteristica."""
    if role == "preferences":
        op = "contains"
        phrase = (soften_post(rng, f"con {value}") if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} con {value}")
    elif role == "constraints" and rng.random() < 0.5:
        op = "excludes"
        phrase = rng.choice([f"senza {value}", f"niente {value}", f"no {value}"])
    elif role == "constraints":
        op = "contains"
        phrase = rng.choice([f"che {rng.choice(OBLIGE)} con {value}",
                             f"con {value} assolutamente",
                             f"deve avere {value}"])
    else:
        op = "contains"
        forms = [f"con {value}", f"con {value}", f"che abbia {value}",
                 f"con dentro {value}"]
        if prop == "ingredient":
            # "un panino ALLA mortadella" è la forma più comune in italiano,
            # e il generatore non ne produceva nemmeno una: su nat-001 e
            # nat-007 il modello perdeva l'ingrediente (reports/042)
            forms += [f"alla {value}", f"con la {value}", f"al {value}",
                      f"alla {value}", f"con il {value}"]
        phrase = rng.choice(forms)
    return Fact(prop, role, op, value, phrase, family=f"{prop}:{op}")


#: proprietà i cui valori sono aggettivi e vanno accordati col target
AGREEING_PROPS = {"color", "condition", "material", "cuisine", "room_type",
                  "spiciness", "coat", "style"}


def f_adjective(rng, prop, value, role, agr="ms"):
    """Proprietà identificante espressa come aggettivo/nome: colore, condizione…"""
    if prop in AGREEING_PROPS:
        value = agree(value, agr)
    if role == "preferences":
        op = "eq"
        phrase = (soften_post(rng, str(value)) if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} {value}")
    elif role == "constraints" and rng.random() < 0.55:
        op = "neq"
        phrase = rng.choice([f"non {value}", f"ma non {value}",
                             f"che non sia {value}", f"tranne {value}"])
    elif role == "constraints":
        op = "eq"
        phrase = rng.choice([f"{rng.choice(OBLIGE)} {value}",
                             f"{value} tassativamente", f"solo {value}"])
    else:
        op = "eq"
        phrase = (f"{value} {rng.choice(ACCEPTANCE_TAILS)}"
                  if rng.random() < 0.12 else value)
    return Fact(prop, role, op, value, phrase, family=f"{prop}:{op}")


_NUM_OPS = {
    "lt": ["sotto i {v}", "meno di {v}", "sotto {v}", "inferiore a {v}"],
    "lte": ["massimo {v}", "non più di {v}", "al massimo {v}", "fino a {v}",
            "max {v}"],
    "gt": ["più di {v}", "oltre {v}", "superiore a {v}", "sopra {v}"],
    "gte": ["almeno {v}", "minimo {v}", "come minimo {v}", "da {v} in su"],
    "eq": ["{v}", "da {v}", "con {v}"],
}

#: per il prezzo la forma "con 30" non esiste in italiano
_PRICE_EQ_FORMS = ["{v}", "da {v}", "per {v}", "sui {v}"]


#: come si SCRIVE un numero piccolo quando lo si dice a voce
_SMALL_WORDS = {1: "un", 2: "due", 3: "tre", 4: "quattro", 5: "cinque",
                6: "sei", 7: "sette", 8: "otto", 9: "nove", 10: "dieci"}


def f_number(rng, prop, value, role, unit=None, noun=""):
    """Proprietà numerica con unità: RAM, età, stelle, stanze, km…"""
    unit_txt = f" {unit}" if unit else ""
    # "compra DUE biglietti", "confrontami TRE amplificatori": in italiano
    # parlato le quantità piccole si scrivono in lettere, e il generatore
    # non ne produceva nessuna. Su nat-020 il modello perdeva del tutto la
    # quantità. §12 dichiara la normalizzazione: il contratto tiene il numero.
    shown = value
    if (not unit and noun and isinstance(value, int) and 2 <= value <= 10
            and rng.random() < 0.45):
        shown = _SMALL_WORDS[value]
    tail = f"{shown}{unit_txt}{(' ' + noun) if noun else ''}"
    if role == "preferences":
        op = "eq"
        phrase = (soften_post(rng, tail) if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} {tail}")
    elif role == "attributes":
        op = "eq"
        phrase = rng.choice(_NUM_OPS["eq"]).format(v=tail)
    else:
        op = rng.choice(["lt", "lte", "gt", "gte"])
        phrase = rng.choice(_NUM_OPS[op]).format(v=tail)
    return Fact(prop, role, op, value, phrase, unit=unit,
                family=f"{prop}:{op}")


def _f_leg_time(rng, prop):
    """Orario di partenza o di arrivo di un viaggio.

    Il gold chiede `arrival_time` ("devo arrivare prima delle 18") e il
    generatore non lo produceva mai: il modello rispondeva `time`, che è la
    proprietà generica. Trovato dal controllo di copertura lessicale."""
    h = rng.choice(["8", "10", "12", "14", "16", "18", "20", "22"])
    verbo = "arrivare" if prop == "arrival_time" else "partire"
    mode = rng.random()
    if mode < 0.55:
        op = "before"
        phrase = rng.choice([f"devo {verbo} prima delle {h}",
                             f"voglio {verbo} entro le {h}",
                             f"con {'arrivo' if prop == 'arrival_time' else 'partenza'} entro le {h}"])
    elif mode < 0.8:
        op = "after"
        phrase = rng.choice([f"devo {verbo} dopo le {h}",
                             f"{'arrivo' if prop == 'arrival_time' else 'partenza'} dopo le {h}"])
    else:
        op = "eq"
        phrase = (f"con {'arrivo' if prop == 'arrival_time' else 'partenza'} "
                  f"alle {h}")
    return Fact(prop, "constraints", op, f"{h}:00", phrase,
                family=f"{prop}:{op}")


def _f_shipping(rng):
    v = rng.choice([5, 8, 10, 15, 20])
    return Fact("shipping_cost", "constraints", "lt", v,
                rng.choice([f"con spedizione sotto i {v} euro",
                            f"spedizione meno di {v}€"]),
                currency="EUR", family="shipping_cost:lt")


def _f_delivery(rng):
    d = rng.choice(["domani", "entro tre giorni", "in giornata",
                    "entro sabato"])
    return Fact("delivery", "constraints", "eq", d,
                rng.choice([f"con consegna {d}", f"consegna {d}"]),
                family="delivery:eq")


def f_price(rng, role, domain=None):
    v = money(rng, domain)
    # §6.3: la valuta si dichiara SOLO se compare nel testo. Con sym vuoto il
    # contratto deve avere currency=None, altrimenti insegniamo ad allucinare.
    sym = rng.choice(["€", " euro", "€", " euro", "", " eur"])
    if role == "preferences":
        op = rng.choice(["lt", "lte"])
        base = rng.choice(_NUM_OPS[op]).format(v=f"{v}{sym}")
        phrase = (soften_post(rng, base) if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} {base}")
    else:
        op = rng.choice(["lt", "lt", "lte", "lte", "gte", "gt", "eq"])
        tail = f"{v}{sym}"
        phrase = rng.choice(_PRICE_EQ_FORMS if op == "eq"
                            else _NUM_OPS[op]).format(v=tail)
        if op == "lt" and rng.random() < 0.45:
            # i modi di dire italiani del rifiuto sono la forma più comune di
            # tetto nel parlato, e il modello li leggeva come `eq`
            phrase = rng.choice([
                f"senza superare i {v}{sym}", f"senza arrivare a {v}{sym}",
                f"senza arrivare a spendere {v}{sym}",
                f"{v}{sym} non li voglio spendere",
                f"{v}{sym} non li spendo", f"{v}{sym} proprio non li spendo",
                f"{v}{sym} però non li spendo",
                f"vorrei restare sotto i {v}{sym}",
                f"vorrei stare sotto i {v}{sym}",
                f"non voglio spendere {v}{sym}"])
        if op == "lte" and rng.random() < 0.25:
            phrase = rng.choice([f"{v}{sym} proprio massimo",
                                 f"entro {v}{sym}", f"budget di {v}{sym}"])
    cur = "EUR" if sym.strip() else None
    return Fact("price", "constraints" if role != "preferences" else "preferences",
                op, v, phrase, currency=cur, family=f"price:{op}")


def _f_diet(rng, role):
    """'Senza glutine' è un vincolo duro (R2); 'possibilmente senza' una
    preferenza (R1). La forma positiva non si usa: nessuno chiede 'con glutine'."""
    v = rng.choice(["glutine", "lattosio", "carne", "cipolla", "frutta secca"])
    if role == "preferences":
        phrase = (soften_post(rng, f"senza {v}") if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} senza {v}")
        return Fact("diet", "preferences", "excludes", v, phrase,
                    family="diet:excludes")
    return Fact("diet", "constraints", "excludes", v,
                rng.choice([f"senza {v}", f"niente {v}", f"no {v}"]),
                family="diet:excludes")


def f_price_range(rng, domain=None):
    a = money(rng, domain)
    b = a + rng.choice([50, 100, 200, 300, 500])
    phrase = rng.choice([f"tra i {a} e i {b} euro", f"da {a} a {b} euro",
                         f"fra {a}€ e {b}€", f"tra {a} e {b}"])
    cur = "EUR" if ("euro" in phrase or "€" in phrase) else None
    return Fact("price", "constraints", "between", a, phrase, currency=cur,
                value_to=b, family="price:between")


_BREEDS = ["Sphynx", "Siamese", "Bengala", "Ragdoll", "Maine Coon",
           "Zwergschnauzer", "Labrador", "Border Collie", "Beagle",
           "Pastore Australiano", "Certosino"]


def f_breeder_breed(rng):
    """"un allevatore di Zwergschnauzer" → `breed eq Zwergschnauzer`.

    Il modello scriveva `specialization contains`, che per un allevatore
    sarebbe pure sensato, ma la razza è una razza (open-004)."""
    b = rng.choice(_BREEDS)
    phrase = rng.choice([f"di {b}", f"di {b}", f"che allevi {b}",
                         f"specializzato in {b}"])
    return Fact("breed", "attributes", "eq", b, phrase, slot="tail",
                family="breed:eq")


INVENTED_VENUES = ["Frobnicato", "Zorbetta", "Da Velun", "Quorix",
                   "Blorch", "Farnex", "Astera", "Movolino", "Kiro"]


def f_venue_brand(rng):
    """Il nome del locale è una MARCA, non un ingrediente (open-006)."""
    name = rng.choice(INVENTED_VENUES)
    phrase = rng.choice([f"da {name}", f"al {name}", f"dal {name}",
                         f"dai ragazzi di {name}", f"dal {name}"])
    return Fact("brand", "attributes", "eq", name, phrase, slot="tail",
                family="brand:eq")


def f_location(rng):
    city = rng.choice(CITIES)
    phrase = rng.choice([f"a {city}", f"a {city}", f"in zona {city}",
                         f"vicino a {city}", f"qui a {city}",
                         f"dalle parti di {city}", f"su {city}"])
    return Fact("location", "constraints", "near", city, phrase, slot="tail",
                family="location:near")


CITIES_ABROAD = ["Londra", "Parigi", "Madrid", "Berlino", "Barcellona",
                 "Amsterdam", "Lisbona", "Vienna", "Praga", "Tokyo",
                 "New York", "Atene", "Dublino"]


def f_destination(rng):
    city = rng.choice(CITIES_ABROAD + CITIES)
    phrase = rng.choice([f"per {city}", f"per {city}", f"verso {city}",
                         f"diretto a {city}", f"con destinazione {city}"])
    return Fact("destination", "constraints", "eq", city, phrase, slot="tail",
                family="destination:eq")


def f_route(rng):
    """"un volo diretto" → `route eq diretto` (§3.1: è un criterio di scelta).

    "senza scali" è la stessa cosa detta al contrario, ed è una
    normalizzazione già dichiarata in §12: non è un'esclusione da
    rendicontare, è il modo normale di dire "diretto". Serve anche come
    condizione di un gate ("se ne trovi uno diretto prenotalo", cond-003)."""
    phrase = rng.choice(["diretto", "diretto", "senza scali", "non stop",
                         "diretta"])
    return Fact("route", "constraints", "eq", "diretto", phrase,
                family="route:eq")


def f_origin(rng):
    city = rng.choice(CITIES)
    phrase = rng.choice([f"da {city}", f"in partenza da {city}",
                         f"che parta da {city}"])
    return Fact("origin", "constraints", "eq", city, phrase, slot="tail",
                family="origin:eq")


def f_distance(rng, city=None):
    km = rng.choice([5, 10, 20, 30, 50, 80, 100, 150])
    if city is not None or rng.random() < 0.4:
        city = city or rng.choice(CITIES)
        phrase = f"entro {km} km da {city}"
        return Fact("distance", "constraints", "within", km, phrase, unit="km",
                    reference=city, family="distance:within")
    phrase = rng.choice([f"entro {km} km", f"non più di {km} km",
                         f"non voglio farmi più di {km} km",
                         f"non me ne voglio fare più di {km}",
                         f"non più di {km} km di strada",
                         f"nel raggio di {km} km", f"massimo {km} km"])
    return Fact("distance", "constraints", "within", km, phrase, unit="km",
                family="distance:within")


def f_date(rng, role="constraints"):
    d = rng.choice(DATES)
    if rng.random() < 0.18:
        op = "before"
        phrase = f"entro {d}"
    else:
        op = "eq"
        phrase = d
    if role == "preferences":
        phrase = (soften_post(rng, d) if rng.random() < 0.45
                  else f"{rng.choice(SOFT)} {d}")
        op = "eq"
    return Fact("date", role, op, d, phrase, family=f"date:{op}")


def f_date_range(rng):
    """'dal 12 al 14' è un intervallo di DATE, non due orari.

    Il modello produceva `time eq 12:00` e `time eq 14:00` (nat-018)."""
    a = rng.choice([3, 5, 8, 10, 12, 15, 18, 20, 22, 25])
    b = a + rng.choice([1, 2, 3, 5, 7])
    phrase = rng.choice([f"dal {a} al {b}", f"dal {a} fino al {b}",
                         f"tra il {a} e il {b}"])
    return Fact("date", "constraints", "between", a, phrase, value_to=b,
                family="date:between")


def f_availability(rng):
    """'aperto la domenica' è una disponibilità, non una data della richiesta.

    Contrasto necessario: 'domenica' da solo È una data (nat-019)."""
    d = rng.choice(["domenica", "sabato", "la sera", "di notte", "a pranzo",
                    "nei festivi", "il lunedì"])
    phrase = rng.choice([f"aperto {d}", f"che sia aperto {d}",
                         f"disponibile {d}"])
    return Fact("availability", "constraints", "contains",
                d.replace("la ", "").replace("di ", "").replace("a ", "")
                 .replace("nei ", "").replace("il ", ""),
                phrase, family="availability:contains")


def f_time(rng):
    h = rng.choice(HOURS)
    mode = rng.random()
    if mode < 0.45:
        op, phrase = "eq", rng.choice([f"alle {h}", f"per le {h}"])
    elif mode < 0.75:
        op, phrase = "before", rng.choice([f"entro le {h}", f"prima delle {h}"])
    else:
        op, phrase = "after", rng.choice([f"dopo le {h}", f"dalle {h} in poi"])
    return Fact("time", "constraints", op, f"{h}:00", phrase,
                family=f"time:{op}")


# ================================================================== domini
# Per ogni dominio: target plausibili e fatti che HANNO SENSO per quel target.

def _pick(rng, pool):
    return rng.choice(pool)


DOMAINS: dict[str, dict[str, Any]] = {
    "food": {
        # i target vaghi si scrivono come li ha detti l'utente: §8 vieta di
        # astrarre, e il modello traduceva "qualcosa da mangiare" in `food`
        "targets": [("panino", "food", ["un panino", "un panino", "qualcosa tipo un panino"]),
                    ("qualcosa da mangiare", "food", ["qualcosa da mangiare"]),
                    ("qualcosa da bere", "drink", ["qualcosa da bere"]),
                    ("pizza", "food", ["una pizza"]),
                    ("piadina", "food", ["una piadina"]),
                    ("poke", "food", ["un poke"]),
                    ("sushi", "food", ["del sushi"]),
                    ("hamburger", "food", ["un hamburger"]),
                    ("ristorante", "place", ["un ristorante"]),
                    ("pasticceria", "place", ["una pasticceria"])],
        "actions": ["find", "book", "order"],
        "facts": [
            ("ingredient", lambda r, role, agr='ms': f_contains(
                r, "ingredient", _pick(r, ["mortadella", "prosciutto crudo", "mozzarella",
                                           "melanzane", "tonno", "salmone", "porchetta",
                                           "stracchino", "rucola", "nduja", "pistacchio"]), role, agr), 0.55),
            ("diet", lambda r, role, agr='ms': _f_diet(r, role), 0.2),
            ("cuisine", lambda r, role, agr='ms': f_adjective(
                r, "cuisine", _pick(r, ["di pesce", "di carne", "giapponese",
                                        "messicana", "pugliese", "vegetariana",
                                        "toscana", "indiana"]), role, agr), 0.2),
            ("spiciness", lambda r, role, agr='ms': f_adjective(
                r, "spiciness", _pick(r, ["piccante", "poco piccante"]), role, agr), 0.1),
        ],
    },
    "electronics": {
        "targets": [("laptop", "product", ["un laptop"]),
                    ("portatile", "product", ["un portatile"]),
                    ("MacBook", "product", ["un MacBook"]),
                    ("smartphone", "product", ["uno smartphone"]),
                    ("telefono", "product", ["un telefono"]),
                    ("monitor", "product", ["un monitor"]),
                    ("scheda video", "product", ["una scheda video"]),
                    ("hard disk", "product", ["un hard disk"]),
                    ("cuffie", "product", ["delle cuffie"]),
                    ("tablet", "product", ["un tablet"])],
        "actions": ["find", "buy", "compare"],
        "facts": [
            ("ram", lambda r, role: f_number(r, "ram", _pick(r, [8, 16, 32, 64]),
                                             role, unit="GB", noun="di RAM"), 0.4),
            ("storage", lambda r, role: f_number(r, "storage", _pick(r, [256, 512, 1, 2]),
                                                 role, unit=_pick(r, ["GB", "TB"])), 0.3),
            ("display", lambda r, role: f_number(r, "display", _pick(r, [13, 14, 15, 17, 24, 27]),
                                                 role, unit="pollici"), 0.25),
            ("condition", lambda r, role, agr='ms': f_adjective(
                r, "condition", _pick(r, ["nuovo", "usato", "ricondizionato"]), role, agr), 0.35),
            ("color", lambda r, role, agr='ms': f_adjective(
                r, "color", _pick(r, ["nero", "bianco", "grigio", "argento", "blu"]), role, agr), 0.2),
            ("brand", lambda r, role, agr='ms': f_adjective(
                r, "brand", _pick(r, ["Lenovo", "Dell", "Asus", "Acer", "HP"]), role, agr), 0.2),
        ],
    },
    "professional_service": {
        "targets": [("elettricista", "professional_service", ["un elettricista"]),
                    ("idraulico", "professional_service", ["un idraulico"]),
                    ("fabbro", "professional_service", ["un fabbro"]),
                    ("imbianchino", "professional_service", ["un imbianchino"]),
                    ("fotografo", "professional_service", ["un fotografo"]),
                    ("traduttore", "professional_service", ["un traduttore"]),
                    ("veterinario", "professional_service", ["un veterinario"]),
                    ("commercialista", "professional_service", ["un commercialista"]),
                    ("dog sitter", "professional_service", ["un dog sitter"])],
        "actions": ["find", "hire", "contact"],
        "facts": [
            ("specialization", lambda r, role, agr='ms': f_contains(
                r, "specialization", _pick(r, ["urgenze", "impianti elettrici", "ritratti",
                                               "matrimoni", "caldaie", "condomini",
                                               "partite IVA", "gatti"]), role, agr), 0.4),
            ("language", lambda r, role, agr='ms': f_adjective(
                r, "language", _pick(r, ["inglese", "tedesco", "spagnolo"]), role, agr), 0.2),
            ("experience", lambda r, role: f_number(
                r, "experience", _pick(r, [3, 5, 10]), role, unit="anni",
                noun="di esperienza"), 0.2),
            ("rating", lambda r, role: f_number(
                r, "rating", _pick(r, [4, 5]), role, noun="stelle"), 0.15),
        ],
    },
    "animal": {
        "targets": [("gatto", "animal", ["un gatto"]),
                    # "un allevatore di Zwergschnauzer": la razza è `breed`,
                    # non una specializzazione del professionista (open-004)
                    ("allevatore", "service", ["un allevatore"]),
                    ("allevamento", "service", ["un allevamento"]),
                    ("addestratore", "service", ["un addestratore"]),
                    ("gattino", "animal", ["un gattino"]),
                    ("cane", "animal", ["un cane"]),
                    ("cucciolo", "animal", ["un cucciolo"]),
                    ("coniglio", "animal", ["un coniglio"]),
                    ("pappagallo", "animal", ["un pappagallo"])],
        "actions": ["find"],
        "facts": [
            ("breed", lambda r, role, agr='ms', species=None: f_adjective(
                r, "breed", _pick(r, BREEDS_BY_SPECIES.get(species, ["meticcio"])),
                role, agr), 0.65),
            ("age", lambda r, role: f_number(
                r, "age", _pick(r, [2, 3, 6, 12, 24]), role, unit="mesi"), 0.35),
            ("coat", lambda r, role, agr='ms': f_adjective(
                r, "coat", _pick(r, ["a pelo lungo", "a pelo corto", "tigrato",
                                     "pepe e sale"]), role, agr), 0.2),
        ],
    },
    "accommodation": {
        # "un albergo" con target.name "hotel" è un sinonimo, cioè una
        # parola che nella richiesta non c'è: §8 lo vieta e il Guard ora lo
        # ferma. I sinonimi nel target erano già stati tolti nel v4 e questo
        # era rientrato di soppiatto.
        "targets": [("hotel", "accommodation", ["un hotel"]),
                    ("albergo", "accommodation", ["un albergo"]),
                    ("bed and breakfast", "accommodation", ["un bed and breakfast"]),
                    ("b&b", "accommodation", ["un b&b"]),
                    ("agriturismo", "accommodation", ["un agriturismo"]),
                    ("casa vacanze", "accommodation", ["una casa vacanze"])],
        "actions": ["find", "book"],
        "facts": [
            ("stars", lambda r, role: f_number(r, "stars", _pick(r, [3, 4, 5]),
                                               role, noun="stelle"), 0.45),
            ("amenity", lambda r, role, agr='ms': f_contains(
                r, "amenity", _pick(r, ["parcheggio", "colazione", "piscina", "spa",
                                        "wifi", "palestra", "vista mare"]), role, agr), 0.5),
            ("guests", lambda r, role: f_number(r, "guests", _pick(r, [2, 3, 4, 5]),
                                                role, noun="persone"), 0.3),
            ("room_type", lambda r, role, agr='ms': f_adjective(
                r, "room_type", _pick(r, ["doppia", "matrimoniale", "singola",
                                          "suite"]), role, agr), 0.2),
        ],
    },
    "transport": {
        "targets": [("volo", "transport", ["un volo"]),
                    ("treno", "transport", ["un treno"]),
                    ("traghetto", "transport", ["un traghetto"]),
                    ("noleggio auto", "transport", ["un noleggio auto"])],
        "actions": ["find", "book"],
        "facts": [
            ("destination", lambda r, role: Fact(
                "destination", "constraints", "eq",
                d := _pick(r, ["Londra", "Parigi", "Madrid", "Berlino", "Catania",
                               "Palermo", "Tokyo", "Barcellona", "Amsterdam"]),
                f"per {d}", family="destination:eq"), 0.8),
            ("carrier", lambda r, role, agr='ms': f_adjective(
                r, "carrier", _pick(r, ["Ryanair", "Wizzair", "ITA", "Lufthansa",
                                        "easyJet"]), role, agr), 0.3),
            ("stops", lambda r, role: f_number(r, "stops", _pick(r, [0, 1, 2]),
                                               role, noun="scali"), 0.2),
            ("baggage", lambda r, role, agr='ms': f_contains(
                r, "baggage", _pick(r, ["bagaglio a mano", "bagaglio da stiva"]), role, agr), 0.25),
            ("arrival_time", lambda r, role, agr='ms': _f_leg_time(
                r, "arrival_time"), 0.3),
            ("departure_time", lambda r, role, agr='ms': _f_leg_time(
                r, "departure_time"), 0.25),
            ("travel_class", lambda r, role, agr='ms': f_adjective(
                r, "travel_class", _pick(r, ["economy", "business", "prima classe"]), role, agr), 0.15),
        ],
    },
    "real_estate": {
        "targets": [("appartamento", "real_estate", ["un appartamento"]),
                    ("monolocale", "real_estate", ["un monolocale"]),
                    ("villetta", "real_estate", ["una villetta"]),
                    ("ufficio", "real_estate", ["un ufficio"])],
        "actions": ["find", "rent", "buy"],
        "facts": [
            ("rooms", lambda r, role: f_number(r, "rooms", _pick(r, [1, 2, 3, 4, 5]),
                                               role, noun="stanze"), 0.5),
            ("size", lambda r, role: f_number(r, "size", _pick(r, [40, 60, 80, 100, 120]),
                                              role, unit="mq"), 0.4),
            ("floor", lambda r, role: Fact(
                "floor", "attributes", "eq", fl := _pick(r, [1, 2, 3, 4]),
                f"al {fl}° piano", family="floor:eq"), 0.15),
            ("amenity", lambda r, role, agr='ms': f_contains(
                r, "amenity", _pick(r, ["balcone", "garage", "ascensore", "giardino",
                                        "terrazzo"]), role, agr), 0.45),
            ("condition", lambda r, role, agr='ms': f_adjective(
                r, "condition", _pick(r, ["arredato", "ristrutturato", "da ristrutturare"]),
                role), 0.3),
        ],
    },
    "ticket": {
        # "due biglietti" NON sta fra le frasi del target: se il numero è
        # nella frase ma non c'è `quantity` nel contratto, il dataset
        # insegna proprio l'omissione che poi misuro (nat-020). La quantità
        # la produce il fatto `quantity`, che scrive anche la frase.
        "targets": [("biglietto", "ticket", ["un biglietto"]),
                    ("abbonamento", "ticket", ["un abbonamento"])],
        "actions": ["find", "buy"],
        "facts": [
            ("event", lambda r, role, agr='ms': f_contains(
                r, "event", _pick(r, ["concerto", "partita", "spettacolo",
                                      "mostra", "teatro"]), role, agr), 0.7),
            ("section", lambda r, role, agr='ms': f_adjective(
                r, "section", _pick(r, ["platea", "tribuna", "curva",
                                        "galleria"]), role, agr), 0.45),
            # il nome NON ripete il target: "un biglietto, 2 biglietti" non
            # lo dice nessuno. La forma "compra due biglietti" — dove il
            # numero sta nella frase del target — la insegna quantity_block
            ("quantity", lambda r, role, agr='ms': f_number(
                r, "quantity", _pick(r, [2, 2, 3, 4]), role,
                noun=_pick(r, ["posti", "ingressi", "persone"])), 0.6),
        ],
    },
    "service": {
        "targets": [("allevatore", "service", ["un allevatore"]),
                    ("assicurazione auto", "service", ["un'assicurazione auto"]),
                    ("trasloco", "service", ["un trasloco"]),
                    ("corriere", "service", ["un corriere"]),
                    ("noleggio attrezzi", "service", ["un noleggio attrezzi"])],
        "actions": ["find", "compare", "contact", "cancel"],
        "facts": [
            ("specialization", lambda r, role, agr='ms': f_contains(
                r, "specialization", _pick(r, ["cani", "gatti", "pianoforti",
                                               "fragile", "internazionale"]),
                role, agr), 0.4),
            ("certification", lambda r, role, agr='ms': f_adjective(
                r, "certification", _pick(r, ["certificato", "ENCI", "assicurato",
                                              "riconosciuto"]), role, agr), 0.35),
            ("rating", lambda r, role, agr='ms': f_number(
                r, "rating", _pick(r, [4, 5]), role, noun="stelle"), 0.2),
        ],
    },
    "media": {
        "targets": [("libro", "media", ["un libro"]),
                    ("fumetto", "media", ["un fumetto"]),
                    ("vinile", "media", ["un vinile"]),
                    ("film", "media", ["un film"]),
                    ("audiolibro", "media", ["un audiolibro"])],
        "actions": ["find", "buy"],
        "facts": [
            ("genre", lambda r, role, agr='ms': f_adjective(
                r, "genre", _pick(r, ["giallo", "fantasy", "storico", "thriller",
                                      "biografia", "fantascienza"]), role, agr), 0.45),
            ("language", lambda r, role, agr='ms': f_adjective(
                r, "language", _pick(r, ["italiano", "inglese", "francese"]),
                role, agr), 0.25),
            ("format", lambda r, role, agr='ms': f_adjective(
                r, "format", _pick(r, ["cartaceo", "ebook", "tascabile",
                                       "rilegato"]), role, agr), 0.3),
            ("author", lambda r, role, agr='ms': f_adjective(
                r, "author", _pick(r, ["Calvino", "Levi", "Ferrante",
                                       "Camilleri"]), role, agr), 0.2),
        ],
    },
    "education": {
        "targets": [("corso", "education", ["un corso"]),
                    ("lezione", "education", ["una lezione"]),
                    ("master", "education", ["un master"])],
        "actions": ["find", "book"],
        "facts": [
            ("subject", lambda r, role, agr='ms': f_contains(
                r, "subject", _pick(r, ["inglese", "chitarra", "yoga", "python",
                                        "fotografia", "cucina"]), role, agr), 0.7),
            ("format", lambda r, role, agr='ms': f_adjective(
                r, "format", _pick(r, ["online", "in presenza", "serale"]),
                role, agr), 0.4),
            ("level", lambda r, role, agr='ms': f_adjective(
                r, "level", _pick(r, ["base", "intermedio", "avanzato"]),
                role, agr), 0.3),
        ],
    },
    "product": {
        "targets": [("bici", "product", ["una bici"]),
                    ("bicicletta", "product", ["una bicicletta"]),
                    ("chitarra", "product", ["una chitarra"]),
                    ("divano", "product", ["un divano"]),
                    ("trapano", "product", ["un trapano"]),
                    ("auto", "product", ["un'auto"]),
                    ("passeggino", "product", ["un passeggino"]),
                    ("tenda da campeggio", "product", ["una tenda da campeggio"])],
        "actions": ["find", "buy", "compare", "rent"],
        "facts": [
            ("color", lambda r, role, agr='ms': f_adjective(
                r, "color", _pick(r, ["rosso", "nero", "bianco", "blu", "verde",
                                      "giallo"]), role, agr), 0.4),
            ("condition", lambda r, role, agr='ms': f_adjective(
                r, "condition", _pick(r, ["nuovo", "usato", "ricondizionato"]), role, agr), 0.4),
            ("material", lambda r, role, agr='ms': f_adjective(
                r, "material", _pick(r, ["in pelle", "in legno", "in alluminio",
                                         "in tessuto"]), role, agr), 0.25),
            ("feature", lambda r, role, agr='ms': f_contains(
                r, "feature", _pick(r, ["elettrico", "pieghevole", "impermeabile",
                                        "garanzia", "telecomando"]), role, agr), 0.3),
            ("brand", lambda r, role, agr='ms': f_adjective(
                r, "brand", _pick(r, ["Bianchi", "Fender", "Ikea", "Bosch",
                                      "Decathlon"]), role, agr), 0.2),
            ("year", lambda r, role, agr='ms': Fact(
                "year", "attributes", "eq", y := _pick(r, [2018, 2019, 2020,
                                                           2021, 2022]),
                f"del {y}", family="year:eq"), 0.2),
            ("mileage", lambda r, role, agr='ms': f_number(
                r, "mileage", _pick(r, [30000, 60000, 90000, 120000]),
                "constraints", unit="km"), 0.15),
            ("seller", lambda r, role, agr='ms': Fact(
                "seller", "attributes", "eq", v := _pick(r, ["Amazon", "Subito",
                                                             "eBay", "Vinted"]),
                f"su {v}", family="seller:eq"), 0.15),
            ("shipping_cost", lambda r, role, agr='ms': _f_shipping(r), 0.12),
            ("delivery", lambda r, role, agr='ms': _f_delivery(r), 0.15),
        ],
    },
}

# target inventati: insegnano a copiare invece che riconoscere
INVENTED_PREFIX = ["Zorba", "Nexa", "Velun", "Kiro", "Aster", "Movo", "Prax",
                   "Nebula", "Orvex", "Talin", "Quorix", "Blorch", "Farnex"]
#: niente "Max"/"Min": in una richiesta telegrafica tutta minuscola
#: "prax max 16gb" è indistinguibile da "al massimo 16 GB". È un'ambiguità
#: che fabbricava il generatore, non una che esiste nel mondo.
INVENTED_SUFFIX = ["X7", "Mini", "Pro", "One", "Nova", "Q", "Lite", "12",
                   "Drive", "S", "Neo", "Plus", "Air", "Go"]
#: code attaccate al prefisso, per nomi che finiscono con un numero
INVENTED_TAILS = ["Drive", "Tech", "Core", "Line", "Pad", "Book", "Wave",
                  "Fold", "Scan"]

ACTION_PHRASES = {
    "find": ["trovami", "cercami", "mi trovi", "sto cercando", "cerco",
             "mi serve", "vorrei", "ho bisogno di", "mi cerchi", "avrei bisogno di",
             "mi servirebbe", "cerco di trovare", "vorrei prendere",
             "mi piacerebbe trovare", "sto cercando di trovare"],
    "buy": ["comprami", "voglio comprare", "vorrei acquistare", "compra",
            "acquistami", "vorrei comprare"],
    "book": ["prenotami", "vorrei prenotare", "prenota", "mi prenoti",
             "vorrei una prenotazione per"],
    "hire": ["vorrei ingaggiare", "ingaggiami", "assumi", "voglio assumere"],
    "contact": ["contatta", "chiama", "scrivi a", "mettiti in contatto con"],
    "compare": ["confrontami", "metti a confronto", "paragona", "fammi confrontare"],
    "cancel": ["disdici", "cancella", "annulla", "disdicimi"],
    "rent": ["affittami", "noleggiami", "vorrei affittare", "prendi a noleggio"],
    "order": ["ordinami", "ordina", "vorrei ordinare"],
}

#: verbi ammessi solo in certi domini, in aggiunta a quelli generici
#: le razze dipendono dalla specie: un pappagallo non è un Labrador
BREEDS_BY_SPECIES = {
    "gatto": ["Sphynx", "Siamese", "Maine Coon", "Bengala", "Ragdoll",
              "Certosino", "Persiano", "Norvegese"],
    "cane": ["Labrador", "Border Collie", "Zwergschnauzer", "Beagle",
             "Pastore Tedesco", "Jack Russell", "Corgi"],
    "coniglio": ["Ariete", "Testa di leone", "Rex"],
    "pappagallo": ["Cocorita", "Calopsite", "Inseparabile", "Ara"],
}

#: "trova e prenota un hotel" è UN goal con azione composta: l'azione finale
#: è quella irreversibile (§10). Il gold lo chiede e il generatore non lo
#: produceva mai.
COMPOSITE_ACTIONS = [
    ("trova e prenota", "book", {"accommodation", "transport", "place",
                                 "education"}),
    ("cerca e prenota", "book", {"accommodation", "transport", "place"}),
    ("trova e compra", "buy", {"product", "electronics", "media", "ticket"}),
    ("cerca e acquista", "buy", {"product", "electronics", "media"}),
    ("trova e contatta", "contact", {"professional_service", "service"}),
]

DOMAIN_ACTION_PHRASES = {
    "animal": {"find": ["vorrei adottare", "vorrei prendere", "sto cercando",
                        "mi trovi", "cercami", "trovami"]},
    "ticket": {
        "targets": [("biglietto", "ticket", ["un biglietto", "due biglietti",
                                             "dei biglietti"]),
                    ("abbonamento", "ticket", ["un abbonamento"])],
        "actions": ["find", "buy"],
        "facts": [
            ("event", lambda r, role, agr='ms': f_contains(
                r, "event", _pick(r, ["concerto", "partita", "spettacolo",
                                      "mostra", "teatro"]), role, agr), 0.7),
            ("section", lambda r, role, agr='ms': f_adjective(
                r, "section", _pick(r, ["platea", "tribuna", "curva",
                                        "galleria"]), role, agr), 0.45),
            ("quantity", lambda r, role, agr='ms': f_number(
                r, "quantity", _pick(r, [1, 2, 3, 4]), role, noun="posti"), 0.4),
        ],
    },
    "service": {
        "targets": [("allevatore", "service", ["un allevatore"]),
                    ("assicurazione auto", "service", ["un'assicurazione auto"]),
                    ("trasloco", "service", ["un trasloco"]),
                    ("corriere", "service", ["un corriere"]),
                    ("noleggio attrezzi", "service", ["un noleggio attrezzi"])],
        "actions": ["find", "compare", "contact", "cancel"],
        "facts": [
            ("specialization", lambda r, role, agr='ms': f_contains(
                r, "specialization", _pick(r, ["cani", "gatti", "pianoforti",
                                               "fragile", "internazionale"]),
                role, agr), 0.4),
            ("certification", lambda r, role, agr='ms': f_adjective(
                r, "certification", _pick(r, ["certificato", "ENCI", "assicurato",
                                              "riconosciuto"]), role, agr), 0.35),
            ("rating", lambda r, role, agr='ms': f_number(
                r, "rating", _pick(r, [4, 5]), role, noun="stelle"), 0.2),
        ],
    },
    "media": {"find": ["cercami", "trovami", "sto cercando", "vorrei leggere",
                       "mi consigli e trovi"]},
    "education": {"find": ["cercami", "trovami", "vorrei imparare",
                           "vorrei iscrivermi a", "sto cercando"]},
}

#: quanti fatti per frase negli scenari liberi. I blocchi correttivi (28%
#: del dataset) sono quasi tutti a fatto singolo e tirano la media in giù:
#: qui si compensa, puntando a una distribuzione finale vicina a quella del
#: gold (0:0.10 1:0.39 2:0.24 3:0.22 4:0.05).
FACT_COUNTS = (0, 1, 2, 3, 4, 5)
FACT_COUNT_WEIGHTS = (0.06, 0.16, 0.29, 0.29, 0.15, 0.05)

FILLER_OPEN = ["", "", "", "ciao, ", "buongiorno, ", "senti, ", "allora, ",
               "ehi, ", "scusa, "]
FILLER_CLOSE = ["", "", "", "", " per favore", " grazie", ", grazie mille",
                " se riesci", ", quando puoi"]


# =========================================================== costruzione
IMPLICIT_ACTION = "(implicito)"


def _sample_facts(rng, dom, domain_name, name, ttype, phrase, k,
                  soft_p, oblige_p) -> list[Fact]:
    """k fatti distinti e COERENTI col target, pescati per peso.

    I generici (luogo, prezzo, data, ora, distanza) stanno nello stesso
    sacchetto dei fatti di dominio: così una frase densa non è mai
    "prezzo + luogo + prezzo", e un volo non riceve mai una `location`."""
    agr = agreement_of(phrase)
    pool: list[tuple[str, Any, float]] = [
        (prop, maker, w) for prop, maker, w in dom["facts"] if prop != "sex"]
    pool.append(("price", lambda r, role: (
        f_price_range(r, domain_name) if r.random() < 0.14
        else f_price(r, role, domain_name)), 1.1))
    if ttype == "transport":
        pool.append(("destination", lambda r, role: f_destination(r), 1.4))
        pool.append(("origin", lambda r, role: f_origin(r), 0.35))
    else:
        pool.append(("location", lambda r, role: f_location(r), 1.2))
        pool.append(("distance", lambda r, role: f_distance(r), 0.35))
    pool.append(("date", lambda r, role: (
        f_date_range(r) if r.random() < 0.18 else f_date(r, role)), 0.6))
    pool.append(("time", lambda r, role: f_time(r), 0.3))
    if ttype in ("place", "professional_service", "service", "accommodation"):
        pool.append(("availability", lambda r, role: f_availability(r), 0.55))
    if ttype in ("place", "food"):
        pool.append(("brand", lambda r, role: f_venue_brand(r), 0.4))
    if name in ("allevatore", "allevamento", "addestratore", "toelettatore"):
        pool.append(("breed", lambda r, role: f_breeder_breed(r), 1.5))
    if ttype == "transport":
        pool.append(("route", lambda r, role: f_route(r), 0.3))

    chosen: list[Fact] = []
    seen: set[str] = set()
    bag = list(pool)
    while bag and len(chosen) < k:
        weights = [w for _p, _m, w in bag]
        idx = rng.choices(range(len(bag)), weights=weights)[0]
        prop, maker, _w = bag.pop(idx)
        if prop in seen:
            continue
        # il ruolo segue la cascata §2: una proprietà di selezione non può
        # finire in attributes, qualunque dado esca. Il generatore ci
        # scriveva `date` e `location` fra gli attributi, e il Guard poi le
        # spostava: il dataset insegnava una cosa e le regole ne dicevano
        # un'altra.
        if rng.random() < soft_p:
            role = "preferences"
        elif prop in SELECTION_PROPERTIES or rng.random() < oblige_p:
            role = "constraints"
        else:
            role = "attributes"
        try:
            f = maker(rng, role, agr, name)
        except TypeError:
            try:
                f = maker(rng, role, agr)
            except TypeError:
                f = maker(rng, role)
        if f is None:
            continue
        seen.add(f.prop)
        chosen.append(f)
    if domain_name == "animal" and "sex" not in seen and rng.random() < 0.35 \
            and len(chosen) < k + 1:
        sx = rng.choice(["female", "male"])
        chosen.append(Fact("sex", "attributes", "eq", sx,
                           {"female": "femmina", "male": "maschio"}[sx],
                           family="sex:eq"))
    return chosen


def build_scenario(rng: random.Random, *, sparse_p=0.24, invented_p=0.10,
                   soft_p=0.18, oblige_p=0.10, implicit_p=0.16,
                   density=1.0, target_facts=None) -> Scenario:
    """`target_facts` fissa quanti fatti deve avere la frase; `density`
    moltiplica le probabilità quando non è fissato.

    La versione a dadi indipendenti produceva una distribuzione bimodale —
    o un fatto o otto — che non somiglia a come parlano le persone. Con
    `target_facts` la forma della distribuzione si decide una volta sola, in
    `generate`, e si può confrontare con quella del gold.

    Misurato su data/v11: il 57% dei goal di training aveva UN SOLO fatto,
    contro il 39% del gold, e il 46% dei casi gold ne ha due o tre contro il
    21% del training. Il modello non sbagliava i fatti: ne scriveva uno e si
    fermava (35 `missing_fact` su 47 fallimenti in reports/038). I blocchi
    correttivi, che sono quasi tutti a fatto singolo, avevano schiacciato la
    distribuzione senza che nessuno la misurasse."""
    domain_name = rng.choice(list(DOMAINS))
    dom = DOMAINS[domain_name]
    invented = rng.random() < invented_p

    if invented:
        if rng.random() < 0.3:
            # il numero fa parte del NOME: "NebulaDrive 12" non è
            # "NebulaDrive" più un 12 da qualche parte (open-002)
            name = (rng.choice(INVENTED_PREFIX) + rng.choice(INVENTED_TAILS)
                    + " " + str(rng.choice([7, 9, 11, 12, 14, 15, 20, 30, 60])))
        else:
            name = f"{rng.choice(INVENTED_PREFIX)} {rng.choice(INVENTED_SUFFIX)}"
        ttype = rng.choice(["product", "product", "electronics", None])
        phrase = rng.choice([f"un {name}", f"il {name}", name])
        dom = DOMAINS[rng.choice(["electronics", "product"])]
    else:
        name, ttype, phrases = rng.choice(dom["targets"])
        phrase = rng.choice(phrases)

    composite = [c for c in COMPOSITE_ACTIONS if ttype in c[2]]
    if composite and rng.random() < 0.05:
        phrase_verb, action, _ = rng.choice(composite)
        sc = Scenario(action, phrase_verb, phrase_verb, name, ttype, phrase, [])
        sc.family = f"composite::{name}:{action}"
        if rng.random() < 0.6:
            sc.facts.append(f_destination(rng) if ttype == "transport"
                            else f_location(rng))
        if rng.random() < 0.4:
            sc.facts.append(f_price(rng, "constraints", domain_name))
        return sc

    action = rng.choice(dom["actions"])
    override = DOMAIN_ACTION_PHRASES.get(domain_name, {}).get(action)
    action_phrase = rng.choice(override if override and rng.random() < 0.5
                               else ACTION_PHRASES[action])
    # una frase su sei non ha verbo ("Un hotel a 4 stelle a Bologna"):
    # raw_action deve essere la costante, non un verbo inventato (§7)
    implicit = rng.random() < implicit_p
    if implicit:
        action = "find"
        action_phrase = ""

    facts: list[Fact] = []
    sparse = rng.random() < sparse_p if target_facts is None else target_facts <= 1

    if target_facts is not None:
        facts = _sample_facts(rng, dom, domain_name, name, ttype, phrase,
                              target_facts, soft_p, oblige_p)
    elif not sparse:
        agr = agreement_of(phrase)
        for _prop, maker, prob in dom["facts"]:
            if _prop == "sex":
                continue
            if rng.random() < min(0.95, prob * density):
                if rng.random() < soft_p:
                    role = "preferences"
                elif rng.random() < oblige_p:
                    role = "constraints"
                else:
                    role = "attributes"
                try:
                    facts.append(maker(rng, role, agr, name))
                except TypeError:
                    try:
                        facts.append(maker(rng, role, agr))
                    except TypeError:
                        facts.append(maker(rng, role))
        # sesso per gli animali: trattato a parte, è quasi sempre attributo
        if domain_name == "animal" and rng.random() < 0.4:
            sx = rng.choice(["female", "male"])
            facts.append(Fact("sex", "attributes", "eq", sx,
                              {"female": "femmina", "male": "maschio"}[sx],
                              family="sex:eq"))
        loc = None
        if rng.random() < min(0.95, 0.55 * density):
            # un volo non ha una "location": ha una destinazione e
            # un'origine. Il generatore ci attaccava `location near Roma`
            # nel 23% dei goal transport e il modello ha imparato quello:
            # su 6 casi gold di volo scriveva `location` al posto di
            # `destination` (reports/038).
            if ttype == "transport":
                if not any(f.prop == "destination" for f in facts):
                    facts.append(f_destination(rng))
                if rng.random() < 0.3:
                    facts.append(f_origin(rng))
            else:
                loc = f_location(rng)
                facts.append(loc)
        if rng.random() < min(0.95, 0.5 * density):
            facts.append(
                f_price_range(rng, domain_name) if rng.random() < 0.12
                else f_price(rng, "preferences" if rng.random() < 0.08 else "constraints",
                             domain_name))
        if rng.random() < min(0.9, 0.2 * density):
            # se c'è già una città, la distanza si misura da quella: due città
            # scorrelate nella stessa frase sono rumore, non varietà
            facts.append(f_distance(rng, loc.value if loc else None))
        if rng.random() < min(0.9, 0.28 * density):
            roll_d = rng.random()
            if roll_d < 0.12:
                facts.append(f_date_range(rng))
            elif roll_d < 0.24 and ttype in ("place", "professional_service",
                                             "service", "accommodation"):
                facts.append(f_availability(rng))
            else:
                facts.append(f_date(rng, "preferences" if rng.random() < 0.1
                                    else "constraints"))
        if rng.random() < min(0.8, 0.15 * density):
            facts.append(f_time(rng))
    elif target_facts is None:
        # scenari poveri: 0 o 1 fatto. Insegnano a NON inventare.
        if rng.random() < 0.45:
            facts.append(f_destination(rng) if ttype == "transport"
                         else f_location(rng))

    # deduplica per proprietà: un fatto, un ruolo (§12)
    seen: set[str] = set()
    unique: list[Fact] = []
    for f in facts:
        if f.prop in seen:
            continue
        seen.add(f.prop)
        unique.append(f)

    sc = Scenario(action, action_phrase or IMPLICIT_ACTION, action_phrase,
                  name, ttype, phrase, unique)
    sc.family = "|".join([domain_name, name] + sorted(f.family for f in unique))
    return sc


# ================================================================ rendering

def render(rng: random.Random, sc: Scenario,
           style: str | None = None) -> tuple[str, str]:
    """Una delle molte superfici linguistiche dello stesso scenario.

    Ritorna (testo, raw_action): alcune rese tolgono il verbo dal testo, e in
    quel caso il contratto DI QUELLA RESA deve dire "(implicito)". Il
    contratto non è più uno per scenario ma uno per superficie: è il prezzo
    di avere raw_action davvero letterale (§7)."""
    style = style or rng.choices(
        ["plain", "plain", "plain", "telegraphic", "split", "question",
         "fronted", "fronted-location", "chatty", "two-sentence"],
        weights=[3, 3, 3, 1, 1.5, 1.5, 1.5, 1.5, 1.5, 2])[0]

    frags = [f.phrase for f in sc.facts]
    rng.shuffle(frags)

    if style == "telegraphic":
        # la forma stringata cancella i marcatori ("possibilmente blu" →
        # "blu"), e un ruolo che si giustificava solo col marcatore resta
        # appeso al nulla: il contratto direbbe `preferences` e nella frase
        # non ci sarebbe niente che lo dica. In quel caso si rende in modo
        # normale, invece di insegnare un ruolo indovinato.
        if any(f.role != bucket_by_cascade(f.predicate()) for f in sc.facts):
            style = "plain"
        else:
            bits = [sc.target_name.lower()] + [
                _telegraph(f) for f in sc.facts]
            return " ".join(b for b in bits if b), IMPLICIT_ACTION

    head_verb = (sc.action_phrase + " ") if sc.action_phrase else ""

    if style == "question":
        head = f"{rng.choice(FILLER_OPEN)}{head_verb}{sc.target_phrase}"
        body = ", ".join(frags)
        q = f"{head}{(' ' + body) if body else ''}?"
        return q[0].upper() + q[1:], sc.raw_action

    if style == "fronted-location":
        loc = next((f for f in sc.facts if f.prop == "location"), None)
        if loc is not None:
            rest = [f.phrase for f in sc.facts if f is not loc]
            rng.shuffle(rest)
            body = ", ".join(rest)
            s = (f"{loc.phrase.capitalize()} {head_verb}{sc.target_phrase}"
                 f"{(', ' + body) if body else ''}")
            s += rng.choice([".", "", " grazie", " per favore"])
            return s[0].upper() + s[1:], sc.raw_action

    if style == "fronted" and frags:
        first = frags.pop(rng.randrange(len(frags)))
        rest = ", ".join(frags)
        s = f"{first}, {head_verb}{sc.target_phrase}"
        if rest:
            s += f", {rest}"
        s += rng.choice(FILLER_CLOSE)
        return (s[0].upper() + s[1:] + rng.choice([".", "", ""]), sc.raw_action)

    if style == "two-sentence" and frags:
        # due frasi, un solo goal: il modello ne creava due
        # ("…qui a Milano? Vorrei restare sotto i dieci." → 2 goal)
        cut = rng.randrange(0, len(frags))
        head_part = ", ".join(frags[:cut])
        tail_part = ", ".join(frags[cut:])
        s1 = (f"{rng.choice(FILLER_OPEN)}{head_verb}{sc.target_phrase}"
              f"{(' ' + head_part) if head_part else ''}")
        s2 = rng.choice(["Poi ", "E ", "Ah, ", "Inoltre ", "", "Comunque "]) + tail_part
        s = s1.strip().rstrip(".") + rng.choice([". ", "? ", ". "]) + \
            (s2[0].upper() + s2[1:] if s2 else "")
        return s.strip()[0].upper() + s.strip()[1:], sc.raw_action

    if style == "split" and len(frags) >= 2:
        cut = rng.randrange(1, len(frags))
        a, b = frags[:cut], frags[cut:]
        s1 = f"{rng.choice(FILLER_OPEN)}{head_verb}{sc.target_phrase}, " + ", ".join(a)
        s2 = rng.choice(["Ah, e ", "Inoltre ", "", "Poi ", "Importante: "]) + ", ".join(b)
        s = s1.strip() + ". " + s2.strip() + "."
        return s[0].upper() + s[1:], sc.raw_action

    if style == "chatty":
        s = (f"{rng.choice(FILLER_OPEN)}{head_verb}{sc.target_phrase}"
             f"{(', ' + ', '.join(frags)) if frags else ''}"
             f"{rng.choice(FILLER_CLOSE)}")
        return s[0].upper() + s[1:], sc.raw_action

    # più virgole: giustapporre frammenti senza pausa crea ambiguità che
    # nemmeno un umano scioglierebbe ("1500 euro proprio massimo 2 GB")
    joiner = rng.choice([", ", ", ", ", ", " "])
    s = f"{rng.choice(FILLER_OPEN)}{head_verb}{sc.target_phrase}"
    if frags:
        s += joiner + joiner.join(frags)
    s += rng.choice(FILLER_CLOSE) + rng.choice([".", "", "", "."])
    return s[0].upper() + s[1:], sc.raw_action


def _telegraph(f: Fact) -> str:
    """Forma stringata: 'max 500', '<10€', 'no rosso', 'milano'."""
    # Solo le abbreviazioni di cui conosciamo l'equivalenza sono ammesse.
    # 'prima delle 9' e 'dopo le 9' non possono diventare entrambe '9:00';
    # anche quantità, unità e ruoli spaziali devono conservare i marcatori.
    if f.prop not in ("price", "location"):
        return f.phrase
    if f.prop == "price":
        sym = "€" if f.currency else ""
        return {"lt": f"<{f.value}{sym}", "lte": f"max {f.value}{sym}",
                "gt": f">{f.value}{sym}", "gte": f"min {f.value}{sym}",
                "eq": f"{f.value}{sym}",
                "between": f"{f.value}-{f.value_to}{sym}"}.get(f.operator, f"{f.value}{sym}")
    return str(f.value).lower()


def _protected_spans(text: str, sc: "Scenario") -> list[tuple[int, int]]:
    """Intervalli del testo che NON vanno corrotti: i valori dei fatti.

    Un refuso dentro un valore romperebbe l'ancoraggio (§12): il contratto
    direbbe 'bianco' mentre il testo dice 'biiianco'."""
    spans: list[tuple[int, int]] = []
    low = text.lower()
    needles = [sc.target_name]
    for f in sc.facts:
        needles.append(str(f.value))
        if f.value_to is not None:
            needles.append(str(f.value_to))
        if f.reference:
            needles.append(str(f.reference))
    for n in needles:
        n = str(n).lower()
        if not n:
            continue
        start = 0
        while True:
            i = low.find(n, start)
            if i < 0:
                break
            spans.append((max(0, i - 1), i + len(n) + 1))
            start = i + 1
    return spans


def typo(rng: random.Random, text: str, sc: "Scenario | None" = None,
         raw_action: str | None = None) -> str:
    """Refusi realistici, ma mai dentro un valore del contratto."""
    if len(text) < 14:
        return text
    spans = _protected_spans(text, sc) if sc is not None else []
    candidates = [i for i in range(4, len(text) - 2)
                  if text[i].isalpha()
                  and not any(a <= i < b for a, b in spans)]
    if not candidates:
        return text
    i = rng.choice(candidates)
    kind = rng.random()
    if kind < 0.45:
        out = text[:i] + text[i + 1:]
    elif kind < 0.85:
        out = text[:i] + text[i] + text[i:]
    else:
        out = text[:i] + text[i + 1] + text[i] + text[i + 2:]
    # rete di sicurezza: i valori del contratto devono restare ancorati al
    # testo. Le maschere di protezione coprono i valori, ma non le superfici
    # che differiscono dal valore ("femmina" per value="female").
    if sc is not None:
        for f in sc.facts:
            # value, value_to E reference: il riferimento spaziale era l'unico
            # campo che la rete non controllava, e infatti ci è passato
            # "100km siena" storpiato in "100k msiena"
            if not all(value_is_grounded(v, out)
                       for v in (f.value, f.value_to, f.reference)):
                return text
            if f.currency and not currency_is_grounded(f.currency, out):
                return text
    if raw_action and not raw_action_is_grounded(raw_action, out):
        return text
    # un refuso non deve cambiare il SENSO: "non più di 20" mutilato in
    # "on più di 20" ribalta il vincolo da tetto a soglia minima
    for f in sc.facts if sc is not None else []:
        if isinstance(f.value, (int, float)) and not isinstance(f.value, bool):
            want = expected_strictness(out, f.value)
            if want is not None and want != f.operator:
                return text
    # ultimo controllo, il più severo: il Guard deve accettare il contratto
    # sul testo storpiato SENZA riparazioni. Un refuso dentro "se riesci"
    # cancella il marcatore debole, e la preferenza nel contratto resta
    # senza niente che la giustifichi ("meglio s questa settimana").
    if sc is not None:
        res = check_contract(json.loads(json.dumps(to_contract(sc, raw_action))),
                             out, ground=True)
        if res.repairs or res.status not in (ACCEPT, REPAIRED):
            return text
    return out


# ============================================================== contratto

def to_contract(sc: Scenario, raw_action: str | None = None) -> dict[str, Any]:
    surface_action = raw_action or sc.raw_action
    goal = {
        "action": "find" if surface_action == IMPLICIT_ACTION else sc.action,
        "raw_action": surface_action,
        "target": {"name": sc.target_name, "type": sc.target_type,
                   "raw": sc.target_phrase},
        "attributes": [f.predicate() for f in sc.facts if f.role == "attributes"],
        "constraints": [f.predicate() for f in sc.facts if f.role == "constraints"],
        "preferences": [f.predicate() for f in sc.facts if f.role == "preferences"],
    }
    return {
        "schema_version": "1.1",
        "goals": [goal],
        "clarification": sc.clarification or {"required": False, "reason": None,
                                              "missing_fields": []},
    }


def merge_contracts(a: dict, b: dict) -> dict:
    return {"schema_version": "1.1",
            "goals": a["goals"] + b["goals"],
            "clarification": {"required": False, "reason": None, "missing_fields": []}}


AMBIGUOUS = [
    ("Prenotamelo per domani.", "book", "prenotamelo", "lo", "target_missing", ["target"],
     [{"property": "date", "operator": "eq", "value": "domani"}]),
    ("Trova quello più vicino.", "find", "trova", "quello più vicino",
     "target_missing", ["target"], []),
    ("Compramelo entro stasera.", "buy", "compramelo", "lo", "target_missing",
     ["target"], [{"property": "date", "operator": "before", "value": "stasera"}]),
    ("Aiutami.", "find", "aiutami", "aiutami", "goal_unintelligible",
     ["target", "action"], []),
    ("Mi serve una cosa.", "find", "mi serve", "una cosa", "target_missing",
     ["target"], []),
    ("Fammi sapere.", "notify", "fammi sapere", "sapere", "goal_unintelligible",
     ["target"], []),
]

#: L'azione condizionata dipende dal DOMINIO: un fabbro non si compra e un
#: bed and breakfast non si acquista, si prenota. Alla prima stesura questo
#: blocco pescava template a caso e produceva "se trovi un fabbro sotto i
#: 1000 compralo" — lo stesso difetto di coerenza già corretto due volte.
_GATE_ACTION_BY_TYPE = {
    "accommodation": ["book"], "transport": ["book"], "place": ["book"],
    "product": ["buy"], "electronics": ["buy"], "media": ["buy"],
    "food": ["buy", "order"], "education": ["book"],
    "professional_service": ["contact"], "service": ["contact"],
    "animal": ["contact"], "real_estate": ["contact", "book"],
    "ticket": ["buy"],
}

#: template per ciascun ramo `then`, con il verbo coerente
_GATE_TEMPLATES = {
    "buy": [
        ("Se {t} costa meno di {v}{sym} compralo, altrimenti avvisami", "notify"),
        ("Cercami {t} e se costa meno di {v}{sym} compralo", None),
        ("Trovami {t}, compralo solo se sta sotto i {v}{sym}", None),
        ("Se trovi {t} sotto i {v}{sym} prendilo, sennò lascia stare", None),
        ("Cercami {t}: se sta sotto i {v}{sym} compralo, se no dimmi dov'è",
         "notify"),
    ],
    "order": [
        ("Cercami {t} e se costa meno di {v}{sym} ordinalo", None),
        ("Trovami {t}, ordinalo solo se sta sotto i {v}{sym}", None),
    ],
    "book": [
        ("Cercami {t} e se sta sotto i {v}{sym} prenotalo", None),
        ("Trovami {t}, se costa meno di {v}{sym} prenotalo, altrimenti dimmelo",
         "notify"),
        ("Cercami {t}, prenotalo solo se sta sotto i {v}{sym}", None),
        ("Se trovi {t} sotto i {v}{sym} prenotalo, sennò avvisami", "notify"),
    ],
    "contact": [
        ("Cercami {t} e se chiede meno di {v}{sym} contattalo", None),
        ("Trovami {t}, contattalo solo se sta sotto i {v}{sym}", None),
        ("Se trovi {t} sotto i {v}{sym} scrivigli, altrimenti dimmi chi c'è",
         "notify"),
    ],
    "notify": [
        ("Controlla {t} e avvisami se scende sotto i {v}{sym}", None),
        ("Tieni d'occhio {t}, se va sotto {v}{sym} fammelo sapere", None),
        ("Monitora {t} e dimmi se cala sotto i {v}{sym}", None),
    ],
}

_GATE_VERBS = {
    "se": "se", "cercami": "cercami", "trovami": "trovami",
    "controlla": "controlla", "tieni": "tieni d'occhio", "monitora": "monitora",
    "comprami": "comprami", "prenotami": "prenotami", "contatta": "contatta",
    "ordinami": "ordinami",
}

#: condizioni NON di prezzo. Il v8 generava solo gate sul prezzo, e sui due
#: casi gold con condizione diversa (`route eq diretto`,
#: `availability eq oggi`) falliva entrambi: il modello non aveva mai visto
#: un gate che non parlasse di soldi.
#: (tipo target, property, value, testo della condizione, target ammessi)
#: Il `value` DEVE comparire letteralmente nel testo della condizione,
#: altrimenti il Guard lo segnala come non ancorato — ed è giusto così.
_GATE_NON_PRICE = [
    ("transport", "route", "diretto", "se è diretto", {"volo", "treno"}),
    ("transport", "baggage", "bagaglio", "se include il bagaglio",
     {"volo"}),
    ("professional_service", "availability", "oggi", "se è libero oggi", None),
    ("professional_service", "availability", "domani", "se è libero domani",
     None),
    ("accommodation", "amenity", "parcheggio", "se ha il parcheggio", None),
    ("accommodation", "amenity", "colazione", "se include la colazione", None),
    ("product", "condition", "nuovo", "se è nuovo", None),
    ("product", "warranty", "garanzia", "se ha la garanzia", None),
    ("food", "diet", "glutine", "se è senza glutine", None),
    ("animal", "vaccination", "vaccinato", "se è vaccinato", None),
]

#: l'utente nomina l'azione CONDIZIONATA in apertura ("Comprami X solo se…"),
#: ma `action` deve restare osservativa: l'acquisto dipende dalla condizione.
#: Sul gold (cond-002) il v8 produceva ancora `action: buy`, e solo il Guard
#: lo fermava.
_GATE_FRONTED = [
    ("Comprami {t} solo se costa meno di {v}{sym}", "buy", "comprami", None),
    ("Comprami {t} se sta sotto i {v}{sym}", "buy", "comprami", None),
    ("Prenotami {t} solo se costa meno di {v}{sym}", "book", "prenotami", None),
    ("Prenotami {t} se sta sotto i {v}{sym}, altrimenti avvisami", "book",
     "prenotami", "notify"),
    ("Ordinami {t} solo se sta sotto i {v}{sym}", "order", "ordinami", None),
    ("Contatta {t} solo se chiede meno di {v}{sym}", "contact", "contatta",
     None),
]

#: riferimenti anaforici: il target non è nominato, solo ripreso da un
#: pronome. Il gate è valido, ma il target va chiesto (§11.4).
_GATE_ANAPHORIC = [
    ("Se la trovi sotto i {v}{sym} comprala, altrimenti dimmi dove l'hai vista",
     "la", "buy", "notify"),
    ("Se lo trovi sotto i {v}{sym} prendilo, sennò lascia stare", "lo",
     "buy", None),
    ("Compramelo se costa meno di {v}{sym}", "lo", "buy", None),
    ("Se costa meno di {v}{sym} prendilo, altrimenti fammi sapere", "lo",
     "buy", "notify"),
    ("Prenotamelo solo se sta sotto i {v}{sym}", "lo", "book", None),
]


def gate_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Richieste condizionali con il campo `gate` (GoalContract v1.1).

    Sostituiscono la vecchia policy `degrade_to_clarification`, che in
    quattro iterazioni di training il modello non ha mai imparato e che,
    sbagliando, produceva `action: buy` — cioè proprio l'azione irreversibile
    che la policy voleva evitare."""
    out = []
    for _ in range(3):
        domain_name = rng.choice(list(DOMAINS))
        dom = DOMAINS[domain_name]
        name, ttype, phrases = rng.choice(dom["targets"])
        phrase = rng.choice(phrases)
        # il ramo `then` deve avere senso per questo target
        # il ramo `notify` produce `monitor` senza gate (§11.3) ed è
        # l'unica forma che insegna l'azione `monitor`: era al 25% e sul
        # gold act-005 il modello scriveva ancora `find`
        then = rng.choice(_GATE_ACTION_BY_TYPE.get(ttype, ["buy"])
                          if rng.random() < 0.6 else ["notify"])
        tmpl, otherwise = rng.choice(_GATE_TEMPLATES[then])
        v = money(rng, domain_name)
        sym = rng.choice(["€", " euro", ""])
        text = tmpl.format(t=phrase, v=v, sym=sym)
        raw_action = _GATE_VERBS.get(text.split(" ", 1)[0].lower(),
                                     text.split(" ", 1)[0].lower())
        action = "monitor" if then == "notify" else "find"
        when = {"property": "price", "operator": "lt", "value": v}
        if sym.strip():
            when["currency"] = "EUR"
        # metà delle volte la richiesta porta ANCHE un vincolo normale prima
        # della condizione. Senza questi esempi il modello impara "il gate è
        # la prima cosa che vedo" e su "Cercami un hotel a Lisbona e se costa
        # meno di 80 prenotalo" produce gate{location near Lisbona}.
        cons = []
        if rng.random() < 0.5:
            extra = rng.choice(["location", "date"])
            if extra == "location":
                # un mezzo di trasporto ha una destinazione, non una
                # "location": la frase e il predicato devono concordare
                transport = ttype == "transport" if "ttype" in dir() else False
                city = rng.choice(CITIES_ABROAD + CITIES if transport else CITIES)
                cons.append({
                    "property": "destination" if transport else "location",
                    "operator": "eq" if transport else "near", "value": city})
                text = text.replace(
                    phrase, f"{phrase} {'per' if transport else 'a'} {city}", 1)
            else:
                d = rng.choice(DATES)
                cons.append({"property": "date", "operator": "eq", "value": d})
                text = text.replace(phrase, f"{phrase} {d}", 1)
        goal = {
            "action": action, "raw_action": raw_action,
            "target": {"name": name, "type": ttype, "raw": phrase},
            "attributes": [], "constraints": cons, "preferences": [],
            "gate": {"when": when, "then": then, "otherwise": otherwise},
        }
        # "Avvisami se scende sotto i 500" NON è un gate: l'unica conseguenza
        # è essere informati, che è già ciò che `monitor` significa. La
        # condizione è il criterio di ricerca, e va in constraints (§11.3).
        # Il generatore ci metteva un gate `then: notify`, cioè una
        # tautologia, e sul gold act-005 il modello produceva `find` + gate
        # invece di `monitor` (reports/038).
        if then == "notify":
            goal.pop("gate")
            goal["constraints"] = cons + [when]
        out.append((text, {"schema_version": "1.1", "goals": [goal],
                           "clarification": {"required": False, "reason": None,
                                             "missing_fields": []}},
                    f"gate::{name}:{then}:{otherwise}:{len(cons)}"))

    # --- condizione NON di prezzo
    ttype_pick, prop, value, cond_txt, only = rng.choice(_GATE_NON_PRICE)
    cands = [(n, t, ph) for dn, d in DOMAINS.items() for (n, t, ph) in d["targets"]
             if t == ttype_pick and (only is None or n in only)]
    if cands:
        name, ttype, phrases = rng.choice(cands)
        phrase = rng.choice(phrases)
        then = rng.choice(_GATE_ACTION_BY_TYPE.get(ttype, ["buy"]))
        verb = rng.choice(["cercami", "trovami"])
        tail = {"buy": "compralo", "order": "ordinalo", "book": "prenotalo",
                "contact": "contattalo", "notify": "avvisami"}[then]
        otherwise = "notify" if rng.random() < 0.3 else None
        text = f"{verb.capitalize()} {phrase} e {cond_txt} {tail}"
        if otherwise:
            text += ", altrimenti dimmelo"
        out.append((text, {
            "schema_version": "1.1",
            "goals": [{"action": "find", "raw_action": verb,
                       "target": {"name": name, "type": ttype, "raw": phrase},
                       "attributes": [], "constraints": [], "preferences": [],
                       "gate": {"when": {
                           "property": prop,
                           "operator": ("excludes" if "senza" in cond_txt
                                        else ("contains" if prop in
                                              ("amenity", "baggage", "warranty",
                                               "vaccination") else "eq")),
                           "value": value},
                           "then": then, "otherwise": otherwise}}],
            "clarification": {"required": False, "reason": None,
                              "missing_fields": []}},
            f"gate::{name}:{prop}:{then}"))

    # --- azione condizionata nominata in apertura
    tmpl, then, verb, otherwise = rng.choice(_GATE_FRONTED)
    cands = [(dn, n, t, ph) for dn, d in DOMAINS.items()
             for (n, t, ph) in d["targets"]
             if then in _GATE_ACTION_BY_TYPE.get(t, ["buy"])]
    if cands:
        dn, name, ttype, phrases = rng.choice(cands)
        phrase = rng.choice(phrases)
        v = money(rng, dn)
        sym = rng.choice(["€", " euro", ""])
        when = {"property": "price", "operator": "lt", "value": v}
        if sym.strip():
            when["currency"] = "EUR"
        out.append((tmpl.format(t=phrase, v=v, sym=sym), {
            "schema_version": "1.1",
            "goals": [{"action": "find", "raw_action": verb,
                       "target": {"name": name, "type": ttype, "raw": phrase},
                       "attributes": [], "constraints": [], "preferences": [],
                       "gate": {"when": when, "then": then,
                                "otherwise": otherwise}}],
            "clarification": {"required": False, "reason": None,
                              "missing_fields": []}},
            f"gate::fronted:{name}:{then}"))

    # --- target ripreso da un pronome: gate valido, target da chiedere
    tmpl, pron, then, otherwise = rng.choice(_GATE_ANAPHORIC)
    v = money(rng)
    sym = rng.choice(["€", " euro", ""])
    when = {"property": "price", "operator": "lt", "value": v}
    if sym.strip():
        when["currency"] = "EUR"
    text = tmpl.format(v=v, sym=sym)
    out.append((text, {
        "schema_version": "1.1",
        "goals": [{"action": "find",
                   "raw_action": text.split(" ", 1)[0].lower(),
                   "target": {"name": "unknown", "type": None, "raw": pron},
                   "attributes": [], "constraints": [], "preferences": [],
                   "gate": {"when": when, "then": then,
                            "otherwise": otherwise}}],
        "clarification": {"required": True, "reason": "target_missing",
                          "missing_fields": ["target"]}},
        f"gate::anaphoric:{pron}:{then}"))

    # contrasto indispensabile: la stessa richiesta SENZA condizione non ha
    # gate, e l'azione irreversibile sta in `action`
    domain_name = rng.choice(list(DOMAINS))
    dom = DOMAINS[domain_name]
    name, ttype, phrases = rng.choice(dom["targets"])
    phrase = rng.choice(phrases)
    direct = rng.choice(_GATE_ACTION_BY_TYPE.get(ttype, ["buy"]))
    verb = {"buy": "comprami", "order": "ordinami", "book": "prenotami",
            "contact": "contatta"}[direct]
    v = money(rng, domain_name)
    text = f"{verb.capitalize()} {phrase} sotto i {v} euro"
    out.append((text, {
        "schema_version": "1.1",
        "goals": [{"action": direct, "raw_action": verb,
                   "target": {"name": name, "type": ttype, "raw": phrase},
                   "attributes": [],
                   "constraints": [{"property": "price", "operator": "lt",
                                    "value": v, "currency": "EUR"}],
                   "preferences": []}],
        "clarification": {"required": False, "reason": None,
                          "missing_fields": []}},
        f"gate::{name}:none"))
    rng.shuffle(out)
    return out


def make_ambiguous(rng) -> tuple[str, dict]:
    text, action, raw, traw, reason, missing, preds = rng.choice(AMBIGUOUS)
    return text, {
        "schema_version": "1.1",
        "goals": [{"action": action, "raw_action": raw,
                   "target": {"name": "unknown", "type": None, "raw": traw},
                   "attributes": [], "constraints": preds, "preferences": []}],
        "clarification": {"required": True, "reason": reason,
                          "missing_fields": missing},
    }


# ==================================================== blocchi correttivi
# Generati dall'evidenza di reports/002-v2-iter1500.json, non a intuito.
# Le tre famiglie di errore dominanti erano:
#   1. lo stesso numero interpretato nel ruolo sbagliato
#      ("Prenotami un tavolo alle 20" → price eq 20 EUR)
#   2. il comparativo su proprietà descrittiva appiattito su `eq`
#      ("da almeno 3 stelle" → stars eq 3, in attributes invece che constraints)
#   3. forma negativa e forma d'obbligo finite nel bucket sbagliato

#: (frase, property, operator, value_fn, unit, currency, ruolo)
_NUMBER_ROLES = [
    ("un tavolo alle {n}",               "time",     "eq",     "{n}:00", None, None, "constraints"),
    ("un tavolo per le {n}",             "time",     "eq",     "{n}:00", None, None, "constraints"),
    ("un tavolo verso le {n}",           "time",     "eq",     "{n}:00", None, None, "constraints"),
    ("un tavolo per {n} persone",        "guests",   "eq",     "{n}",    None, None, "attributes"),
    ("un tavolo da {n} persone",         "guests",   "eq",     "{n}",    None, None, "attributes"),
    ("un tavolo per {n} adulti",         "guests",   "eq",     "{n}",    None, None, "attributes"),
    ("un tavolo a {n} euro a testa",     "price",    "eq",     "{n}",    None, "EUR", "constraints"),
    ("un tavolo entro le {n}",           "time",     "before", "{n}:00", None, None, "constraints"),
    ("un tavolo dopo le {n}",            "time",     "after",  "{n}:00", None, None, "constraints"),
    ("un tavolo per {n}",                "guests",   "eq",     "{n}",    None, None, "attributes"),
    ("un tavolo per {n} persone",        "guests",   "eq",     "{n}",    None, None, "attributes"),
    ("un tavolo sotto i {n} euro",       "price",    "lt",     "{n}",    None, "EUR", "constraints"),
    ("un tavolo entro {n} euro",         "price",    "lte",    "{n}",    None, "EUR", "constraints"),
    ("un tavolo entro {n} km",           "distance", "within", "{n}",    "km", None, "constraints"),
    ("un tavolo a {n} km da qui",        "distance", "within", "{n}",    "km", None, "constraints"),
]

_NUMBER_ROLES_HOTEL = [
    ("un hotel per {n} persone",         "guests", "eq",  "{n}", None, None,  "attributes"),
    ("un hotel per {n} notti",           "duration", "eq", "{n}", "notti", None, "attributes"),
    ("un hotel sotto i {n} euro",        "price",  "lt",  "{n}", None, "EUR", "constraints"),
    ("un hotel massimo {n} euro",        "price",  "lte", "{n}", None, "EUR", "constraints"),
    ("un hotel entro le {n}",            "time",   "before", "{n}:00", None, None, "constraints"),
    ("un hotel a {n} km dal centro",     "distance", "within", "{n}", "km", None, "constraints"),
]

_NUMBER_ROLES_TECH = [
    ("un laptop da {n} pollici",            "display", "eq",  "{n}", "pollici", None, "attributes"),
    ("un laptop sotto i {n} euro",          "price",   "lt",  "{n}", None, "EUR", "constraints"),
    ("un laptop a {n} euro",                "price",   "eq",  "{n}", None, "EUR", "constraints"),
    ("un laptop entro {n} km",              "distance","within", "{n}", "km", None, "constraints"),
    ("un laptop entro le {n}",              "time",    "before", "{n}:00", None, None, "constraints"),
]

_NUMBER_ROLES_ANIMAL = [
    ("un gattino di {n} mesi",              "age",   "eq",  "{n}", "mesi", None, "attributes"),
    ("un gattino sotto i {n} mesi",         "age",   "lt",  "{n}", "mesi", None, "constraints"),
    ("un gattino entro {n} km",             "distance", "within", "{n}", "km", None, "constraints"),
    ("un gattino sotto i {n} euro",         "price", "lt",  "{n}", None, "EUR", "constraints"),
    ("un gattino entro le {n}",             "time",  "before", "{n}:00", None, None, "constraints"),
]

_BLOCK_TARGETS = {
    "tavolo": (_NUMBER_ROLES, "tavolo", "place", ["prenotami", "mi prenoti",
                                                  "vorrei prenotare"]),
    "hotel": (_NUMBER_ROLES_HOTEL, "hotel", "accommodation",
              ["prenotami", "trovami", "cercami"]),
    "laptop": (_NUMBER_ROLES_TECH, "laptop", "product",
               ["trovami", "cercami", "vorrei comprare"]),
    "gattino": (_NUMBER_ROLES_ANIMAL, "gattino", "animal",
                ["trovami", "mi trovi", "vorrei adottare"]),
}

_ACTION_OF = {"prenotami": "book", "mi prenoti": "book",
              "vorrei prenotare": "book", "vorrei comprare": "buy",
              "trovami": "find", "cercami": "find", "mi trovi": "find",
              "vorrei adottare": "find"}

#: numeri che possono essere sia un'ora sia un prezzo: è lì che il modello sbaglia
_AMBIGUOUS_NUMBERS = [8, 9, 10, 12, 14, 16, 18, 19, 20, 21, 22]
_HOUR_WORDS = {8: "otto", 9: "nove", 10: "dieci", 12: "dodici", 20: "venti"}


def number_role_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Stesso numero, ruoli diversi, frasi quasi identiche.

    È il contrasto minimo che insegna che il ruolo di un numero lo decide la
    preposizione che lo precede (o l'unità che lo segue), non il numero.
    Dopo due iterazioni resta l'errore più ostinato: `alle 20` continuava a
    diventare `guests eq 20` o `price eq 20`."""
    key = rng.choice(list(_BLOCK_TARGETS))
    rows_spec, name, ttype, verbs = _BLOCK_TARGETS[key]
    n = rng.choice(_AMBIGUOUS_NUMBERS)
    verb = rng.choice(verbs)
    action = _ACTION_OF[verb]
    out = []
    for template, prop, op, val_t, unit, cur, role in rows_spec:
        phrase = template.format(n=n)
        raw_value = val_t.format(n=n)
        value = int(raw_value) if raw_value.isdigit() else raw_value
        pred = {"property": prop, "operator": op, "value": value}
        if unit:
            pred["unit"] = unit
        if cur:
            pred["currency"] = cur
        target_raw = phrase.split(" ", 1)[0] + " " + name
        text = f"{verb.capitalize()} {phrase}"
        if rng.random() < 0.3:
            text += rng.choice([" per favore", ".", " grazie", ""])
        goal = {"action": action, "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": target_raw},
                "attributes": [pred] if role == "attributes" else [],
                "constraints": [pred] if role == "constraints" else [],
                "preferences": []}
        contract = {"schema_version": "1.1", "goals": [goal],
                    "clarification": {"required": False, "reason": None,
                                      "missing_fields": []}}
        out.append((text, contract, f"numrole::{key}::{prop}:{op}"))

        # variante con l'ora scritta a lettere: "alle dieci" non è un prezzo
        if prop == "time" and n in _HOUR_WORDS and rng.random() < 0.6:
            word_text = text.replace(str(n), _HOUR_WORDS[n])
            out.append((word_text, contract, f"numrole::{key}::{prop}:{op}:word"))
    rng.shuffle(out)
    return out


#: (target, tipo, property, unit, nome nel testo, valori plausibili)
_COMPARATIVE_ITEMS = [
    ("hotel", "accommodation", "stars", None, "stelle", [3, 4, 5]),
    ("laptop", "product", "ram", "GB", "GB di RAM", [8, 16, 32, 64]),
    ("laptop", "product", "storage", "GB", "GB di spazio", [256, 512]),
    ("monitor", "product", "display", "pollici", "pollici", [24, 27, 32]),
    ("appartamento", "real_estate", "rooms", None, "stanze", [2, 3, 4, 5]),
    ("appartamento", "real_estate", "size", "mq", "mq", [50, 70, 90, 120]),
    ("gattino", "animal", "age", "mesi", "mesi", [2, 3, 6, 12]),
    ("volo", "transport", "stops", None, "scali", [0, 1, 2]),
    ("fotografo", "professional_service", "experience", "anni",
     "anni di esperienza", [3, 5, 10]),
]

#: forma piana → attributo `eq`; con comparativo → vincolo con l'operatore giusto
_COMPARATIVE_FORMS = [
    ("da {n} {noun}",            "eq",  "attributes"),
    ("con {n} {noun}",           "eq",  "attributes"),
    ("da almeno {n} {noun}",     "gte", "constraints"),
    ("con almeno {n} {noun}",    "gte", "constraints"),
    ("minimo {n} {noun}",        "gte", "constraints"),
    ("massimo {n} {noun}",       "lte", "constraints"),
    ("non più di {n} {noun}",    "lte", "constraints"),
    ("sotto i {n} {noun}",       "lt",  "constraints"),
    ("meno di {n} {noun}",       "lt",  "constraints"),
    ("più di {n} {noun}",        "gt",  "constraints"),
    ("oltre {n} {noun}",         "gt",  "constraints"),
]


def comparative_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Forma piana contro forma comparativa sulla STESSA proprietà.

    È l'errore `operator:gte->eq` + `bucket:constraints->attributes` che
    dominava il report 002: 'da almeno 3 stelle' finiva come
    `stars eq 3` dentro attributes."""
    name, ttype, prop, unit, noun, values = rng.choice(_COMPARATIVE_ITEMS)
    n = rng.choice(values)
    verb = rng.choice(["trovami", "cercami", "vorrei", "mi serve", "prenotami"]
                      if ttype == "accommodation"
                      else ["trovami", "cercami", "vorrei", "mi serve"])
    action = "book" if verb == "prenotami" else "find"
    art = "una" if name in ("auto", "chitarra") else "un"
    out = []
    for tmpl, op, role in _COMPARATIVE_FORMS:
        frag = tmpl.format(n=n, noun=noun)
        text = f"{verb.capitalize()} {art} {name} {frag}"
        pred = {"property": prop, "operator": op, "value": n}
        if unit:
            pred["unit"] = unit
        goal = {"action": action, "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": f"{art} {name}"},
                "attributes": [pred] if role == "attributes" else [],
                "constraints": [pred] if role == "constraints" else [],
                "preferences": []}
        out.append((text, {"schema_version": "1.1", "goals": [goal],
                           "clarification": {"required": False, "reason": None,
                                             "missing_fields": []}},
                    f"comp::{name}:{prop}:{op}"))
    rng.shuffle(out)
    return out


_TWO_NUMBER_TEMPLATES = [
    ("Prenotami un tavolo per {a} alle {b}", "tavolo", "place", "book",
     "prenotami", [("guests", "eq", "{a}", None, None, "attributes"),
                   ("time", "eq", "{b}:00", None, None, "constraints")]),
    ("Prenotami un tavolo per {a} persone alle {b}", "tavolo", "place", "book",
     "prenotami", [("guests", "eq", "{a}", None, None, "attributes"),
                   ("time", "eq", "{b}:00", None, None, "constraints")]),
    ("Trovami un idraulico entro {a} km sotto i {b} euro", "idraulico",
     "professional_service", "find", "trovami",
     [("distance", "within", "{a}", "km", None, "constraints"),
      ("price", "lt", "{b}", None, "EUR", "constraints")]),
    ("Cercami un hotel per {a} persone sotto i {b} euro", "hotel",
     "accommodation", "find", "cercami",
     [("guests", "eq", "{a}", None, None, "attributes"),
      ("price", "lt", "{b}", None, "EUR", "constraints")]),
    ("Prenotami un hotel da {a} stelle entro le {b}", "hotel",
     "accommodation", "book", "prenotami",
     [("stars", "eq", "{a}", None, None, "attributes"),
      ("time", "before", "{b}:00", None, None, "constraints")]),
    ("Trovami un volo con {a} scali sotto i {b} euro", "volo", "transport",
     "find", "trovami", [("stops", "eq", "{a}", None, None, "attributes"),
                         ("price", "lt", "{b}", None, "EUR", "constraints")]),
    ("Cercami un appartamento di {a} mq sotto i {b} euro", "appartamento",
     "real_estate", "find", "cercami",
     [("size", "eq", "{a}", "mq", None, "attributes"),
      ("price", "lt", "{b}", None, "EUR", "constraints")]),
    ("Trovami un gattino di {a} mesi entro {b} km", "gattino", "animal",
     "find", "trovami", [("age", "eq", "{a}", "mesi", None, "attributes"),
                         ("distance", "within", "{b}", "km", None, "constraints")]),
]


def two_number_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Due numeri nella stessa frase, con ruoli diversi.

    Il caso singolo non basta: il modello deve tenere separati due numeri
    contemporaneamente, ed è lì che collassa
    ("un tavolo per 4 alle 21" → guests 21, oppure price 4)."""
    out = []
    for tmpl, name, ttype, action, verb, specs in rng.sample(
            _TWO_NUMBER_TEMPLATES, k=min(3, len(_TWO_NUMBER_TEMPLATES))):
        a = rng.choice([2, 3, 4, 5, 6])
        b = rng.choice([18, 19, 20, 21, 22]) if "{b}:00" in str(specs) \
            else rng.choice([50, 80, 100, 150, 200, 300])
        if "km" in tmpl and "{a} km" in tmpl:
            a = rng.choice([5, 10, 20, 30, 50])
        if "mq" in tmpl:
            a = rng.choice([40, 60, 80, 100])
        if "mesi" in tmpl:
            a = rng.choice([2, 3, 6, 12])
        if "{b} km" in tmpl:
            b = rng.choice([10, 20, 50, 100])
        goal = {"action": action, "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": f"un {name}"},
                "attributes": [], "constraints": [], "preferences": []}
        for prop, op, val_t, unit, cur, role in specs:
            raw = val_t.format(a=a, b=b)
            pred = {"property": prop, "operator": op,
                    "value": int(raw) if raw.isdigit() else raw}
            if unit:
                pred["unit"] = unit
            if cur:
                pred["currency"] = cur
            goal[role].append(pred)
        text = tmpl.format(a=a, b=b)
        out.append((text, {"schema_version": "1.1", "goals": [goal],
                           "clarification": {"required": False, "reason": None,
                                             "missing_fields": []}},
                    f"twonum::{name}"))
    return out


_SHARED_MULTI = [
    ("professional_service", ["idraulico", "elettricista", "fabbro",
                              "imbianchino"], "mi servono", "find"),
    ("professional_service", ["un idraulico", "un elettricista"],
     "cercami", "find"),
    ("place", ["ristorante", "bar", "parcheggio"], "trovami", "find"),
    ("product", ["trapano", "avvitatore", "sega"], "cercami", "find"),
    ("food", ["panino", "insalata", "dolce"], "ordinami", "order"),
]


def shared_multi_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Due target coordinati con UN modificatore che vale per entrambi.

    "Mi servono un idraulico e un elettricista a Torino" sono due goal, e il
    vincolo di luogo si ripete in tutti e due (§10). Il generatore univa due
    scenari indipendenti con "e poi", che è un'altra cosa: sul gold il
    modello perdeva il secondo goal."""
    ttype, pool, verb, action = rng.choice(_SHARED_MULTI)
    a, b = rng.sample([x.replace("un ", "") for x in pool], 2)
    city = rng.choice(CITIES)
    art = "un"
    shared = rng.choice(["location", "price", "date"])
    if shared == "location":
        frag = f"a {city}"
        pred = {"property": "location", "operator": "near", "value": city}
    elif shared == "price":
        v = money(rng, ttype if ttype in PRICE_BY_DOMAIN else None)
        frag = f"sotto i {v} euro"
        pred = {"property": "price", "operator": "lt", "value": v,
                "currency": "EUR"}
    else:
        d = rng.choice(DATES)
        frag = d
        pred = {"property": "date", "operator": "eq", "value": d}

    def mk(text, preds_each, fam):
        goals = [{"action": action, "raw_action": verb,
                  "target": {"name": n, "type": ttype, "raw": f"{art} {n}"},
                  "attributes": [], "constraints": list(preds_each),
                  "preferences": []} for n in (a, b)]
        return (text, {"schema_version": "1.1", "goals": goals,
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}}, fam)

    out = [
        mk(f"{verb.capitalize()} {art} {a} e {art} {b} {frag}", [pred],
           f"sharedmulti::{a}:{b}:{shared}"),
        mk(f"{verb.capitalize()} {art} {a} e anche {art} {b} {frag}", [pred],
           f"sharedmulti::{a}:{b}:{shared}b"),
        mk(f"{verb.capitalize()} {art} {a} {frag}, e {art} {b}", [pred],
           f"sharedmulti::{a}:{b}:{shared}c"),
    ]
    # contrasto: un solo target, nessun secondo goal
    out.append((f"{verb.capitalize()} {art} {a} {frag}",
                {"schema_version": "1.1",
                 "goals": [{"action": action, "raw_action": verb,
                            "target": {"name": a, "type": ttype,
                                       "raw": f"{art} {a}"},
                            "attributes": [], "constraints": [pred],
                            "preferences": []}],
                 "clarification": {"required": False, "reason": None,
                                   "missing_fields": []}},
                f"sharedmulti::{a}:single"))
    rng.shuffle(out)
    return out


#: le coppie in cui i due aggettivi parlano della STESSA proprietà sono il
#: caso che il generatore non sapeva produrre: `_sample_facts` tiene un
#: fatto per proprietà, quindi "usata ma non incidentata" — `condition eq`
#: più `condition neq` — non compariva in nessun esempio. Sul gold neg-003
#: il modello perdeva sempre il secondo pezzo, in tutte le versioni.
_NEG_ATTACH_ITEMS = [
    ("auto", "product", ("condition", "usata"), ("condition", "incidentata")),
    ("auto", "product", ("condition", "usata"), ("condition", "fumata")),
    ("moto", "product", ("condition", "restaurata"), ("condition", "modificata")),
    ("appartamento", "real_estate", ("condition", "arredato"),
     ("condition", "da ristrutturare")),
    ("laptop", "product", ("condition", "ricondizionato"),
     ("condition", "riparato")),
    ("stampante 3D", "product", ("condition", "usata"), ("brand", "cinese")),
    ("auto", "product", ("condition", "usata"), ("color", "rossa")),
    ("laptop", "product", ("condition", "ricondizionato"), ("brand", "Asus")),
    ("panino", "food", ("ingredient", "mortadella"), ("ingredient", "cipolla")),
    ("hotel", "accommodation", ("amenity", "piscina"), ("amenity", "discoteca")),
    ("volo", "transport", ("baggage", "bagaglio"), ("carrier", "Ryanair")),
]


def negation_attachment_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """La negazione vale per UN elemento, non per tutta la frase.

    "una stampante 3d usata, che non sia cinese": `usata` resta positivo.
    Il modello propagava la negazione all'indietro e produceva
    `condition neq usata` (nat-017)."""
    name, ttype, (p1, v1), (p2, v2) = rng.choice(_NEG_ATTACH_ITEMS)
    verb = rng.choice(["trovami", "cercami", "vorrei", "mi serve"])
    art = "un'" if name == "auto" else ("una" if name in ("stampante 3D", "moto") else "un")
    op1 = "contains" if p1 in ("ingredient", "amenity", "baggage") else "eq"
    op2n = "excludes" if p2 in ("ingredient", "amenity", "baggage") else "neq"
    op2p = "contains" if p2 in ("ingredient", "amenity", "baggage") else "eq"

    def mk(text, preds):
        attrs = [p for p in preds if p["operator"] in ("eq", "contains")]
        cons = [p for p in preds if p["operator"] in ("neq", "excludes")]
        return (text, {"schema_version": "1.1",
                       "goals": [{"action": "find", "raw_action": verb,
                                  "target": {"name": name, "type": ttype,
                                             "raw": f"{art}{sep}{name}"},
                                  "attributes": attrs, "constraints": cons,
                                  "preferences": []}],
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}},
                f"negattach::{name}")

    A1 = {"property": p1, "operator": op1, "value": v1}
    A2p = {"property": p2, "operator": op2p, "value": v2}
    A2n = {"property": p2, "operator": op2n, "value": v2}
    sep = "" if art.endswith("'") else " "
    head = f"{verb.capitalize()} {art}{sep}{name}"
    conn1 = "con" if op1 == "contains" else ""
    conn2 = "con" if op2p == "contains" else ""
    f1 = f"{conn1} {v1}".strip()
    out = [
        # negazione SOLO sul secondo
        mk(f"{head} {f1}, che non sia {v2}", [A1, A2n]),
        mk(f"{head} {f1}, ma non {v2}", [A1, A2n]),
        mk(f"{head} {f1} senza {v2}" if op2n == "excludes"
           else f"{head} {f1}, non {v2}", [A1, A2n]),
    ]
    # Le varianti in cui entrambi i valori restano positivi hanno senso solo
    # se le proprietà sono DIVERSE, o se la proprietà ammette più valori
    # (`contains`: un hotel può avere piscina e palestra). Su una proprietà
    # a valore singolo darebbero "una moto restaurata, modificata", che non
    # è una richiesta: è una contraddizione. Ci ho messo un'iterazione
    # intera a scoprirlo, perché il dato era coerente con sé stesso e solo
    # il benchmark protestava.
    if p1 != p2 or op2p == "contains":
        out += [
            # nessuna negazione
            mk(f"{head} {f1}, {conn2} {v2}".replace("  ", " "), [A1, A2p]),
            # negazione SOLO sul primo
            mk(f"{head} che non sia {v1}, {conn2} {v2}".replace("  ", " "),
               [{"property": p1, "operator":
                 "excludes" if op1 == "contains" else "neq", "value": v1}, A2p]),
        ]
    rng.shuffle(out)
    return out


#: terne forza: descrizione / obbligo / preferenza / negazione sullo stesso fatto
_FORCE_ITEMS = [
    ("panino", "food", "ingredient", "mortadella", "con {v}", "contains"),
    ("pizza", "food", "ingredient", "bufala", "con la {v}", "contains"),
    ("hotel", "accommodation", "amenity", "parcheggio", "con {v}", "contains"),
    ("hotel", "accommodation", "amenity", "colazione", "con {v}", "contains"),
    ("laptop", "product", "feature", "tastiera retroilluminata", "con {v}", "contains"),
    ("volo", "transport", "baggage", "bagaglio a mano", "con {v}", "contains"),
    ("auto", "product", "color", "rossa", "{v}", "eq"),
    ("divano", "product", "material", "in pelle", "{v}", "eq"),
    ("laptop", "product", "condition", "ricondizionato", "{v}", "eq"),
    ("appartamento", "real_estate", "amenity", "balcone", "con {v}", "contains"),
]

_FORCE_VERBS = ["trovami", "cercami", "vorrei", "mi serve", "sto cercando"]


def force_contrast_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Lo stesso fatto con quattro forze diverse: descrizione, obbligo,
    preferenza, esclusione. Cambia una parola, cambia il bucket."""
    name, ttype, prop, value, tmpl, pos_op = rng.choice(_FORCE_ITEMS)
    verb = rng.choice(_FORCE_VERBS)
    art = "una" if name in ("pizza", "auto") else "un"
    head = f"{verb.capitalize()} {art} {name}"
    pos = tmpl.format(v=value)
    neg_op = "excludes" if pos_op == "contains" else "neq"

    variants = [
        (f"{head} {pos}", "attributes", pos_op, value),
        (f"{head} {pos}", "attributes", pos_op, value),
        (f"{head} {pos}, {rng.choice(['grazie', 'per favore', 'quando puoi'])}",
         "attributes", pos_op, value),
        (f"{head} che deve avere {value}" if pos_op == "contains"
         else f"{head} che deve essere {value}", "constraints", pos_op, value),
        (f"{head} {rng.choice(['assolutamente', 'tassativamente'])} {pos}",
         "constraints", pos_op, value),
        (f"{head} {rng.choice(SOFT)} {pos}", "preferences", pos_op, value),
        (f"{head} senza {value}" if pos_op == "contains"
         else f"{head} non {value}", "constraints", neg_op, value),
        (f"{head} {rng.choice(SOFT)} senza {value}" if pos_op == "contains"
         else f"{head} {rng.choice(SOFT)} non {value}", "preferences", neg_op, value),
    ]
    out = []
    for text, role, op, val in variants:
        pred = {"property": prop, "operator": op, "value": val}
        goal = {"action": "find", "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": f"{art} {name}"},
                "attributes": [pred] if role == "attributes" else [],
                "constraints": [pred] if role == "constraints" else [],
                "preferences": [pred] if role == "preferences" else []}
        out.append((text, {"schema_version": "1.1", "goals": [goal],
                           "clarification": {"required": False, "reason": None,
                                             "missing_fields": []}},
                    f"force::{name}:{prop}:{role}:{op}"))
    rng.shuffle(out)
    return out


#: "un gatto Sphynx" (genere + razza) vs "uno Sphynx" (solo entità specifica)
_SPECIFICITY_ITEMS = [
    ("gatto", "animal", "breed", ["Sphynx", "Siamese", "Bengala", "Ragdoll"]),
    ("cane", "animal", "breed", ["Labrador", "Border Collie", "Zwergschnauzer"]),
    ("auto", "product", "model", ["Panda", "Golf", "Clio"]),
    ("laptop", "product", "model", ["ThinkPad", "MacBook", "Zorba X7"]),
    ("chitarra", "product", "model", ["Telecaster", "Stratocaster", "Kiro Q"]),
]


def specificity_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Il livello di specificità del target (§8).

    'un gatto Sphynx' → target gatto + breed Sphynx.
    'uno Sphynx'      → target Sphynx, e breed NON si ripete: sarebbe un
                        duplicato del target. Astrarre 'Sphynx' in 'gatto'
                        richiederebbe conoscenza del mondo, che il
                        vocabolario aperto non assume."""
    genus, ttype, prop, specifics = rng.choice(_SPECIFICITY_ITEMS)
    spec = rng.choice(specifics)
    verb = rng.choice(["trovami", "cercami", "vorrei", "mi trovi"])
    art = "una" if genus in ("auto", "chitarra") else "un"
    art_spec = rng.choice(["un", "uno"]) if genus in ("gatto", "cane") else art

    def mk(name, raw, preds):
        goal = {"action": "find", "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": raw},
                "attributes": preds, "constraints": [], "preferences": []}
        return {"schema_version": "1.1", "goals": [goal],
                "clarification": {"required": False, "reason": None,
                                  "missing_fields": []}}

    out = [
        (f"{verb.capitalize()} {art} {genus} {spec}",
         mk(genus, f"{art} {genus} {spec}",
            [{"property": prop, "operator": "eq", "value": spec}]),
         f"spec::{genus}:genus"),
        (f"{verb.capitalize()} {art_spec} {spec}",
         mk(spec, f"{art_spec} {spec}", []),
         f"spec::{genus}:specific"),
    ]
    return out



# ------------------------------------------------ marcatori debole posticipati
#: il v4 metteva il marcatore SEMPRE prima del fatto ("possibilmente con X").
#: Il gold lo ha spesso dopo ("oggi pomeriggio se possibile") o come coda
#: ("magari inglese, ma non è indispensabile"), e su quelle forme il modello
#: sbagliava 7 casi su 7. Vedi reports/011.
SOFT_POST = ["se possibile", "se riesci", "se ci riesci", "possibilmente",
             "ma non è indispensabile", "ma non è obbligatorio",
             "se capita", "sarebbe l'ideale", "se si riesce"]


def soften_post(rng: random.Random, phrase: str) -> str:
    """Sposta la desiderabilità DOPO il fatto invece che prima."""
    return f"{phrase} {rng.choice(SOFT_POST)}"


# ------------------------------------------------------- operatori di lista
#: "o X o Y" → in,  "né X né Y" → not_in. Il v4 non ne conteneva nemmeno uno
#: e sui 3 casi gold ha sbagliato tutti e 3 (reports/011).
_LIST_ITEMS = [
    ("volo", "transport", "destination", ["Londra", "Manchester", "Parigi",
                                          "Lione", "Madrid", "Siviglia"]),
    ("volo", "transport", "carrier", ["Ryanair", "Wizzair", "ITA", "easyJet"]),
    ("gatto", "animal", "breed", ["Siamese", "Sphynx", "Bengala", "Ragdoll"]),
    ("laptop", "product", "brand", ["Lenovo", "Dell", "Asus", "HP"]),
    ("panino", "food", "ingredient", ["mortadella", "prosciutto", "porchetta"]),
    ("hotel", "accommodation", "amenity", ["piscina", "spa", "palestra"]),
    ("appartamento", "real_estate", "amenity", ["garage", "balcone", "giardino"]),
]


def list_operator_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Alternative accettate (`in`) ed escluse (`not_in`).

    Il valore è una stringa con `|` perché lo schema v1 ammette solo scalari
    (§6.6). La forma linguistica cambia, la serializzazione no."""
    name, ttype, prop, pool = rng.choice(_LIST_ITEMS)
    a, b = rng.sample(pool, 2)
    verb = rng.choice(["trovami", "cercami", "cerco", "vorrei", "mi serve"])
    art = "una" if name in ("auto", "pizza") else "un"
    head = f"{verb.capitalize()} {art} {name}"
    prep = "per" if prop == "destination" else ("con" if prop in
                                                ("amenity", "ingredient") else "")
    sep = (prep + " ") if prep else ""

    setop = "contains" if prop in ("amenity", "ingredient", "diet",
                                   "baggage", "specialization") else "eq"
    negop = "excludes" if setop == "contains" else "neq"

    # "né A né B" è UN insieme chiuso di alternative escluse → not_in.
    # "senza A e senza B" ripete la negazione: sono DUE fatti separati, e il
    # gold li chiede così (neg-002). Il generatore li appiattiva entrambi su
    # not_in, cioè insegnava l'errore che poi misuravo.
    variants = [
        (f"{head} {sep}{a} o {sep}{b}", [(prop, "in", f"{a}|{b}")]),
        (f"{head} {sep}{a} oppure {sep}{b}", [(prop, "in", f"{a}|{b}")]),
        (f"{head} o {sep}{a} o {sep}{b}", [(prop, "in", f"{a}|{b}")]),
        (f"{head} né {sep}{a} né {sep}{b}", [(prop, "not_in", f"{a}|{b}")]),
        (f"{head} tranne {a} e {b}", [(prop, "not_in", f"{a}|{b}")]),
        (f"{head} senza {a} e senza {b}",
         [(prop, negop, a), (prop, negop, b)]),
        (f"{head} niente {a} e niente {b}",
         [(prop, negop, a), (prop, negop, b)]),
        # il contrasto: un solo valore non è una lista
        (f"{head} {sep}{a}", [(prop, setop, a)]),
        (f"{head} senza {a}", [(prop, negop, a)]),
    ]
    out = []
    for text, preds in variants:
        attrs, cons = [], []
        for pr, op, value in preds:
            pred = {"property": pr, "operator": op, "value": value}
            # `in` è una lista di alternative ACCETTATE: descrive, non
            # seleziona (gold alt-002). Lo decide la cascata §2, non una
            # lista di operatori scritta qui a mano.
            role = bucket_by_cascade(pred)
            (attrs if role == "attributes" else cons).append(pred)
        goal = {"action": "find", "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": f"{art} {name}"},
                "attributes": attrs, "constraints": cons, "preferences": []}
        ops = ":".join(op for _pr, op, _v in preds)
        out.append((text, {"schema_version": "1.1", "goals": [goal],
                           "clarification": {"required": False, "reason": None,
                                             "missing_fields": []}},
                    f"list::{name}:{prop}:{ops}:{len(preds)}"))
    rng.shuffle(out)
    return out


#: un solo marcatore che copre PIÙ fatti: "se possibile con parcheggio e
#: colazione inclusa" — due preferenze, non una preferenza e un vincolo.
#: Sul gold (pref-001) il v4 ne metteva una in attributes e una in constraints.
_SHARED_SOFT_ITEMS = [
    ("hotel", "accommodation", "amenity",
     ["parcheggio", "colazione", "piscina", "spa", "wifi"], "con"),
    ("appartamento", "real_estate", "amenity",
     ["balcone", "garage", "ascensore", "giardino"], "con"),
    ("panino", "food", "ingredient",
     ["mortadella", "rucola", "stracchino", "pomodoro"], "con"),
    ("laptop", "product", "feature",
     ["tastiera retroilluminata", "lettore SD", "webcam"], "con"),
]


_AVAIL_ITEMS = [
    ("veterinario", "professional_service", "un veterinario"),
    ("farmacia", "place", "una farmacia"),
    ("meccanico", "professional_service", "un meccanico"),
    ("ristorante", "place", "un ristorante"),
    ("dentista", "professional_service", "un dentista"),
    ("supermercato", "place", "un supermercato"),
]


_DAYWORDS = ["stasera", "domani", "sabato", "domenica", "stamattina",
             "dopodomani", "lunedì", "oggi pomeriggio"]


def date_vs_threshold_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Una parola di tempo accanto a una soglia di prezzo.

    "Mi serve un elettricista stasera entro 100 euro": `entro` governa i
    cento euro, non la sera. Il modello attaccava il comparativo alla data
    e produceva `date before stasera` (nat-013). Le quattro varianti
    cambiano quale dei due elementi porta il marcatore."""
    name, ttype, art = rng.choice(_AVAIL_ITEMS)
    day = rng.choice(_DAYWORDS)
    v = rng.choice([50, 80, 100, 150, 200, 300])
    verb = rng.choice(["mi serve", "trovami", "cercami", "vorrei"])
    head = f"{verb.capitalize()} {art}"
    date = {"property": "date", "operator": "eq", "value": day}

    def mk(text, price_op, sym):
        price = {"property": "price", "operator": price_op, "value": v}
        if sym:
            price["currency"] = "EUR"
        goal = {"action": "find", "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": art},
                "attributes": [], "constraints": [date, price],
                "preferences": []}
        return (text, {"schema_version": "1.1", "goals": [goal],
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}},
                f"datethr::{name}:{price_op}")

    out = [
        mk(f"{head} {day} entro {v} euro", "lte", True),
        mk(f"{head} {day} sotto i {v} euro", "lt", True),
        mk(f"{head} {day}, massimo {v}€", "lte", True),
        mk(f"{head} per {day}, non più di {v} euro", "lte", True),
        mk(f"{head} entro {v} euro {day}", "lte", True),
    ]
    rng.shuffle(out)
    return out


_QTY_ITEMS = [
    ("biglietto", "ticket", "biglietti", "compra", "buy",
     [("event", "contains", "concerto", "per il concerto"),
      ("section", "eq", "platea", "in platea")]),
    ("biglietto", "ticket", "biglietti", "prendimi", "buy",
     [("event", "contains", "partita", "per la partita"),
      ("section", "eq", "curva", "in curva")]),
    ("amplificatore", "product", "amplificatori", "confrontami", "compare", []),
    ("cuffie", "product", "paia di cuffie", "cercami", "find", []),
    ("camera", "accommodation", "camere", "prenotami", "book",
     [("room_type", "eq", "doppia", "doppie")]),
    ("posto", "place", "posti", "prenotami", "book",
     [("date", "eq", "sabato", "per sabato")]),
]
_QTY_WORDS = {2: "due", 3: "tre", 4: "quattro", 5: "cinque", 6: "sei"}


def quantity_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """La quantità sta nel sintagma del target: "compra DUE biglietti".

    È il caso in cui il numero non ha né unità né preposizione: è
    attaccato al nome della cosa. Il generatore produceva solo la forma
    staccata ("un biglietto, 2 posti") e su nat-020 il modello perdeva del
    tutto la quantità, in tutte le versioni dalla nona in poi.

    Il contrasto è il singolare: "compra un biglietto" NON porta nessun
    `quantity` — un oggetto solo è il caso di default, non un fatto."""
    name, ttype, plural, verb, action, extras = rng.choice(_QTY_ITEMS)
    n = rng.choice([2, 2, 3, 4, 5])
    word = _QTY_WORDS[n]
    extra_preds, extra_txt = [], []
    for prop, op, val, frag in extras:
        if rng.random() < 0.6:
            extra_preds.append({"property": prop, "operator": op, "value": val})
            extra_txt.append(frag)
    tail = (" " + " ".join(extra_txt)) if extra_txt else ""

    def mk(text, raw, preds, fam):
        attrs = [p for p in preds if p["property"] not in SELECTION_PROPERTIES]
        cons = [p for p in preds if p["property"] in SELECTION_PROPERTIES]
        goal = {"action": action, "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": raw},
                "attributes": attrs, "constraints": cons, "preferences": []}
        return (text, {"schema_version": "1.1", "goals": [goal],
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}}, fam)

    qty = {"property": "quantity", "operator": "eq", "value": n}
    art = "una" if name in ("camera", "cuffie") else "un"
    out = [
        mk(f"{verb.capitalize()} {word} {plural}{tail}", f"{word} {plural}",
           [qty] + extra_preds, f"qty::{name}:parola"),
        mk(f"{verb.capitalize()} {n} {plural}{tail}", f"{n} {plural}",
           [qty] + extra_preds, f"qty::{name}:cifra"),
        # il singolare non porta quantità: è il caso di default
        mk(f"{verb.capitalize()} {art} {name}{tail}", f"{art} {name}",
           list(extra_preds), f"qty::{name}:singolare"),
    ]
    rng.shuffle(out)
    return out


def availability_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """'aperto la domenica' contro 'domenica' da solo.

    Sono due fatti diversi con la stessa parola: nel primo caso la domenica
    è una caratteristica del posto (`availability`), nel secondo è quando
    l'utente vuole andarci (`date`). Sul gold nat-019 il modello scriveva
    `date eq domenica` per "un veterinario aperto la domenica", cioè
    prendeva l'orario del negozio per l'orario dell'appuntamento."""
    name, ttype, art = rng.choice(_AVAIL_ITEMS)
    day = rng.choice(["domenica", "sabato", "il lunedì", "la sera",
                      "a pranzo", "nei festivi"])
    bare = day.replace("il ", "").replace("la ", "").replace("a ", "") \
              .replace("nei ", "")
    verb = rng.choice(["trovami", "cercami", "mi serve", "mi cerchi"])
    city = rng.choice(CITIES)
    head = f"{verb.capitalize()} {art}"

    def mk(text, preds, fam):
        goal = {"action": "find", "raw_action": verb,
                "target": {"name": name, "type": ttype, "raw": art},
                "attributes": [], "constraints": list(preds), "preferences": []}
        return (text, {"schema_version": "1.1", "goals": [goal],
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}}, fam)

    loc = {"property": "location", "operator": "near", "value": city}
    avail = {"property": "availability", "operator": "contains", "value": bare}
    date = {"property": "date", "operator": "eq", "value": bare}
    out = [
        mk(f"{head} aperto {day} a {city}", [avail, loc],
           f"avail::{name}:availability"),
        mk(f"{head} che sia aperto {day}, sono a {city}", [avail, loc],
           f"avail::{name}:availability"),
        mk(f"{head} {day} a {city}", [date, loc], f"avail::{name}:date"),
        mk(f"{head} a {city} per {day}", [date, loc], f"avail::{name}:date"),
    ]
    rng.shuffle(out)
    return out


def shared_soft_block(rng: random.Random) -> list[tuple[str, dict, str]]:
    """Un marcatore debole che governa due fatti coordinati."""
    name, ttype, prop, pool, prep = rng.choice(_SHARED_SOFT_ITEMS)
    a, b = rng.sample(pool, 2)
    verb = rng.choice(["trovami", "cercami", "vorrei", "mi serve", "prenotami"])
    action = "book" if verb == "prenotami" else "find"
    city = rng.choice(CITIES)
    art = "un"
    head = f"{verb.capitalize()} {art} {name} a {city}"
    loc = {"property": "location", "operator": "near", "value": city}
    pa = {"property": prop, "operator": "contains", "value": a}
    pb = {"property": prop, "operator": "contains", "value": b}

    def mk(text, role_a, role_b, fam):
        g = {"action": action, "raw_action": verb,
             "target": {"name": name, "type": ttype, "raw": f"{art} {name}"},
             "attributes": [], "constraints": [loc], "preferences": []}
        for pred, role in ((pa, role_a), (pb, role_b)):
            g[role].append(pred)
        return (text, {"schema_version": "1.1", "goals": [g],
                       "clarification": {"required": False, "reason": None,
                                         "missing_fields": []}}, fam)

    marker = rng.choice(["se possibile", "possibilmente", "magari"])
    return [
        # il marcatore precede la coordinazione: copre ENTRAMBI i fatti
        mk(f"{head}, {marker} {prep} {a} e {b}",
           "preferences", "preferences", f"sharedsoft::{name}:both"),
        mk(f"{head}, {prep} {a} e {b} {rng.choice(SOFT_POST)}",
           "preferences", "preferences", f"sharedsoft::{name}:both-post"),
        # senza marcatore: entrambi attributi
        mk(f"{head}, {prep} {a} e {b}",
           "attributes", "attributes", f"sharedsoft::{name}:none"),
        # marcatore solo sul secondo: scope stretto
        mk(f"{head}, {prep} {a}, e {marker} {prep} {b}",
           "attributes", "preferences", f"sharedsoft::{name}:second"),
    ]


# ------------------------------------------------------------------ inglese
#: il dataset era interamente in italiano e sul gold inglese faceva 1 su 3.
#: Blocco a template: non replica tutta la macchina dei domini, ma copre le
#: forme che contano — comparativi, negazione, preferenza, luogo, prezzo.
#: ogni target inglese porta SOLO le proprietà che hanno senso per lui.
#: Alla prima stesura questo blocco era combinatorio e produceva "a plumber
#: with at least 4 rooms" e "a restaurant red": lo stesso difetto del dataset
#: v1 che questa riscrittura doveva eliminare. Coerenza di dominio anche qui.
_EN_DOMAINS = [
    {"name": "sandwich", "type": "food", "art": "a sandwich",
     "actions": ["find", "buy"],
     "attrs": [("ingredient", ["mortadella", "cheese", "bacon", "ham"], "with {v}")],
     "excl": [("diet", ["gluten", "lactose"], "without {v}")],
     "nums": [], "prices": [5, 8, 10, 15, 20]},
    {"name": "pizza", "type": "food", "art": "a pizza",
     "actions": ["find", "order"],
     "attrs": [("ingredient", ["mushrooms", "pepperoni", "buffalo mozzarella"],
                "with {v}")],
     "excl": [("diet", ["gluten", "lactose"], "without {v}")],
     "nums": [], "prices": [8, 10, 12, 15, 20]},
    {"name": "hotel", "type": "accommodation", "art": "a hotel",
     "actions": ["find", "book"],
     "attrs": [("amenity", ["a pool", "parking", "breakfast", "a gym"], "with {v}"),
               ("room_type", ["a double room", "a suite"], "with {v}")],
     "excl": [], "nums": [("stars", "stars", [3, 4, 5]),
                          ("guests", "people", [2, 3, 4])],
     "prices": [50, 80, 100, 150, 200, 300]},
    {"name": "flight", "type": "transport", "art": "a flight",
     "actions": ["find", "book"],
     "attrs": [("baggage", ["cabin baggage", "checked baggage"], "with {v}"),
               ("travel_class", ["economy", "business"], "in {v}")],
     "excl": [("carrier", ["Ryanair", "Wizzair"], "not on {v}")],
     "nums": [("stops", "stops", [0, 1, 2])],
     "prices": [40, 60, 90, 120, 200, 400]},
    {"name": "laptop", "type": "product", "art": "a laptop",
     "actions": ["find", "buy", "compare"],
     "attrs": [("condition", ["used", "new", "refurbished"], "{v}"),
               ("color", ["black", "silver"], "{v}")],
     "excl": [], "nums": [("ram", "GB of RAM", [8, 16, 32, 64]),
                          ("display", "inches", [13, 14, 15])],
     "prices": [300, 500, 800, 1000, 1500]},
    {"name": "electrician", "type": "professional_service", "art": "an electrician",
     "actions": ["find"],
     "attrs": [("specialization", ["emergencies", "rewiring", "fuse boxes"],
                "for {v}"),
               ("language", ["English", "Italian"], "who speaks {v}")],
     "excl": [], "nums": [("experience", "years of experience", [3, 5, 10])],
     "prices": [50, 80, 100, 150]},
    {"name": "plumber", "type": "professional_service", "art": "a plumber",
     "actions": ["find"],
     "attrs": [("specialization", ["boilers", "leaks", "bathrooms"], "for {v}")],
     "excl": [], "nums": [("experience", "years of experience", [3, 5, 10])],
     "prices": [50, 80, 100, 150]},
    {"name": "apartment", "type": "real_estate", "art": "an apartment",
     "actions": ["find", "rent"],
     "attrs": [("amenity", ["a balcony", "a garage", "an elevator"], "with {v}"),
               ("condition", ["furnished", "renovated"], "{v}")],
     "excl": [], "nums": [("rooms", "rooms", [2, 3, 4]),
                          ("size", "sqm", [50, 70, 90])],
     "prices": [700, 900, 1200, 1500]},
    {"name": "restaurant", "type": "place", "art": "a restaurant",
     "actions": ["find", "book"],
     "attrs": [("cuisine", ["Japanese", "Italian", "Indian", "vegetarian"], "{v}")],
     "excl": [], "nums": [("guests", "people", [2, 4, 6])],
     "prices": [20, 30, 50, 80]},
    {"name": "book", "type": "media", "art": "a book",
     "actions": ["find", "buy"],
     "attrs": [("genre", ["crime", "fantasy", "historical"], "{v}"),
               ("format", ["paperback", "hardcover", "ebook"], "in {v}")],
     "excl": [], "nums": [], "prices": [10, 15, 20, 30]},
    {"name": "cat", "type": "animal", "art": "a cat",
     "actions": ["find"],
     "attrs": [("breed", ["Sphynx", "Siamese", "Ragdoll"], "{v}"),
               ("sex", ["female", "male"], "{v}")],
     "excl": [], "nums": [("age", "months", [3, 6, 12])],
     "prices": [200, 400, 800]},
]

_EN_VERBS = {
    "find": ["find me", "i'm looking for", "i need", "look for",
             "can you find me", "i want", "could you find"],
    "book": ["book me", "book", "i'd like to book", "can you book"],
    "buy": ["buy me", "i want to buy", "purchase", "get me"],
    "order": ["order me", "order"],
    "rent": ["rent me", "i'd like to rent"],
    "compare": ["compare", "compare for me"],
}
_EN_CITIES = ["London", "Rome", "Milan", "Berlin", "Dublin", "Manchester",
              "Lisbon", "Madrid", "Boston", "Edinburgh", "Bristol"]
_EN_COMP = [("at least {n} {noun}", "gte", "constraints"),
            ("with at least {n} {noun}", "gte", "constraints"),
            ("{n} {noun}", "eq", "attributes"),
            ("with {n} {noun}", "eq", "attributes"),
            ("no more than {n} {noun}", "lte", "constraints"),
            ("more than {n} {noun}", "gt", "constraints"),
            ("under {n} {noun}", "lt", "constraints")]
_EN_PRICE = [("under {v} euros", "lt", "EUR"), ("under {v} dollars", "lt", "USD"),
             ("at most {v} euros", "lte", "EUR"), ("max {v} euros", "lte", "EUR"),
             ("less than {v} pounds", "lt", "GBP"),
             ("no more than {v} euros", "lte", "EUR"),
             ("for {v} euros", "eq", "EUR")]
_EN_SOFT_PRE = ["preferably", "ideally", "if possible"]
_EN_SOFT_POST = ["if possible", "if you can", "but it's not essential",
                 "would be nice"]


def english_block(rng: random.Random, n: int = 10) -> list[tuple[str, dict, str]]:
    """Richieste in inglese, con le stesse regole semantiche dell'italiano.

    Ne produce diverse per chiamata: gli altri blocchi ne restituiscono 5-7
    ciascuno, e a una per volta l'inglese restava allo 0.4% del dataset."""
    return [_english_one(rng) for _ in range(n)]


def _english_one(rng: random.Random) -> tuple[str, dict, str]:
    dom = rng.choice(_EN_DOMAINS)
    name, ttype, art = dom["name"], dom["type"], dom["art"]
    action = rng.choice(dom["actions"])
    verb = rng.choice(_EN_VERBS[action])
    attrs, cons, prefs = [], [], []
    parts = [f"{verb} {art}"]

    if dom["attrs"] and rng.random() < 0.7:
        prop, pool, tmpl = rng.choice(dom["attrs"])
        v = rng.choice(pool)
        frag = tmpl.format(v=v)
        op = "contains" if prop in ("ingredient", "amenity", "baggage",
                                    "specialization") else "eq"
        pred = {"property": prop, "operator": op, "value": v}
        if rng.random() < 0.28:
            prefs.append(pred)
            frag = (f"{rng.choice(_EN_SOFT_PRE)} {frag}" if rng.random() < 0.5
                    else f"{frag} {rng.choice(_EN_SOFT_POST)}")
        else:
            attrs.append(pred)
        parts.append(frag)

    if dom["excl"] and rng.random() < 0.3:
        prop, pool, tmpl = rng.choice(dom["excl"])
        v = rng.choice(pool)
        cons.append({"property": prop,
                     "operator": "excludes" if prop == "diet" else "neq",
                     "value": v})
        parts.append(tmpl.format(v=v))

    if dom["nums"] and rng.random() < 0.55:
        prop, noun, pool = rng.choice(dom["nums"])
        n = rng.choice(pool)
        tmpl, op, role = rng.choice(_EN_COMP)
        pred = {"property": prop, "operator": op, "value": n}
        if noun in ("GB of RAM", "inches", "sqm", "months"):
            pred["unit"] = {"GB of RAM": "GB", "inches": "inches",
                            "sqm": "sqm", "months": "months"}[noun]
        (cons if role == "constraints" else attrs).append(pred)
        parts.append(tmpl.format(n=n, noun=noun))

    if rng.random() < 0.55:
        city = rng.choice(_EN_CITIES)
        if ttype == "transport":
            cons.append({"property": "destination", "operator": "eq",
                         "value": city})
            parts.append(rng.choice([f"to {city}", f"bound for {city}"]))
        else:
            cons.append({"property": "location", "operator": "near",
                         "value": city})
            parts.append(rng.choice([f"in {city}", f"near {city}",
                                     f"around {city}"]))

    if rng.random() < 0.55:
        v = rng.choice(dom["prices"])
        tmpl, op, cur = rng.choice(_EN_PRICE)
        cons.append({"property": "price", "operator": op, "value": v,
                     "currency": cur})
        parts.append(tmpl.format(v=v))

    text = parts[0] + (", " + ", ".join(parts[1:]) if len(parts) > 1 else "")
    text = text[0].upper() + text[1:]
    goal = {"action": action, "raw_action": verb,
            "target": {"name": name, "type": ttype, "raw": art},
            "attributes": attrs, "constraints": cons, "preferences": prefs}
    fam = f"en::{name}:{action}:" + ":".join(sorted(
        p["property"] + ":" + p["operator"] for p in attrs + cons + prefs))
    return (text, {"schema_version": "1.1", "goals": [goal],
                   "clarification": {"required": False, "reason": None,
                                     "missing_fields": []}}, fam)


# ============================================================== produzione

_TYPE_TO_DOMAIN: dict[str, list[str]] = {}
for _dn, _d in DOMAINS.items():
    for _n, _tt, _ph in _d.get("targets", []):
        _TYPE_TO_DOMAIN.setdefault(_tt, []).append(_dn)


def _domain_of_type(ttype):
    names = _TYPE_TO_DOMAIN.get(ttype)
    return names[0] if names else None


def _one_domain_fact(rng, domain_name, raw_phrase):
    """Un fatto descrittivo a caso del dominio, per fare da contorno."""
    dom = DOMAINS[domain_name]
    pool = [(prop, maker) for prop, maker, _w in dom["facts"]
            # destination/origin sono escluse perché il ramo successivo di
            # `_context_facts` ne aggiunge già una: prendendole da qui si
            # ottenevano due destinazioni ("un volo per Amsterdam per Atene")
            if prop not in ("price", "location", "date", "time", "sex",
                            "destination", "origin")]
    if not pool:
        return None
    prop, maker = rng.choice(pool)
    agr = agreement_of(raw_phrase or "un oggetto")
    role = "attributes" if rng.random() < 0.8 else "preferences"
    for args in ((rng, role, agr, prop), (rng, role, agr), (rng, role)):
        try:
            return maker(*args)
        except TypeError:
            continue
    return None


def _context_facts(rng, ttype, raw_phrase, lang="it"):
    """Uno o due fatti di contorno coerenti col target."""
    extras = []
    dom_name = _domain_of_type(ttype)
    if lang == "it" and dom_name and rng.random() < 0.45:
        f = _one_domain_fact(rng, dom_name, raw_phrase)
        if f is not None:
            extras.append(f)
    if not extras or rng.random() < 0.5:
        if ttype == "transport":
            city = rng.choice(CITIES_ABROAD)
            extras.append(Fact(
                "destination", "constraints", "eq", city,
                (f"per {city}" if lang == "it" else f"to {city}"),
                family="destination:eq"))
        else:
            city = rng.choice(CITIES)
            extras.append(Fact(
                "location", "constraints", "near", city,
                (rng.choice([f"a {city}", f"in zona {city}", f"vicino a {city}"])
                 if lang == "it" else rng.choice([f"in {city}", f"near {city}"])),
                family="location:near"))
    if rng.random() < 0.25:
        if lang == "it":
            extras.append(f_date(rng, "constraints"))
        else:
            d = rng.choice(["tonight", "tomorrow", "on Saturday", "this weekend"])
            extras.append(Fact("date", "constraints", "eq",
                               d.replace("on ", ""), d, family="date:eq"))
    # una proprietà sola per contorno, sempre
    seen, unique = set(), []
    for f in extras:
        if f.prop in seen:
            continue
        seen.add(f.prop)
        unique.append(f)
    return unique


def enrich_block(rng: random.Random, items, p=0.6, lang="it"):
    """Aggiunge un fatto di contorno agli esempi di un blocco correttivo.

    I blocchi insegnano una cosa per volta, e per farlo hanno frasi con un
    fatto solo. Messi insieme erano metà del dataset, e il modello ha
    imparato benissimo anche la lezione che non volevo dargli: scrivi un
    fatto e fermati (35 `missing_fact` su 47 fallimenti in reports/038).

    Il contorno dipende dal TIPO del target — un volo riceve una
    destinazione, un ristorante una città — ed è identico per tutti gli
    esempi che condividono il target, così fra le due frasi di una coppia
    contrastiva continua a cambiare una cosa sola. Il primo tentativo
    calcolava il contorno una volta per blocco, e i blocchi che mescolano
    target diversi (l'inglese ne produce dieci) finivano con "a Bologna"
    appiccicato a un volo per Madrid."""
    if not items or rng.random() >= p:
        return items
    seed = rng.random()
    out = []
    for text, contract, family in items:
        # copia: certi blocchi riusano lo STESSO dizionario in più varianti,
        # e senza copia il secondo passaggio trovava i predicati già
        # presenti, non riscriveva il testo e produceva un contratto con
        # fatti che nella frase non c'erano (18 `hallucinated_value` su 6000)
        contract = copy.deepcopy(contract)
        try:
            goal = contract["goals"][0]
            ttype = goal["target"]["type"]
            raw = goal["target"].get("raw") or ""
        except (KeyError, IndexError, TypeError):
            out.append((text, contract, family))
            continue
        sub = random.Random(f"{seed}|{ttype}")
        extras = _context_facts(sub, ttype, raw, lang)
        present = {pr["property"]
                   for b in ("attributes", "constraints", "preferences")
                   for pr in goal.get(b, [])}
        added = []
        for f in extras:
            if f.prop in present:
                continue
            goal.setdefault(f.role, []).append(f.predicate())
            added.append(f)
        if not added:
            out.append((text, contract, family))
            continue
        tail = "".join(c for c in reversed(text) if c in ".?!")
        body = text.rstrip(".?! ")
        body = body + " " + " ".join(f.phrase for f in added) + tail[::-1]
        out.append((body, contract,
                    family))
    return out


def as_chat(request: str, contract: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": request},
        {"role": "assistant",
         "content": json.dumps(contract, ensure_ascii=False, separators=(",", ":"))},
    ]}


def generate(rng: random.Random, n: int, *, contrastive_ratio=0.22,
             multi_ratio=0.07, ambiguous_ratio=0.04, conditional_ratio=0.06,
             typo_ratio=0.10, paraphrases=2, corrective_ratio=0.17,
             ) -> list[tuple[str, dict, str]]:
    """Ritorna una lista di (testo, contratto, family)."""
    out: list[tuple[str, dict, str]] = []
    while len(out) < n:
        roll = rng.random()

        # blocchi correttivi mirati sugli errori misurati (vedi sopra)
        if roll < corrective_ratio:
            which = rng.random()
            if which < 0.30:
                out.extend(enrich_block(rng, number_role_block(rng)))
            elif which < 0.38:
                out.extend(enrich_block(rng, two_number_block(rng)))
            elif which < 0.52:
                out.extend(enrich_block(rng, comparative_block(rng)))
            elif which < 0.66:
                out.extend(enrich_block(rng, force_contrast_block(rng)))
            elif which < 0.74:
                out.extend(enrich_block(rng, specificity_block(rng)))
            elif which < 0.78:
                out.extend(enrich_block(rng, list_operator_block(rng)))
            elif which < 0.84:
                # multi-goal con vincolo condiviso: sul gold il secondo goal
                # spariva ("un idraulico e un elettricista a Torino")
                out.extend(shared_multi_block(rng))
            elif which < 0.87:
                out.extend(enrich_block(rng, negation_attachment_block(rng)))
            elif which < 0.89:
                out.extend(enrich_block(rng, shared_soft_block(rng)))
            elif which < 0.92:
                out.extend(availability_block(rng))
            elif which < 0.94:
                out.extend(date_vs_threshold_block(rng))
            elif which < 0.97:
                out.extend(enrich_block(rng, quantity_block(rng), p=0.45))
            else:
                out.extend(enrich_block(rng, english_block(rng), lang="en"))
            continue
        # fuori dai blocchi correttivi si rilancia il dado sulle altre
        # categorie, invece di riscalare il primo (più semplice da leggere)
        roll = rng.random()

        if roll < ambiguous_ratio:
            text, c = make_ambiguous(rng)
            out.append((text, c, "ambiguous::" + text.casefold()))
            continue
        if roll < ambiguous_ratio + conditional_ratio:
            out.extend(gate_block(rng))
            continue
        if roll < ambiguous_ratio + conditional_ratio + multi_ratio:
            a = build_scenario(rng, target_facts=rng.choice([2, 2, 3, 3, 4]))
            b = build_scenario(rng, target_facts=rng.choice([0, 1, 1, 2]))
            # il secondo goal non eredita i fatti del primo: se entrambi
            # hanno un luogo e per caso è lo stesso, l'esempio insegna a
            # copiare. Il modello lo faceva su multi-001 ("un hotel a Roma
            # e prenotami un treno per sabato" → treno con destinazione
            # Roma). Qui si garantisce che i luoghi siano diversi.
            aloc = {f.value for f in a.facts
                    if f.prop in ("location", "destination")}
            b.facts = [f for f in b.facts
                       if not (f.prop in ("location", "destination")
                               and f.value in aloc)]
            ta, ra = render(rng, a)
            tb, rb = render(rng, b)
            text = ta.rstrip(". ") + rng.choice([" e poi ", ". Poi ", " e anche ",
                                                 ". Inoltre "]) + tb[0].lower() + tb[1:]
            out.append((text, merge_contracts(to_contract(a, ra),
                                              to_contract(b, rb)),
                        f"multi::{a.family}::{b.family}"))
            continue

        # quanti fatti avrà questa frase: la forma si decide qui, una volta
        # sola, e si confronta con quella del gold (scripts/coverage_check.py)
        k = rng.choices(FACT_COUNTS, weights=FACT_COUNT_WEIGHTS)[0]
        if rng.random() < 0.22:
            # niente marcatori: i ruoli discendono solo dalla cascata, e la
            # resa telegrafica — che nel gold è il 30% dei casi — può usarla
            # senza lasciare preferenze appese al nulla
            sc = build_scenario(rng, target_facts=k, soft_p=0.0, oblige_p=0.0)
        else:
            sc = build_scenario(rng, target_facts=k)
        # parafrasi multiple dello STESSO scenario: la varianza è nella lingua
        k = paraphrases if rng.random() < 0.5 else 1
        for _ in range(k):
            text, raw_action = render(rng, sc)
            if rng.random() < typo_ratio:
                text = typo(rng, text, sc, raw_action)
            out.append((text, to_contract(sc, raw_action), sc.family))
            if len(out) >= n:
                break

        # coppie contrastive: stesso scenario, un fatto cambia ruolo/segno
        if rng.random() < contrastive_ratio and sc.facts:
            for variant in contrastive_variants(rng, sc):
                vtext, vraw = render(rng, variant)
                out.append((vtext, to_contract(variant, vraw), variant.family))
                if len(out) >= n:
                    break
    # ultima passata: le unità si scrivono in un modo solo (§6.2). Molti
    # blocchi costruiscono i predicati a mano e scrivevano "mesi", "pollici",
    # "gb" — tre grafie per due unità, e il benchmark ne chiede una.
    for _text, contract, _fam in out:
        for g in contract.get("goals") or []:
            preds = [p for b in ("attributes", "constraints", "preferences")
                     for p in (g.get(b) or [])]
            gt = g.get("gate")
            if isinstance(gt, dict) and isinstance(gt.get("when"), dict):
                preds.append(gt["when"])
            for p in preds:
                if p.get("unit") is not None:
                    p["unit"] = display_unit(p["unit"])
    return out[:n]


def contrastive_variants(rng: random.Random, sc: Scenario) -> list[Scenario]:
    """Varianti minime: stesso scenario, un solo fatto cambia forza o verso."""
    import copy as _copy
    variants: list[Scenario] = []
    idx = rng.randrange(len(sc.facts))
    base = sc.facts[idx]
    if base.prop in ("location", "destination"):
        return variants

    for role in ("attributes", "constraints", "preferences"):
        if role == base.role:
            continue
        v = _copy.deepcopy(sc)
        dom_maker = None
        for dom in DOMAINS.values():
            for prop, maker, _ in dom["facts"]:
                if prop == base.prop:
                    dom_maker = maker
                    break
            if dom_maker:
                break
        if base.prop == "price":
            newf = f_price(rng, role)
        elif dom_maker is not None:
            newf = dom_maker(rng, role)
        else:
            continue
        # la variante deve parlare dello STESSO valore, cambiando solo la
        # forza: si sostituisce il valore NELLA FRASE prima di riassegnarlo,
        # come token intero (una sottostringa trasformerebbe 2500 in 2525)
        replaced = re.sub(rf"\b{re.escape(str(newf.value))}\b",
                          str(base.value), newf.phrase, count=1)
        if replaced == newf.phrase and str(newf.value) != str(base.value):
            continue  # il valore non è sostituibile nella frase: si salta
        newf.phrase = replaced
        newf.value = base.value
        # value_to vive solo con `between`: se l'operatore cambia, sparisce
        newf.value_to = base.value_to if newf.operator == "between" else None
        newf.unit = base.unit
        # la valuta NON si eredita dal fatto base: dipende dalla frase di
        # QUESTA variante, che può non contenere il simbolo (§6.3)
        newf.currency = ("EUR" if ("euro" in newf.phrase or "€" in newf.phrase
                                   or " eur" in newf.phrase) else None)
        v.facts[idx] = newf
        # Tutte le derivazioni dello scenario devono rimanere nello split
        # del genitore, anche quando cambia la forza del fatto.
        v.family = sc.family
        variants.append(v)
    return variants[:2]


def family_split(rows: list[tuple[str, dict, str]], rng: random.Random,
                 valid_frac=0.08) -> tuple[list, list]:
    """Split per FAMIGLIA di scenario: niente parafrasi dello stesso scenario
    a cavallo tra train e valid."""
    families = sorted({r[2] for r in rows})
    rng.shuffle(families)
    n_valid = max(1, int(len(families) * valid_frac))
    valid_fams = set(families[:n_valid])
    train = [r for r in rows if r[2] not in valid_fams]
    valid = [r for r in rows if r[2] in valid_fams]
    return train, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v2")
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--valid-frac", type=float, default=0.08)
    ap.add_argument("--check", type=int, default=400,
                    help="quanti esempi verificare col Semantic Guard")
    ap.add_argument(
        "--decontaminate-against", default="benchmarks/gold_v1.jsonl",
        help="rimuove dal dataset gli exact duplicate del benchmark indicato; "
             "usa stringa vuota per disattivare")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = generate(rng, args.n)

    # Decontaminazione PRIMA dello split: i regression block possono
    # ricreare letteralmente frasi del benchmark. Quei casi sono utili come
    # famiglie semantiche, ma la frase gold non deve mai finire nel train.
    removed_leaks: list[str] = []
    if args.decontaminate_against:
        bp = Path(args.decontaminate_against)
        if not bp.is_absolute():
            bp = ROOT / bp
        if bp.exists():
            rows, removed_leaks = decontaminate_exact(rows, load_examples(bp))
        else:
            raise FileNotFoundError(f"benchmark di decontaminazione non trovato: {bp}")

    train, valid = family_split(rows, random.Random(args.seed + 1), args.valid_frac)

    # controllo di coerenza: il Guard deve accettare ciò che generiamo
    bad = 0
    sample = random.Random(args.seed + 2).sample(rows, min(args.check, len(rows)))
    for text, contract, _fam in sample:
        res = check_contract(contract, text, ground=True)
        if res.status not in (ACCEPT, REPAIRED):
            bad += 1
            if bad <= 5:
                print(f"  GUARD {res.status}: {text!r}")
                for i in res.issues:
                    if i.severity == "error":
                        print(f"      {i.code}: {i.message}")
        elif res.repairs and bad <= 5:
            print(f"  RIPARATO ({len(res.repairs)}): {text!r} → {res.repairs[0]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train.jsonl", train), ("valid.jsonl", valid)):
        with (out / name).open("w", encoding="utf-8") as f:
            for text, contract, _fam in data:
                f.write(json.dumps(as_chat(text, contract), ensure_ascii=False) + "\n")

    fam_train = {r[2] for r in train}
    fam_valid = {r[2] for r in valid}
    print(f"train={len(train)} valid={len(valid)}")
    if args.decontaminate_against:
        print(f"decontaminazione: rimossi {len(removed_leaks)} exact duplicate "
              f"di {args.decontaminate_against}")
    print(f"famiglie: train={len(fam_train)} valid={len(fam_valid)} "
          f"sovrapposte={len(fam_train & fam_valid)}")
    print(f"guard: {len(sample) - bad}/{len(sample)} accettati")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
