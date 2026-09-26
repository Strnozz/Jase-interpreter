"""Build a balanced V1.2 continuation set with held-out surface forms.

Cases are programmatically authored from explicit semantic templates, then
validated. The V20 benchmark panels and historical dev are exclusion sets.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.goal_contract_v1_2 import validate_contract  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.guard import ACCEPT, check_contract  # noqa: E402
from scripts.migrate_v19_to_v20 import convert  # noqa: E402

DEST = ROOT / "data/v21_9b"
SYSTEM = (ROOT / "V21_SYSTEM_PROMPT.txt").read_text(encoding="utf-8").strip()
TRAIN_CITIES = ["Roma", "Napoli", "Firenze", "Palermo", "Bari", "Trieste", "Verona", "Cagliari", "Pisa", "Modena", "Ravenna", "Perugia", "Lecce"]
VALID_CITIES = ["Ancona", "Catania", "Arezzo", "Pavia", "Taranto", "Lucca", "Salerno"]
NAMES = ["Marta", "Luca", "Sara", "Davide", "Elena", "Marco", "Chiara", "Paolo", "Anna", "Roberto", "Laura"]
FOODS = ["ramen", "tacos", "sushi", "falafel", "pizza", "cous cous", "gnocchi"]
CUISINES = ["messicano", "greco", "vietnamita", "indiano", "coreano", "libanese", "spagnolo"]
AMENITIES = ["wifi", "colazione", "piscina", "giardino", "parcheggio", "terrazza", "ascensore"]
MESSAGES = ["arrivo alle 19", "sono in ritardo", "ci vediamo domani", "la riunione è annullata", "parto venerdì", "ti richiamo stasera"]
COMMANDS = ["prenota un treno", "compra una bicicletta", "manda una mail", "ordina una pizza",
            "assumi un elettricista", "cancella il volo", "trova un hotel", "noleggia un furgone",
            "avvisami domani", "confronta due tablet", "acquista una lampada", "prenota un traghetto",
            "contatta un idraulico", "cerca una farmacia", "compra un biglietto", "prenota una camera",
            "ordina un libro", "trova un parcheggio", "disdici la visita", "scrivi a un medico",
            "cerca un agriturismo", "noleggia una bici", "avvisa il collega", "trova una libreria",
            "prenota un museo", "ordina dei fiori", "compra una giacca", "cerca un taxi",
            "contatta un fotografo", "confronta due telefoni", "prenota una cena", "cancella la lezione"]
TOPICS = ["prenotazioni ferroviarie", "acquisti online", "promemoria", "ricerca di alberghi",
          "invio delle email", "assunzione di professionisti", "cancellazione dei biglietti",
          "noleggio di automobili", "prenotazione dei musei", "confronto dei prezzi",
          "ordine di cibo", "ricerca di farmacie", "servizi idraulici", "biglietti aerei",
          "prenotazione dei ristoranti", "ricerca dei taxi", "acquisto di libri",
          "consegna dei fiori", "ricerca di appartamenti", "noleggio di biciclette",
          "invio di messaggi", "avvisi sul calendario", "prenotazione di visite mediche",
          "confronto degli hotel"]


def fact(field, op, value=None, *, strength="hard", **extra):
    return {"field": field, "op": op, "value": value, "strength": strength, **extra}


def goal(gid, action, name, typ, facts=None, **extra):
    return {"id": gid, "action": action, "target": {"name": name, "type": typ},
            "facts": facts or [], **extra}


def contract(*goals, context=None, kind="task"):
    out = {"schema_version": "1.2", "kind": kind, "goals": list(goals)}
    if context:
        out["context"] = context
    return out


def at(seq, i, stride=1):
    return seq[(i * stride) % len(seq)]


def authored(category: str, i: int, split: str):
    city = at(TRAIN_CITIES if split == "train" else VALID_CITIES, i, 3)
    name = at(NAMES, i, 5)
    form = i % (3 if split == "train" else 2)
    if category == "non_actionable_quote":
        phrase = COMMANDS[(i // (3 if split == "train" else 2)) % len(COMMANDS)]
        text = ([f"La stringa '{phrase}' è solo un esempio: non eseguire nulla.",
                 f"Sto citando il comando «{phrase}», non ti sto chiedendo di farlo.",
                 f"Nel manuale compare '{phrase}'; è una citazione, non una richiesta."] if split == "train" else
                [f"Non è un ordine: '{phrase}' è soltanto una frase di prova.",
                 f"Riporto per esempio '{phrase}', senza chiederti alcuna azione."])[form]
        return text, contract(kind="non_actionable")
    if category == "non_actionable_theory":
        topic = TOPICS[(i // (3 if split == "train" else 2)) % len(TOPICS)]
        text = ([f"In generale, come funzionano le {topic}? Non fare operazioni.",
                 f"Spiegami solo in teoria le {topic}.",
                 f"Sto studiando le {topic}: quali dati servono di solito?"] if split == "train" else
                [f"È una domanda teorica sulle {topic}, non una richiesta di esecuzione.",
                 f"Parliamo in astratto delle {topic}; non agire per me."])[form]
        return text, contract(kind="non_actionable")
    if category == "missing_table":
        guests = [2, 3, 4, 5][(i // 6) % 4] if i % 2 else None
        if guests is None:
            text = ([f"Prenota un tavolo a {city}, i dettagli te li dico dopo.",
                     f"Mi serve una prenotazione per un tavolo a {city}.",
                     f"Puoi riservarmi un tavolo a {city}? Non ho ancora scelto giorno e orario."] if split == "train" else
                    [f"Vorrei un tavolo prenotato a {city}; ora e numero di persone sono da definire.",
                     f"Occupati della prenotazione di un tavolo a {city}, poi ti darò i particolari."])[form]
        else:
            text = ([f"Prenota un tavolo a {city} per {guests} persone; giorno e ora li decido dopo.",
                     f"Per {guests} ospiti mi serve un tavolo a {city}, senza data e orario fissati.",
                     f"Riservami un tavolo a {city} per {guests}; ti darò data e ora più tardi."] if split == "train" else
                    [f"Vorrei un tavolo per {guests} a {city}; devo ancora scegliere quando.",
                     f"A {city} prenota per {guests} persone, ma ti confermerò giorno e ora."])[form]
        facts = [fact("location", "near", city)] + ([fact("guests", "eq", guests)] if guests else [])
        missing = ["date", "time"] + (["guests"] if guests is None else [])
        return text, contract(goal("g1", "book", "tavolo", "place", facts,
            missing=[{"field": x, "blocks": "action", "reason": "unspecified"} for x in missing]))
    if category == "missing_flight":
        origin = at(TRAIN_CITIES if split == "train" else VALID_CITIES, i + 4, 5) if i % 2 else None
        if origin == city:
            origin = at(TRAIN_CITIES if split == "train" else VALID_CITIES, i + 5, 5)
        if origin is None:
            text = ([f"Prenotami un volo per {city}; partenza e data non sono ancora fissate.",
                     f"Vorrei un biglietto aereo per {city}, ma devo ancora dirti da dove e quando parto.",
                     f"Riserva un volo diretto a {city}; ti comunicherò origine e giorno."] if split == "train" else
                    [f"Mi prenoti un volo verso {city}? Non ho indicato né città di partenza né data.",
                     f"Organizza la prenotazione aerea per {city}; origine e giorno sono ancora ignoti."])[form]
        else:
            text = ([f"Prenotami un volo da {origin} a {city}; non so ancora la data.",
                     f"Voglio un biglietto aereo {origin}-{city}, ma il giorno lo deciderò dopo.",
                     f"Riserva un volo con partenza da {origin} e arrivo a {city}; manca la data."] if split == "train" else
                    [f"Mi prenoti un volo da {origin} verso {city}? Non ho scelto il giorno.",
                     f"Per un volo {origin}-{city} fammi la prenotazione, data ancora da stabilire."])[form]
        facts = [fact("destination", "eq", city)] + ([fact("origin", "eq", origin)] if origin else [])
        missing = ["date"] + (["origin"] if origin is None else [])
        return text, contract(goal("g1", "book", "volo", "transport", facts,
            missing=[{"field": x, "blocks": "action", "reason": "unspecified"} for x in missing]))
    if category == "missing_reference":
        when = ["domani", "lunedì", "stasera", "venerdì", "tra due giorni", "sabato", "domenica",
                "martedì", "mercoledì", "giovedì"][(i // 6) % 10]
        if i % 2:
            text = ([f"Ricordamelo {when}; sai a cosa mi riferisco?", f"Metti un promemoria per quella cosa {when}.",
                     f"{when.capitalize()} ricordami ciò di cui parlavamo."] if split == "train" else
                    [f"Mi serve un promemoria {when} per quello che avevo detto prima.",
                     f"{when.capitalize()} avvisami di quella faccenda, anche se non l'ho specificata."])[form]
            return text, contract(goal("g1", "notify", "promemoria", "reminder", [fact("date", "eq", when)],
                missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}]))
        item = [("libro", "product"), ("scarpa", "product"), ("biglietto", "ticket"),
                ("tablet", "product"), ("maglione", "product"), ("orologio", "product"),
                ("quaderno", "product"), ("biglietto del treno", "ticket"),
                ("telefono", "product"), ("cappello", "product")][(i // 6) % 10]
        text = ([f"Compra il secondo {item[0]} che mi avevi mostrato prima.",
                 f"Acquista il {item[0]} numero due tra quelli visti in precedenza.",
                 f"Prendi il secondo {item[0]} della lista di prima, che ora non è disponibile."] if split == "train" else
                [f"Vorrei comprare il secondo {item[0]} che avevamo guardato in un'altra conversazione.",
                 f"Acquistami il secondo {item[0]} della vecchia lista, senza averla qui."])[form]
        return text, contract(goal("g1", "buy", item[0], item[1], [],
            missing=[{"field": "referent", "blocks": "action", "reason": "unresolved_reference"}]))
    if category == "missing_message":
        medium = ["messaggio", "email", "SMS"][(i // 33) % 3]
        text = ([f"Scrivi a {name} un {medium}; il testo te lo fornirò più tardi.",
                 f"Contatta {name} con un {medium} che ancora devo dettarti.",
                 f"Invia un {medium} a {name}, ma non ho detto cosa scrivere."] if split == "train" else
                [f"Manda un {medium} a {name}; il contenuto è ancora da decidere.",
                 f"Puoi scrivere un {medium} a {name}? Non ti ho fornito il testo."])[form]
        return text, contract(goal("g1", "contact", name, "person", [],
            missing=[{"field": "message", "blocks": "action", "reason": "unspecified"}]))
    if category == "confirm_contact":
        message = at(MESSAGES, i, 5)
        confirm = i % 4 != 0
        ending = (" ma solo dopo una mia conferma" if confirm else " senza altre condizioni")
        text = ([f"Scrivi a {name} che {message}{ending}.",
                 f"Manda a {name} il messaggio '{message}'{ending}.",
                 f"Contatta {name}: digli che {message}{ending}."] if split == "train" else
                [f"A {name} comunica '{message}'{ending}.",
                 f"Per favore, avvisa {name} che {message}{ending}."])[form]
        extra = {"policy": {"confirm_before": ["contact"]}} if confirm else {}
        return text, contract(goal("g1", "contact", name, "person", [fact("message", "eq", message)], **extra))
    if category == "forbid_side_effect":
        hotel = i % 2 == 0
        if hotel:
            text = ([f"Cerca un hotel a {city}, ma non prenotare nulla.",
                     f"Mostrami alberghi a {city}; la prenotazione è vietata.",
                     f"Trova un hotel a {city} senza riservare camere."] if split == "train" else
                    [f"Mi interessano hotel a {city}, però non devi effettuare prenotazioni.",
                     f"Fammi vedere un hotel a {city} e fermati prima di prenotare."])[form]
            return text, contract(goal("g1", "find", "hotel", "accommodation", [fact("location", "near", city)],
                                  policy={"forbid": ["book"]}))
        item = at(["bicicletta", "tablet", "giacca", "lampada"], i, 3)
        text = ([f"Cerca una {item} a {city}, senza comprarla.",
                 f"Mostrami una {item} disponibile a {city}: non effettuare acquisti.",
                 f"Trova una {item} a {city}, ma non procedere al pagamento."] if split == "train" else
                [f"Voglio vedere una {item} a {city}, non comprarla.",
                 f"Consulta le offerte per una {item} a {city}; nessun acquisto."])[form]
        return text, contract(goal("g1", "find", item, "product", [fact("location", "near", city)],
                              policy={"forbid": ["buy"]}))
    if category == "independent_goals":
        cities = TRAIN_CITIES if split == "train" else VALID_CITIES
        forms = 3 if split == "train" else 2
        city = cities[(i // forms) % len(cities)]
        other = cities[(i // (forms * 5) + 2) % len(cities)]
        if city == other:
            other = cities[(cities.index(other) + 1) % len(cities)]
        first, second = [("ristorante", "museo"), ("hotel", "farmacia"),
                         ("bar", "libreria"), ("parcheggio", "teatro")][(i // (forms * 3)) % 4]
        type_map = {"hotel": "accommodation", "ristorante": "place", "museo": "place",
                    "farmacia": "place", "bar": "place", "libreria": "place", "parcheggio": "place", "teatro": "place"}
        second_article = "una" if second in ("farmacia", "libreria") else "un"
        text = ([f"Trova un {first} a {city} e, separatamente, {second_article} {second} a {other}.",
                 f"Cerca due cose indipendenti: un {first} a {city} e {second_article} {second} a {other}.",
                 f"Mi servono un {first} a {city} e {second_article} {second} a {other}; nessuno dipende dall'altro."] if split == "train" else
                [f"Per due richieste distinte, cerca un {first} a {city} e {second_article} {second} a {other}.",
                 f"Mostrami un {first} a {city}; cerca anche {second_article} {second} a {other}, a parte."])[form]
        return text, contract(goal("g1", "find", first, type_map[first], [fact("location", "near", city)]),
                              goal("g2", "find", second, type_map[second], [fact("location", "near", other)]))
    if category == "ordinal_chain":
        forms = 3 if split == "train" else 2
        count = [3, 4, 5][(i // forms) % 3]
        ordinal = [(1, "primo"), (2, "secondo"), (3, "terzo")][(i // (forms * 3)) % 3]
        city = (TRAIN_CITIES if split == "train" else VALID_CITIES)[
            (i // (forms * 9)) % (len(TRAIN_CITIES) if split == "train" else len(VALID_CITIES))]
        text = ([f"Cerca {count} ristoranti a {city}, poi prenota il {ordinal[1]} per domani alle 20 per due persone.",
                 f"Mostrami {count} ristoranti a {city}; fra i risultati riserva l'opzione numero {ordinal[0]} domani alle 20 per due.",
                 f"Trova {count} ristoranti a {city} e dopo prenota il risultato {ordinal[0]} per due persone domani alle 20."] if split == "train" else
                [f"Elenca {count} ristoranti a {city}; scegli il {ordinal[1]} e prenotalo domani alle 20 per due ospiti.",
                 f"Voglio vedere {count} ristoranti a {city}, quindi riservare il numero {ordinal[0]} per domani alle 20, due persone."])[form]
        first = goal("g1", "find", "ristorante", "place", [fact("quantity", "eq", count), fact("location", "near", city)])
        selection = {"field": "selection", "op": "eq", "value_ref": {"goal": "g1", "field": "result", "ordinal": ordinal[0]}, "strength": "hard"}
        second = goal("g2", "book", "ristorante", "place", [selection, fact("date", "eq", "domani"),
            fact("time", "eq", "20:00"), fact("guests", "eq", 2)], depends_on=["g1"])
        return text, contract(first, second)
    if category == "fallback":
        food, cuisine = at(FOODS, i, 3), at(CUISINES, i, 5)
        text = ([f"A {city} cerca {food}; se non ne trovi, cerca un ristorante {cuisine} sempre a {city}.",
                 f"Prova a trovare {food} a {city}; in assenza di risultati ripiega su un ristorante {cuisine} a {city}.",
                 f"Voglio {food} a {city}; solo se la ricerca è vuota proponi un ristorante {cuisine} a {city}."] if split == "train" else
                [f"Cercami {food} a {city}; se non c'è, va bene un ristorante {cuisine} a {city}.",
                 f"Prima cerca {food} a {city}; nel caso non trovi nulla, passa a un ristorante {cuisine} a {city}."])[form]
        first = goal("g1", "find", food, "food", [fact("location", "near", city)])
        second = goal("g2", "find", "ristorante", "place", [fact("cuisine", "eq", cuisine),
            fact("location", "near", city)], depends_on=["g1"],
            condition={"goal": "g1", "test": {"field": "result_count", "op": "eq", "value": 0}})
        return text, contract(first, second)
    if category == "positive_condition":
        price = at([180, 220, 260, 300, 340, 380], i, 5)
        text = ([f"Se trovi un volo per {city} sotto {price} euro, cerca anche una stanza.",
                 f"Cerca un volo per {city} a meno di {price} euro; soltanto se c'è, trova un albergo.",
                 f"Vorrei un volo verso {city} sotto {price} euro; se la ricerca riesce, cerca poi un hotel."] if split == "train" else
                [f"Solo nel caso esista un volo per {city} sotto {price} euro, cerca una camera.",
                 f"Trova un volo per {city} entro {price} euro; se lo trovi, passa a cercare un alloggio."])[form]
        first = goal("g1", "find", "volo", "transport", [fact("destination", "eq", city),
            fact("price", "lt" if (split == "train" or form == 0) else "lte", price, currency="EUR")])
        second_name = ("stanza" if form == 0 and split == "train" else
                       "camera" if form == 0 else
                       "alloggio" if split == "valid" else
                       "albergo" if form == 1 else "hotel")
        second = goal("g2", "find", second_name, "accommodation", [], depends_on=["g1"],
            condition={"goal": "g1", "test": {"field": "result_count", "op": "gt", "value": 0}})
        return text, contract(first, second)
    if category == "context":
        if i % 2 == 0:
            text = ([f"Ho già il biglietto; trovami un hotel a {city}.",
                     f"Il biglietto è già mio. Cerca un albergo a {city}.",
                     f"Possiedo già il ticket: mi serve un hotel a {city}."] if split == "train" else
                    [f"Il biglietto l'ho acquistato prima; ora cerca un hotel a {city}.",
                     f"Sono già munito di biglietto, trovami un albergo a {city}."])[form]
            name = "albergo" if (split == "train" and form == 1) or (split == "valid" and form == 1) else "hotel"
            return text, contract(goal("g1", "find", name, "accommodation", [fact("location", "near", city)]),
                context=[{"field": "ticket_owned", "op": "eq", "value": True}])
        count = at([2, 3, 4, 5], i, 3)
        text = ([f"Siamo in {count} e non abbiamo un'auto; cerca un ristorante a {city}.",
                 f"Per {count} persone senza macchina, trova un ristorante a {city}.",
                 f"Non abbiamo la macchina e il gruppo è di {count}: cercaci un ristorante a {city}."] if split == "train" else
                [f"Il gruppo conta {count} persone, nessuno ha l'auto. Cerca un ristorante a {city}.",
                 f"Senza automobile, noi {count} vorremmo un ristorante a {city}."])[form]
        return text, contract(goal("g1", "find", "ristorante", "place", [fact("location", "near", city)]),
            context=[{"field": "party_size", "op": "eq", "value": count},
                     {"field": "has_car", "op": "eq", "value": False}])
    if category == "ranking":
        mode = i % 2
        if mode == 0:
            text = ([f"Cerca treni per {city} domani e ordina dal meno caro al più caro.",
                     f"Trova un treno per {city} domani, mostrando prima quelli più economici.",
                     f"Elenca i treni diretti a {city} domani in ordine di prezzo crescente."] if split == "train" else
                    [f"Per {city} domani cerca treni, classificati dal prezzo più basso.",
                     f"Mostrami treni verso {city} domani ordinati per costo crescente."])[form]
            return text, contract(goal("g1", "find", "treno", "transport", [fact("destination", "eq", city),
                fact("date", "eq", "domani")], ranking={"field": "price", "direction": "asc"}))
        text = ([f"Cerca hotel a {city}, dai più cari ai meno cari.",
                 f"Mostrami gli hotel a {city} ordinati per prezzo decrescente.",
                 f"Trova un albergo a {city} e metti prima quelli che costano di più."] if split == "train" else
                [f"Elenca hotel a {city} in ordine di costo dal maggiore al minore.",
                 f"Per {city} cerca alberghi, con i prezzi più alti in cima."])[form]
        name = "albergo" if (split == "train" and form == 2) or (split == "valid" and form == 1) else "hotel"
        return text, contract(goal("g1", "find", name, "accommodation", [fact("location", "near", city)],
            ranking={"field": "price", "direction": "desc"}))
    if category == "hard_soft":
        price, amenity = at([70, 85, 100, 115, 130, 150], i, 5), at(AMENITIES, i, 3)
        text = ([f"Cerca un hotel a {city} entro {price} euro, preferibilmente con {amenity}.",
                 f"Mi serve un hotel a {city} da massimo {price} euro; se c'è {amenity}, meglio.",
                 f"Trova un hotel a {city} sotto o uguale a {price} euro, idealmente con {amenity}."] if split == "train" else
                [f"Per {city} cerco un hotel non oltre {price} euro; {amenity} sarebbe un vantaggio.",
                 f"Un hotel a {city} da non più di {price} euro, possibilmente con {amenity}."])[form]
        return text, contract(goal("g1", "find", "hotel", "accommodation", [fact("location", "near", city),
            fact("price", "lte", price, currency="EUR"), fact("amenity", "eq", amenity, strength="soft")]))
    raise ValueError(category)


CATEGORIES = ("non_actionable_quote", "non_actionable_theory", "missing_table", "missing_flight",
              "missing_reference", "missing_message", "confirm_contact", "forbid_side_effect",
              "independent_goals", "ordinal_chain", "fallback", "positive_condition", "context",
              "ranking", "hard_soft")


def row(text, gold, *, category, split, source_id=None):
    issues = validate_contract(gold)
    if issues:
        raise ValueError(f"Invalid {category}: {text}: {issues}")
    answer = json.dumps(gold, ensure_ascii=False, separators=(",", ":"))
    return {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": answer}],
            "category": category, "source_split": split,
            "source_id": source_id or hashlib.sha256(f"{category}:{split}:{text}".encode()).hexdigest()[:20]}


def text_of(item):
    return item["messages"][1]["content"]


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    rng = random.Random(2027)
    data = {"train": [], "valid": []}
    for split, count in (("train", 180), ("valid", 30)):
        for category in CATEGORIES:
            for i in range(count):
                text, gold = authored(category, i, split)
                data[split].append(row(text, gold, category=category, split=split))
    source = ROOT / "data/v20_9b"
    for split, count in (("train", 1000), ("valid", 90)):
        old = [item for item in read_jsonl(source / f"{split}.jsonl")
               if json.loads(item["messages"][-1]["content"])["goals"][0]["action"]
               in ("find", "compare", "monitor")]
        selected = rng.sample(old, count)
        for item in selected:
            item = dict(item)
            item["messages"] = [dict(message) for message in item["messages"]]
            item["messages"][0]["content"] = SYSTEM
            item["category"] = "v20_rehearsal"
            data[split].append(item)
    # Real V19 conjunctions with two independent read-only goals. Source Guard
    # must accept the whole request; each goal must convert without inference.
    for split, cap in (("train", 300), ("valid", 60)):
        multi = []
        for item in read_jsonl(ROOT / f"data/v19_clean/{split}.jsonl"):
            text = next(m["content"] for m in item["messages"] if m["role"] == "user")
            old = json.loads(item["messages"][-1]["content"])
            goals = old.get("goals", [])
            if (len(goals) != 2 or any(g.get("action") != "find" or g.get("gate") is not None for g in goals)
                    or any(g.get("target", {}).get("type") != "place" for g in goals)
                    or old.get("clarification", {}).get("required")
                    or any(word in normalize_text(text).split() for word in
                           ("poi", "dopo", "prima", "quindi", "se", "terzo", "secondo"))):
                continue
            if check_contract(old, text).status != ACCEPT:
                continue
            converted = []
            for index, old_goal in enumerate(goals, 1):
                partial = {"schema_version": "1.1", "goals": [old_goal],
                           "clarification": {"required": False, "reason": None, "missing_fields": []}}
                single, reasons = convert(partial, text)
                if reasons:
                    break
                new_goal = single["goals"][0]
                new_goal["id"] = f"g{index}"
                converted.append(new_goal)
            if len(converted) == 2:
                gold = contract(*converted)
                normalized = normalize_text(text)
                first_name = normalize_text(converted[0]["target"]["name"])
                second_name = normalize_text(converted[1]["target"]["name"])
                second_at = normalized.rfind(second_name)
                facts = converted[0]["facts"]
                shared_after_second = (first_name != second_name and second_at > normalized.find(first_name)
                    and facts == converted[1]["facts"] and all(
                        "value" in fact and normalized.find(normalize_text(str(fact["value"])), second_at) >= 0
                        for fact in facts))
                if shared_after_second and not validate_contract(gold):
                    multi.append(row(text, gold, category="v19_multi_independent", split=split))
        data[split].extend(rng.sample(multi, min(cap, len(multi))))
    excluded = set()
    benchmark_paths = [ROOT / "benchmarks/v20/hard_dev.jsonl", ROOT / "benchmarks/v20/holdout.jsonl",
                       ROOT / "data/v20_quality/dev.jsonl", ROOT / "benchmarks/release_holdout_v1.jsonl"]
    benchmark_tokens = []
    for path in benchmark_paths:
        for case in read_jsonl(path):
            key = normalize_text(case["text"])
            excluded.add(key)
            benchmark_tokens.append(token_set(case["text"]))
    seen = {}
    for split in ("train", "valid"):
        unique = []
        for item in data[split]:
            text = text_of(item)
            key = normalize_text(text)
            if key in excluded:
                continue
            tokens = token_set(text)
            if any(tokens and other and len(tokens & other) / len(tokens | other) >= .85
                   for other in benchmark_tokens):
                continue
            if key in seen:
                if seen[key][0] != item["messages"][-1]["content"]:
                    raise RuntimeError(f"Conflicting labels for {text}")
                continue
            seen[key] = (item["messages"][-1]["content"], split)
            unique.append(item)
        data[split] = unique
    # Keep variant-heavy synthetic validation honest: quarantine close lexical
    # neighbors of train, including old V19 prompt-family variants.
    postings = defaultdict(set)
    train_tokens = [token_set(text_of(item)) for item in data["train"]]
    for index, tokens in enumerate(train_tokens):
        for token in tokens:
            postings[token].add(index)
    clean_valid = []
    near_validation_quarantined = 0
    for item in data["valid"]:
        tokens = token_set(text_of(item))
        candidates = set()
        for token in sorted(tokens, key=lambda t: len(postings[t]))[:3]:
            candidates.update(postings[token])
        if any(tokens and train_tokens[j]
               and len(tokens & train_tokens[j]) / len(tokens | train_tokens[j]) >= .85
               for j in candidates):
            near_validation_quarantined += 1
        else:
            clean_valid.append(item)
    data["valid"] = clean_valid
    rng.shuffle(data["train"])
    rng.shuffle(data["valid"])
    for split in ("train", "valid"):
        with (DEST / f"{split}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for item in data[split]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    manifest = {"version": "V21_9B", "schema_version": "1.2", "seed": 2027,
                "source": "V20 adapter continuation + authored V1.2 templates + V20 rehearsal",
                "source_sha256": {split: sha256(source / f"{split}.jsonl") for split in ("train", "valid")},
                "excluded_benchmark_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in benchmark_paths},
                "generator_sha256": sha256(Path(__file__)),
                "system_prompt_sha256": sha256(ROOT / "V21_SYSTEM_PROMPT.txt"),
                "rows": {split: len(data[split]) for split in data},
                "near_validation_quarantined": near_validation_quarantined,
                "category_counts": {split: dict(Counter(item["category"] for item in data[split])) for split in data},
                "output_sha256": {split: sha256(DEST / f"{split}.jsonl") for split in data},
                "limitations": ["Synthetic templates require real-world review", "No contradiction/blocked contract in V1.2",
                                "Prior-conversation reference lacks a state ID"]}
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
