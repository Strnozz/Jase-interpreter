"""Semantic Guard — barriera deterministica fail-closed.

    user text → Interpreter → contratto candidato → GUARD → Planner

Il Guard non "migliora" un contratto: lo accetta, lo ripara solo con
trasformazioni dimostrabilmente sicure, oppure lo ferma.
Nessun contratto sospetto arriva al Planner.

Esiti:
    ACCEPT    il contratto è valido così com'è
    REPAIRED  riparato con trasformazioni deterministiche elencate
    CLARIFY   non utilizzabile, ma si sa cosa chiedere all'utente
    REJECT    output inutilizzabile (JSON rotto, degenerato, fuori schema)
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # jsonschema >= 4.x
    from jsonschema import Draft202012Validator as _Validator
except ImportError:  # ambienti con jsonschema 3.x: Draft7 copre const e $ref
    from jsonschema import Draft7Validator as _Validator

from .canon import (
    ExtractionError,
    canon_predicate,
    extract_contract,
    fact_key,
    norm_text,
    norm_value,
    predicate_key,
    strip_accents,
)
from .lexicon import (
    CLARIFICATION_REASONS,
    OBSERVATIONAL_ACTIONS,
    CURRENCY_MAP,
    DESCRIPTIVE_PROPERTIES,
    LIST_OPERATORS,
    MONETARY_PROPERTIES,
    NEGATION_MARKERS,
    NEGATIVE_OPERATORS,
    OBLIGATION_MARKERS,
    NUMBER_WORDS,
    NUMERIC_PROPERTIES,
    ORDERING_OPERATORS,
    SELECTION_PROPERTIES,
    SOFT_MARKERS,
    TEMPORAL_OPERATORS,
    TEMPORAL_PROPERTIES,
)

ROOT = Path(__file__).resolve().parents[1]
#: lo schema attivo è la v1.1 (aggiunge il `gate` opzionale, §11). La v1
#: resta su disco come baseline storica dei report già prodotti.
_SCHEMA_PATH = ROOT / "schema" / "goal_contract_v1_1.schema.json"
_SCHEMA_V1_PATH = ROOT / "schema" / "goal_contract_v1.schema.json"
SCHEMA = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
SCHEMA_V1 = json.loads(_SCHEMA_V1_PATH.read_text(encoding="utf-8"))
VALIDATOR = _Validator(SCHEMA)
VALIDATOR_V1 = _Validator(SCHEMA_V1)

ACCEPT, REPAIRED, CLARIFY, REJECT = "ACCEPT", "REPAIRED", "CLARIFY", "REJECT"
BUCKETS = ("attributes", "constraints", "preferences")
#: priorità in caso di duplicato tra bucket (§13 di SEMANTICS.md)
BUCKET_PRIORITY = {"constraints": 0, "attributes": 1, "preferences": 2}

#: normalizzazioni ammesse: valore prodotto → forme accettabili nel testo utente
VALUE_ALIASES: dict[str, tuple[str, ...]] = {
    "female": ("femmina", "femminile", "f", "female"),
    "male": ("maschio", "maschile", "m", "male"),
    "new": ("nuovo", "nuova", "new"),
    "used": ("usato", "usata", "used", "second hand", "seconda mano"),
    "refurbished": ("ricondizionato", "ricondizionata", "refurbished"),
    "direct": ("diretto", "diretta", "senza scalo", "senza scali", "direct",
               "nonstop", "non stop"),
    # la forma italiana è un valore a sé: il modello scrive quella, e
    # "senza scali" / "non stop" sono lo stesso fatto detto al contrario
    # (normalizzazione dichiarata in §12)
    "diretto": ("diretto", "diretta", "senza scalo", "senza scali",
                "direct", "nonstop", "non stop", "no stopover"),
    "eur": ("euro", "eur", "€", "euri"),
    "usd": ("dollari", "dollaro", "usd", "$"),
    "gbp": ("sterline", "gbp", "£"),
}


@dataclass
class Issue:
    code: str
    severity: str  # "error" | "warn"
    message: str
    path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "severity": self.severity,
                "message": self.message, "path": self.path}


@dataclass
class GuardResult:
    status: str
    contract: dict[str, Any] | None
    issues: list[Issue] = field(default_factory=list)
    repairs: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in (ACCEPT, REPAIRED)

    def error_codes(self) -> list[str]:
        return [i.code for i in self.issues if i.severity == "error"]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "issues": [i.as_dict() for i in self.issues],
            "repairs": list(self.repairs),
            "contract": self.contract,
        }


# ----------------------------------------------------------------- grounding

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _text_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(strip_accents(text.lower())))


def _text_numbers(text: str) -> set[float]:
    flat = strip_accents(text.lower())
    nums: set[float] = set()
    for m in re.findall(r"\d+(?:[.,]\d+)?", flat):
        try:
            nums.add(float(m.replace(",", ".")))
        except ValueError:
            pass
    for word, val in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", flat):
            nums.add(float(val))
    # "20:30" → anche 20 e 30 sono ancorati
    for h, mnt in re.findall(r"(\d{1,2})[:.](\d{2})", flat):
        nums.add(float(h))
        nums.add(float(mnt))
    return nums


def _fuzzy_token_match(token: str, hay: set[str]) -> bool:
    """Match tollerante a flessione: `rosso`↔`rossa`, `panini`↔`panino`."""
    if token in hay:
        return True
    if len(token) < 4:
        return False
    stem = token[:-1]
    for h in hay:
        if h == token or h.startswith(stem) or token.startswith(h[:-1] if len(h) > 3 else h):
            if abs(len(h) - len(token)) <= 3 and h[:3] == token[:3]:
                return True
    return False


def value_is_grounded(value: Any, user_text: str) -> bool:
    """Un valore è ancorato se risale al testo utente o a §12 di SEMANTICS.md."""
    if value is None or isinstance(value, bool):
        return True
    tokens = _text_tokens(user_text)
    numbers = _text_numbers(user_text)

    if isinstance(value, (int, float)):
        return float(value) in numbers

    s = str(value).strip()
    if not s:
        return True
    # liste serializzate: ogni elemento deve essere ancorato
    if "|" in s:
        return all(value_is_grounded(part.strip(), user_text)
                   for part in s.split("|") if part.strip())
    # valore numerico espresso come stringa
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", s):
        return float(s.replace(",", ".")) in numbers
    # orari "18:00" → basta l'ora
    m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", s)
    if m:
        return float(m.group(1)) in numbers
    low = strip_accents(s.lower())
    aliases = VALUE_ALIASES.get(low, ())
    for alias in aliases:
        if strip_accents(alias) in strip_accents(user_text.lower()):
            return True
    # tutti i token significativi del valore devono comparire nel testo
    val_tokens = [t for t in _TOKEN_RE.findall(low) if len(t) > 1]
    if not val_tokens:
        return True
    return all(_fuzzy_token_match(t, tokens) for t in val_tokens)


#: marcatori di strettezza che PRECEDONO il numero → operatore atteso (§5.1)
PREFIX_STRICTNESS: tuple[tuple[str, str], ...] = (
    ("non piu di", "lte"), ("non oltre", "lte"), ("al massimo", "lte"),
    ("al piu", "lte"), ("fino a", "lte"), ("entro", "lte"),
    ("massimo", "lte"), ("max", "lte"),
    ("come minimo", "gte"), ("a partire da", "gte"), ("almeno", "gte"),
    ("minimo", "gte"), ("min ", "gte"),
    ("al di sotto di", "lt"), ("meno di", "lt"), ("inferiore a", "lt"),
    ("sotto i", "lt"), ("sotto le", "lt"), ("sotto ai", "lt"), ("sotto", "lt"),
    ("superiore a", "gt"), ("piu di", "gt"), ("oltre", "gt"), ("sopra", "gt"),
)

#: marcatori che SEGUONO il numero ("cento euro proprio massimo")
SUFFIX_STRICTNESS: tuple[tuple[str, str], ...] = (
    ("al massimo", "lte"), ("massimo", "lte"), ("max", "lte"),
    ("al minimo", "gte"), ("minimo", "gte"), ("in su", "gte"),
    ("o piu", "gte"), ("o meno", "lte"),
)

#: modi di dire italiani del rifiuto: "dieci euro non li spendo" è un tetto
#: STRETTO, non un prezzo. Il modello li leggeva come `eq` e nessun controllo
#: se ne accorgeva (nat-001, nat-006, nat-007 in reports/035). Cercati nella
#: coda della frase, dopo il numero.
REFUSAL_IDIOMS: tuple[str, ...] = (
    r"non li spendo", r"non li voglio spendere", r"non li spenderei",
    r"non ce li ho", r"non li ho", r"proprio non li spendo",
    r"non voglio spender\w*", r"non arriverei a", r"non ci arrivo",
    r"non ci voglio arrivare", r"non li butto",
)
_REFUSAL_RE = re.compile("|".join(REFUSAL_IDIOMS))

#: "senza arrivare a X", "senza superare X": tetto stretto, prefisso
PREFIX_STRICT_PHRASES: tuple[tuple[str, str], ...] = (
    ("senza arrivare a", "lt"), ("senza superare", "lt"),
    ("senza sforare", "lt"), ("senza spendere", "lt"),
    ("restare sotto", "lt"), ("stare sotto", "lt"), ("rimanere sotto", "lt"),
)

#: marcatori corti che sono anche parole comuni o nomi propri ("Nebula Max"):
#: valgono solo se incollati al numero, senza nemmeno una parola in mezzo
_STRICT_ADJACENCY = {"max", "min "}
_AMBIGUOUS_PREFIX: set[str] = set()

#: verbi di intenzione: "non voglio spendere più di X" = X è un tetto
_NEGATED_INTENT = (
    "voglio", "vorrei", "voglia", "intendo", "posso", "riesco", "spendere",
    "spendo", "superare", "supero", "sforare", "pagare", "pago", "farmi",
    "fare", "andare", "arrivare", "want", "spend", "pay", "go",
)


def _negation_flips(window: str) -> bool:
    """La negazione ribalta davvero il verso del comparativo?

    Sì in "non voglio spendere più di 50" (50 è un tetto) e in "non meno di 3"
    (3 è un minimo). No in "niente gatti magari meno di 120": lì il `niente`
    governa i gatti, non la soglia — falso positivo trovato generando il
    dataset. Serve o l'adiacenza, o un verbo d'intenzione di mezzo."""
    m = None
    for cand in re.finditer(r"\b(non|mai|no)\b", window):
        # "se no" / "sennò" è un connettivo ("altrimenti"), non una negazione:
        # in "...ma non è un problema se no, sotto 800 euro" quel `no` non
        # ribalta niente
        if cand.group(1) == "no" and re.search(r"\bse\s*$", window[:cand.start()]):
            continue
        m = cand
    if m is None:
        return False
    gap = window[m.end():]
    if not gap.strip():
        return True  # "non meno di 3": negazione attaccata al comparativo
    # il verbo d'intenzione vale solo nella STESSA clausola: in "300 euro
    # però non li spendo, superiore a 4 stelle" la negazione governa i 300,
    # e senza questa barriera ribaltava anche le 4 stelle
    if any(c in gap for c in ",;.!?"):
        return False
    # La negazione deve governare il verbo d'intenzione, non soltanto
    # precederlo da qualche parte nella stessa clausola. In
    # "no glutine vorrei restare sotto i 5 euro" il `no` riguarda glutine:
    # cercare `vorrei` ovunque nel gap ribaltava erroneamente `sotto` in gte.
    tokens = re.findall(r"[a-z]+", strip_accents(gap.lower()))
    clitics = {"ci", "ce", "li", "lo", "la", "le", "mi", "me", "ne",
               "proprio", "davvero", "assolutamente"}
    while tokens and tokens[0] in clitics:
        tokens.pop(0)
    if not tokens:
        return False
    return tokens[0] in _NEGATED_INTENT


#: i marcatori vanno cercati come PAROLE: senza confini, "o meno" si
#: annidava dentro "sabatO MENO di 250" e ribaltava l'operatore.
_MARKER_RE_CACHE: dict[str, "re.Pattern[str]"] = {}


def _marker_re(marker: str) -> "re.Pattern[str]":
    rx = _MARKER_RE_CACHE.get(marker)
    if rx is None:
        pat = re.escape(marker.strip())
        if marker.strip()[:1].isalnum():
            pat = r"\b" + pat
        if marker.strip()[-1:].isalnum():
            pat = pat + r"\b"
        rx = re.compile(pat)
        _MARKER_RE_CACHE[marker] = rx
    return rx


def _rfind_marker(hay: str, marker: str) -> int:
    last = -1
    for m in _marker_re(marker).finditer(hay):
        last = m.start()
    return last


def _find_marker(hay: str, marker: str) -> int:
    m = _marker_re(marker).search(hay)
    return m.start() if m else -1


def _number_spans(flat: str, value) -> list[int]:
    """Posizioni di inizio in cui `value` compare nel testo, in cifre o in lettere."""
    num = int(value) if float(value).is_integer() else value
    # (?<!\d)…(?!\d) invece di \b…\b: con i confini di parola "50" dentro
    # "50km" non veniva trovato (la `k` è un carattere di parola), e il
    # rilevatore vedeva una sola occorrenza invece di due, perdendo
    # l'ambiguità che doveva farlo tacere.
    spans = [m.start() for m in
             re.finditer(rf"(?<!\d){re.escape(str(num))}(?!\d)", flat)]
    for word, val in NUMBER_WORDS.items():
        if float(val) == float(value):
            spans += [m.start() for m in re.finditer(rf"\b{re.escape(word)}\b", flat)]
    return spans


#: "un/una/uno" sono numerali solo sulla carta: nel 99% dei testi sono
#: articoli, e trattarli da numeri faceva tacere il rilevatore su
#: "Trovami UN panino sotto i 10 euro"
_NOT_REALLY_NUMBERS = {"un", "una", "uno", "one", "a", "an"}
_ANY_NUMBER_RE = re.compile(
    r"\d+|\b(?:" + "|".join(sorted(
        (re.escape(w) for w in NUMBER_WORDS if w not in _NOT_REALLY_NUMBERS),
        key=len, reverse=True)) + r")\b")


def _looks_like_proper_noun(user_text: str, flat_pos: int, marker: str) -> bool:
    """Il marcatore, nel testo ORIGINALE, ha l'iniziale maiuscola a metà frase?

    "Talin Max 13 pollici": quel `Max` è parte del nome del prodotto. In
    minuscolo l'informazione si perde, quindi si guarda il testo originale."""
    if flat_pos < 0 or flat_pos >= len(user_text):
        return False
    ch = user_text[flat_pos]
    if not ch.isupper():
        return False
    prefix = user_text[:flat_pos].rstrip()
    return bool(prefix) and not prefix.endswith((".", "!", "?", ":"))


def expected_strictness(user_text: str, value) -> str | None:
    """Operatore atteso per un valore numerico, dedotto dal marcatore vicino.

    Regola di attribuzione: un marcatore governa il PRIMO numero che lo segue.
    Se fra il marcatore e il nostro numero ce n'è un altro, il marcatore è già
    "consumato" da quello e non vale per noi — senza questo controllo,
    'sotto i 300 euro e confrontami tre amplificatori' faceva sembrare che
    anche il 3 avesse una soglia.

    Se lo stesso numero compare più volte con indicazioni discordi, si
    restituisce None: meglio non segnalare che segnalare a caso, perché
    questo controllo BLOCCA il contratto.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    flat = strip_accents(user_text.lower())
    verdicts: set[str | None] = set()
    for start in _number_spans(flat, value):
        # la finestra DEVE iniziare a inizio parola: tagliando a 34 caratteri
        # secchi, "Buongiorno, prenota ... sotto i 500" lasciava il frammento
        # "no, prenota ...", e quel "no" veniva letto come negazione che
        # ribalta la soglia. Falso positivo trovato generando il dataset.
        win = max(0, start - 34)
        while 0 < win < start and flat[win - 1] not in " \t\n":
            win += 1
        before = flat[win:start]
        after = flat[start:start + 40]
        best_op, best_dist = None, 10 ** 6
        for marker, op in PREFIX_STRICTNESS:
            pos = _rfind_marker(before, marker)
            if pos < 0:
                continue
            if marker in _AMBIGUOUS_PREFIX and pos < len(before) - 6:
                continue
            gap = before[pos + len(marker.strip()):]
            if _ANY_NUMBER_RE.search(gap):
                continue  # il marcatore governa quell'altro numero, non il nostro
            # né può scavalcare una virgola o un'altra parola piena: un
            # marcatore di soglia sta incollato al suo numero. In
            # "entro lunedì, 1000€ non li voglio spendere" e in
            # "entro dopodomani da 6 mesi" l'`entro` è della data, non del
            # numero che segue. Entrambi falsi positivi trovati generando
            # il dataset.
            if re.search(r"[,.;:!?]", gap):
                continue
            # ADIACENZA STRETTA per tutti i marcatori prefissi: fra il
            # marcatore e il suo numero non ci va nemmeno una parola. Ogni
            # volta che ho concesso una parola di margine è entrato un falso
            # positivo ("entro dopodomani da 6 mesi", "massimo domenica 2 GB",
            # "Nebula Max con 16 GB"). Si perde qualche rilevazione — "meno di
            # ben 600" non scatta più — ma questo controllo BLOCCA contratti,
            # quindi la precisione vale più della copertura.
            if gap.strip():
                continue
            if _looks_like_proper_noun(user_text, win + pos, marker):
                continue
            # un marcatore stretto fra DUE numeri è ambiguo: in
            # "1500 euro proprio massimo 2 GB" quel `massimo` è il suffisso
            # dei 1500, non il prefisso del 2. In dubbio, tace.
            if _ANY_NUMBER_RE.search(before[max(0, pos - 20):pos]):
                continue
            dist = len(gap)
            if op in ("gt", "lt") and _negation_flips(before[max(0, pos - 30):pos]):
                op = {"gt": "lte", "lt": "gte"}[op]
            if dist < best_dist:
                best_op, best_dist = op, dist
        for marker, op in SUFFIX_STRICTNESS:
            pos = _find_marker(after, marker)
            if pos < 0:
                continue
            # la lunghezza del numero va misurata sul TESTO, non su
            # len(str(value)): con un float "80.0" conta 4 caratteri e il
            # punto della frase finiva fuori dal controllo di punteggiatura
            num_match = _ANY_NUMBER_RE.match(after)
            num_len = num_match.end() if num_match else 0
            gap_after = after[num_len:pos]
            if _ANY_NUMBER_RE.search(gap_after):
                continue
            # un marcatore in un'altra proposizione non è nostro:
            # "32 GB di RAM, avrei bisogno di Zorba Max" — quel `Max` è un
            # nome proprio dopo la virgola
            if re.search(r"[,.;:!?]", gap_after):
                continue
            # il resto si guarda sul testo INTERO, non sulla finestra di 40
            # caratteri: altrimenti il taglio della finestra crea una fine di
            # frase che non esiste
            rest = flat[start + pos + len(marker.strip()):]
            if _ANY_NUMBER_RE.match(rest.lstrip()[:12] or ""):
                continue
            # "sarebbe IL massimo" è un modo di dire, non una soglia:
            # un articolo davanti al marcatore lo rende un sostantivo
            if re.search(r"\b(il|lo|la|un|una)\s*$", gap_after):
                continue
            # un marcatore suffisso vale solo se CHIUDE la frase: "100 euro
            # massimo" sì, "il Talin Max sotto i 700" no.
            if not re.match(r"\s*($|[.,;:!?)])", rest):
                continue
            if pos < best_dist:
                best_op, best_dist = op, pos
        # modi di dire del rifiuto nella coda della frase.
        # Qui serve una finestra più lunga dei 40 caratteri usati per i
        # marcatori: in "dieci euro per un panino proprio non li spendo"
        # l'idioma sta a 33 caratteri dal numero e la finestra corta lo
        # tagliava a metà.
        tail = flat[start:start + 90]
        if best_op is None and _REFUSAL_RE.search(tail):
            gap = tail[:_REFUSAL_RE.search(tail).start()]
            # la lunghezza del numero si misura sul testo: con `gap[2:]` il
            # "0" di "500" veniva scambiato per un secondo numero
            nm = _ANY_NUMBER_RE.match(tail)
            rest = gap[nm.end():] if nm else gap
            # l'idioma non può scavalcare una virgola: in "…con 1 posti, a
            # Trieste, venerdì, non voglio spendere 150€" il rifiuto è dei
            # 150, non del numero di posti
            if (not re.search(r"[,.;:!?]", rest)
                    and not _ANY_NUMBER_RE.search(rest)):
                best_op = "lt"
        # locuzioni prefisse a più parole ("senza arrivare a 10€")
        if best_op is None:
            for phrase, op in PREFIX_STRICT_PHRASES:
                pos = _rfind_marker(before, phrase)
                if pos < 0:
                    continue
                gap = before[pos + len(phrase):]
                if _ANY_NUMBER_RE.search(gap) or re.search(r"[,.;:!?]", gap):
                    continue
                if len(gap.strip().split()) <= 1:
                    best_op = op
                    break
        verdicts.add(best_op)
    # un solo verdetto, non ambiguo, su tutte le occorrenze del numero
    if len(verdicts) != 1:
        return None
    return verdicts.pop()


#: ancore che autorizzano una `currency` (§6.3)
CURRENCY_ANCHORS: dict[str, tuple[str, ...]] = {
    "EUR": ("\u20ac", "euro", "eur", "euri"),
    "USD": ("$", "dollar", "usd"),
    "GBP": ("\u00a3", "sterlin", "gbp", "pound"),
    "CHF": ("chf", "franch"),
}


def currency_is_grounded(currency: str | None, user_text: str) -> bool:
    """Una valuta è ammessa solo se il testo la nomina (simbolo o parola).

    Un verbo monetario ('spendere') giustifica la PROPRIETÀ price, non la
    valuta: quella dipende dalla localizzazione dell'utente e si risolve a
    valle, come 'stasera' (§6.3)."""
    if not currency:
        return True
    flat = strip_accents(user_text.lower())
    anchors = CURRENCY_ANCHORS.get(str(currency).upper())
    if anchors is None:
        return False
    return any(strip_accents(a) in flat for a in anchors)


#: parole di apertura e cortesia: non sono MAI valori di un predicato.
#: Misurato su reports/014: da "ciao! senti, mi servirebbe una pizza…" il
#: modello estraeva `ingredient contains senti`. Il controllo di ancoraggio
#: non poteva vederlo, perché "senti" è davvero nel testo.
DISCOURSE_FILLERS = frozenset({
    "senti", "allora", "ciao", "salve", "scusa", "scusi", "ehi", "ehilà",
    "buongiorno", "buonasera", "grazie", "per favore", "prego", "guarda",
    "dunque", "insomma", "praticamente", "diciamo", "ecco", "beh", "boh",
    "hey", "hi", "hello", "please", "thanks", "thank you", "so", "well",
})

#: marcatori che fissano il RUOLO di un numero, non la sua soglia.
#: (regex sul testo prima / dopo il numero) → classe attesa
_ROLE_BEFORE: tuple[tuple[str, str], ...] = (
    (r"\balle\s*$", "time"), (r"\ball\s*$", "time"),
    (r"\bper le\s*$", "time"), (r"\bentro le\s*$", "time"),
    (r"\bdopo le\s*$", "time"), (r"\bdalle\s*$", "time"),
    (r"\bprima delle\s*$", "time"), (r"\bverso le\s*$", "time"),
    (r"\bat\s*$", "time"), (r"\bbefore\s*$", "time"),
)
_ROLE_AFTER: tuple[tuple[str, str], ...] = (
    (r"^\s*(?:€|euro|eur\b|euri|dollar|usd\b|sterlin|£|\$)", "money"),
    (r"^\s*(?:km|chilometri|metri|miglia|minuti di|miles)\b", "distance"),
    (r"^\s*(?:person[ea]|adulti|ospiti|people|guests|pax)\b", "quantity"),
    (r"^\s*(?:stelle|stars)\b", "rating"),
    (r"^\s*(?:notti|nights)\b", "duration"),
)

#: a quale classe appartiene ciascuna proprietà, per il confronto
_PROPERTY_ROLE: dict[str, str] = {
    **{p: "time" for p in TEMPORAL_PROPERTIES},
    **{p: "money" for p in MONETARY_PROPERTIES},
    "distance": "distance", "radius": "distance", "mileage": "distance",
    "guests": "quantity", "quantity": "quantity",
    "stars": "rating", "rating": "rating",
    "duration": "duration",
}


def expected_number_role(user_text: str, value) -> str | None:
    """Classe semantica che il TESTO assegna a un numero.

    `alle 20` è un orario, `20 euro` è denaro, `20 km` una distanza,
    `20 persone` una quantità. È un'informazione che sta nella preposizione e
    nell'unità, non nel numero: il modello la sbagliava ancora
    (`Prenotami un tavolo alle 20` → `guests eq 20`), e l'ancoraggio non
    poteva accorgersene perché il 20 nel testo c'è davvero.

    Come per le soglie: se il numero compare più volte con indicazioni
    discordi, si tace."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    flat = strip_accents(user_text.lower())
    verdicts: set[str | None] = set()
    for start in _number_spans(flat, value):
        num_match = _ANY_NUMBER_RE.match(flat[start:])
        num_len = num_match.end() if num_match else len(str(int(value)))
        before, after = flat[:start], flat[start + num_len:]
        role = None
        for pat, r in _ROLE_BEFORE:
            if re.search(pat, before):
                role = r
                break
        if role is None:
            for pat, r in _ROLE_AFTER:
                if re.match(pat, after):
                    role = r
                    break
        verdicts.add(role)
    if len(verdicts) != 1:
        return None
    return verdicts.pop()


IMPLICIT_ACTION = "(implicito)"


# ------------------------------------------------- fatti non rendicontati
#
# Il Guard sa dire se ciò che il modello ha prodotto è sbagliato. Non sa dire
# se manca qualcosa — e l'omissione è la causa di 30 dei 44 contratti
# sbagliati che superano il Guard (reports/030). Questo controllo guarda il
# TESTO e cerca frammenti che devono per forza aver prodotto un predicato.
#
# È volutamente conservativo: segnala solo due forme ad altissima precisione,
# perché un falso positivo qui blocca un contratto corretto.

#: "senza X", "niente X": una negazione esplicita DEVE produrre un predicato
#: negativo. Se non c'è, il modello ha ignorato una richiesta dell'utente.
#:
#: Non vale davanti a un infinito: "senza arrivare a spendere 10€" e
#: "senza superare i 500" non escludono niente, esprimono una soglia. Era
#: l'unico falso positivo su 102 casi gold.
_EXPLICIT_NEGATION_RE = re.compile(
    r"\b(?:senza|niente|without)\s+"
    r"(?!\w*(?:are|ere|ire|arsi|ersi|irsi)\b)"
    r"([a-zà-ÿ]{3,}(?:\s+[a-zà-ÿ]{3,})?)")

#: locuzioni in cui "senza" non introduce un'esclusione richiesta
_NEGATION_STOPWORDS = frozenset({
    "problemi", "fretta", "impegno", "dubbio", "altro", "troppo", "esagerare",
    "dover", "spendere", "troppa", "troppi",
    # avverbi temporali: "niente prima delle 8" non esclude un ingrediente
    "prima", "dopo", "oltre", "meno", "piu", "più", "male", "importa",
    # "senza scali" non è un'esclusione da rendicontare: è il modo normale
    # di dire "diretto", e §12 dichiara la normalizzazione
    "scali", "scalo", "stops", "sosta", "soste",
})

#: espressioni di tempo che, se compaiono nel testo, DEVONO comparire nel
#: contratto. Lista chiusa: niente inferenze, solo parole che non possono
#: voler dire altro. "ora", "oggi come oggi", "prima" e simili restano
#: fuori di proposito — sono troppo spesso modi di dire.
TEMPORAL_WORDS = (
    "stasera", "stamattina", "stanotte", "stamane", "dopodomani",
    "lunedi", "martedi", "mercoledi", "giovedi", "venerdi", "sabato",
    "domenica", "weekend", "ferragosto", "capodanno",
    "tonight", "tomorrow", "this weekend", "tonite",
)

#: nomi propri a metà frase: se l'utente scrive una parola con l'iniziale
#: maiuscola in mezzo a una frase, quasi sempre è un'entità — una città, una
#: marca, una compagnia — e deve comparire nel contratto. Misurato: 0 falsi
#: positivi sui 102 casi gold e 1 su 4000 esempi generati (quello era un
#: refuso). Le sigle e le parole di cortesia restano fuori.
_PROPER_NOUN_RE = re.compile(r"\b([A-Z][\w'&-]{2,})")
_PROPER_NOUN_STOP = frozenset({
    "gb", "tb", "mb", "kb", "ram", "ssd", "hdd", "tv", "pc", "usb", "ok",
    "spa", "wifi", "km", "mq", "eur", "usd", "gbp", "chf", "cpu", "gpu",
    "suv", "bnb", "b&b", "iva", "dvd", "cd", "led", "oled", "lcd", "hd",
    "poi", "ah", "eh", "mah", "boh",
})

#: un numero seguito da valuta o unità è un fatto, non rumore
_MEASURED_NUMBER_RE = re.compile(
    r"(?<!\d)(\d+(?:[.,]\d+)?)\s*"
    r"(€|\$|£|euro|eur\b|dollar\w*|sterlin\w*|km\b|mq\b|gb\b|tb\b|"
    r"pollici|mesi|anni|stelle|notti|persone)", re.I)

#: Numerali scritti in lettere davanti a sostantivi che quasi certamente
#: esprimono una quantità/misura. Il controllo è volutamente chiuso: non
#: proviamo a interpretare ogni numero in lettere, ma solo combinazioni ad
#: alta precisione che in produzione non possono sparire dal contratto.
_QUANTIFIED_NOUNS = (
    "biglietti", "biglietto", "tickets", "ticket",
    "persone", "persona", "people", "guests",
    "notti", "notte", "nights", "night",
    "anni", "anno", "years", "year",
    "mesi", "mese", "months", "month",
    "stelle", "stella", "stars", "star",
    "scali", "scalo", "stops", "stop",
    "camere", "camera", "rooms", "room",
    "posti", "posto", "seats", "seat",
    "pezzi", "pezzo", "items", "item",
    "ore", "ora", "hours", "hour",
    "minuti", "minuto", "minutes", "minute",
)
_NUMBER_WORD_KEYS = tuple(
    sorted((w for w, n in NUMBER_WORDS.items() if n >= 2), key=len, reverse=True)
)
_QUANTIFIED_WORD_RE = re.compile(
    rf"\b({'|'.join(map(re.escape, _NUMBER_WORD_KEYS))})\s+"
    rf"({'|'.join(map(re.escape, _QUANTIFIED_NOUNS))})\b", re.I)


def unaccounted_facts(contract: dict[str, Any], user_text: str) -> list[str]:
    """Frammenti del testo che non hanno prodotto nessun predicato.

    Ritorna le descrizioni dei fatti apparentemente ignorati. Lista vuota
    quando tutto è rendicontato."""
    if not user_text:
        return []
    flat = strip_accents(user_text.lower())
    values: list[str] = []
    numbers: set[float] = set()
    has_negative = False
    target_text = ""
    actions: list[str] = []
    for g in contract.get("goals") or []:
        t = g.get("target") or {}
        target_text += " " + norm_text(t.get("name")) + " " + norm_text(t.get("raw"))
        actions.append(strip_accents(str(g.get("raw_action") or "").lower()))
        actions.append(strip_accents(str(g.get("action") or "").lower()))
        gt = g.get("gate")
        if isinstance(gt, dict):
            actions.append(strip_accents(str(gt.get("then") or "").lower()))
            actions.append(strip_accents(str(gt.get("otherwise") or "").lower()))
        preds = [p for b in BUCKETS for p in (g.get(b) or []) if isinstance(p, dict)]
        gate = g.get("gate")
        if isinstance(gate, dict) and isinstance(gate.get("when"), dict):
            preds.append(gate["when"])
        for p in preds:
            if canon_predicate(p)["operator"] in NEGATIVE_OPERATORS:
                has_negative = True
            for field in ("value", "value_to", "reference"):
                v = p.get(field)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    numbers.add(float(v))
                elif isinstance(v, str):
                    values.append(strip_accents(v.lower()))
                    m = re.fullmatch(r"(\d{1,2})[:.](\d{2})", v.strip())
                    if m:
                        numbers.add(float(m.group(1)))

    missing: list[str] = []
    for m in _EXPLICIT_NEGATION_RE.finditer(flat):
        what = m.group(1).strip()
        if what.split()[0] in _NEGATION_STOPWORDS:
            continue
        covered = has_negative and any(
            what.split()[0][:4] in v for v in values)
        if not covered:
            missing.append(f"esclusione non rendicontata: 'senza {what}'")
            break  # una segnalazione basta: il contratto si ferma comunque

    # espressioni temporali: lista CHIUSA, quindi ad altissima precisione.
    # Se l'utente dice "sabato" o "stasera" e nel contratto non c'è nessun
    # valore che lo contenga, un fatto è andato perso. Era il caso di
    # nat-020 ("sabato"), en-003 ("tonight") e nat-019 ("domenica"): tre
    # contratti sbagliati che arrivavano al Planner senza che nulla li
    # fermasse.
    for word in TEMPORAL_WORDS:
        if not re.search(rf"\b{re.escape(word)}\b", flat):
            continue
        if any(word in v for v in values):
            continue
        if word in strip_accents(target_text.lower()):
            continue
        missing.append(f"espressione temporale non rendicontata: '{word}'")
        break

    for sentence in re.split(r"(?<=[.!?])\s+", user_text):
        found = None
        for m in _PROPER_NOUN_RE.finditer(sentence):
            if m.start() == 0:      # inizio frase: la maiuscola è d'obbligo
                continue
            word = m.group(1)
            low = strip_accents(word.lower())
            if low in _PROPER_NOUN_STOP or low in DISCOURSE_FILLERS:
                continue
            if low in strip_accents(target_text.lower()):
                continue
            if any(low in v for v in values) or any(low in a for a in actions):
                continue
            found = word
            break
        if found:
            missing.append(f"nome proprio non rendicontato: '{found}'")
            break

    for m in _MEASURED_NUMBER_RE.finditer(flat):
        try:
            n = float(m.group(1).replace(",", "."))
        except ValueError:
            continue
        if n in numbers:
            continue
        # un numero che fa parte del nome del target non è un vincolo
        # ("iPhone 15", "RTX 6080", "Fiat 500")
        if re.search(rf"(?<!\d){re.escape(m.group(1))}(?!\d)", target_text):
            continue
        missing.append(
            f"numero non rendicontato: '{m.group(0).strip()}'")
        break

    # Numeri in lettere: `due biglietti`, `tre persone`, `cinque stelle`...
    # Prima questa informazione poteva sparire senza che il Guard avesse modo
    # di accorgersene (il caso storico nat-020).
    for m in _QUANTIFIED_WORD_RE.finditer(flat):
        n = float(NUMBER_WORDS.get(m.group(1).lower(), -1))
        if n in numbers:
            continue
        missing.append(f"quantità non rendicontata: '{m.group(0).strip()}'")
        break
    return missing



def raw_action_is_grounded(raw_action: str, user_text: str) -> bool:
    """`raw_action` è per definizione testo LETTERALE dell'utente (§7).

    Se non compare nel testo, il modello se l'è inventato — ed è
    l'allucinazione peggiore, perché mette in bocca all'utente parole che non
    ha detto. Misurato in reports/005: su 'Un hotel a 4 stelle a Bologna',
    che non contiene verbi, il modello produceva raw_action='avrei bisogno di'.
    """
    if not raw_action:
        return False
    ra = norm_text(raw_action)
    if not ra or ra == norm_text(IMPLICIT_ACTION):
        return True
    text = norm_text(user_text)
    if ra in text:
        return True
    # tolleranza alla flessione su una sola parola ("trova"/"trovami")
    tokens = ra.split()
    if len(tokens) == 1 and len(tokens[0]) >= 4:
        return any(h.startswith(tokens[0][:4]) for h in text.split())
    return all(t in text for t in tokens)


def has_soft_marker(user_text: str) -> bool:
    flat = strip_accents(user_text.lower())
    return any(strip_accents(m) in flat for m in SOFT_MARKERS)


def has_obligation_marker(user_text: str) -> bool:
    flat = strip_accents(user_text.lower())
    return any(strip_accents(m) in flat for m in OBLIGATION_MARKERS)


def bucket_by_cascade(p: dict[str, Any]) -> str:
    """Bucket imposto da SEMANTICS.md §2, per le regole che NON leggono il testo.

    R1 (marcatore debole) e R3 (obbligo) non stanno qui: dipendono da dove
    cade una parola nella frase, e il Guard non sa a quale fatto si
    riferisca. Il chiamante applica la riparazione solo quando nel testo non
    compare nessuno dei due, cioè quando restano in gioco solo R2, R4, R5 e
    R6 — e allora il bucket è funzione di operatore e proprietà, niente di
    più. Meglio non riparare che riparare a caso."""
    cp = canon_predicate(p)
    prop, op = norm_text(p.get("property")), cp["operator"]
    if op in NEGATIVE_OPERATORS:                                    # R2
        return "constraints"
    if prop in SELECTION_PROPERTIES:                                # R4
        return "constraints"
    if op in ORDERING_OPERATORS or op in TEMPORAL_OPERATORS:        # R5
        return "constraints"
    return "attributes"                                             # R6


#: R2 — una negazione attaccata al valore ne ribalta la polarità. Da
#: "cerco zorba x7 ... no rosso" il modello produceva `color eq rosso` e
#: nessun controllo se ne accorgeva, perché "rosso" nel testo c'è davvero
#: (nat-004 in reports/038). Si guardano solo i caratteri PRIMA del valore,
#: senza attraversare una virgola: più in là la negazione parla d'altro.
_POLARITY_NEGATIONS = (
    "no", "non", "niente", "nessun", "nessuna", "nessuno", "senza",
    "escluso", "esclusa", "escluse", "tranne", "eccetto", "evita",
    "evitare", "ne", "not", "without", "except", "avoid", "neither", "nor",
)
_POSITIVE_OPERATORS = frozenset({"eq", "contains", "in"})
_POLARITY_WINDOW = 24
#: fra la negazione e il valore possono esserci solo parole vuote. Con
#: "fino a tre parole qualsiasi", da "una pizza con la bufala SENZA CIPOLLA
#: dopodomani" risultava negato anche "dopodomani": la negazione aveva già
#: il suo bersaglio, e la finestra non se ne accorgeva.
_FUNCTION_WORDS = frozenset({
    "sia", "siano", "e", "essere", "di", "del", "dello", "della", "dei",
    "degli", "delle", "il", "lo", "la", "i", "gli", "le", "un", "uno",
    "una", "che", "con", "a", "al", "allo", "alla", "ai", "agli", "alle",
    "in", "nel", "nella", "da", "dal", "dalla", "per", "su", "sul", "o",
    "is", "be", "the", "an", "of", "with", "to", "any", "it",
})


def negation_attached_to_value(user_text: str, value: Any) -> bool:
    """Il valore compare nel testo preceduto da una negazione, ogni volta?

    Come per la strettezza: se il valore compare più volte e le occorrenze
    non concordano, si tace."""
    if not isinstance(value, str) or not value.strip():
        return False
    flat = strip_accents(user_text.lower())
    needle = strip_accents(value.lower().strip())
    if not needle:
        return False
    verdicts: set[bool] = set()
    start = flat.find(needle)
    while start != -1:
        # la finestra parte dall'inizio della CLAUSOLA, non da un offset
        # fisso: tagliare a 24 caratteri cadeva in mezzo a una parola e
        # "un pani|no possibilmente con la mortadella" diventava una
        # negazione. È lo stesso errore di "Buongior-no": \b crede sempre
        # che l'inizio di una stringa sia un confine di parola.
        cut = max([flat.rfind(c, 0, start) for c in ",;.!?"] + [-1]) + 1
        prefix = flat[cut:start]
        hit = False
        for m in _POLARITY_NEGATIONS:
            mt = re.search(rf"\b{re.escape(m)}\b((?:\s+\w+){{0,3}})\W*$", prefix)
            if not mt or (start - cut - mt.start()) > _POLARITY_WINDOW:
                continue
            between = mt.group(1).split()
            if all(w in _FUNCTION_WORDS for w in between):
                hit = True
                break
        verdicts.add(hit)
        start = flat.find(needle, start + 1)
    return len(verdicts) == 1 and verdicts.pop()


_HHMM_RE = re.compile(r"^(\d{1,2})[:.](\d{2})$")


def time_hour(value: Any) -> float | None:
    """Ora contenuta in un valore temporale normalizzato ('20:00' → 20).

    Serve al controllo di ruolo: la normalizzazione `20` → `20:00` (§12)
    rendeva il numero invisibile, e da "un idraulico entro 20 euro" passava
    indisturbato `time before 20:00` (con-024 in reports/038)."""
    if not isinstance(value, str):
        return None
    m = _HHMM_RE.match(value.strip())
    return float(m.group(1)) if m else None


# --------------------------------------------------------------- validazioni

def _predicate_issues(p: dict[str, Any], path: str) -> list[Issue]:
    issues: list[Issue] = []
    c = canon_predicate(p)
    prop, op, val = c["property"], c["operator"], c["value"]

    if op in ORDERING_OPERATORS and op != "within":
        if not isinstance(val, (int, float)) and not _looks_temporal(p.get("value")):
            issues.append(Issue("operator_value_mismatch", "error",
                                f"operatore '{op}' su valore non ordinabile {p.get('value')!r}",
                                path))
    if op == "within" and not isinstance(val, (int, float)):
        issues.append(Issue("operator_value_mismatch", "error",
                            f"'within' richiede un valore numerico, trovato {p.get('value')!r}",
                            path))
    if op in TEMPORAL_OPERATORS and prop and prop not in TEMPORAL_PROPERTIES:
        issues.append(Issue("temporal_operator_on_non_temporal", "warn",
                            f"operatore '{op}' su proprietà non temporale '{prop}'", path))
    if op == "between" and p.get("value_to") is None:
        issues.append(Issue("between_without_value_to", "error",
                            "'between' senza value_to", path))
    if op != "between" and p.get("value_to") is not None:
        issues.append(Issue("value_to_without_between", "error",
                            f"value_to presente con operatore '{op}'", path))
    if p.get("currency") is not None and prop not in MONETARY_PROPERTIES:
        issues.append(Issue("currency_on_non_monetary", "error",
                            f"currency su proprietà non monetaria '{prop}'", path))
    if prop in MONETARY_PROPERTIES and not isinstance(val, (int, float)):
        issues.append(Issue("non_numeric_money", "error",
                            f"valore monetario non numerico: {p.get('value')!r}", path))
    if prop in NUMERIC_PROPERTIES and not isinstance(val, (int, float)) and op not in LIST_OPERATORS:
        issues.append(Issue("non_numeric_value", "warn",
                            f"proprietà numerica '{prop}' con valore {p.get('value')!r}", path))
    if op in LIST_OPERATORS:
        raw = str(p.get("value") or "")
        if not raw.strip():
            issues.append(Issue("empty_list_value", "error",
                                f"operatore '{op}' con valore vuoto", path))
    if prop and not str(prop).strip():
        issues.append(Issue("empty_property", "error", "property vuota", path))
    if prop != norm_text(prop):
        pass  # solo cosmetico
    return issues


_TIME_RE = re.compile(r"^\d{1,2}([:.]\d{2})?$")


def _looks_temporal(value: Any) -> bool:
    if value is None:
        return False
    s = str(value).strip().lower()
    if _TIME_RE.match(s):
        return True
    keywords = ("oggi", "domani", "stasera", "stamattina", "sabato", "domenica",
                "lunedì", "martedì", "mercoledì", "giovedì", "venerdì",
                "settimana", "mese", "weekend", "today", "tomorrow", "tonight")
    return any(k in strip_accents(s) or k in s for k in keywords)


def _conflicting(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Due predicati sulla stessa proprietà si contraddicono in modo palese?"""
    ca, cb = canon_predicate(a), canon_predicate(b)
    if ca["property"] != cb["property"]:
        return False
    va, vb = ca["value"], cb["value"]
    oa, ob = ca["operator"], cb["operator"]
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        upper = {"lt", "lte"}
        lower = {"gt", "gte"}
        if oa in upper and ob in lower and va <= vb:
            return True
        if oa in lower and ob in upper and vb <= va:
            return True
        if oa == "eq" and ob in upper and va > vb:
            return True
        if oa == "eq" and ob in lower and va < vb:
            return True
    if va == vb and (
        (oa == "eq" and ob == "neq") or (oa == "neq" and ob == "eq")
        or (oa == "contains" and ob == "excludes") or (oa == "excludes" and ob == "contains")
    ):
        return True
    # due valori DIVERSI dati per certi sulla stessa proprietà: "un volo per
    # Amsterdam per Atene" non è una richiesta, è un errore. `eq` e `near`
    # designano un valore solo; `contains` no (un hotel può avere piscina e
    # palestra), e nemmeno le soglie (un prezzo ha un minimo e un massimo).
    # Nessuno dei 102 casi gold ne ha uno: il generatore invece ne produceva
    # 61 su 28 000, e insegnava una cosa impossibile.
    if oa in ("eq", "near") and ob in ("eq", "near") and va != vb:
        return True
    return False


# --------------------------------------------------------------- guard vero

def check_contract(contract: dict[str, Any], user_text: str = "",
                   *, ground: bool = True,
                   check_omissions: bool = True) -> GuardResult:
    """Valida e ripara un contratto già estratto.

    `check_omissions` esiste per i test mirati: un contratto-fixture ridotto
    all'osso serve a verificare UN controllo alla volta, e il rilevatore di
    omissioni lo segnalerebbe giustamente come incompleto. In produzione
    resta sempre attivo."""
    issues: list[Issue] = []
    repairs: list[str] = []
    c = copy.deepcopy(contract)

    # --- R0: un contratto v1 è un contratto v1.1 senza gate (§11.5).
    # La migrazione è una riparazione deterministica e senza perdita.
    if "goals" in c and c.get("schema_version") != "1.1":
        if c.get("schema_version") in (None, "", "1", "1.0", 1, 1.0, "1.1"):
            old = c.get("schema_version")
            c["schema_version"] = "1.1"
            if old not in (None, ""):
                repairs.append(f"schema_version {old!r} → '1.1' (migrazione v1→v1.1)")
            else:
                repairs.append("schema_version normalizzata a '1.1'")

    # --- riparazioni pre-schema sui predicati
    for gi, goal in enumerate(c.get("goals") or []):
        if not isinstance(goal, dict):
            continue
        for bucket in BUCKETS:
            preds = goal.get(bucket)
            if not isinstance(preds, list):
                continue
            for pi, p in enumerate(preds):
                if not isinstance(p, dict):
                    continue
                path = f"goals[{gi}].{bucket}[{pi}]"
                # campi extra vuoti → rimossi
                for extra in [k for k in list(p)
                              if k not in ("property", "operator", "value", "value_to",
                                           "unit", "currency", "reference")]:
                    if p[extra] in (None, "", [], {}):
                        p.pop(extra)
                        repairs.append(f"{path}: rimosso campo extra vuoto '{extra}'")
                # negazione finita dentro il valore ("color eq 'non rossa'"):
                # si riporta il segno sull'operatore, dove appartiene (§5.4)
                val = p.get("value")
                if isinstance(val, str):
                    low = val.strip().lower()
                    for marker, neg_op in (("non ", "neq"), ("senza ", "excludes"),
                                           ("niente ", "excludes"), ("no ", "neq"),
                                           ("not ", "neq"), ("without ", "excludes")):
                        if low.startswith(marker) and len(low) > len(marker) + 1:
                            stripped = val.strip()[len(marker):].strip()
                            old_op = p.get("operator")
                            if old_op in ("eq", "contains"):
                                p["value"] = stripped
                                p["operator"] = neg_op if old_op == "eq" else "excludes"
                                repairs.append(
                                    f"{path}: negazione spostata dal valore "
                                    f"all'operatore ({old_op} {val!r} → "
                                    f"{p['operator']} {stripped!r})")
                            break

                # currency normalizzata
                cur = p.get("currency")
                if isinstance(cur, str) and cur.strip():
                    key = cur.strip().lower()
                    if key in CURRENCY_MAP and CURRENCY_MAP[key] != cur:
                        p["currency"] = CURRENCY_MAP[key]
                        repairs.append(f"{path}: currency '{cur}' → '{p['currency']}'")
                # stringa numerica su proprietà numerica → numero
                prop = norm_text(p.get("property"))
                if prop in NUMERIC_PROPERTIES and isinstance(p.get("value"), str):
                    s = p["value"].strip().replace(",", ".")
                    if re.fullmatch(r"-?\d+(?:\.\d+)?", s):
                        p["value"] = int(float(s)) if float(s).is_integer() else float(s)
                        repairs.append(f"{path}: valore '{s}' convertito in numero")
                # value_to spurio
                if p.get("operator") != "between" and p.get("value_to") is not None:
                    if p.get("value_to") in (None, "", 0):
                        p["value_to"] = None
                # currency su proprietà non monetaria e valore None → rimossa
                if p.get("currency") is None:
                    p.pop("currency", None)
                if p.get("unit") is None:
                    p.pop("unit", None)
                if p.get("reference") is None:
                    p.pop("reference", None)
                if p.get("value_to") is None:
                    p.pop("value_to", None)

    # --- schema
    schema_errors = sorted(VALIDATOR.iter_errors(c), key=lambda e: list(e.path))
    if schema_errors:
        for e in schema_errors[:8]:
            issues.append(Issue("schema_violation", "error", e.message,
                                "/".join(str(x) for x in e.path)))
        return GuardResult(REJECT, None, issues, repairs)

    goals = c["goals"]

    # --- invarianti per goal
    for gi, goal in enumerate(goals):
        gpath = f"goals[{gi}]"
        if user_text and not raw_action_is_grounded(goal["raw_action"], user_text):
            issues.append(Issue(
                "hallucinated_raw_action", "error",
                f"raw_action {goal['raw_action']!r} non compare nel testo "
                f"utente: usare '{IMPLICIT_ACTION}' quando non c'è un verbo",
                gpath))

        # --- gate (§11)
        gate = goal.get("gate")
        if gate is not None:
            if norm_text(goal["action"]) not in OBSERVATIONAL_ACTIONS:
                issues.append(Issue(
                    "gate_on_non_observational", "error",
                    f"un gate su '{goal['action']}' non condiziona niente: "
                    f"l'azione è già quella irreversibile (§11.3)", gpath))
            when = gate.get("when") or {}
            issues.extend(_predicate_issues(when, f"{gpath}.gate.when"))
            if ground and user_text and not value_is_grounded(
                    when.get("value"), user_text):
                issues.append(Issue(
                    "hallucinated_value", "error",
                    f"condizione del gate non ancorata: {when.get('value')!r}",
                    f"{gpath}.gate.when"))
            if ground and user_text and when.get("currency") is not None \
                    and not currency_is_grounded(when.get("currency"), user_text):
                issues.append(Issue(
                    "hallucinated_currency", "error",
                    f"valuta {when.get('currency')!r} del gate non nel testo",
                    f"{gpath}.gate.when"))
            # la condizione del gate non può essere anche un filtro (§11.3.2)
            wk = fact_key(when) if when else None
            for bucket in BUCKETS:
                for p2 in goal[bucket]:
                    if wk and fact_key(p2) == wk:
                        issues.append(Issue(
                            "gate_duplicated_in_bucket", "error",
                            f"la condizione del gate compare anche in "
                            f"'{bucket}': un fatto, un ruolo (§12)", gpath))
            if norm_text(gate.get("then")) == norm_text(goal["action"]):
                issues.append(Issue(
                    "gate_then_equals_action", "warn",
                    "il ramo 'then' ripete l'azione del goal", gpath))

        target = goal["target"]
        if norm_text(target["name"]) in ("", "un", "una", "il", "lo", "la"):
            issues.append(Issue("empty_target", "error",
                                f"target.name non informativo: {target['name']!r}", gpath))

        # predicati: validità locale
        for bucket in BUCKETS:
            for pi, p in enumerate(goal[bucket]):
                issues.extend(_predicate_issues(p, f"{gpath}.{bucket}[{pi}]"))

        # duplicati esatti dentro lo stesso bucket
        for bucket in BUCKETS:
            seen: dict[tuple, int] = {}
            keep: list[dict[str, Any]] = []
            for p in goal[bucket]:
                k = predicate_key(p)
                if k in seen:
                    repairs.append(f"{gpath}.{bucket}: rimosso duplicato esatto {k[0]}/{k[1]}")
                    continue
                seen[k] = 1
                keep.append(p)
            goal[bucket] = keep

        # stesso FATTO in bucket diversi → si tiene quello a priorità più alta
        placement: dict[tuple, str] = {}
        for bucket in sorted(BUCKETS, key=lambda b: BUCKET_PRIORITY[b]):
            keep = []
            for p in goal[bucket]:
                fk = fact_key(p)
                if fk in placement:
                    repairs.append(
                        f"{gpath}: fatto '{fk[0]}={fk[1]}' duplicato in "
                        f"'{bucket}', tenuto in '{placement[fk]}'")
                    continue
                placement[fk] = bucket
                keep.append(p)
            goal[bucket] = keep

        # conflitti diretti
        all_preds = [(b, p) for b in BUCKETS for p in goal[b]]
        for i in range(len(all_preds)):
            for j in range(i + 1, len(all_preds)):
                if _conflicting(all_preds[i][1], all_preds[j][1]):
                    issues.append(Issue(
                        "conflicting_predicates", "error",
                        f"conflitto tra {all_preds[i][1]} e {all_preds[j][1]}", gpath))

        # policy di bucket: la cascata R1-R6 di SEMANTICS.md §2.
        #
        # Si ripara SOLO se nel testo non c'è né un marcatore debole né un
        # obbligo: allora R1 e R3 sono fuori gioco e il bucket dipende solo
        # da operatore e classe della proprietà. Se un marcatore c'è, il
        # Guard lascia stare: non sa a quale fatto si riferisce.
        if user_text and not has_soft_marker(user_text) \
                and not has_obligation_marker(user_text):
            for src in BUCKETS:
                for p in list(goal[src]):
                    dst = bucket_by_cascade(p)
                    if dst == src:
                        continue
                    prop = norm_text(p.get("property"))
                    # verso attributes si scende solo su proprietà che
                    # SEMANTICS.md dichiara descrittive: il vocabolario è
                    # aperto, e di una proprietà inventata non si sa se
                    # selezioni o descriva
                    if dst == "attributes" and prop not in DESCRIPTIVE_PROPERTIES:
                        continue
                    goal[src].remove(p)
                    if predicate_key(p) not in {predicate_key(q) for q in goal[dst]}:
                        goal[dst].append(p)
                    repairs.append(
                        f"{gpath}: '{prop}' spostato da {src} a {dst} (cascata §2)")

        # negazioni mai in attributes
        for p in list(goal["attributes"]):
            if canon_predicate(p)["operator"] in NEGATIVE_OPERATORS:
                goal["attributes"].remove(p)
                if predicate_key(p) not in {predicate_key(q) for q in goal["constraints"]}:
                    goal["constraints"].append(p)
                repairs.append(
                    f"{gpath}: predicato negativo '{norm_text(p.get('property'))}' "
                    "spostato in constraints")

        # strettezza dell'operatore (diagnostico, severity=warn finché non
        # se ne misura la precisione sul benchmark gold)
        if user_text:
            for bucket in BUCKETS:
                for pi, p in enumerate(goal[bucket]):
                    op = canon_predicate(p)["operator"]
                    # `eq` DEVE essere incluso: l'errore più frequente misurato
                    # (reports/002) è proprio l'appiattimento di 'almeno 3'
                    # su `eq`. Escludendolo, il detector non scattava mai.
                    if op not in ("lt", "lte", "gt", "gte", "eq"):
                        continue
                    want = expected_strictness(user_text, canon_predicate(p)["value"])
                    if want and want != op:
                        issues.append(Issue(
                            "operator_strictness_mismatch", "error",
                            f"il testo suggerisce '{want}' ma l'operatore è '{op}' "
                            f"su '{norm_text(p.get('property'))}'",
                            f"{gpath}.{bucket}[{pi}]"))

        # polarità: un valore negato nel testo non può reggere un operatore
        # positivo ("no rosso" non è `color eq rosso`)
        if user_text:
            for bucket in BUCKETS:
                for pi, p in enumerate(goal[bucket]):
                    cp = canon_predicate(p)
                    if cp["operator"] not in _POSITIVE_OPERATORS:
                        continue
                    if negation_attached_to_value(user_text, cp["value"]):
                        issues.append(Issue(
                            "operator_polarity_mismatch", "error",
                            f"nel testo '{cp['value']}' è preceduto da una "
                            f"negazione ma l'operatore è '{cp['operator']}'",
                            f"{gpath}.{bucket}[{pi}]"))

        # ruolo del numero e parole di cortesia usate come valori
        if user_text:
            for bucket in BUCKETS:
                for pi, p in enumerate(goal[bucket]):
                    cp = canon_predicate(p)
                    path = f"{gpath}.{bucket}[{pi}]"
                    if isinstance(cp["value"], str) and \
                            cp["value"] in DISCOURSE_FILLERS:
                        issues.append(Issue(
                            "filler_as_value", "error",
                            f"'{p.get('value')}' è una parola di apertura, "
                            f"non un valore", path))
                    have = _PROPERTY_ROLE.get(cp["property"])
                    probe = cp["value"]
                    if have == "time" and time_hour(probe) is not None:
                        probe = time_hour(probe)
                    want = expected_number_role(user_text, probe)
                    if want and have and want != have:
                        issues.append(Issue(
                            "number_role_mismatch", "error",
                            f"il testo usa {cp['value']} come '{want}' ma la "
                            f"proprietà '{cp['property']}' è '{have}'", path))

        # grounding
        if ground and user_text:
            for bucket in BUCKETS:
                for pi, p in enumerate(goal[bucket]):
                    for fieldname in ("value", "value_to", "reference"):
                        v = p.get(fieldname)
                        if v is None:
                            continue
                        if not value_is_grounded(v, user_text):
                            issues.append(Issue(
                                "hallucinated_value", "error",
                                f"valore non ancorato al testo utente: "
                                f"{norm_text(p.get('property'))}.{fieldname}={v!r}",
                                f"{gpath}.{bucket}[{pi}]"))
                    cur = p.get("currency")
                    if cur is not None and not currency_is_grounded(cur, user_text):
                        issues.append(Issue(
                            "hallucinated_currency", "error",
                            f"valuta {cur!r} non nominata nel testo: nessun "
                            f"simbolo né parola di valuta (§6.3)",
                            f"{gpath}.{bucket}[{pi}]"))
            tname = target.get("name")
            # `unknown` è il segnaposto dichiarato per un target che manca
            # (§9): non è un'invenzione, è una dichiarazione di ignoranza.
            if tname and norm_text(tname) != "unknown" \
                    and not value_is_grounded(tname, user_text):
                # promosso da `warn` a `error` nell'iterazione 10: il modello
                # traduceva il target ("qualcosa da mangiare" → `food`,
                # "giovane" → `young`) e il contratto arrivava al Planner con
                # un nome che nella richiesta non c'era. Misurato: 0 falsi
                # positivi sui 102 casi gold (l'unico era un errore del gold,
                # amb-004, corretto).
                issues.append(Issue("hallucinated_target", "error",
                                    f"target.name non ancorato al testo: "
                                    f"{tname!r}", gpath))

    # --- fatti dell'utente rimasti fuori dal contratto
    #
    # Va valutato sul contratto INTERO, non goal per goal: in una richiesta
    # multi-goal il prezzo sta in un goal e il secondo goal, guardato da solo,
    # lo vedrebbe come non rendicontato. Baco trovato generando il dataset.
    if ground and check_omissions and user_text:
        for msg in unaccounted_facts(c, user_text):
            issues.append(Issue("unaccounted_fact", "error", msg, "goals"))

    # --- clarification coerente
    clar = c["clarification"]
    if not clar["required"]:
        if clar["reason"] is not None or clar["missing_fields"]:
            clar["reason"] = None
            clar["missing_fields"] = []
            repairs.append("clarification azzerata (required=false)")
    else:
        if clar["reason"] is None:
            issues.append(Issue("clarification_without_reason", "error",
                                "clarification.required=true senza reason", "clarification"))
        elif clar["reason"] not in CLARIFICATION_REASONS:
            issues.append(Issue("unknown_clarification_reason", "warn",
                                f"reason non canonica: {clar['reason']!r}", "clarification"))
        if not clar["missing_fields"]:
            issues.append(Issue("clarification_without_fields", "error",
                                "clarification.required=true senza missing_fields",
                                "clarification"))

    errors = [i for i in issues if i.severity == "error"]
    if errors:
        # errori "strutturali" → REJECT; errori semantici → CLARIFY
        hard = {"schema_violation", "empty_target", "empty_property"}
        status = REJECT if any(e.code in hard for e in errors) else CLARIFY
        return GuardResult(status, None, issues, repairs)

    status = REPAIRED if repairs else ACCEPT
    return GuardResult(status, c, issues, repairs)


def guard(raw_output: str, user_text: str = "", *, ground: bool = True) -> GuardResult:
    """Pipeline completa: testo grezzo del modello → esito del Guard."""
    try:
        contract = extract_contract(raw_output)
    except ExtractionError as exc:
        return GuardResult(REJECT, None,
                           [Issue(f"extraction:{exc.code}", "error", str(exc))], [])
    return check_contract(contract, user_text, ground=ground)


def clarification_contract(user_text: str, reason: str,
                           missing: list[str],
                           candidate: dict[str, Any] | None = None) -> dict[str, Any]:
    """Contratto di chiarimento v1.1 che conserva solo informazioni sicure.

    In precedenza ogni errore degradava a ``target=unknown`` e schema 1.0,
    anche quando il modello aveva già estratto correttamente il soggetto.
    Questo non è pericoloso (lo status resta CLARIFY) ma distrugge contesto
    utile all'UX. Qui preserviamo *solo* action/target ancorati al testo; i
    predicati vengono sempre svuotati perché sono proprio la parte che può
    aver causato il blocco.
    """
    raw = (user_text or "richiesta").strip()[:120] or "richiesta"
    action, raw_action = "find", "(implicito)"
    target = {"name": "unknown", "type": None, "raw": raw}

    if isinstance(candidate, dict):
        goals = candidate.get("goals") or []
        if goals and isinstance(goals[0], dict):
            g = goals[0]
            ra = str(g.get("raw_action") or "").strip()
            act = str(g.get("action") or "").strip()
            if ra and act and raw_action_is_grounded(ra, user_text):
                action, raw_action = act, ra

            t = g.get("target") or {}
            if isinstance(t, dict):
                name = str(t.get("name") or "").strip()
                if name and norm_text(name) != "unknown" \
                        and value_is_grounded(name, user_text):
                    traw = str(t.get("raw") or name).strip() or name
                    # `raw` è testo descrittivo: se il ritaglio del modello
                    # non è ancorato, conserviamo il nome già verificato.
                    if not value_is_grounded(traw, user_text):
                        traw = name
                    target = {"name": name, "type": t.get("type"), "raw": traw}

    return {
        "schema_version": "1.1",
        "goals": [{
            "action": action,
            "raw_action": raw_action,
            "target": target,
            "attributes": [], "constraints": [], "preferences": [],
        }],
        "clarification": {"required": True, "reason": reason,
                          "missing_fields": missing or ["goal"]},
    }
