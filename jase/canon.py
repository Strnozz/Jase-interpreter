"""Estrazione e canonicalizzazione dei GoalContract.

Due responsabilità:
  1. tirare fuori UN oggetto JSON dall'output grezzo del modello, in modo
     onesto: se l'output è degenerato o troncato lo si dichiara, non lo si
     ripara silenziosamente prendendo la prima sottostringa plausibile;
  2. portare un contratto in forma canonica, così che confronti e ricerche
     di duplicati ignorino le differenze cosmetiche.
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any

from .lexicon import (
    ARTICLES, CURRENCY_MAP, LEADING_PREPOSITIONS, NUMBER_WORDS,
    UNIT_ALIASES, UNIT_DISPLAY,
)


class ExtractionError(Exception):
    """L'output del modello non contiene un GoalContract utilizzabile."""

    def __init__(self, message: str, code: str, raw: str = ""):
        super().__init__(message)
        self.code = code
        self.raw = raw


# --------------------------------------------------------------- estrazione

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def _strip_fences(text: str) -> str:
    text = _FENCE_RE.sub("", text.strip())
    return text.strip()


def _scan_balanced(text: str, start: int) -> int:
    """Indice (esclusivo) della chiusura del primo oggetto JSON a `start`.

    Ritorna -1 se l'oggetto non si chiude mai (output troncato).
    Rispetta le stringhe e gli escape, così una `}` dentro una stringa non
    chiude l'oggetto per sbaglio.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def detect_degeneration(text: str, min_repeats: int = 4, window: int = 240) -> str | None:
    """Rileva loop ripetitivi tipici della degenerazione (es. `"preferences":` a ripetizione).

    Ritorna una descrizione del problema, oppure None.
    """
    if not text:
        return None
    # 1) chiave JSON ripetuta troppe volte
    keys = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', text)
    if keys:
        counts: dict[str, int] = {}
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        # `property`/`operator`/`value` sono legittimamente ripetuti (un
        # predicato per volta), le chiavi di primo livello no.
        for k in ("schema_version", "clarification", "preferences",
                  "constraints", "attributes", "goals", "target",
                  "raw_action", "missing_fields"):
            if counts.get(k, 0) > 40:
                return f"repeated_key:{k}x{counts[k]}"
        # property/operator/value si ripetono legittimamente (uno per
        # predicato), ma oltre una certa soglia è un loop
        for k in ("property", "operator", "value"):
            if counts.get(k, 0) > 45:
                return f"repeated_key:{k}x{counts[k]}"
    # 2) n-gramma di caratteri ripetuto consecutivamente
    tail = text[-window * 4:]
    # Si cerca una ripetizione consecutiva OVUNQUE nella coda, non solo
    # esattamente in fondo: un output degenerato spesso finisce con qualche
    # carattere di chiusura ("...},]}]}") che sfaserebbe un confronto ancorato.
    for size in range(6, 161):
        if len(tail) < size * min_repeats:
            break
        run = 1
        i = 0
        limit = len(tail) - 2 * size
        while i <= limit:
            if tail[i:i + size] == tail[i + size:i + 2 * size]:
                run += 1
                if run >= min_repeats:
                    return f"repeated_ngram:{size}x{run}"
                i += size
            else:
                run = 1
                i += 1
    return None


def extract_contract(text: str) -> dict[str, Any]:
    """Estrae l'unico oggetto JSON atteso. Fail-closed su output sospetto."""
    if text is None:
        raise ExtractionError("output vuoto", "empty_output", "")
    cleaned = _strip_fences(text)
    if not cleaned:
        raise ExtractionError("output vuoto", "empty_output", text)

    degen = detect_degeneration(cleaned)
    if degen is not None:
        raise ExtractionError(f"generazione degenerata ({degen})", "degenerate", text)

    start = cleaned.find("{")
    if start < 0:
        raise ExtractionError("nessun oggetto JSON nell'output", "no_json", text)

    end = _scan_balanced(cleaned, start)
    if end < 0:
        raise ExtractionError("JSON troncato: oggetto mai chiuso", "truncated", text)

    candidate = cleaned[start:end]
    trailing = cleaned[end:].strip()
    if trailing.startswith("{"):
        raise ExtractionError("più di un oggetto JSON nell'output", "multiple_objects", text)
    if trailing and not re.fullmatch(r"[\s`]*", trailing):
        raise ExtractionError(f"testo extra dopo il JSON: {trailing[:60]!r}", "trailing_text", text)

    try:
        obj = json.loads(candidate, object_pairs_hook=_reject_duplicate_keys)
    except _DuplicateKey as exc:
        raise ExtractionError(f"chiave JSON duplicata: {exc.key}", "duplicate_key", text) from exc
    except json.JSONDecodeError as exc:
        raise ExtractionError(f"JSON non valido: {exc.msg}", "invalid_json", text) from exc

    if not isinstance(obj, dict):
        raise ExtractionError("il JSON non è un oggetto", "not_an_object", text)
    return obj


class _DuplicateKey(Exception):
    def __init__(self, key: str):
        super().__init__(key)
        self.key = key


def _reject_duplicate_keys(pairs):
    seen = set()
    for key, _ in pairs:
        if key in seen:
            raise _DuplicateKey(key)
        seen.add(key)
    return dict(pairs)


# --------------------------------------------------- canonicalizzazione testo

def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm_text(s: Any) -> str:
    """Forma comparabile di una stringa: minuscola, senza accenti, senza
    articoli iniziali, spazi normalizzati, punteggiatura di bordo rimossa."""
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = strip_accents(s)
    s = re.sub(r"[^\w\s\|\.\-/:]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    # preposizione e articolo iniziali cadono, ma solo finché resta qualcosa:
    # "in pelle" → "pelle", "a pool" → "pool", "di" → "di"
    while len(tokens) > 1 and (tokens[0] in ARTICLES
                               or tokens[0] in LEADING_PREPOSITIONS):
        tokens.pop(0)
    while len(tokens) > 1 and tokens[0] in ARTICLES:
        tokens.pop(0)
    if len(tokens) == 1 and tokens[0] in ARTICLES:
        tokens = []
    return " ".join(tokens)


def norm_value(v: Any) -> Any:
    """Valore in forma canonica: i numeri restano numeri, le stringhe
    numeriche diventano numeri, il resto passa da `norm_text`."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", s):
        return float(s.replace(",", "."))
    word = strip_accents(s.lower())
    if word in NUMBER_WORDS:
        return float(NUMBER_WORDS[word])
    return norm_text(s)


def norm_currency(c: Any) -> str | None:
    if c is None:
        return None
    key = str(c).strip().lower()
    return CURRENCY_MAP.get(key, str(c).strip().upper() or None)


def norm_unit(u: Any) -> str | None:
    """Unità in forma canonica (§6.2): il vocabolario è chiuso e inglese."""
    if u is None:
        return None
    s = str(u).strip().lower()
    if not s:
        return None
    return UNIT_ALIASES.get(s, s)


def display_unit(u: Any) -> str | None:
    """Grafia canonica di un'unità, per ciò che si SCRIVE nel dataset."""
    n = norm_unit(u)
    return UNIT_DISPLAY.get(n, n) if n else n


# ------------------------------------------------- canonicalizzazione oggetti

PREDICATE_FIELDS = ("property", "operator", "value", "value_to", "unit",
                    "currency", "reference")


def _sort_key(d: dict[str, Any]) -> tuple:
    return tuple(str(d.get(f)) for f in PREDICATE_FIELDS)


def canon_predicate(p: dict[str, Any]) -> dict[str, Any]:
    return {
        "property": norm_text(p.get("property")),
        "operator": str(p.get("operator") or "").strip().lower(),
        "value": norm_value(p.get("value")),
        "value_to": norm_value(p.get("value_to")) if p.get("value_to") is not None else None,
        "unit": norm_unit(p.get("unit")),
        "currency": norm_currency(p.get("currency")),
        "reference": norm_text(p.get("reference")) or None,
    }


def predicate_key(p: dict[str, Any]) -> tuple:
    """Chiave di identità semantica di un predicato (per i duplicati)."""
    c = canon_predicate(p)
    return (c["property"], c["operator"], c["value"], c["value_to"],
            c["unit"], c["currency"], c["reference"])


def fact_key(p: dict[str, Any]) -> tuple:
    """Chiave del FATTO sottostante, senza l'operatore.

    Serve a individuare lo stesso fatto messo in due bucket diversi con
    operatori equivalenti (§12 di SEMANTICS.md)."""
    c = canon_predicate(p)
    return (c["property"], c["value"], c["unit"], c["currency"])


def _sorted_preds(raw) -> list[dict[str, Any]]:
    preds = [canon_predicate(p) for p in (raw or []) if isinstance(p, dict)]
    return sorted(preds, key=_sort_key)


def canon_gate(gate: Any) -> dict[str, Any] | None:
    """Forma canonica di un `gate` (§11 di SEMANTICS.md)."""
    if not isinstance(gate, dict):
        return None
    when = gate.get("when")
    return {
        "when": canon_predicate(when) if isinstance(when, dict) else None,
        "then": norm_text(gate.get("then")) or None,
        "otherwise": norm_text(gate.get("otherwise")) or None,
    }


def canon_goal(g: dict[str, Any]) -> dict[str, Any]:
    target = g.get("target") or {}
    return {
        "gate": canon_gate(g.get("gate")),
        "action": norm_text(g.get("action")),
        "target_name": norm_text(target.get("name")),
        "target_type": norm_text(target.get("type")) or None,
        "attributes": _sorted_preds(g.get("attributes")),
        "constraints": _sorted_preds(g.get("constraints")),
        "preferences": _sorted_preds(g.get("preferences")),
    }


def canon_contract(c: dict[str, Any]) -> dict[str, Any]:
    clar = c.get("clarification") or {}
    return {
        "schema_version": c.get("schema_version"),
        "goals": sorted(
            (canon_goal(g) for g in c.get("goals") or [] if isinstance(g, dict)),
            key=lambda cg: (cg["target_name"], cg["action"], str(cg["constraints"]))),
        "clarification": {
            "required": bool(clar.get("required")),
            "reason": norm_text(clar.get("reason")) or None,
            "missing_fields": sorted(norm_text(x) for x in clar.get("missing_fields") or []),
        },
    }


