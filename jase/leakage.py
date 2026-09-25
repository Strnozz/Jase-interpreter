"""Utility di decontaminazione e controllo leakage fra split/dataset.

Il benchmark usato per scegliere checkpoint o correggere il modello non deve
comparire letteralmente nel training. Questo modulo contiene solo funzioni
pure, così generatore e checker CLI condividono la stessa normalizzazione.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canon import strip_accents

_WS_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Example:
    text: str
    contract: dict[str, Any] | None = None
    id: str | None = None


def normalize_text(text: str) -> str:
    """Forma stabile per rilevare duplicati testuali banali.

    Ignora maiuscole, accenti e punteggiatura ma NON parafrasa: una frase
    semanticamente simile resta distinta e viene gestita dal near-check.
    """
    flat = strip_accents((text or "").lower())
    flat = _NON_WORD_RE.sub(" ", flat)
    return _WS_RE.sub(" ", flat).strip()


def token_set(text: str) -> set[str]:
    return set(normalize_text(text).split())


def token_jaccard(a: str, b: str) -> float:
    aa, bb = token_set(a), token_set(b)
    if not aa or not bb:
        return 0.0
    return len(aa & bb) / len(aa | bb)


def load_examples(path: Path) -> list[Example]:
    """Carica sia chat-jsonl sia benchmark gold Jase."""
    out: list[Example] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "messages" in row:
            user = next((m.get("content", "") for m in row["messages"]
                         if m.get("role") == "user"), "")
            assistant = next((m.get("content", "") for m in reversed(row["messages"])
                              if m.get("role") == "assistant"), "")
            contract = None
            if assistant:
                try:
                    contract = json.loads(assistant)
                except json.JSONDecodeError:
                    pass
            out.append(Example(user, contract, row.get("id")))
        else:
            out.append(Example(str(row.get("text", "")), row.get("contract"),
                               row.get("id")))
    return out


def exact_texts(rows: Iterable[Example]) -> set[str]:
    return {normalize_text(r.text) for r in rows if normalize_text(r.text)}


def decontaminate_exact(rows: list[tuple[str, Any, Any]],
                        against: Iterable[Example]) -> tuple[list[tuple[str, Any, Any]], list[str]]:
    """Rimuove da tuple ``(text, contract, family)`` gli exact duplicate."""
    forbidden = exact_texts(against)
    kept: list[tuple[str, Any, Any]] = []
    removed: list[str] = []
    for row in rows:
        if normalize_text(row[0]) in forbidden:
            removed.append(row[0])
        else:
            kept.append(row)
    return kept, removed
