"""Deterministic dataset checks; no model or benchmark-specific rules."""
from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict

from .canon import canon_contract


def text_key(text: str) -> str:
    """Preserve semantic symbols: '<10', '>10' and '10€' are distinct."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def contract_key(contract: dict) -> str:
    return json.dumps(canon_contract(contract), sort_keys=True, ensure_ascii=False)


def tags(contract: dict) -> set[str]:
    out = set()
    goals = contract.get("goals", [])
    if contract.get("clarification", {}).get("required"):
        out.add("clarification")
    if len(goals) > 1:
        out.add("multi-goal")
    for goal in goals:
        if goal.get("gate"):
            out.add("gate")
        if goal.get("preferences"):
            out.add("preference")
        if goal.get("raw_action") == "(implicito)":
            out.add("implicit-action")
        predicates = [p for b in ("attributes", "constraints", "preferences")
                      for p in goal.get(b, [])]
        if len(predicates) >= 3:
            out.add("three-plus-facts")
        if any(p.get("operator") in ("neq", "excludes", "not_in") for p in predicates):
            out.add("negation")
    return out


def clean_rows(rows: list[tuple[str, dict, str]]) -> tuple[list, dict, list]:
    """Union families BEFORE deduplication; quarantine all conflicting labels.

    Duplicate text may originate from two generator branches. Discarding one
    before connecting those branches would lose the split dependency.
    """
    parents = {family: family for _, _, family in rows}

    def find(family):
        while parents[family] != family:
            parents[family] = parents[parents[family]]
            family = parents[family]
        return family

    by_text = defaultdict(list)
    for row in rows:
        by_text[text_key(row[0])].append(row)
    for group in by_text.values():
        root = find(group[0][2])
        for _, _, family in group[1:]:
            other = find(family)
            if root != other:
                smaller, larger = sorted((root, other))
                parents[larger] = smaller
                root = smaller

    kept, quarantine = [], []
    duplicates = 0
    conflicts = 0
    for group in by_text.values():
        if len({contract_key(contract) for _, contract, _ in group}) > 1:
            conflicts += 1
            quarantine.extend(group)
            continue
        text, contract, family = group[0]
        kept.append((text, contract, "family-" + digest(find(family))[:24]))
        duplicates += len(group) - 1
    return kept, {"input_rows": len(rows), "duplicate_rows_removed": duplicates,
                  "conflicting_prompts": conflicts,
                  "conflicting_rows_quarantined": len(quarantine),
                  "unique_rows": len(kept)}, quarantine


def grouped_split(rows: list, seed: int, valid_frac: float = 0.08):
    """Keep components intact, stratifying clarification, multi-goal and gates."""
    if not 0 < valid_frac < 1:
        raise ValueError("valid_frac must be between 0 and 1")
    groups = defaultdict(list)
    for row in rows:
        groups[row[2]].append(row)
    strata = defaultdict(list)
    for family, group in sorted(groups.items()):
        feats = set().union(*(tags(c) for _, c, _ in group))
        key = tuple(f in feats for f in ("clarification", "multi-goal", "gate"))
        strata[key].append(family)
    rng = random.Random(seed)
    valid_families = set()
    for families in strata.values():
        rng.shuffle(families)
        if len(families) < 2:
            continue
        target = sum(len(groups[f]) for f in families) * valid_frac
        chosen, size = [], 0
        for family in families:
            n = len(groups[family])
            if abs(size + n - target) < abs(size - target):
                chosen.append(family)
                size += n
        if not chosen:
            chosen = [min(families, key=lambda f: abs(len(groups[f]) - target))]
        if len(chosen) == len(families):
            chosen.pop()
        valid_families.update(chosen)
    train = [row for row in rows if row[2] not in valid_families]
    valid = [row for row in rows if row[2] in valid_families]
    rng.shuffle(train)
    rng.shuffle(valid)
    return train, valid


def audit_splits(train: list, valid: list) -> dict:
    report = {}
    for name, rows in (("train", train), ("valid", valid)):
        seen = defaultdict(set)
        counts = Counter()
        for text, contract, _ in rows:
            seen[text_key(text)].add(contract_key(contract))
            counts.update(tags(contract))
        report[name] = {"rows": len(rows), "unique_prompts": len(seen),
                        "duplicate_rows": len(rows) - len(seen),
                        "conflicting_prompts": sum(len(v) > 1 for v in seen.values()),
                        "families": len({r[2] for r in rows}),
                        "tags": dict(sorted(counts.items()))}
    report["shared_prompts"] = len({text_key(r[0]) for r in train} &
                                    {text_key(r[0]) for r in valid})
    report["shared_families"] = len({r[2] for r in train} & {r[2] for r in valid})
    report["missing_validation_tags"] = sorted(
        set(report["train"]["tags"]) - set(report["valid"]["tags"]))
    report["passed"] = bool(train and valid) and not (
        report["shared_prompts"] or report["shared_families"] or
        report["missing_validation_tags"] or any(
            report[s]["duplicate_rows"] or report[s]["conflicting_prompts"]
            for s in ("train", "valid")))
    return report


def select_dev(rows: list, n: int, seed: int, preferred: set[str] | None = None) -> list:
    """A fixed development panel with coverage, selected before inference."""
    if len(rows) < n:
        raise ValueError(f"Only {len(rows)} eligible dev rows; need {n}")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    if preferred is not None:
        shuffled.sort(key=lambda r: r[0] not in preferred)
    chosen, used = [], set()
    # Ten examples per available feature, then fill by seeded sampling.
    for tag in ("clarification", "multi-goal", "gate", "negation", "preference",
                "implicit-action", "three-plus-facts"):
        candidates = [r for r in shuffled if tag in tags(r[1])]
        if not candidates:
            raise ValueError(f"No eligible dev examples for {tag}")
        for row in candidates[:10]:
            if row[0] not in used and len(chosen) < n:
                chosen.append(row)
                used.add(row[0])
    for row in shuffled:
        if row[0] not in used and len(chosen) < n:
            chosen.append(row)
            used.add(row[0])
    return chosen
