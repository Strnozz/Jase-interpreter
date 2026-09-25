#!/usr/bin/env python3
"""Suite di regressione del Semantic Guard e dell'evaluator.

Non richiede MLX né il modello: gira ovunque ci sia Python + jsonschema.
    python tests/test_guard.py

Ogni bug trovato durante lo sviluppo diventa un caso qui, PRIMA di essere
considerato risolto (§29 della specifica).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.canon import ExtractionError, extract_contract  # noqa: E402
from jase.guard import (  # noqa: E402
    ACCEPT, CLARIFY, REJECT, REPAIRED,
    bucket_by_cascade, check_contract, clarification_contract, expected_strictness, guard,
    negation_attached_to_value, time_hour, value_is_grounded,
)
from jase.metrics import compare  # noqa: E402

PASSED = FAILED = 0
FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
    else:
        FAILED += 1
        FAILURES.append(f"{name}: {detail}")


def contract(goals, clar=None):
    return {"schema_version": "1.1", "goals": goals,
            "clarification": clar or {"required": False, "reason": None,
                                      "missing_fields": []}}


def goal(name="panino", ttype="food", raw="un panino", action="find",
         raw_action="(implicito)", A=None, C=None, P=None):
    # default "(implicito)": così le fixture non inciampano nel controllo di
    # ancoraggio di raw_action, che ha un test dedicato più sotto
    return {"action": action, "raw_action": raw_action,
            "target": {"name": name, "type": ttype, "raw": raw},
            "attributes": A or [], "constraints": C or [], "preferences": P or []}


def pr(prop, op, value, **kw):
    d = {"property": prop, "operator": op, "value": value}
    d.update(kw)
    return d


# ===================================================== estrazione robusta

def test_extraction():
    ok = '{"schema_version":"1.0","goals":[],"clarification":{}}'
    check("extract/ok", isinstance(extract_contract(ok), dict))
    check("extract/fences",
          isinstance(extract_contract("```json\n" + ok + "\n```"), dict))

    for raw, code in [
        ('{"a":1', "truncated"),
        ("nessun json qui", "no_json"),
        ('{"a":1}{"b":2}', "multiple_objects"),
        ('{"a":1} ecco il contratto', "trailing_text"),
        ('{"a":1,"a":2}', "duplicate_key"),
        ("", "empty_output"),
    ]:
        try:
            extract_contract(raw)
            check(f"extract/{code}", False, "nessuna eccezione")
        except ExtractionError as e:
            check(f"extract/{code}", e.code == code, f"code={e.code}")

    # degenerazione: loop ripetitivo (il bug osservato con iter300)
    loop = '{"goals":[{"preferences":[' + '{"property":"x","operator":"eq","value":1},' * 60 + "]}]}"
    try:
        extract_contract(loop)
        check("extract/degenerate", False, "loop non rilevato")
    except ExtractionError as e:
        check("extract/degenerate", e.code == "degenerate", f"code={e.code}")


# ===================================================== invarianti del Guard

def test_duplicates_and_buckets():
    # il fallimento storico di iter400: prezzo in due bucket, location in attributes
    c = contract([goal(
        A=[pr("location", "near", "Milano"),
           pr("price", "lt", 10, currency="EUR")],
        C=[pr("price", "lt", 10, currency="EUR")])])
    r = check_contract(c, "Trovami un panino a Milano sotto i 10 euro")
    check("guard/iter400-repair", r.status == REPAIRED, r.status)
    g = r.contract["goals"][0]
    check("guard/no-dup-price",
          sum(1 for b in ("attributes", "constraints", "preferences")
              for p in g[b] if p["property"] == "price") == 1)
    check("guard/location-in-constraints",
          any(p["property"] == "location" for p in g["constraints"]))
    check("guard/attributes-empty", g["attributes"] == [], g["attributes"])

    # duplicato esatto nello stesso bucket
    c = contract([goal(C=[pr("price", "lt", 10, currency="EUR"),
                          pr("price", "lt", 10, currency="EUR")])])
    r = check_contract(c, "un panino sotto i 10 euro")
    check("guard/exact-dup", len(r.contract["goals"][0]["constraints"]) == 1)

    # negazione mai in attributes
    c = contract([goal(A=[pr("ingredient", "excludes", "mortadella")])])
    r = check_contract(c, "un panino senza mortadella")
    check("guard/negation-moved",
          r.contract["goals"][0]["constraints"] and not r.contract["goals"][0]["attributes"])


def test_hallucination():
    c = contract([goal("hotel", "accommodation", "un hotel", "book", "prenotami",
                       C=[pr("location", "near", "Roma"),
                          pr("guests", "eq", 2),
                          pr("time", "eq", "20:00")])])
    r = check_contract(c, "Cercami un hotel a Roma")
    check("guard/hallucination-blocks", r.status == CLARIFY, r.status)
    check("guard/hallucination-codes",
          r.error_codes().count("hallucinated_value") == 2, r.error_codes())

    # 'stasera' NON si risolve in un orario
    c = contract([goal("elettricista", "professional_service", "un elettricista",
                       C=[pr("time", "eq", "20:00")])])
    r = check_contract(c, "Mi serve un elettricista stasera")
    check("guard/stasera-not-2000", r.status == CLARIFY, r.status)

    # 'dieci euro' → 10 è normalizzazione ammessa, non invenzione
    c = contract([goal(C=[pr("price", "eq", 10, currency="EUR")])])
    r = check_contract(c, "Trovami un panino per dieci euro")
    check("guard/number-word-grounded", r.ok, r.status)


def test_raw_action_grounding():
    """Regressione da reports/005: su testi senza verbo il modello inventava
    il raw_action ('avrei bisogno di' da 'Un hotel a 4 stelle a Bologna')."""
    c = contract([goal("hotel", "accommodation", "un hotel",
                       raw_action="avrei bisogno di")])
    r = check_contract(c, "Un hotel a 4 stelle a Bologna")
    check("rawaction/blocked", r.status == CLARIFY, r.status)
    check("rawaction/code",
          "hallucinated_raw_action" in r.error_codes(), r.error_codes())

    # fixture minimali: qui si verifica solo l'ancoraggio di raw_action
    for ra, text, ok_expected in [
        ("(implicito)", "Un hotel a 4 stelle a Bologna", True),
        ("trovami", "Trovami un panino", True),
        ("trova", "Trovami un panino", True),
        ("mi serve", "Mi serve un elettricista", True),
        ("paragona", "Un hard disk da 2 TB", False),
        ("ordinami", "panino mortadella milano", False),
    ]:
        c = contract([goal("x", None, "x", raw_action=ra)])
        r = check_contract(c, text, check_omissions=False)
        check(f"rawaction/{ra}", r.ok == ok_expected, f"{r.status} su {text!r}")


def test_currency_grounding():
    """Regressione da reports/002: 'Prenotami un tavolo alle 20' produceva
    price eq 20 EUR e nulla lo fermava. La valuta va ancorata (§6.3)."""
    c = contract([goal("tavolo", "place", "un tavolo", "book", "prenotami",
                       C=[pr("price", "eq", 20, currency="EUR")])])
    r = check_contract(c, "Prenotami un tavolo alle 20")
    check("currency/alle-20-blocked", r.status == CLARIFY, r.status)
    check("currency/code", "hallucinated_currency" in r.error_codes(), r.error_codes())

    # con l'ancora esplicita passa
    c = contract([goal(C=[pr("price", "lt", 10, currency="EUR")])])
    check("currency/euro-ok", check_contract(c, "un panino sotto i 10 euro").ok)
    c = contract([goal(C=[pr("price", "lt", 10, currency="EUR")])])
    check("currency/simbolo-ok", check_contract(c, "un panino sotto i 10€").ok)
    c = contract([goal(C=[pr("price", "lt", 900, currency="USD")])])
    check("currency/usd-ok", check_contract(c, "un panino sotto i 900 dollari").ok)
    # valuta sbagliata rispetto al testo
    c = contract([goal(C=[pr("price", "lt", 900, currency="USD")])])
    check("currency/mismatch",
          not check_contract(c, "un panino sotto i 900 euro").ok)
    # niente valuta dichiarata: nessun problema
    c = contract([goal(C=[pr("price", "lt", 10)])])
    check("currency/null-ok",
          check_contract(c, "un panino, vorrei restare sotto i dieci").ok)


def test_negation_in_value():
    """Regressione da reports/002: il modello produceva color eq 'non rossa'
    invece di color neq 'rossa'. Riparazione deterministica (§5.4)."""
    c = contract([goal("auto", "product", "un'auto",
                       A=[pr("color", "eq", "non rossa")])])
    r = check_contract(c, "Trovami un'auto non rossa")
    check("negval/repaired", r.status == REPAIRED, r.status)
    preds = r.contract["goals"][0]["constraints"]
    check("negval/operator", preds and preds[0]["operator"] == "neq", preds)
    check("negval/value", preds and preds[0]["value"] == "rossa", preds)

    c = contract([goal("pizza", "food", "una pizza",
                       A=[pr("diet", "contains", "senza glutine")])])
    r = check_contract(c, "una pizza senza glutine")
    preds = r.contract["goals"][0]["constraints"] if r.contract else []
    check("negval/excludes", preds and preds[0]["operator"] == "excludes", preds)


def test_predicate_validity():
    for preds, expect_code in [
        ([pr("price", "lt", "economico", currency="EUR")], "non_numeric_money"),
        ([pr("price", "between", 10, currency="EUR")], "between_without_value_to"),
        ([pr("color", "eq", "rosso", value_to=5)], "value_to_without_between"),
        ([pr("color", "eq", "rosso", currency="EUR")], "currency_on_non_monetary"),
    ]:
        c = contract([goal(C=preds)])
        r = check_contract(c, "un panino rosso economico tra 10 e 5 euro", ground=False)
        check(f"guard/{expect_code}", expect_code in r.error_codes(), r.error_codes())

    # conflitto diretto
    c = contract([goal(C=[pr("price", "lt", 10, currency="EUR"),
                          pr("price", "gt", 20, currency="EUR")])])
    r = check_contract(c, "sotto i 10 euro e oltre i 20 euro")
    check("guard/conflict", "conflicting_predicates" in r.error_codes(), r.error_codes())


def test_schema_violations():
    # campo inventato dentro target (fallimento storico)
    bad = {"schema_version": "1.0",
           "goals": [{"action": "book", "raw_action": "prenotami",
                      "target": {"name": "hotel", "type": "accommodation",
                                 "raw": "un hotel", "city": "Roma"},
                      "attributes": [], "constraints": [], "preferences": []}],
           "clarification": {"required": False, "reason": None, "missing_fields": []}}
    r = check_contract(bad, "Prenotami un hotel a Roma")
    check("guard/extra-target-field", r.status == REJECT, r.status)

    # predicato come stringa invece che oggetto
    bad = contract([goal(C=["a Roma"])])
    r = check_contract(bad, "Prenotami un hotel a Roma")
    check("guard/string-predicate", r.status == REJECT, r.status)

    # predicato malformato {"property":"Roma","type":"location"}
    bad = contract([goal(C=[{"property": "Roma", "type": "location"}])])
    r = check_contract(bad, "Prenotami un hotel a Roma")
    check("guard/malformed-predicate", r.status == REJECT, r.status)


def test_clarification_coherence():
    c = contract([goal()], clar={"required": False, "reason": "target_missing",
                                 "missing_fields": ["target"]})
    r = check_contract(c, "un panino")
    check("guard/clar-reset", r.contract["clarification"]["reason"] is None)

    c = contract([goal()], clar={"required": True, "reason": None,
                                 "missing_fields": []})
    r = check_contract(c, "un panino")
    check("guard/clar-no-reason",
          "clarification_without_reason" in r.error_codes(), r.error_codes())


def test_strictness():
    cases = [
        ("Trovami un panino sotto i 10 euro", 10, "lt"),
        ("un elettricista entro 100 euro massimo", 100, "lte"),
        ("hotel da almeno 3 stelle", 3, "gte"),
        ("un hotel 4 stelle", 4, None),
        ("sono a Verbania, cento euro proprio massimo", 100, "lte"),
        ("non voglio farmi più di 80 km", 80, "lte"),
        ("un hard disk da 2 TB", 2, None),
        ("max 500", 500, "lte"),
        ("meno di 600", 600, "lt"),
        ("sotto i sei mesi", 6, "lt"),
    ]
    # falsi positivi scoperti generando il dataset: ognuno è un modo diverso
    # in cui un marcatore si attribuisce al numero sbagliato
    cases += [
        # "Buongiorno" tagliato dalla finestra lasciava "no," = negazione
        ("Buongiorno, prenota un traghetto, sotto i 500€", 500, "lt"),
        # "o meno" annidato dentro "sabatO MENO di 250"
        ("Una casa vacanze da 2 persone idealmente sabato meno di 250 eur", 2, None),
        ("Una casa vacanze da 2 persone idealmente sabato meno di 250 eur", 250, "lt"),
        # marcatore suffisso che ha già il suo numero
        ("preferibilmente 24 pollici max 200", 24, None),
        # un marcatore stretto fra due numeri tace, anche quando per un umano
        # l'attribuzione sarebbe chiara: qui si perde una rilevazione, ma
        # questo controllo BLOCCA contratti e la precisione conta di più
        ("preferibilmente 24 pollici max 200", 200, None),
        # modi di dire italiani del rifiuto: tetto STRETTO, non un prezzo
        ("dieci euro per un panino proprio non li spendo", 10, "lt"),
        ("Dieci euro però non li spendo.", 10, "lt"),
        ("cerco una bici. 500 euro non li spendo", 500, "lt"),
        ("senza arrivare a spendere 10€", 10, "lt"),
        ("Vorrei restare sotto i dieci.", 10, "lt"),
        # ma un prezzo normale resta senza verdetto
        ("Trovami un panino a Milano per dieci euro", 10, None),
        ("Prenotami un tavolo alle dieci", 10, None),
        ("500 euro non li spendo, ma 300 sì", 300, None),
        # marcatore prefisso già consumato da un altro numero
        ("Cercami una chitarra sotto i 300 euro e confrontami tre amplificatori",
         3, None),
        ("Un volo il 3 per 2 persone sotto i 3 scali", 3, None),
    ]
    for text, val, want in cases:
        got = expected_strictness(text, val)
        check(f"strict/{text[:24]}|{val}", got == want, f"{got} != {want}")


def test_strictness_blocking():
    """Regressione da reports/002: l'errore più frequente era 'almeno 3
    stelle' appiattito su `stars eq 3`, e il Guard non lo vedeva perché il
    controllo escludeva l'operatore `eq`."""
    def one(text, pred):
        # fixture a un predicato: il rilevatore di omissioni segnalerebbe
        # giustamente gli altri fatti della frase, ma qui si testa altro
        return check_contract(contract([goal("x", None, "x", A=[pred])]), text,
                              check_omissions=False)

    blocca = [
        ("Prenotami un hotel da almeno 3 stelle", pr("stars", "eq", 3)),
        ("Cercami una bici sotto i 500 euro", pr("price", "lte", 500, currency="EUR")),
        ("Cercami un MacBook con almeno 32 GB di RAM",
         pr("ram", "eq", 32, unit="GB")),
        ("Un divano massimo 800 euro", pr("price", "eq", 800, currency="EUR")),
        ("Un appartamento di almeno 80 mq", pr("size", "eq", 80, unit="mq")),
        ("una bici sotto i 500 euro", pr("price", "lte", 500, currency="EUR")),
    ]
    passa = [
        ("Prenotami un hotel da almeno 3 stelle", pr("stars", "gte", 3)),
        ("Un hotel a 4 stelle a Bologna", pr("stars", "eq", 4)),
        ("Un hard disk da 2 TB", pr("storage", "eq", 2, unit="TB")),
        ("Prenotami un tavolo per 4 alle 21", pr("guests", "eq", 4)),
        ("Voglio una Fiat 500 del 2020", pr("year", "eq", 2020)),
        # marcatore già consumato da un altro numero: non deve scattare
        ("Cercami una chitarra sotto i 300 euro e confrontami tre amplificatori",
         pr("quantity", "eq", 3)),
        ("Un volo il 3 per 2 persone sotto i 3 scali", pr("date", "eq", 3)),
        ("Cercami un MacBook con almeno 32 GB di RAM",
         pr("ram", "gte", 32, unit="GB")),
    ]
    for text, pred in blocca:
        check(f"strictblock/{text[:22]}", not one(text, pred).ok,
              one(text, pred).status)
    for text, pred in passa:
        r = one(text, pred)
        check(f"strictpass/{text[:22]}", r.ok, f"{r.status} {r.error_codes()}")


def test_number_role_and_fillers():
    """Regressione da reports/014: due allucinazioni che l'ancoraggio non
    poteva vedere, perché i valori erano davvero nel testo."""
    from jase.guard import expected_number_role

    for text, val, want in [
        ("Prenotami un tavolo alle 20", 20, "time"),
        ("Prenotami un tavolo alle dieci", 10, "time"),
        ("Cercami un idraulico entro 20 euro", 20, "money"),
        ("Cercami un idraulico entro 20 km", 20, "distance"),
        ("un tavolo per 4 persone", 4, "quantity"),
        ("un hotel da almeno 3 stelle", 3, "rating"),
        ("un hard disk da 2 TB", 2, None),
        ("Prenotami un tavolo per 4 alle 21 sotto i 30 euro", 21, "time"),
        ("Prenotami un tavolo per 4 alle 21 sotto i 30 euro", 30, "money"),
    ]:
        got = expected_number_role(text, val)
        check(f"numrole/{text[:20]}|{val}", got == want, f"{got} != {want}")

    # "alle 20" non può diventare né ospiti né prezzo
    for pred in (pr("guests", "eq", 20), pr("price", "eq", 20, currency="EUR")):
        c = contract([goal("tavolo", "place", "un tavolo", "book", "prenotami",
                           C=[pred])])
        r = check_contract(c, "Prenotami un tavolo alle 20")
        check(f"numrole/block-{pred['property']}", not r.ok, r.status)

    # il caso giusto passa
    c = contract([goal("tavolo", "place", "un tavolo", "book", "prenotami",
                       C=[pr("time", "eq", "20:00")])])
    check("numrole/time-ok", check_contract(c, "Prenotami un tavolo alle 20").ok)

    # "entro 20 km" non è un prezzo, "entro 20 euro" non è una distanza
    c = contract([goal(C=[pr("price", "lte", 20, currency="EUR")])])
    check("numrole/km-not-price",
          not check_contract(c, "Cercami un idraulico entro 20 km").ok)
    c = contract([goal(C=[pr("distance", "within", 20, unit="km")])])
    check("numrole/euro-not-distance",
          not check_contract(c, "Cercami un idraulico entro 20 euro").ok)

    # parole di cortesia usate come valori
    c = contract([goal("pizza", "food", "una pizza",
                       A=[pr("ingredient", "contains", "senti")])])
    r = check_contract(c, "ciao! senti, mi servirebbe una pizza")
    check("filler/blocked", not r.ok, r.status)
    check("filler/code", "filler_as_value" in r.error_codes(), r.error_codes())


def test_grounding_helpers():
    check("ground/number", value_is_grounded(10, "sotto i 10 euro"))
    check("ground/number-word", value_is_grounded(10, "per dieci euro"))
    check("ground/absent", not value_is_grounded(42, "sotto i 10 euro"))
    check("ground/inflection", value_is_grounded("rosso", "un'auto rossa"))
    check("ground/alias-sex", value_is_grounded("female", "uno Sphynx femmina"))
    check("ground/invented", value_is_grounded("Blorch", "una bici color Blorch"))
    check("ground/list", value_is_grounded("Ryanair|Wizzair",
                                           "né Ryanair né Wizzair"))
    check("ground/hallucinated", not value_is_grounded("piscina", "un hotel a Roma"))


# ========================================================== metriche v2

def test_metrics():
    exp = contract([goal(
        A=[pr("ingredient", "contains", "mortadella")],
        C=[pr("location", "near", "Milano"),
           pr("price", "lt", 10, currency="EUR")])])
    # bucket sbagliato = errore semantico
    got = contract([goal(
        A=[pr("ingredient", "contains", "mortadella"),
           pr("price", "lt", 10, currency="EUR")],
        C=[pr("location", "near", "Milano")])])
    m = compare(exp, got, "Trovami un panino con la mortadella a Milano sotto i 10 euro")
    check("metrics/bucket-error", m["bucket_ok"] == 2 and m["bucket_total"] == 3,
          f"{m['bucket_ok']}/{m['bucket_total']}")
    check("metrics/not-exact", m["semantic_exact"] == 0)

    # differenza solo cosmetica (target.raw) = NON errore semantico
    got2 = json.loads(json.dumps(exp))
    got2["goals"][0]["target"]["raw"] = "panino"
    m2 = compare(exp, got2, "x")
    check("metrics/cosmetic-ok", m2["semantic_exact"] == 1, m2["errors"])

    # operatore sbagliato
    got3 = json.loads(json.dumps(exp))
    got3["goals"][0]["constraints"][1]["operator"] = "lte"
    m3 = compare(exp, got3, "x")
    check("metrics/operator-error", m3["operator_ok"] == 2, m3["operator_ok"])

    # multi-goal disordinato: l'allineamento è per target, non per indice
    a = goal("hotel", "accommodation", "un hotel")
    b = goal("treno", "transport", "un treno", action="book", raw_action="prenotami")
    m4 = compare(contract([a, b]), contract([b, a]), "x")
    check("metrics/goal-alignment", m4["semantic_exact"] == 1, m4["errors"])


# ================================================ coerenza del gold benchmark

def test_gold_selfconsistency():
    path = ROOT / "benchmarks" / "gold_v1.jsonl"
    if not path.exists():
        check("gold/exists", False, "benchmarks/gold_v1.jsonl mancante")
        return
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    check("gold/nonempty", len(rows) >= 90, str(len(rows)))
    bad = []
    for row in rows:
        r = check_contract(row["contract"], row["text"], ground=False)
        if r.status != ACCEPT:
            bad.append(f"{row['id']}={r.status}")
    check("gold/all-accepted", not bad, ", ".join(bad[:5]))

    # un contratto gold confrontato con se stesso deve dare 100%
    off = []
    for row in rows:
        m = compare(row["contract"], row["contract"], row["text"])
        if not m["semantic_exact"]:
            off.append(row["id"])
    check("gold/self-identity", not off, ", ".join(off[:5]))


def test_gate():
    """GoalContract v1.1: il campo `gate` per i condizionali (§11).

    Nasce da un fallimento misurato: su quattro iterazioni la policy di
    ripiego `degrade_to_clarification` non è mai stata imparata (0 su 3), e
    l'errore prodotto era `action: buy` — l'azione irreversibile senza
    condizione, cioè esattamente ciò che la policy voleva evitare."""
    when = pr("price", "lt", 400, currency="EUR")
    text = "Comprami la scheda video solo se costa meno di 400 euro"

    def mk(action="find", gate=None, cons=None):
        gl = goal("scheda video", "product", "la scheda video",
                  action=action, raw_action="comprami", C=cons or [])
        if gate is not None:
            gl["gate"] = gate
        return contract([gl])

    ok = mk(gate={"when": when, "then": "buy", "otherwise": None})
    check("gate/accept", check_contract(ok, text).ok,
          check_contract(ok, text).status)

    # un gate su un'azione già irreversibile non condiziona niente
    bad = mk(action="buy", gate={"when": when, "then": "buy", "otherwise": None})
    r = check_contract(bad, text)
    check("gate/on-buy-blocked", not r.ok, r.status)
    check("gate/on-buy-code",
          "gate_on_non_observational" in r.error_codes(), r.error_codes())

    # la condizione non può essere anche un filtro di ricerca (§11.3.2)
    dup = mk(gate={"when": when, "then": "buy", "otherwise": None}, cons=[when])
    r = check_contract(dup, text)
    check("gate/duplicated", not r.ok, r.status)
    check("gate/duplicated-code",
          "gate_duplicated_in_bucket" in r.error_codes(), r.error_codes())

    # la condizione del gate è soggetta all'ancoraggio come ogni valore
    inv = mk(gate={"when": pr("price", "lt", 900, currency="EUR"),
                   "then": "buy", "otherwise": None})
    check("gate/ungrounded", not check_contract(inv, text).ok)

    # migrazione v1 → v1.1: un contratto senza gate resta valido
    v1 = {"schema_version": "1.0",
          "goals": [goal("panino", "food", "un panino", raw_action="trovami")],
          "clarification": {"required": False, "reason": None,
                            "missing_fields": []}}
    r = check_contract(v1, "Trovami un panino")
    check("gate/migration", r.ok and r.contract["schema_version"] == "1.1",
          f"{r.status} {r.repairs}")

    # metriche: il gate entra nel confronto
    a = mk(gate={"when": when, "then": "buy", "otherwise": None})
    b = mk(gate={"when": when, "then": "buy", "otherwise": "notify"})
    m = compare(a, b, text)
    check("gate/metric-otherwise", "gate_otherwise" in m["errors"], m["errors"])
    check("gate/metric-same", compare(a, a, text)["gate_ok"] == 1)


def test_unaccounted_facts():
    """Il Guard sa dire se ciò che c'è è sbagliato; questo controllo dice se
    manca qualcosa. L'omissione era la causa di 30 dei 44 contratti sbagliati
    che superavano il Guard (reports/030), e nessun altro controllo poteva
    vederla: nell'output non c'è niente di storto, c'è solo un buco."""
    from jase.guard import unaccounted_facts

    # esclusione esplicita ignorata
    c = contract([goal("pizza", "food", "una pizza",
                       C=[pr("location", "near", "Modena")])])
    msgs = unaccounted_facts(c, "Una pizza senza glutine a Modena")
    check("unacct/negation", bool(msgs), msgs)
    # se il predicato c'è, nessuna segnalazione
    c = contract([goal("pizza", "food", "una pizza",
                       C=[pr("diet", "excludes", "glutine"),
                          pr("location", "near", "Modena")])])
    check("unacct/negation-ok",
          not unaccounted_facts(c, "Una pizza senza glutine a Modena"))

    # numero misurato ignorato
    c = contract([goal("hotel", "accommodation", "un hotel",
                       C=[pr("location", "near", "Lisbona")])])
    check("unacct/number",
          bool(unaccounted_facts(c, "Cercami un hotel a Lisbona sotto 80 euro")))
    c = contract([goal("hotel", "accommodation", "un hotel",
                       C=[pr("location", "near", "Lisbona"),
                          pr("price", "lt", 80, currency="EUR")])])
    check("unacct/number-ok",
          not unaccounted_facts(c, "Cercami un hotel a Lisbona sotto 80 euro"))

    # un numero nel gate conta come rendicontato
    gl = goal("hotel", "accommodation", "un hotel")
    gl["gate"] = {"when": pr("price", "lt", 80, currency="EUR"),
                  "then": "book", "otherwise": None}
    check("unacct/gate-counts",
          not unaccounted_facts(contract([gl]),
                                "Cercami un hotel e se costa meno di 80 euro prenotalo"))

    # un numero che fa parte del NOME del target non è un vincolo
    c = contract([goal("iPhone 15", "product", "un iPhone 15")])
    check("unacct/target-number",
          not unaccounted_facts(c, "Cerco un iPhone 15"))

    # "senza" davanti a un infinito è una soglia, non un'esclusione
    c = contract([goal("panino", "food", "un panino",
                       A=[pr("ingredient", "contains", "mortadella")],
                       C=[pr("location", "near", "Milano"),
                          pr("price", "lt", 10, currency="EUR")])])
    check("unacct/infinitive",
          not unaccounted_facts(
              c, "A Milano vorrei un panino con la mortadella, "
                 "senza arrivare a spendere 10€"))

    # zero falsi positivi su tutto il benchmark gold
    path = ROOT / "benchmarks" / "gold_v1.jsonl"
    if path.exists():
        bad = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if unaccounted_facts(row["contract"], row["text"]):
                bad.append(row["id"])
        check("unacct/no-false-positives", not bad, ", ".join(bad[:5]))


def test_guard_pipeline():
    good = json.dumps(contract([goal(
        A=[pr("ingredient", "contains", "mortadella")],
        C=[pr("location", "near", "Milano"),
           pr("price", "lt", 10, currency="EUR")])]), ensure_ascii=False)
    r = guard(good, "Trovami un panino con la mortadella a Milano sotto i 10 euro")
    check("pipeline/accept", r.status == ACCEPT, f"{r.status} {r.repairs}")

    r = guard("mi dispiace, non ho capito", "qualcosa")
    check("pipeline/reject-nonjson", r.status == REJECT, r.status)


def test_bucket_cascade() -> None:
    """La cascata §2 applicata come riparazione, e i casi in cui NON si tocca."""
    check("cascata/selezione", bucket_by_cascade(pr("price", "lte", 100)) == "constraints")
    check("cascata/descrittiva", bucket_by_cascade(pr("breed", "in", "Siamese|Sphynx")) == "attributes")
    check("cascata/comparativo", bucket_by_cascade(pr("size", "gte", 80)) == "constraints")
    check("cascata/negazione", bucket_by_cascade(pr("color", "neq", "rosso")) == "constraints")

    # preferenza inventata: nel testo non c'è nessun marcatore debole
    c = contract([goal(P=[pr("price", "lte", 100, currency="EUR")],
                       C=[pr("date", "eq", "stasera")])])
    r = check_contract(c, "Mi serve un panino stasera entro 100 euro")
    check("cascata/pref-senza-marcatore", r.status == REPAIRED
          and not r.contract["goals"][0]["preferences"]
          and len(r.contract["goals"][0]["constraints"]) == 2, f"{r.status} {r.repairs}")

    # proprietà descrittiva finita in constraints senza ragione
    c = contract([goal(C=[pr("breed", "in", "Siamese|Sphynx")])])
    r = check_contract(c, "Cerco un gatto Siamese o Sphynx", ground=False)
    check("cascata/desc-in-constraints", r.status == REPAIRED
          and r.contract["goals"][0]["attributes"], f"{r.status} {r.repairs}")

    # col marcatore debole nel testo il Guard non tocca NIENTE: non sa a
    # quale fatto si riferisca
    c = contract([goal(P=[pr("price", "lte", 100, currency="EUR")])])
    r = check_contract(c, "Un panino, possibilmente entro 100 euro")
    check("cascata/marcatore-blocca", r.status == ACCEPT and not r.repairs,
          f"{r.status} {r.repairs}")

    # proprietà fuori dal vocabolario noto: non si scende mai in attributes
    c = contract([goal(C=[pr("blorchness", "eq", "alta")])])
    r = check_contract(c, "Trovami un panino con blorchness alta", ground=False)
    check("cascata/vocabolario-aperto", not r.repairs, str(r.repairs))


def test_polarity() -> None:
    """Un valore negato nel testo non può reggere un operatore positivo."""
    check("polarita/no-rosso", negation_attached_to_value("zorba x7 no rosso max 500", "rosso"))
    check("polarita/che-non-sia", negation_attached_to_value(
        "un ristorante che non sia di pesce", "pesce"))
    # il falso positivo che ha insegnato la lezione: "un pani|no possibilmente"
    check("polarita/panino", not negation_attached_to_value(
        "Vorrei un panino possibilmente con la mortadella", "mortadella"))
    check("polarita/virgola", not negation_attached_to_value(
        "Non ho fretta, trovami una pizza margherita", "margherita"))
    check("polarita/lontano", not negation_attached_to_value(
        "Non voglio spendere troppo e vorrei una pizza margherita", "margherita"))

    # La negazione di un altro fatto non deve ribaltare una soglia successiva.
    check("strictness/no-glutine-sotto",
          expected_strictness("cerco sushi no glutine vorrei restare sotto i 5 euro", 5) == "lt",
          str(expected_strictness("cerco sushi no glutine vorrei restare sotto i 5 euro", 5)))

    c = contract([goal(A=[pr("color", "eq", "rosso")])])
    r = check_contract(c, "cerco un panino no rosso", ground=True)
    check("polarita/blocca", r.status == CLARIFY
          and "operator_polarity_mismatch" in r.error_codes(), f"{r.status} {r.error_codes()}")


def test_time_role() -> None:
    """'entro 20 euro' non è un orario, nemmeno dopo la normalizzazione."""
    check("ora/estrazione", time_hour("20:00") == 20 and time_hour("rosso") is None)
    c = contract([goal(C=[pr("price", "lte", 20, currency="EUR"),
                          pr("time", "before", "20:00")])])
    r = check_contract(c, "Cercami un panino entro 20 euro")
    check("ora/ruolo-denaro", r.status == CLARIFY
          and "number_role_mismatch" in r.error_codes(), f"{r.status} {r.error_codes()}")
    # quando il testo l'orario ce l'ha davvero, deve passare
    c = contract([goal(C=[pr("time", "before", "20:00")])])
    r = check_contract(c, "Prenotami un panino prima delle 20")
    check("ora/ruolo-orario", r.ok, f"{r.status} {r.error_codes()}")


def test_omission_new_forms() -> None:
    """Due segnali di omissione ad alta precisione: tempo e nomi propri."""
    # "sabato" nel testo, niente nel contratto
    c = contract([goal(name="biglietto", ttype="ticket", raw="due biglietti",
                       action="buy", raw_action="compra",
                       A=[pr("section", "eq", "platea")],
                       C=[pr("price", "lte", 60, currency="EUR")])])
    r = check_contract(c, "compra due biglietti per il concerto di sabato, "
                          "in platea, non più di 60 euro l'uno")
    check("omissione/temporale", not r.ok
          and "unaccounted_fact" in r.error_codes(), f"{r.status} {r.error_codes()}")

    # quantità scritta in lettere: anche senza il segnale temporale deve
    # bloccare se `quantity=2` scompare dal contratto.
    c = contract([goal(name="biglietto", ttype="ticket", raw="due biglietti",
                       action="buy", raw_action="compra",
                       A=[pr("section", "eq", "platea")])])
    r = check_contract(c, "compra due biglietti in platea")
    check("omissione/quantita-lettere", not r.ok
          and "unaccounted_fact" in r.error_codes(), f"{r.status} {r.error_codes()}")

    c = contract([goal(name="biglietto", ttype="ticket", raw="due biglietti",
                       action="buy", raw_action="compra",
                       A=[pr("quantity", "eq", 2), pr("section", "eq", "platea")])])
    r = check_contract(c, "compra due biglietti in platea")
    check("omissione/quantita-lettere-ok", r.ok, f"{r.status} {r.error_codes()}")

    # lo stesso testo con il fatto al suo posto deve passare
    c = contract([goal(name="biglietto", ttype="ticket", raw="due biglietti",
                       action="buy", raw_action="compra",
                       A=[pr("section", "eq", "platea"), pr("quantity", "eq", 2)],
                       C=[pr("date", "eq", "sabato"),
                          pr("price", "lte", 60, currency="EUR")])])
    r = check_contract(c, "compra due biglietti per il concerto di sabato, "
                          "in platea, non più di 60 euro l'uno")
    check("omissione/temporale-ok", r.ok, f"{r.status} {r.error_codes()}")

    # marca ignorata
    c = contract([goal(name="tavolo", ttype="place", raw="un tavolo",
                       action="book", raw_action="prenotami",
                       C=[pr("location", "near", "Lecce"),
                          pr("date", "eq", "stasera")])])
    r = check_contract(c, "Prenotami un tavolo da Frobnicato a Lecce per stasera")
    check("omissione/nome-proprio", not r.ok
          and "unaccounted_fact" in r.error_codes(), f"{r.status} {r.error_codes()}")

    # la maiuscola di inizio frase non è un nome proprio
    c = contract([goal(name="panino", ttype="food", raw="un panino",
                       raw_action="trovami")])
    r = check_contract(c, "Trovami un panino")
    check("omissione/maiuscola-iniziale", r.ok, f"{r.status} {r.error_codes()}")


def test_single_valued_conflict() -> None:
    """Due valori diversi dati per certi sulla stessa proprietà."""
    c = contract([goal(name="volo", ttype="transport", raw="un volo",
                       raw_action="trovami",
                       C=[pr("destination", "eq", "Amsterdam"),
                          pr("destination", "eq", "Atene")])])
    r = check_contract(c, "Trovami un volo per Amsterdam per Atene", ground=False)
    check("conflitto/due-destinazioni", not r.ok
          and "conflicting_predicates" in r.error_codes(), f"{r.status} {r.error_codes()}")

    # `contains` invece ammette più valori: un hotel ha piscina E palestra
    c = contract([goal(name="hotel", ttype="accommodation", raw="un hotel",
                       raw_action="trovami",
                       A=[pr("amenity", "contains", "piscina"),
                          pr("amenity", "contains", "palestra")])])
    r = check_contract(c, "Trovami un hotel con piscina e palestra", ground=False)
    check("conflitto/due-amenity-ok", r.ok, f"{r.status} {r.error_codes()}")

    # e nemmeno una forbice di prezzo è un conflitto
    c = contract([goal(C=[pr("price", "gte", 100), pr("price", "lte", 300)])])
    r = check_contract(c, "un panino fra 100 e 300", ground=False)
    check("conflitto/forbice-prezzo-ok", r.ok, f"{r.status} {r.error_codes()}")


def test_clarification_preserves_grounded_context() -> None:
    """Il fallback deve essere v1.1 e non cancellare target già affidabili."""
    candidate = contract([goal(name="Sphynx", ttype="animal", raw="uno Sphynx",
                               action="find", raw_action="mi trovi",
                               C=[pr("distance", "within", 80, unit="km")])])
    c = clarification_contract(
        "Mi trovi uno Sphynx? Non voglio farmi più di 80 km.",
        "conflicting_constraints", ["constraints"], candidate=candidate)
    check("clarify/schema-11", c["schema_version"] == "1.1", c["schema_version"])
    check("clarify/target-preserved", c["goals"][0]["target"]["name"] == "Sphynx",
          str(c["goals"][0]["target"]))
    check("clarify/action-preserved", c["goals"][0]["action"] == "find",
          str(c["goals"][0]))
    check("clarify/predicates-cleared", not c["goals"][0]["constraints"],
          str(c["goals"][0]["constraints"]))

    # un target inventato non viene mai preservato.
    bad = contract([goal(name="Mario", ttype="person", raw="Mario",
                         action="find", raw_action="(implicito)")])
    c = clarification_contract("Aiutami.", "goal_unintelligible", ["goal"],
                               candidate=bad)
    check("clarify/hallucinated-target-dropped",
          c["goals"][0]["target"]["name"] == "unknown",
          str(c["goals"][0]["target"]))


def main() -> int:
    for fn in [test_extraction, test_duplicates_and_buckets, test_hallucination,
               test_currency_grounding, test_raw_action_grounding,
               test_negation_in_value,
               test_predicate_validity, test_schema_violations,
               test_clarification_coherence, test_strictness,
               test_grounding_helpers, test_strictness_blocking,
               test_number_role_and_fillers, test_gate,
               test_unaccounted_facts, test_metrics, test_gold_selfconsistency,
               test_bucket_cascade, test_polarity, test_time_role,
               test_omission_new_forms, test_single_valued_conflict,
               test_clarification_preserves_grounded_context,
               test_guard_pipeline]:
        fn()
    print(f"\n{PASSED} passati, {FAILED} falliti")
    for f in FAILURES:
        print("  FAIL " + f)
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
