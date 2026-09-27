"""Freeze a new 100-case cross-foundation panel before Granite weight/training work.

Requests and gold are agent-authored. The review queue explicitly records that
independent human adjudication is pending; it is never silently treated as done.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.build_v28_blind import c, ctx, f, fallback, g, missing, selection, t  # noqa: E402
from jase.leakage import normalize_text, token_set  # noqa: E402
from jase.multimodel import read_jsonl, sha256  # noqa: E402
from jase.semantic_contract_v1_3 import validate_contract  # noqa: E402
from jase.semantic_guard_v1_3_routing_v5 import check_contract  # noqa: E402

DEST = ROOT / "benchmarks/v33"
PRIOR = sorted(set([*ROOT.glob("benchmarks/v*/**/*.jsonl"),
                    *ROOT.glob("data/v*_9b/train.jsonl"),
                    *ROOT.glob("data/v*_9b/valid.jsonl"),
                    *ROOT.glob("data/v19_clean/train.jsonl"),
                    *ROOT.glob("data/v19_clean/valid.jsonl"),
                    ROOT / "benchmarks/release_holdout_v1.jsonl",
                    ROOT / "data/v20_quality/dev.jsonl"]))


def cases():
    rows = []

    def add(family, text, gold, *, review_flags=()):
        issues = validate_contract(gold, user_text=text)
        if issues:
            raise ValueError((family, text, [i.as_dict() for i in issues]))
        rows.append({"id": f"v33-{len(rows)+1:03d}", "family": family,
                     "text": text, "contract": gold,
                     "review_flags": list(review_flags)})

    # 1. Commands quoted as data are not instructions to Jase.
    add("non_actionable", "Sto correggendo un dialogo teatrale: la battuta 'prenotami un tavolo subito' è grammaticale?", c())
    add("non_actionable", "Nel manuale compare ‘assumi un elettricista’. Spiegami il significato, non cercare nessuno.", c())
    add("non_actionable", "Questa è una prova di traduzione: 'rent me a car tomorrow'. Dammi solo la traduzione italiana.", c())
    add("non_actionable", "Ho fotografato un cartello con scritto 'annulla l'appuntamento'. Voglio archiviarne il testo.", c())
    add("non_actionable", "Un amico ha scritto «ordina tre pizze» nel racconto. Non è una richiesta mia.", c())

    # 2. Plain retrieval, including colloquial English.
    add("search", "Mi trovi un'edicola nei dintorni di Orvieto?", c(g("find", "edicola", "place", [f("location", "Orvieto", "near")])))
    add("search", "Cerco un agriturismo a Cuneo per sabato; fammi vedere le opzioni.", c(g("find", "agriturismo", "accommodation", [f("location", "Cuneo", "near")], temporal=[t("search_date", "sabato")])))
    add("search", "Could you find a pharmacy near the station in Siena?", c(g("find", "farmacia", "place", [f("location", "stazione di Siena", "near")])), review_flags=("translation_of_target",))
    add("search", "Per favore cerca un negozio di strumenti musicali a Lodi, senza fare acquisti.", c(g("find", "negozio di strumenti musicali", "place", [f("location", "Lodi", "near")], policy={"forbid": ["buy"]})))
    add("search", "Sai trovare un parcheggio vicino all'ospedale di Pistoia?", c(g("find", "parcheggio", "place", [f("location", "ospedale di Pistoia", "near")])))

    # 3. Local constraints and access.
    add("local_constraints", "Siamo sotto il Castello Estense a Ferrara; mi indichi una farmacia ancora aperta qui intorno?", c(g("find", "farmacia", "place", [f("location", "Castello Estense a Ferrara", "near"), f("open_now", True)])))
    add("local_constraints", "Trova un bar a piedi dalla stazione di Lucca, massimo tre opzioni.", c(g("find", "bar", "place", [f("location", "stazione di Lucca", "near"), f("access", "a piedi")], modifiers={"limit": 3})))
    add("local_constraints", "A Biella oggi vorrei una trattoria senza glutine, possibilmente aperta ora.", c(g("find", "trattoria", "place", [f("location", "Biella", "near"), f("diet", "senza glutine"), f("open_now", True)], temporal=[t("search_date", "oggi")])))
    add("local_constraints", "Cerca due officine aperte adesso nella zona del porto di Ancona.", c(g("find", "officina", "professional_service", [f("location", "porto di Ancona", "near"), f("open_now", True)], modifiers={"limit": 2})))
    add("local_constraints", "Siamo in sei: mostrami pizzerie a Novara che abbiano posto stasera, senza prenotare.", c(g("find", "pizzeria", "place", [f("location", "Novara", "near"), f("availability", "stasera")], policy={"forbid": ["book"]}), context=[ctx("party_size", 6)]), review_flags=("availability_field",))

    # 4. Hotel and restaurant bookings.
    add("booking", "Prenota una stanza a Matera per venerdì notte, ma chiedimi il via libera prima di confermare.", c(g("book", "stanza", "accommodation", [f("location", "Matera", "near")], temporal=[t("action_date", "venerdì")], policy={"confirm_before": ["book"]})))
    add("booking", "Per quattro persone blocca un tavolo a Perugia domenica alle 20:30; voglio approvare il locale.", c(g("book", "ristorante", "place", [f("location", "Perugia", "near"), f("guests", 4)], temporal=[t("action_date", "domenica"), t("action_time", "20:30")], missing=[missing("provider")], policy={"confirm_before": ["book"]})))
    add("booking", "Mi prenoti un albergo a Lecce per lunedì? Prima però dimmi quale struttura hai scelto.", c(g("book", "albergo", "accommodation", [f("location", "Lecce", "near")], temporal=[t("action_date", "lunedì")], missing=[missing("provider")], policy={"confirm_before": ["book"]})))
    add("booking", "Vorrei fissare un tavolo a cena a Ravenna; non ho ancora deciso il giorno.", c(g("book", "ristorante", "place", [f("location", "Ravenna", "near")], missing=[missing("date"), missing("provider")])) )
    add("booking", "A Trento trovami un hotel per il 9 ottobre e prenotalo soltanto dopo il mio sì.", c(g("find", "hotel", "accommodation", [f("location", "Trento", "near")], temporal=[t("search_date", "9 ottobre")], idx=1), g("book", "hotel", "accommodation", [selection()], temporal=[t("action_date", "9 ottobre")], depends_on=["g1"], policy={"confirm_before": ["book"]}, idx=2)), review_flags=("selection_unspecified",))

    # 5. Appointment creation.
    add("appointment_create", "Fissa una visita dermatologica a Viterbo il 17 novembre alle 09:40, previa mia conferma.", c(g("book", "visita dermatologica", "appointment", [f("location", "Viterbo", "near")], temporal=[t("action_date", "17 novembre"), t("action_time", "09:40")], policy={"confirm_before": ["book"]})))
    add("appointment_create", "Prendimi appuntamento dal dentista a Fano martedì pomeriggio; devo scegliere lo studio.", c(g("book", "dentista", "appointment", [f("location", "Fano", "near")], temporal=[t("action_date", "martedì"), t("action_time", "pomeriggio")], missing=[missing("provider")])))
    add("appointment_create", "Mi organizzi un controllo della vista a Terni giovedì alle 15, ma prima fammi approvare.", c(g("book", "controllo della vista", "appointment", [f("location", "Terni", "near")], temporal=[t("action_date", "giovedì"), t("action_time", "15:00")], policy={"confirm_before": ["book"]})))
    add("appointment_create", "Serve una visita fisioterapica a Rimini; la data ancora non ce l'ho.", c(g("book", "visita fisioterapica", "appointment", [f("location", "Rimini", "near")], missing=[missing("date")])) )
    add("appointment_create", "Puoi fissare il richiamo dal veterinario per il cane a Pavia il 6 dicembre alle 11?", c(g("book", "visita veterinaria", "appointment", [f("location", "Pavia", "near")], temporal=[t("action_date", "6 dicembre"), t("action_time", "11:00")]), context=[ctx("pet", "cane")]), review_flags=("context_ontology",))

    # 6. Cancellation of user-owned appointments, always explicit about identity.
    add("appointment_cancel", "Cancella la mia visita ortopedica del 7 maggio alle 10 a Pesaro, dopo che ti avrò confermato.", c(g("cancel", "visita ortopedica", "appointment", [f("original_date", "7 maggio"), f("original_time", "10:00"), f("location", "Pesaro", "near")], policy={"confirm_before": ["cancel"]})))
    add("appointment_cancel", "L'appuntamento per la pulizia dei denti a Modena del 22 giugno non mi serve più: annullalo, ma mostramelo prima.", c(g("cancel", "pulizia dei denti", "appointment", [f("original_date", "22 giugno"), f("location", "Modena", "near")], policy={"confirm_before": ["cancel"]})))
    add("appointment_cancel", "Vorrei disdire la visita dal cardiologo che ho il 3 luglio alle 08:30; aspetta il mio ok.", c(g("cancel", "visita cardiologica", "appointment", [f("original_date", "3 luglio"), f("original_time", "08:30")], policy={"confirm_before": ["cancel"]})))
    add("appointment_cancel", "Annulla il tagliando auto prenotato a Cremona per venerdì, con mia autorizzazione.", c(g("cancel", "tagliando auto", "appointment", [f("original_date", "venerdì"), f("location", "Cremona", "near")], policy={"confirm_before": ["cancel"]})))
    add("appointment_cancel", "Non prenotare altro: elimina solo la visita oculistica del 12 settembre a Rovigo, chiedendomi conferma.", c(g("cancel", "visita oculistica", "appointment", [f("original_date", "12 settembre"), f("location", "Rovigo", "near")], policy={"confirm_before": ["cancel"]})))

    # 7. Rescheduling keeps old/new time roles separate.
    add("appointment_reschedule", "Sposta il mio appuntamento dal logopedista del 4 aprile alle 14 al 9 aprile alle 16; prima fammi confermare.", c(g("reschedule", "logopedista", "appointment", [f("original_date", "4 aprile"), f("original_time", "14:00")], temporal=[t("action_date", "9 aprile"), t("action_time", "16:00")], policy={"confirm_before": ["reschedule"]})))
    add("appointment_reschedule", "La visita neurologica del 6 novembre va rinviata a lunedì prossimo, stesso orario; non cambiare nulla senza il mio sì.", c(g("reschedule", "visita neurologica", "appointment", [f("original_date", "6 novembre")], temporal=[t("action_date", "lunedì prossimo")], policy={"confirm_before": ["reschedule"]})), review_flags=("same_time_reference",))
    add("appointment_reschedule", "Riprogramma la fisioterapia di domani alle 12 per venerdì alle 09:00, previa conferma.", c(g("reschedule", "fisioterapia", "appointment", [f("original_date", "domani"), f("original_time", "12:00")], temporal=[t("action_date", "venerdì"), t("action_time", "09:00")], policy={"confirm_before": ["reschedule"]})))
    add("appointment_reschedule", "Devo posticipare il controllo dentistico del 15 gennaio, ma la nuova data te la dirò dopo.", c(g("reschedule", "controllo dentistico", "appointment", [f("original_date", "15 gennaio")], missing=[missing("new_date")])))
    add("appointment_reschedule", "Porta l'appuntamento per la revisione moto del 2 agosto alle 10:30 al 5 agosto alle 08:00; chiedi autorizzazione.", c(g("reschedule", "revisione moto", "appointment", [f("original_date", "2 agosto"), f("original_time", "10:30")], temporal=[t("action_date", "5 agosto"), t("action_time", "08:00")], policy={"confirm_before": ["reschedule"]})))

    # 8. Rental actions and search-only prohibitions.
    add("rental", "A Belluno noleggia una bicicletta per sabato, solo dopo avermi chiesto conferma.", c(g("rent", "bicicletta", "vehicle", [f("location", "Belluno", "near")], temporal=[t("action_date", "sabato")], policy={"confirm_before": ["rent"]})))
    add("rental", "Per il weekend guarda i furgoni disponibili a Udine; per ora non prenderne uno.", c(g("find", "furgone", "vehicle", [f("location", "Udine", "near")], temporal=[t("search_date", "weekend")], policy={"forbid": ["rent"]})))
    add("rental", "Mi serve un monopattino in affitto a Piacenza dalle 18, ma prima mostrami l'offerta.", c(g("rent", "monopattino", "vehicle", [f("location", "Piacenza", "near")], temporal=[t("action_time", "18:00")], policy={"confirm_before": ["rent"]})))
    add("rental", "Trova tre auto a noleggio per domani a Grosseto, senza impegnarti con nessuna agenzia.", c(g("find", "auto", "vehicle", [f("location", "Grosseto", "near")], temporal=[t("search_date", "domani")], modifiers={"limit": 3}, policy={"forbid": ["rent"]})))
    add("rental", "Affittami una canoa a Como domenica, ma solo se ti do il permesso dopo aver visto il prezzo.", c(g("rent", "canoa", "vehicle", [f("location", "Como", "near")], temporal=[t("action_date", "domenica")], policy={"confirm_before": ["rent"]})))

    # 9. Provider search vs hiring.
    add("provider_hire", "Ingaggia un imbianchino a Potenza per lunedì, ma il professionista non l'ho scelto; chiedimi conferma.", c(g("hire", "imbianchino", "professional_service", [f("location", "Potenza", "near")], temporal=[t("action_date", "lunedì")], missing=[missing("provider")], policy={"confirm_before": ["hire"]})))
    add("provider_hire", "A Livorno cerca un idraulico libero oggi; non assumerlo ancora.", c(g("find", "idraulico", "professional_service", [f("location", "Livorno", "near"), f("availability", "oggi")], policy={"forbid": ["hire"]})))
    add("provider_hire", "Fai venire un elettricista a Jesolo domani mattina, però voglio decidere io chi ingaggiare.", c(g("hire", "elettricista", "professional_service", [f("location", "Jesolo", "near")], temporal=[t("action_date", "domani"), t("action_time", "mattina")], missing=[missing("provider")], policy={"confirm_before": ["hire"]})), review_flags=("verb_hire_or_contact",))
    add("provider_hire", "Mi mostri dei tecnici per la caldaia a Forlì? Nessun incarico senza il mio via libera.", c(g("find", "tecnico caldaia", "professional_service", [f("location", "Forlì", "near")], policy={"forbid": ["hire"]})))
    add("provider_hire", "Assumi un giardiniere per casa mia a Rieti sabato; non so ancora quale professionista.", c(g("hire", "giardiniere", "professional_service", [f("location", "Rieti", "near")], temporal=[t("action_date", "sabato")], missing=[missing("provider")]), context=[ctx("place", "casa mia")]), review_flags=("context_ontology",))

    # 10. Reminders separate notification time from text content.
    add("reminder", "Alle 07:15 di martedì ricordami 'ritirare il pacco entro le 13'.", c(g("notify", "promemoria", "reminder", [f("message", "ritirare il pacco entro le 13")], temporal=[t("notification_date", "martedì"), t("notification_time", "07:15")])))
    add("reminder", "Domattina verso le nove avvisami di chiamare mia sorella.", c(g("notify", "promemoria", "reminder", [f("message", "chiamare mia sorella")], temporal=[t("notification_date", "domani"), t("notification_time", "09:00")])), review_flags=("approximate_time"))
    add("reminder", "Imposta un promemoria per il 19 marzo alle 18:40: 'partenza del bus alle 20'.", c(g("notify", "promemoria", "reminder", [f("message", "partenza del bus alle 20")], temporal=[t("notification_date", "19 marzo"), t("notification_time", "18:40")])))
    add("reminder", "Tra due giorni ricordami di pagare la bolletta, ma non so a che ora.", c(g("notify", "promemoria", "reminder", [f("message", "pagare la bolletta")], temporal=[t("notification_date", "tra due giorni")], missing=[missing("notification_time")])))
    add("reminder", "Puoi avvisarmi alle 16:10 di venerdì che il medico arriva alle 17:30?", c(g("notify", "promemoria", "reminder", [f("message", "il medico arriva alle 17:30")], temporal=[t("notification_date", "venerdì"), t("notification_time", "16:10")])))

    # 11. Compare count must not become purchase quantity.
    add("comparison", "Metti a confronto due lavatrici sotto i 500 euro, senza comprarle.", c(g("compare", "lavatrice", "product", [f("price", 500, "lt", currency="EUR")], modifiers={"compare_count": 2}, policy={"forbid": ["buy"]})))
    add("comparison", "Vorrei confrontare quattro offerte di fibra a Savona per canone mensile.", c(g("compare", "fibra", "service", [f("location", "Savona", "near")], modifiers={"compare_count": 4, "sort": {"field": "price", "direction": "asc"}})), review_flags=("monthly_price_sort",))
    add("comparison", "Fammi vedere un confronto tra tre hotel di Bolzano per prezzo, ma non riservare camere.", c(g("compare", "hotel", "accommodation", [f("location", "Bolzano", "near")], modifiers={"compare_count": 3, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["book"]})))
    add("comparison", "Compare two second-hand bikes in Arezzo, cheapest first; don't buy either.", c(g("compare", "bicicletta usata", "vehicle", [f("location", "Arezzo", "near")], modifiers={"compare_count": 2, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})), review_flags=("english_vehicle_buy_policy",))
    add("comparison", "Tra cinque stampanti laser sotto 300 euro, quali costano meno? Solo confronto.", c(g("compare", "stampante laser", "product", [f("price", 300, "lt", currency="EUR")], modifiers={"compare_count": 5, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})))

    # 12. Ranking and result limits, not order actions.
    add("sorting", "A Macerata mostrami al massimo quattro alberghi dal più economico al più caro.", c(g("find", "albergo", "accommodation", [f("location", "Macerata", "near")], modifiers={"limit": 4, "sort": {"field": "price", "direction": "asc"}})))
    add("sorting", "Ordina per distanza tre farmacie intorno a piazza del Duomo a Firenze.", c(g("find", "farmacia", "place", [f("location", "piazza del Duomo a Firenze", "near")], modifiers={"limit": 3, "sort": {"field": "distance", "direction": "asc"}})))
    add("sorting", "Trova cinque offerte di PC portatile a Verona e metti prima le più economiche; niente acquisti.", c(g("find", "PC portatile", "product", [f("location", "Verona", "near")], modifiers={"limit": 5, "sort": {"field": "price", "direction": "asc"}}, policy={"forbid": ["buy"]})))
    add("sorting", "Per il pranzo cerca due ristoranti a Gubbio ordinati per valutazione migliore.", c(g("find", "ristorante", "place", [f("location", "Gubbio", "near")], modifiers={"limit": 2, "sort": {"field": "rating", "direction": "desc"}})))
    add("sorting", "Quali sono i tre treni meno costosi per Vicenza domani? Partenza da Padova.", c(g("find", "treno", "transport", [f("origin", "Padova"), f("destination", "Vicenza")], temporal=[t("search_date", "domani")], modifiers={"limit": 3, "sort": {"field": "price", "direction": "asc"}})))

    # 13. Departure, arrival and appointment times have different scopes.
    add("transport_time", "Da Novara a Milano cerca un treno che arrivi non oltre le 08:45 di martedì.", c(g("find", "treno", "transport", [f("origin", "Novara"), f("destination", "Milano")], temporal=[t("search_date", "martedì"), t("arrival_time", "08:45", "before")])))
    add("transport_time", "Il colloquio è alle 15 a Trieste: da Monfalcone servono autobus in arrivo prima delle 14:30.", c(g("find", "autobus", "transport", [f("origin", "Monfalcone"), f("destination", "Trieste")], temporal=[t("arrival_time", "14:30", "before")])))
    add("transport_time", "Fammi vedere i regionali da Parma per Reggio Emilia che partono dopo le 19:10.", c(g("find", "treno regionale", "transport", [f("origin", "Parma"), f("destination", "Reggio Emilia")], temporal=[t("departure_time", "19:10", "after")])))
    add("transport_time", "Per domenica cerca collegamenti in treno da Imola a Bologna con arrivo entro mezzogiorno.", c(g("find", "treno", "transport", [f("origin", "Imola"), f("destination", "Bologna")], temporal=[t("search_date", "domenica"), t("arrival_time", "12:00", "before")])))
    add("transport_time", "Voglio essere a Lecce alle 09:30, partendo da Brindisi in treno; controlla le corse utili.", c(g("find", "treno", "transport", [f("origin", "Brindisi"), f("destination", "Lecce")], temporal=[t("arrival_time", "09:30", "before")])))

    # 14. Flights: directness, booking, and explicit non-execution.
    add("flights", "Cerca voli senza scali da Genova a Madrid per il 12 dicembre. Per ora voglio solo guardare.", c(g("find", "volo", "transport", [f("origin", "Genova"), f("destination", "Madrid"), f("stops", 0)], temporal=[t("search_date", "12 dicembre")], policy={"forbid": ["book", "buy"]})))
    add("flights", "Prenota per me un volo Palermo-Lisbona il 14 aprile, ma chiedimi l'ok prima.", c(g("book", "volo", "transport", [f("origin", "Palermo"), f("destination", "Lisbona")], temporal=[t("action_date", "14 aprile")], policy={"confirm_before": ["book"]})))
    add("flights", "Mostrami i voli diretti più economici da Napoli a Vienna per giovedì, massimo due risultati.", c(g("find", "volo", "transport", [f("origin", "Napoli"), f("destination", "Vienna"), f("stops", 0)], temporal=[t("search_date", "giovedì")], modifiers={"limit": 2, "sort": {"field": "price", "direction": "asc"}})))
    add("flights", "Voglio un volo per Amsterdam il 23 febbraio, ma non ti ho detto da quale aeroporto parto.", c(g("find", "volo", "transport", [f("destination", "Amsterdam")], temporal=[t("search_date", "23 febbraio")], missing=[missing("origin", "search")])) )
    add("flights", "Find a nonstop flight from Turin to Berlin next Friday; do not purchase anything.", c(g("find", "volo", "transport", [f("origin", "Torino"), f("destination", "Berlino"), f("stops", 0)], temporal=[t("search_date", "venerdì prossimo")], policy={"forbid": ["buy", "book"]})), review_flags=("translated_city_names",))

    # 15. Quoted message content does not set the send time.
    add("contact", "Scrivi a Elisa 'Arrivo alle 21', ma mostrami la bozza prima di mandarla.", c(g("contact", "Elisa", "person", [f("message", "Arrivo alle 21")], policy={"confirm_before": ["contact"]})))
    add("contact", "Alle 08:10 manda a Pietro il testo 'La visita è alle 12:30'; chiedimi conferma sull'invio.", c(g("contact", "Pietro", "person", [f("message", "La visita è alle 12:30")], temporal=[t("action_time", "08:10")], policy={"confirm_before": ["contact"]})))
    add("contact", "Invia a Marta una mail con scritto 'Ho spedito i documenti ieri', dopo il mio assenso.", c(g("contact", "Marta", "person", [f("message", "Ho spedito i documenti ieri")], policy={"confirm_before": ["contact"]})))
    add("contact", "Chiedi a Davide se il sopralluogo si può fare giovedì; voglio leggere il messaggio prima.", c(g("contact", "Davide", "person", [f("message", "Il sopralluogo si può fare giovedì?")], policy={"confirm_before": ["contact"]})), review_flags=("message_paraphrase",))
    add("contact", "Per favore manda a Nora: 'Le chiavi sono sotto il vaso'. Non serve prenotare nulla.", c(g("contact", "Nora", "person", [f("message", "Le chiavi sono sotto il vaso")], policy={"forbid": ["book"]})), review_flags=("policy_scope",))

    # 16. Two independent goals must not acquire a false dependency.
    add("multi_independent", "Mi servono un'edicola a Pescara e un supermercato a Chieti; cerca entrambi.", c(g("find", "edicola", "place", [f("location", "Pescara", "near")], idx=1), g("find", "supermercato", "place", [f("location", "Chieti", "near")], idx=2)))
    add("multi_independent", "Trova una farmacia a Fermo e, separatamente, un parcheggio a Civitanova Marche.", c(g("find", "farmacia", "place", [f("location", "Fermo", "near")], idx=1), g("find", "parcheggio", "place", [f("location", "Civitanova Marche", "near")], idx=2)))
    add("multi_independent", "Per sabato mostrami hotel a Chioggia e ristoranti a Adria, senza prenotare nessuno dei due.", c(g("find", "hotel", "accommodation", [f("location", "Chioggia", "near")], temporal=[t("search_date", "sabato")], policy={"forbid": ["book"]}, idx=1), g("find", "ristorante", "place", [f("location", "Adria", "near")], temporal=[t("search_date", "sabato")], policy={"forbid": ["book"]}, idx=2)))
    add("multi_independent", "Cerca un treno Verona-Venezia per domani e un albergo a Mestre per la notte.", c(g("find", "treno", "transport", [f("origin", "Verona"), f("destination", "Venezia")], temporal=[t("search_date", "domani")], idx=1), g("find", "albergo", "accommodation", [f("location", "Mestre", "near")], temporal=[t("search_date", "domani")], idx=2)))
    add("multi_independent", "Mi dai tre opzioni di bici a noleggio a Pisa e due officine a Pontedera? Solo ricerche.", c(g("find", "bicicletta", "vehicle", [f("location", "Pisa", "near")], modifiers={"limit": 3}, policy={"forbid": ["rent"]}, idx=1), g("find", "officina", "professional_service", [f("location", "Pontedera", "near")], modifiers={"limit": 2}, policy={"forbid": ["hire"]}, idx=2)))

    # 17. Sequential selections and ordinal references.
    add("ordinal_chain", "Cerca hotel a Urbino; se mi piace il secondo, prenotalo solo dopo avermi chiesto conferma e la data.", c(g("find", "hotel", "accommodation", [f("location", "Urbino", "near")], idx=1), g("book", "hotel", "accommodation", [selection(ordinal=2)], depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]}, idx=2)))
    add("ordinal_chain", "Trova tre meccanici a Prato e chiedi al primo se può venire domani; la domanda va inviata dopo il mio ok.", c(g("find", "meccanico", "professional_service", [f("location", "Prato", "near")], modifiers={"limit": 3}, idx=1), g("contact", "meccanico", "professional_service", [selection(), f("message", "Puoi venire domani?")], depends_on=["g1"], policy={"confirm_before": ["contact"]}, idx=2)))
    add("ordinal_chain", "Fammi vedere due stampanti a Cesena, poi compra la prima solo se ti autorizzo.", c(g("find", "stampante", "product", [f("location", "Cesena", "near")], modifiers={"limit": 2}, idx=1), g("buy", "stampante", "product", [selection()], depends_on=["g1"], policy={"confirm_before": ["buy"]}, idx=2)))
    add("ordinal_chain", "Seleziona un ristorante tra quelli di Carpi e prenota il terzo dopo che avrò scelto la data e confermato.", c(g("find", "ristorante", "place", [f("location", "Carpi", "near")], idx=1), g("book", "ristorante", "place", [selection(ordinal=3)], depends_on=["g1"], missing=[missing("date")], policy={"confirm_before": ["book"]}, idx=2)), review_flags=("selection_then_third",))
    add("ordinal_chain", "Trovami quattro fiorai ad Asti e manda al secondo 'Avete rose bianche?' dopo che leggo la bozza.", c(g("find", "fioraio", "professional_service", [f("location", "Asti", "near")], modifiers={"limit": 4}, idx=1), g("contact", "fioraio", "professional_service", [selection(ordinal=2), f("message", "Avete rose bianche?")], depends_on=["g1"], policy={"confirm_before": ["contact"]}, idx=2)))

    # 18. Conditional fallback, not parallel calls.
    add("conditional", "Prima guarda i treni per Arezzo domani; se non ce ne sono, cerca pullman per la stessa città.", c(g("find", "treno", "transport", [f("destination", "Arezzo")], temporal=[t("search_date", "domani")], idx=1), g("find", "pullman", "transport", [f("destination", "Arezzo")], temporal=[t("search_date", "domani")], depends_on=["g1"], condition=fallback(), idx=2)))
    add("conditional", "Cerca un ostello a Mantova per sabato; solo se non ne trovi passa agli alberghi.", c(g("find", "ostello", "accommodation", [f("location", "Mantova", "near")], temporal=[t("search_date", "sabato")], idx=1), g("find", "albergo", "accommodation", [f("location", "Mantova", "near")], temporal=[t("search_date", "sabato")], depends_on=["g1"], condition=fallback(), idx=2)))
    add("conditional", "Controlla farmacie aperte a Vasto; se non ne risulta nessuna, cerca quelle a San Salvo.", c(g("find", "farmacia", "place", [f("location", "Vasto", "near"), f("open_now", True)], idx=1), g("find", "farmacia", "place", [f("location", "San Salvo", "near"), f("open_now", True)], depends_on=["g1"], condition=fallback(), idx=2)))
    add("conditional", "Per dopodomani verifica voli diretti da Bari a Londra; se non ci sono, mostrami quelli con scalo.", c(g("find", "volo", "transport", [f("origin", "Bari"), f("destination", "Londra"), f("stops", 0)], temporal=[t("search_date", "dopodomani")], idx=1), g("find", "volo", "transport", [f("origin", "Bari"), f("destination", "Londra"), f("stops", 1, "gte")], temporal=[t("search_date", "dopodomani")], depends_on=["g1"], condition=fallback(), idx=2)), review_flags=("stops_operator",))
    add("conditional", "Se non trovi biciclette a noleggio a Pisa, prova i monopattini; per ora non affittare niente.", c(g("find", "bicicletta", "vehicle", [f("location", "Pisa", "near")], policy={"forbid": ["rent"]}, idx=1), g("find", "monopattino", "vehicle", [f("location", "Pisa", "near")], depends_on=["g1"], condition=fallback(), policy={"forbid": ["rent"]}, idx=2)))

    # 19. User context, references and pronouns.
    add("context", "Non guido: a Caltanissetta cerca una libreria raggiungibile a piedi dal centro.", c(g("find", "libreria", "place", [f("location", "centro di Caltanissetta", "near"), f("access", "a piedi")]), context=[ctx("has_car", False)]))
    add("context", "Siamo in cinque e cerchiamo un ristorante a Spoleto aperto ora; niente prenotazioni per il momento.", c(g("find", "ristorante", "place", [f("location", "Spoleto", "near"), f("open_now", True)], policy={"forbid": ["book"]}), context=[ctx("party_size", 5)]))
    add("context", "Sto a Empoli: trovami un negozio di elettronica vicino a me, senza ordinare prodotti.", c(g("find", "negozio di elettronica", "place", [f("location", "Empoli", "near")], policy={"forbid": ["order"]}), context=[ctx("user_location", "Empoli")]), review_flags=("context_duplicate_location",))
    add("context", "La mia auto è a Frosinone: cerca lì un gommista aperto stamattina.", c(g("find", "gommista", "professional_service", [f("location", "Frosinone", "near"), f("availability", "stamattina")]), context=[ctx("vehicle_location", "Frosinone")]), review_flags=("context_field",))
    add("context", "Trova un hotel a Pordenone e mandami il nome del primo; non prenotarlo.", c(g("find", "hotel", "accommodation", [f("location", "Pordenone", "near")], policy={"forbid": ["book"]})), review_flags=("send_me_result_not_external_contact",))

    # 20. Incomplete requests must not fabricate execution-sensitive values.
    add("incomplete", "Prenota una stanza a Bellaria-Igea Marina; ti dirò più tardi per quale notte.", c(g("book", "stanza", "accommodation", [f("location", "Bellaria-Igea Marina", "near")], missing=[missing("date")])) )
    add("incomplete", "Vorrei un biglietto del treno per Venezia domani, ma non so ancora da dove partirò.", c(g("find", "treno", "transport", [f("destination", "Venezia")], temporal=[t("search_date", "domani")], missing=[missing("origin", "search")])), review_flags=("find_or_book_ambiguous",))
    add("incomplete", "Manda un messaggio a Giulia, però non ho ancora deciso cosa scriverle.", c(g("contact", "Giulia", "person", missing=[missing("message")])) )
    add("incomplete", "Assumi un tecnico per la lavatrice a Isernia; non ho scelto chi chiamare.", c(g("hire", "tecnico lavatrice", "professional_service", [f("location", "Isernia", "near")], missing=[missing("provider")])))
    add("incomplete", "Ricordamelo domani mattina, ma non ti ho detto di che cosa si tratta.", c(g("notify", "promemoria", "reminder", temporal=[t("notification_date", "domani"), t("notification_time", "mattina")], missing=[missing("message")])))

    return rows


def main():
    if DEST.exists():
        raise FileExistsError("V33 is already frozen; do not regenerate")
    rows = cases()
    if len(rows) < 100 or len({normalize_text(r["text"]) for r in rows}) != len(rows):
        raise RuntimeError("V33 requires at least 100 unique cases")
    previous = [(path, r.get("text") or next(m["content"] for m in r["messages"]
                                        if m["role"] == "user"))
                for path in PRIOR if path.is_file() for r in read_jsonl(path)]
    prior_exact = {normalize_text(text) for _, text in previous}
    prior_tokens = [(path, text, token_set(text)) for path, text in previous]
    maximum, nearest = 0.0, None
    for row in rows:
        if normalize_text(row["text"]) in prior_exact:
            raise RuntimeError(f"Exact prior overlap: {row['id']}")
        current = token_set(row["text"])
        for path, prior_text, old in prior_tokens:
            score = len(current & old) / len(current | old) if current | old else 0.0
            if score > maximum:
                maximum = score
                nearest = {"case": row["id"], "source": str(path.relative_to(ROOT)),
                           "prior_text": prior_text}
    within = 0.0
    for i, row in enumerate(rows):
        a = token_set(row["text"])
        for other in rows[:i]:
            b = token_set(other["text"])
            within = max(within, len(a & b) / len(a | b) if a | b else 0.0)
    if maximum >= 0.8 or within >= 0.8:
        raise RuntimeError(f"Near duplicate: prior={maximum:.3f}, within={within:.3f}, nearest={nearest}")
    DEST.mkdir(parents=True, exist_ok=False)
    panel = DEST / "blind.jsonl"
    with panel.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    review = DEST / "human_review_queue.jsonl"
    with review.open("w", encoding="utf-8", newline="\n") as output:
        for row in rows:
            contract = row["contract"]
            guard = check_contract(contract, row["text"])
            output.write(json.dumps({
                "id": row["id"], "request": row["text"], "family": row["family"],
                "intended_interpretation": "no action" if contract["kind"] == "non_actionable"
                    else "; ".join(f"{g['action']} {g['target']['name']}" for g in contract["goals"]),
                "goal_contract_gold": contract,
                "goals": [{key: goal.get(key) for key in
                           ("action", "target", "facts", "temporal", "modifiers", "depends_on",
                            "condition", "missing", "policy") if goal.get(key) is not None}
                          for goal in contract["goals"]],
                "context": contract.get("context", []),
                "expected_planner_without_registry": "NO_ACTION" if contract["kind"] == "non_actionable"
                    else "HOLD_NO_CAPABILITY",
                "guard_v5_on_gold": {"status": guard.status, "codes": guard.codes},
                "review_flags": row["review_flags"], "human_review_status": "pending",
                "human_reviewer": None, "human_notes": None,
            }, ensure_ascii=False) + "\n")
    manifest = {
        "version": "V33", "schema_version": "1.3", "cases": len(rows),
        "families": dict(sorted(Counter(r["family"] for r in rows).items())),
        "panel_sha256": sha256(panel), "review_queue_sha256": sha256(review),
        "generator_sha256": sha256(Path(__file__)),
        "max_prior_token_jaccard": round(maximum, 6),
        "max_within_panel_token_jaccard": round(within, 6),
        "nearest_prior_source": nearest,
        "prior_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in PRIOR if p.is_file()},
        "gold_review": "pending independent human review; agent-authored",
        "limitations": ["Not a random production sample", "All 100 gold cases require human adjudication",
                        "Flagged ontology choices must not be silently normalized after freeze"],
    }
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps({"cases": len(rows), "panel_sha256": manifest["panel_sha256"],
                      "max_prior_token_jaccard": maximum,
                      "max_within_panel_token_jaccard": within}, ensure_ascii=False))


if __name__ == "__main__":
    main()
