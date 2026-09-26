# V25 ambiguity review — draft for human adjudication

This is an **AI-prepared review**, not a completed human review. The frozen V25 gold and V24 raw predictions are unchanged. `Canonical` refers only to documented rules in `jase/canonical_semantics.py`; none of the disputed provider mappings below is assumed.

| ID and request | Frozen gold | V24 output | Normalized comparison | Assessment and ontology decision needed |
|---|---|---|---|---|
| **b07** “Noleggia un motorino a Varese domani, meglio se elettrico.” | Soft `feature=eq elettrico` | Soft `product_model=eq elettrico` | Still different | Representation may change the provider query: `elettrico` is a propulsion feature, not a model ID. Keep as unresolved field mismatch; define vehicle feature vocabulary. |
| **b10** “Domani alle 8 avvisami di prendere le medicine.” | `time=08:00` | `time=8:00` | **Equal** after safe clock zero-padding | Formatting only. Admit `H:MM → HH:MM` for valid 24-hour clock strings; preserve the original text separately. |
| **b15** “Controlla se il portatile Zenbook Q77 scende sotto 900 euro.” | Target `portatile Zenbook Q77` | Target `portatile`, hard `model=Zenbook Q77` | Still different | Potentially equivalent only if Planner joins target+model into one exact catalog query. Require a provider mapping and test before admitting this alias. |
| **b16** “Compra il caricatore VoltPro M2 nero, massimo 35 euro.” | Target `caricatore VoltPro M2` | Target `caricatore`, hard `model=VoltPro M2` | Still different | Same unresolved identity representation as b15, with higher consequence because action is `buy`. Preserve a hold until exact model routing is demonstrated. |
| **b20** “Trova un agriturismo a Cremona con parcheggio; la piscina sarebbe un bonus.” | Hard/soft `amenity=eq` | Hard/soft `amenity=contains` | Still different | For boolean amenities this might match the same inventory, but `contains` may also match partial text. Do not equate until the provider semantics are explicit. |

The canonical comparator changes V24 from **13/24 legacy exact to 14/24 canonical exact**, solely because of b10. The other four cases remain unresolved. No canonicalization rule changes hard versus soft strength, arrival versus departure, action versus modifier, context versus filter, or target identity without a proven mapping.
