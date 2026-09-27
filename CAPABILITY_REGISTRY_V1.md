# Capability Registry V1 — dry run

`configs/capabilities/v1.json` defines 21 explicit capability interfaces. `jase/capability_registry_v1.py` validates and resolves them. These are **theoretical mock interfaces**, independent of Interpreter output: no provider adapter or frontend uses them, and `execution_status` must be `mock_only`.

The registry covers restaurant/hotel/flight/train/vehicle/provider search; restaurant/hotel/flight booking; vehicle rental; appointment creation, change and cancellation; reminders; contact; hiring; and comparisons. Each entry records action, target type and names, declared aliases, required/optional slots, modifiers, temporal roles, references, policies, provider binding class, consequence level and confirmation requirement. It deliberately contains no city names or benchmark-specific people. A wildcard target name is allowed only for contacting a `person`, and `fact.provider_ref` is still mandatory.

Resolution compares **action + target type + registered target name/alias**. Unknown combinations return `HOLD_NO_CAPABILITY`; two matching specifications return `HOLD_AMBIGUOUS_CAPABILITY`. `treno regionale` is an explicit alias of `find.train`; `direct` is not silently converted to `stops=0`. Other seemingly equivalent fields require explicit, tested binding in the Planner. An action cannot be reinterpreted as a different action to make it routable.

The registry's required slots are conservative. For example, `book.hotel` requires a concrete selection, check-in, check-out and guests; `hire.service_provider` requires a verified provider reference and location; `appointment.create` requires provider reference, date and time. The Planner must check these regardless of what the model puts in `missing`. Consequential actions require confirmation at the final policy boundary. `required_policies` records this requirement; ordinary searches do not require an unrequested `forbid` policy.

This version is suitable only for contract resolution and planning experiments. Provider result binding, human-reviewed ontology, consent state and execution safety have not been established. No registry entry grants execution permission.
