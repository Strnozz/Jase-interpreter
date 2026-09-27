# Jase Interpreter 9B V25 benchmark

## Result

The new QLoRA run completed at step 600. On the **same frozen 120-case V26
panel**, V25 has 34/120 canonical exact contracts versus 21/120 for V24.
That is 21 paired gains and eight paired losses. The comparison measures the
combined change in contract, prompt, data and adapter; it cannot attribute
the gain to fine-tuning alone. V24 emits GoalContract 1.2 and its predictions
were conservatively projected to 1.3. That projection held 51/120 V24
outputs, so exact comparison is a lower bound for the old representation.

| V26 measure | V24 | V25 |
| --- | ---: | ---: |
| Valid JSON | 119/120 | 116/120 |
| Canonical exact after 1.2→1.3 projection | 21/120 | 34/120 |
| Raw action correct, independent of projection | 106/132 goals | 116/132 goals |
| Raw target type correct | 115/132 | 115/132 |
| Raw goal count correct | 101/120 | 100/120 |
| Candidate projection/schema holds | 51/120 | 22/120 (four non-JSON) |
| Experimental Guard ACCEPT | 47/120 | 92/120 |
| ACCEPT but canonical mismatch, **triage only** | 38 | 58 |
| Output tokens/s | 18.03 | 18.14 |
| Inference peak VRAM | 8.19 GB | 8.18 GB |

The lower projected fact and temporal scores also rose (fact recall 57/156 to
100/156; temporal recall 21/78 to 35/78), but the V24 projection holds bias
these figures. Raw action, target type and goal count above are the more
directly comparable semantic measures.

V25 handles all six appointment cancellations and all six incomplete table
bookings exactly. It gets five of six group-context cases, five reminder cases,
and four direct vehicle rentals. It fixes the earlier V25 case `b19`, where
“ordina i mezzi per prezzo” had been mistaken for the `order` action.

Several core behaviors remain unreliable. V25 gets **zero of six** quoted
non-actionable requests right and turns each into a booking request. It gets
zero of six two-role flight/search/booking requests and zero of six appointment
reschedules exactly. Two accepted find-then-book outputs omit the requested
booking goal. The Guard accepts these false or incomplete interpretations.
The provisional case audit identifies **at least 18/120** Guard-accepted
material omissions or false actions: six quoted commands turned into actions,
two missing booking goals, six omitted direct-flight constraints and four
omitted explicit no-booking policies. This is a lower bound, not a complete
human-labelled risk rate; see `benchmarks/v26/v25_operational_review_draft.json`.

Some exact mismatches are representational. All six transport-sort outputs
preserve `find` and ascending price but call the target `trasporto` rather
than the gold `alternative di viaggio`. Six food-order outputs preserve the
order and confirmation but use `delivery` instead of `delivery_location`.
Five fallback outputs use `result_count` instead of `results_count`; unlike a
mere label difference this may break condition routing if Planner expects
the gold field. V25 often stores reschedule dates as free-form facts rather
than typed `action_date`; the Guard holds reschedule anyway because no
provider capability has been verified.

The frozen V26 panel itself has quality limits: some Italian phrases have
incorrect articles or agreement (for example “una sushi” and “mia controllo”),
and several flight destinations are small towns without an airport. The panel
is six variants per family, AI-authored and targeted; it is not a production
sample. Keep the gold immutable, but create a reviewed successor panel before
using these numbers for a release decision.

## Historical diagnostic

On the older 24-case V25 panel, two gold contracts cannot be safely projected
to 1.3. On the remaining 22, both V24 and V25 get 12/22 canonical exact.
V25 gains `b16` (product model identity) and `b19` (sort versus order), but
loses `b12` (message punctuation/casing; likely harmless) and `b13` (comparison
count remains a fact instead of the V1.3 `compare_count` modifier, a routing
regression). The historical panel has been consulted repeatedly and is only
diagnostic; V24's original native 1.2 result was 13/24 and is a different metric.

## Training and problems

Model `Qwen/Qwen3.5-9B`, pinned revision
`c202236235762e1c871ad0ccb60c8ee5ba337b9a`; BF16 NF4 4-bit QLoRA,
rank 16/alpha 32/dropout 0.05 on the final 16 text layers. V25 has 1,870
train and 276 validation rows, length cap 512, effective batch 4, learning
rate 8e-6, and 600 steps (1.28 effective epochs). Best and final adapter
are the same at step 600. Best validation loss is 0.009704 and final train
loss 0.000174. Low loss on template-derived validation does not predict V26
semantic success. Training took 2,792 seconds (46.5 minutes), processed
361.3 input tokens/s, used 14.09 GB peak VRAM (13.12 GiB) and 2.65 GB
peak process RAM. The final adapter is 106.6 MB.

The first attempt failed at steps 25 and 75 because Windows denied a directory
rename immediately after writing checkpoints. Those checkpoints and run were
preserved. A retry around the checkpoint rename was versioned in commit
`7e1ce4a`; a new smoke and provenance-clean full run completed all checkpoints
without OOM. The selected adapter is in
`training/runs/qwen35-9b-v25/20260926T230452Z-54b5d764/adapter`.
There was no frontend integration, weight merge or replacement of earlier
adapters. The Windows stack still reports fallback PyTorch implementations
for `causal_conv1d` and gated delta kernels, which limits speed.

## Next gate

Do not promote V25. The next iteration should first define canonical field
names and Planner mappings, strengthen the Guard against quoted/non-actionable
requests and missing explicit policies, and create a separately frozen,
linguistically reviewed V27 panel with realistic travel requests. It should
test accepted material errors and false holds, not just exact JSON. Additional
QLoRA is justified only after these deterministic and evaluation fixes are
measured; a lower validation loss alone is insufficient.
