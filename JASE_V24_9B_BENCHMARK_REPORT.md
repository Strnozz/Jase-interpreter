# Jase Interpreter 9B V24 — training and benchmark

V24 QLoRA completed without OOM. It starts from the **V22 final adapter**, keeps the pinned Qwen/Qwen3.5-9B base revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, and trains on the versioned V24 GoalContract 1.2 corpus. V20–V23 adapters, the 2B baseline and the frontend were not changed.

## Training

| Item | Result |
|---|---:|
| Dataset | 5,005 train / 905 validation; no exact or token-Jaccard ≥0.85 overlap with the new V25 panel |
| New coverage | 120 appointment, 150 vehicle and 60 explicit reminder-message train examples, plus quantity, model identity and temporal-role examples |
| Sequence/loss | max 512; observed max 509; completion-only target mask verified |
| QLoRA | 4-bit NF4, BF16, r16/alpha32/dropout 0.05, last 16 language layers, 21.64M trainable parameters (0.2242%) |
| Steps | 1,250, effective batch 4, ~1 epoch; learning rate 8e-6; AdamW, 50 warmup steps, cosine decay |
| Loss | train last 0.00544; best validation 0.01749 at step 1,250 |
| Resources | 89.0 min; peak allocated VRAM 14.13 GB, process RAM 2.73 GB; 406 input tokens/s |
| Adapter | `training/runs/qwen35-9b-v24/20260926T180548Z-28e20ea2/adapter`, 106.6 MB |

Smoke (3 steps), backward, adapter save/reload and post-reload inference passed. The best and final adapter are the **same step 1,250**, so their duplicate benchmark directories are one result, not independent trials. Validation loss is on the new V24 sample and cannot be compared numerically with V22/V23 validation losses. Logs show no NaN or CUDA OOM. The installed Transformers stack uses reference implementations for several Qwen recurrent/conv kernels, reducing throughput but not correctness.

## Frozen V25 panel (24 cases)

V25 was frozen before corpus construction (SHA-256 `acb5dbfdf2f849bc8e0e16d5ee447052142dd60e891e48a8a2dccfdaaa04de7a`). Same requests, gold and V1.2 metric for all adapters. The panel name is `transfer` in the V22/V24 runner outputs, but its hash identifies V25; V23's separate evaluation calls it `blind`.

| Metric | V22 final | V23 final | V24 final |
|---|---:|---:|---:|
| Full exact interpretation | **5/24** | **7/24** | **13/24** |
| JSON valid | 24/24 | 24/24 | 24/24 |
| Schema valid | 22/24 | 23/24 | 24/24 |
| Target exact accuracy | 36.4% | 56.5% | 83.3% |
| Fact recall on scored outputs | 75.6% | 79.5% | 84.8% |
| Generation speed | 17.84 tok/s | 17.79 tok/s | 17.99 tok/s |
| Inference peak VRAM | 8.18 GB | 8.18 GB | 8.19 GB |

V24 gains eight exact cases over V22 and six over V23, with **no paired exact losses on V25**. Appointment cancellation improves from 0/2 to 2/2; vehicle rental from 0/4 to 2/4. Reminder and comparison cases also improve. This is evidence of transfer to unseen surface forms, but 13/24 is still far below dependable interpretation.

## Historical panels, identical gold

| Full exact interpretation | V22 final | V23 final | V24 final |
|---|---:|---:|---:|
| V20 hard dev | 7/16 | 10/16 | **9/16** |
| V20 holdout | 6/12 | 6/12 | **5/12** |
| 198 V19 convertible | 178/198 | 172/198 | **177/198** |

Relative to V22, V24 gains two hard-dev cases and loses none, loses one holdout case, and has five historical wins versus six losses. The historical 2B result (186/198) is a **projection from earlier V1.1 results**, not a new V1.2 inference and not comparable on V25. Earlier panels have been inspected repeatedly, so they provide regression checks rather than fresh validation.

## Remaining semantic and safety failures

- V25 **b03** invents `date=oggi` alongside the requested Tuesday and loses “morning”; the V1.2 validator does not catch conflicting string equalities.
- **b04** applies `confirm_before` to `contact` when the requested action is `book`; wrong confirmation target.
- **b06** turns a rental request into `find`, **b19** reads “ordina i mezzi per prezzo” as action `order` instead of sorting a `find` result.
- **b14** loses comparison cardinality. **b23** moves group and owned-hotel context into search filters. The experimental routing Guard holds both b14 and b23, but still accepts b03, b06 and b19.
- Holdout **s04** loses missing flight origin/date; Guard 1.2 holds it. **s05** emits unsupported action `propose` for a fallback. Hard-dev **d16** loses user context, and **d05** attaches “tomorrow” to both the search and subsequent booking.
- Five V25 exact disagreements need ontology/normalization review: `8:00` versus `08:00`, model identity as a separate `model` fact versus in target name, and amenity/feature operator or field aliases. Do not silently count them as successes or rewrite the frozen gold.

The experimental routing Guard accepts 20/24 V24 outputs on V25; eight accepted outputs differ from exact gold. A provisional AI audit labels six outputs as material errors or routing risks, of which **three are still accepted**. Base Guard 1.2 accepts 21/24 and four of those six. This is a diagnostic, not a human-reviewed safety rate. The V24 safety panel was already consulted before training and is reported separately as a diagnostic only.

On that previously consulted V24 safety panel, exact success is 10/20 for V22, 11/20 for V23 and **15/20 for V24**; V24 has five paired wins and no losses versus V22. Its routing Guard accepts 15/20, including four exact disagreements. This is useful for regression inspection but cannot validate generalization independently of V25.

## Conclusion

V24 clearly improves the targeted V25 cases compared with both earlier 9B adapters, while slightly regressing on the old holdout and historical baseline. It is **not ready for production or frontend promotion**. The next iteration should prioritize typed GoalContract 1.3 invariants for conflicting facts, action–policy alignment, action/target routing, and context preservation; obtain human review for ambiguous gold; then freeze another genuinely separate panel before any further training. More synthetic examples alone are unlikely to fix the accepted semantic errors.
