# Review changes — 23 Sep 2026

Changes made after an independent review of the current Jase Interpreter tree.

1. Added `jase/leakage.py` and `scripts/leakage_check.py`.
2. `generate_dataset_v2.py` now removes exact duplicates of `gold_v1` before
   train/valid splitting by default.
3. Generated `data/v19_clean`: 27,504 train + 2,310 valid rows; 186 exact gold
   duplicates removed; full Guard sweep passes all 29,814 rows.
4. Fixed `validate_dataset.py` to validate GoalContract schema v1.1 instead of
   the obsolete v1.0 schema.
5. Clarification fallback now emits schema v1.1 and preserves grounded target /
   action context while discarding suspect predicates.
6. Clarification reasons no longer map every semantic problem to missing target.
7. Added omission detection for written quantities such as `due biglietti`.
8. Fixed a strictness false positive where `no glutine ... sotto i 5 euro`
   could flip `lt` into `gte` because the earlier `no` was incorrectly treated
   as governing the price comparison.
9. Added `benchmark_role` to evaluator reports and a `--production` mode to
   `release_check.py`; a dev benchmark can no longer be mistaken for an
   independent production holdout.
10. Added controlled v19 training configs. The first run keeps 8 layers/rank 16
    so the only major training-data change versus v18 is decontamination.

Static checks completed in this environment:

- `tests/test_guard.py`: 170/170 pass.
- `validate_dataset.py --data data/v19_clean`: 29,814 rows, 0 failures.
- full Semantic Guard sweep over v19 train+valid: 29,814/29,814 accepted.
- exact train↔gold duplicates: 0 after decontamination.
- coverage checker: no structural/lexical holes relative to current dev gold.

MLX training itself cannot run in this Linux environment. Jobs 200–209 have
therefore been prepared in `runs/queue/` for the existing macOS `jase_runner`.
They first re-evaluate the current best adapter with the new Guard, then train
and evaluate `jase-v19-clean` at 10,500 / 12,000 / 13,000 iterations.

The 16-layer capacity config is prepared as
`configs/mlx_lora_v19_capacity16.yaml` but intentionally not queued yet: first
measure the clean baseline, then decide whether the extra multi-hour run is
worth doing.
