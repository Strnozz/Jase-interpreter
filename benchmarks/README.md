# Benchmark policy

## `gold_v1.jsonl`

`gold_v1` is now a **development/regression benchmark**. It has been inspected
repeatedly during model development and some of its historical cases were also
reproduced by corrective training blocks. It remains extremely useful for
regressions, but it must not be presented as an independent production holdout.

Before every new training run, execute:

```bash
python scripts/leakage_check.py data/<version>/train.jsonl benchmarks/gold_v1.jsonl
```

Exact duplicates are a hard error. Near duplicates are reported for review.

## Production release holdout

A real release candidate should additionally be evaluated on a separate file,
ideally 500+ human-authored requests, that is never used to change the dataset,
Guard, prompt, schema, checkpoint choice, or hyperparameters. Run the evaluator
with:

```bash
python scripts/eval_v2.py --data benchmarks/release_holdout.jsonl \
  --benchmark-role release_holdout ...
python scripts/release_check.py reports/<report>.json --production
```

If the holdout is inspected and then used to make fixes, it becomes a dev set
and a fresh holdout must be created for the next production claim.
