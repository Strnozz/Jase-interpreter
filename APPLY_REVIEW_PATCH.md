# Apply this overlay

Unzip the archive **inside the root of your existing `jase-interpreter-starter`**
and allow files with the same name to be replaced. The archive does not contain
`.venv` or existing adapters/checkpoints.

After applying it on the Mac, if `scripts/jase_runner.py` is already running it
will discover jobs `200`–`209` automatically. Otherwise start it with:

```bash
cd ~/Desktop/jase-interpreter-starter
.venv/bin/python scripts/jase_runner.py
```

The queued sequence runs static preflight checks, re-evaluates the current best
adapter with the new Guard, trains `jase-v19-clean`, extracts checkpoints at
10,500 and 12,000 iterations, and evaluates them plus the final 13,000 model.

Do not queue the 16-layer capacity run yet. First inspect reports 077–080. The
capacity config is included and can be queued only if the clean baseline makes
it worth the extra several hours.
