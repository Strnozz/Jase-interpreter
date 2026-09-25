#!/usr/bin/env python3
"""Build and audit the v20 data-only experiment without touching v19/holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from generate_dataset_v2 import as_chat, generate
from jase.dataset_quality import audit_splits, clean_rows, grouped_split, select_dev, tags
from jase.guard import ACCEPT, check_contract
from jase.leakage import load_examples, normalize_text


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/v20_quality")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--dev-size", type=int, default=256)
    args = ap.parse_args()
    out = ROOT / args.out
    if out.exists():
        raise SystemExit(f"Refusing to replace existing dataset: {out}")
    rows = generate(random.Random(args.seed), args.n)
    benchmarks = [ROOT / "benchmarks/gold_v1.jsonl",
                  ROOT / "benchmarks/release_holdout_v1.jsonl"]
    # Only aggregate decontamination statistics leave this block. The release
    # holdout is neither inspected for errors nor used to design new examples.
    excluded = {}
    for path in benchmarks:
        forbidden = {normalize_text(ex.text) for ex in load_examples(path)}
        kept = [r for r in rows if normalize_text(r[0]) not in forbidden]
        excluded[path.name] = len(rows) - len(kept)
        rows = kept
    rows, cleaning, quarantine = clean_rows(rows)
    train, valid = grouped_split(rows, args.seed + 1)
    audit = audit_splits(train, valid)
    issues = Counter()
    for i, (text, contract, _) in enumerate(rows, 1):
        res = check_contract(contract, text, ground=True)
        if res.status != ACCEPT or res.repairs:
            issues.update(issue.code for issue in res.issues if issue.severity == "error")
            if res.repairs:
                issues["repairs_required"] += 1
            if not res.issues and not res.repairs:
                issues[res.status] += 1
        if i % 5000 == 0:
            print(f"Guard checked {i}/{len(rows)}", flush=True)
    prior = ROOT / "data/v19_clean/train.jsonl"
    prior_texts = {normalize_text(ex.text) for ex in load_examples(prior)}
    eligible = [r for r in valid if normalize_text(r[0]) not in prior_texts]
    dev = select_dev(valid, args.dev_size, args.seed + 2,
                     preferred={r[0] for r in eligible})
    seen_v19 = {t for t, _, _ in dev if normalize_text(t) in prior_texts}
    manifest = {
        "name": "v20_quality", "seed": args.seed, "requested_rows": args.n,
        "purpose": "Synthetic development experiment, not an independent release test",
        "cleaning": cleaning, "decontamination": excluded, "audit": audit,
        "guard_checked": len(rows), "guard_issues": dict(issues),
        "dev_size": len(dev), "dev_eligible_unseen_v19": len(eligible),
        "dev_exact_overlap_v19_train": len(seen_v19),
        "dev_comparison": "Use v19-exact-unseen for comparison; seen cases are diagnostics only. "
                          "All dev cases are held out from v20 training. Same generator, not OOD.",
        "dev_tags": dict(Counter(t for _, c, _ in dev for t in tags(c))),
        "source_hashes": {str(p.relative_to(ROOT)): sha(p) for p in [
            ROOT / "scripts/generate_dataset_v2.py", ROOT / "jase/dataset_quality.py",
            ROOT / "scripts/prepare_v20.py", ROOT / "jase/guard.py", prior, *benchmarks]},
    }
    if not audit["passed"] or issues:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise SystemExit("Dataset checks failed; no training dataset written")
    out.mkdir(parents=True)
    for name, split in (("train", train), ("valid", valid)):
        write_jsonl(out / f"{name}.jsonl", [
            {**as_chat(t, c), "source_scenario_id": f, "tags": sorted(tags(c))}
            for t, c, f in split])
    write_jsonl(out / "dev.jsonl", [
        {"id": f"v20-dev-{i:03d}", "text": t, "contract": c,
         "source_scenario_id": f,
         "tags": sorted(tags(c) | {"v19-exact-seen" if t in seen_v19 else "v19-exact-unseen"}),
         "mandatory": True}
        for i, (t, c, f) in enumerate(dev, 1)])
    write_jsonl(out / "quarantine.jsonl", [
        {"text": t, "contract": c, "family": f, "reason": "conflicting_labels"}
        for t, c, f in quarantine])
    manifest["files"] = {p.name: sha(p) for p in sorted(out.glob("*.jsonl"))}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
