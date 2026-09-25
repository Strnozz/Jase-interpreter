"""Audit existing frozen splits without moving or rewriting their examples."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jase.multimodel import case, canonical_sha256, clean_validation, dataset_stats, read_jsonl  # noqa: E402
from jase.leakage import normalize_text  # noqa: E402


def keys(path):
    return {normalize_text(case(row)["text"]) for row in read_jsonl(path)}


def main():
    paths = {"train": ROOT / "data/v19_clean/train.jsonl",
             "validation": ROOT / "data/v19_clean/valid.jsonl",
             "dev_panel": ROOT / "data/v20_quality/dev.jsonl",
             "test_holdout": ROOT / "benchmarks/release_holdout_v1.jsonl"}
    stats = {name: dataset_stats(path) for name, path in paths.items()}
    texts = {name: keys(path) for name, path in paths.items()}
    intersections = {f"{a}_x_{b}": len(texts[a] & texts[b]) for a, b in
                     (("train", "validation"), ("train", "dev_panel"),
                      ("train", "test_holdout"), ("validation", "test_holdout"))}
    effective_valid, cleaning = clean_validation(read_jsonl(paths["train"]), read_jsonl(paths["validation"]))
    manifest = {"dataset_version": "v19_clean", "training_seed": 2026,
                "generation_seed": {"value": None,
                                    "note": "Original v19 invocation was not persisted; generator default is 2026"},
                "source_manifest_sha256": None,
                "splits": stats, "canonical_lf_sha256": {
                    name: canonical_sha256(path) for name, path in paths.items()},
                "exact_text_intersections": intersections,
                "effective_validation_rows": len(effective_valid),
                "validation_cleaning": cleaning,
                "roles": {"train": "training", "validation": "loss and tuning",
                          "dev_panel": "v20 validation subset; 10 cases overlap v19 training and are excluded from the primary panel",
                          "test_holdout": "independent release set; sealed, never for tuning"},
                "limitations": ["Four historical train-validation text overlaps are filtered from validation in the new trainer; the 2B training corpus is untouched",
                                "Synthetic train and validation share a generator",
                                "Release holdout has 27 examples and was written by the project author",
                                "No human-sourced independent test set is present"]}
    if intersections["train_x_test_holdout"] or intersections["validation_x_test_holdout"]:
        raise SystemExit(f"Unexpected split text overlap: {intersections}")
    out = ROOT / "docs" / "MULTIMODEL_DATA_MANIFEST.json"
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
