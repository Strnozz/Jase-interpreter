"""Distribution and conservative cross-split leakage audit, without inference."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jase.leakage import normalize_text, token_set  # noqa: E402


def load(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        text = row.get("text") or next(m["content"] for m in row["messages"] if m["role"] == "user")
        contract = row.get("contract") or json.loads(next(m["content"] for m in row["messages"] if m["role"] == "assistant"))
        out.append((row.get("id") or row.get("source_id") or hashlib.sha256(text.encode()).hexdigest()[:16], text, contract))
    return out


def distribution(rows):
    counts = Counter()
    actions = Counter()
    facts = Counter()
    for _, text, c in rows:
        counts["rows"] += 1
        counts["non_actionable"] += c["kind"] == "non_actionable"
        counts["multi_goal"] += len(c["goals"]) > 1
        counts["context"] += bool(c.get("context"))
        counts["long_input_100_chars"] += len(text) >= 100
        for g in c["goals"]:
            actions[g["action"]] += 1
            counts["goals"] += 1
            counts["facts"] += len(g["facts"])
            counts["soft_facts"] += sum(f["strength"] == "soft" for f in g["facts"])
            counts["hard_facts"] += sum(f["strength"] == "hard" for f in g["facts"])
            counts["missing"] += len(g.get("missing", []))
            counts["dependency"] += bool(g.get("depends_on"))
            counts["condition"] += "condition" in g
            counts["confirmation"] += bool(g.get("policy", {}).get("confirm_before"))
            counts["prohibition"] += bool(g.get("policy", {}).get("forbid"))
            counts["ranking"] += "ranking" in g
            counts["value_reference"] += sum("value_ref" in f for f in g["facts"])
            facts[len(g["facts"])] += 1
    return {"counts": dict(counts), "actions": dict(actions), "facts_per_goal": dict(sorted(facts.items()))}


def near_audit(left, right, threshold=.85):
    """Approximate candidates via three rare tokens, then exact Jaccard."""
    index = defaultdict(set)
    sets = [token_set(text) for _, text, _ in left]
    for i, tokens in enumerate(sets):
        for token in tokens:
            index[token].add(i)
    exact = {normalize_text(text) for _, text, _ in left}
    exact_count = 0
    near_count = 0
    max_score = 0.0
    examples = []
    for rid, text, _ in right:
        key = normalize_text(text)
        exact_count += key in exact
        tokens = token_set(text)
        candidates = set()
        for token in sorted(tokens, key=lambda t: len(index[t]))[:3]:
            candidates.update(index[token])
        best = (0.0, None)
        for i in candidates:
            other = sets[i]
            if not tokens or not other:
                continue
            if min(len(tokens), len(other)) / max(len(tokens), len(other)) < threshold:
                continue
            value = len(tokens & other) / len(tokens | other)
            if value > best[0]:
                best = (value, left[i][0])
        max_score = max(max_score, best[0])
        if best[0] >= threshold and key not in exact:
            near_count += 1
            if len(examples) < 10:
                examples.append({"candidate_id": rid, "source_id": best[1], "jaccard": round(best[0], 3)})
    return {"exact_normalized_overlap": exact_count, "near_candidates_ge_0_85": near_count,
            "max_token_jaccard": round(max_score, 3), "near_examples": examples,
            "method": "token-set Jaccard after rare-token candidate selection; heuristic, not proof of independence"}


def main():
    paths = {"train": ROOT / "data/v20/train.jsonl", "valid": ROOT / "data/v20/valid.jsonl",
             "hard_dev": ROOT / "benchmarks/v20/hard_dev.jsonl",
             "holdout": ROOT / "benchmarks/v20/holdout.jsonl",
             "v19_train": ROOT / "data/v19_clean/train.jsonl",
             "v19_valid": ROOT / "data/v19_clean/valid.jsonl",
             "v19_dev": ROOT / "data/v20_quality/dev.jsonl",
             "v19_release": ROOT / "benchmarks/release_holdout_v1.jsonl"}
    data = {name: load(path) for name, path in paths.items()}
    report = {"input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in paths.items()},
              "distribution": {name: distribution(data[name]) for name in ("train", "valid", "hard_dev", "holdout")},
              "leakage": {}}
    for left, right in (("train", "valid"), ("train", "hard_dev"), ("valid", "hard_dev"),
                        ("train", "holdout"), ("valid", "holdout"), ("hard_dev", "holdout"),
                        ("v19_train", "holdout"), ("v19_valid", "holdout"),
                        ("v19_dev", "holdout"), ("v19_release", "holdout")):
        report["leakage"][f"{left}_to_{right}"] = near_audit(data[left], data[right])
    (ROOT / "data/v20/audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
