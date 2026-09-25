"""Probe a completed adapter on predeclared requests without benchmark tuning."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluation.compare import evaluate_output, load_backend  # noqa: E402
from jase.multimodel import load_config, read_jsonl, sha256  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("HF_HOME", str(ROOT / "hf-cache"))
    path = Path(args.manifest)
    if not path.is_absolute():
        path = ROOT / path
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "full" or manifest.get("status") != "complete":
        parser.error("Expected a completed full training manifest")
    cfg = load_config(manifest["model"])
    adapter = ROOT / manifest["adapter"]
    schema = json.loads((ROOT / cfg["schema"]).read_text(encoding="utf-8"))
    requests = ROOT / "evaluation" / "posttrain_requests.jsonl"
    infer = load_backend(cfg, adapter)
    rows = []
    for case in read_jsonl(requests):
        performance = infer(case["text"])
        evaluation = evaluate_output(performance["output"], None, case["text"], schema)
        rows.append({"id": case["id"], "category": case["category"],
                     "text": case["text"], "output": performance["output"],
                     "performance": {k: v for k, v in performance.items() if k != "output"},
                     "evaluation": evaluation})
        print(json.dumps({"id": case["id"], "json_valid": evaluation["json_valid"],
                          "schema_valid": evaluation["schema_valid"],
                          "guard": evaluation["guard"]}), flush=True)
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "run_id": manifest["run_id"], "model": cfg["model_id"],
              "revision": cfg["revision"], "adapter": str(adapter),
              "requests_sha256": sha256(requests), "evaluator_sha256": sha256(Path(__file__)),
              "cases": rows}
    out = Path(args.output)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
