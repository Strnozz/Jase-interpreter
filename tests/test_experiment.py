"""Exercise failure boundaries before committing hours of GPU time."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_v20_experiment as experiment


class ExperimentTests(unittest.TestCase):
    def setup_tree(self, root):
        out = root / "reports/v20"
        out.mkdir(parents=True)
        spec = root / "spec.json"
        spec.write_text(json.dumps({"input_hashes": {}, "config": "config.yaml",
            "candidate_adapter": "adapters/new", "baseline_adapter": "adapters/old",
            "dev": "dev.jsonl"}))
        return out, spec

    def test_failed_training_never_runs_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, spec = self.setup_tree(root)
            with patch.multiple(experiment, ROOT=root, OUT=out, SPEC=spec, STATE=out / "status.json"), \
                 patch.object(experiment, "preflight"), \
                 patch.object(experiment, "command_phase", side_effect=RuntimeError("GPU failed")) as run, \
                 patch.object(sys, "argv", ["run_v20_experiment.py"]):
                with self.assertRaisesRegex(RuntimeError, "GPU failed"):
                    experiment.main()
            self.assertEqual([c.args[0] for c in run.call_args_list], ["train"])
            state = json.loads((out / "status.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["completed_phases"], ["preflight"])

    def test_existing_adapter_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, spec = self.setup_tree(root)
            (root / "adapters/new").mkdir(parents=True)
            with patch.multiple(experiment, ROOT=root, OUT=out, SPEC=spec, STATE=out / "status.json"), \
                 patch.object(experiment, "preflight"), \
                 patch.object(experiment, "command_phase") as run, \
                 patch.object(sys, "argv", ["run_v20_experiment.py"]):
                with self.assertRaisesRegex(RuntimeError, "already exists"):
                    experiment.main()
            run.assert_not_called()

    def test_changed_dataset_stops_run(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            data = root / "train.jsonl"
            data.write_text("original")
            spec = {"input_hashes": {"train.jsonl": experiment.sha(data)}}
            data.write_text("changed")
            with patch.object(experiment, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "Frozen input changed"):
                    experiment.verify_inputs(spec)

    def test_preflight_only_does_not_train(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            out, spec = self.setup_tree(root)
            with patch.multiple(experiment, ROOT=root, OUT=out, SPEC=spec, STATE=out / "status.json"), \
                 patch.object(experiment, "preflight") as preflight, \
                 patch.object(experiment, "command_phase") as run, \
                 patch.object(sys, "argv", ["run_v20_experiment.py", "--preflight-only"]):
                experiment.main()
            preflight.assert_called_once()
            run.assert_not_called()
            self.assertEqual(json.loads((out / "status.json").read_text())["status"], "preflight_passed")


if __name__ == "__main__":
    unittest.main()
