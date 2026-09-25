from __future__ import annotations

import unittest
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jase.multimodel import ROOT, clean_validation, encode_supervised, read_jsonl


def row(text, answer):
    return {"messages": [{"role": "system", "content": "system"},
                         {"role": "user", "content": text},
                         {"role": "assistant", "content": answer}]}


class DummyTokenizer:
    def apply_chat_template(self, turns, add_generation_prompt=False, **_):
        prefix = [1, 2, 3]
        return prefix if add_generation_prompt else prefix + [4, 5]


class MultiModelTests(unittest.TestCase):
    def test_validation_excludes_leakage_conflicts_and_repetitions(self):
        train = [row("Trovami un hotel", "A")]
        valid = [row("Trovami un hotel!", "A"), row("Voglio un volo", "B"),
                 row("Voglio un volo", "C"), row("Trova un treno", "D"),
                 row("Trova un treno", "D")]
        kept, audit = clean_validation(train, valid)
        self.assertEqual([r["messages"][1]["content"] for r in kept], ["Trova un treno"])
        self.assertEqual(audit["overlap_prompts_removed"], 1)
        self.assertEqual(audit["conflicting_prompts_removed"], 1)
        self.assertEqual(audit["duplicate_rows_removed"], 1)

    def test_supervised_labels_only_answer_and_never_truncates(self):
        encoded = encode_supervised(DummyTokenizer(), row("x", "y"), 5)
        self.assertEqual(encoded["labels"], [-100, -100, -100, 4, 5])
        with self.assertRaisesRegex(ValueError, "refusing truncation"):
            encode_supervised(DummyTokenizer(), row("x", "y"), 4)

    def test_lora_selection_never_includes_vision_tower(self):
        import bitsandbytes as bnb
        import torch
        from training.train import select_language_linears

        class Tower(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([
                    torch.nn.ModuleDict({"proj": bnb.nn.Linear4bit(4, 4)}) for _ in range(3)])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = Tower()
                self.visual = Tower()

        names = select_language_linears(Model(), 2)
        self.assertEqual(len(names), 2)
        self.assertTrue(all("language_model" in name for name in names))
        self.assertTrue(all("layers.0." not in name for name in names))

    def test_checkpoint_preserves_optimizer_scheduler_and_rng_for_resume(self):
        import random
        import torch
        from training.train import write_checkpoint

        class Tiny(torch.nn.Linear):
            def save_pretrained(self, path, safe_serialization=True):
                path.mkdir()
                (path / "adapter_config.json").write_text("{}", encoding="utf-8")
                torch.save(self.state_dict(), path / "adapter.pt")

        torch.manual_seed(9)
        random.seed(9)
        model = Tiny(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .9 ** step)
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        expected_rng = torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as folder:
            path = write_checkpoint(Path(folder), 1, model, optimizer, scheduler,
                                    1.5, 0.4, 1, [{"step": 1, "loss": 0.4}], torch)
            saved = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
            self.assertEqual(saved["global_step"], 1)
            self.assertEqual(saved["scheduler"]["last_epoch"], scheduler.state_dict()["last_epoch"])
            self.assertEqual(len(saved["optimizer"]["state"]), len(optimizer.state_dict()["state"]))
            self.assertTrue(torch.equal(saved["torch_rng"], expected_rng))
            self.assertEqual(saved["validation_history"], [{"step": 1, "loss": 0.4}])

    def test_posttrain_probe_prompts_are_not_exact_train_prompts(self):
        from jase.leakage import normalize_text

        train = read_jsonl(ROOT / "data" / "v19_clean" / "train.jsonl")
        prompts = {normalize_text(next(m["content"] for m in row["messages"] if m["role"] == "user"))
                   for row in train}
        probes = read_jsonl(ROOT / "evaluation" / "posttrain_requests.jsonl")
        self.assertEqual(len(probes), len({row["id"] for row in probes}))
        self.assertTrue(all(normalize_text(row["text"]) not in prompts for row in probes))


if __name__ == "__main__":
    unittest.main()
