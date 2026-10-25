import json
import tempfile
import unittest
from pathlib import Path

from cpu_ai_lab.data import make_extraction_splits, load_splits, save_splits, score_response, validate_splits


class DataTests(unittest.TestCase):
    def test_deterministic_disjoint_values_and_correct_targets(self):
        splits = make_extraction_splits(24, 8, 8, seed=7)
        self.assertEqual(splits, make_extraction_splits(24, 8, 8, seed=7))
        seen = set()
        for rows in splits.values():
            values = {value for row in rows for value in row["fields"].values()}
            self.assertFalse(seen & values)
            seen |= values
            for row in rows:
                self.assertEqual(json.loads(row["response"]), row["fields"])
                for value in row["fields"].values():
                    self.assertIn(str(value), row["prompt"])

    def test_rejects_cross_split_leakage(self):
        splits = make_extraction_splits(4, 2, 2)
        splits["test"][0] = splits["train"][0].copy()
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_splits(splits)

    def test_semantic_json_metric_is_strict(self):
        target = '{"user":1234,"ticket":5678}'
        self.assertTrue(score_response('{"ticket": 5678, "user": 1234}', target)["json_exact"])
        self.assertFalse(score_response('```json\n' + target + '\n```', target)["valid_json"])
        self.assertFalse(score_response('{"user":"1234","ticket":5678}', target)["json_exact"])
        self.assertFalse(score_response('{"user":true}', '{"user":1}')["json_exact"])
        self.assertFalse(score_response(target + ' extra', target)["valid_json"])

    def test_round_trip_and_checksums(self):
        splits = make_extraction_splits(4, 2, 2)
        with tempfile.TemporaryDirectory() as temp:
            hashes = save_splits(splits, Path(temp))
            self.assertEqual(load_splits(Path(temp)), splits)
            self.assertTrue(all(len(value) == 64 for value in hashes.values()))

    def test_json_rejects_nonstandard_numbers_and_duplicate_keys(self):
        for text in ('{"user":NaN}', '{"user":Infinity}', '{"user":-Infinity}', '{"user":1,"user":2}'):
            self.assertFalse(score_response(text, '{"user":2}')["valid_json"])


class TrainingMeasurementTests(unittest.TestCase):
    def test_heldout_loss_weights_target_tokens(self):
        import torch
        from types import SimpleNamespace
        from cpu_ai_lab.training import evaluate_loss

        class Model:
            def eval(self):
                return self

            def __call__(self, input_ids, **kwargs):
                return SimpleNamespace(loss=input_ids[0, 0].float())

        encoded = [
            {"input_ids": [1, 2], "labels": [-100, 2]},
            {"input_ids": [3, 2, 3, 4], "labels": [-100, 2, 3, 4]},
        ]
        metric = evaluate_loss(Model(), encoded, pad_token_id=0, batch_size=1)
        self.assertEqual(metric["target_tokens"], 4)
        self.assertAlmostEqual(metric["loss"], 2.5)

    def test_prompt_and_padding_are_excluded_from_training(self):
        from cpu_ai_lab.training import collate_batch, encode_examples

        class Tokenizer:
            eos_token_id = 9
            chat_template = None

            def encode(self, text, **kwargs):
                return [1, 2, 3] if text.startswith("User:") else [4, 5]

        encoded = encode_examples(Tokenizer(), [{"prompt": "record", "response": "answer"}], 6)
        self.assertEqual(encoded[0]["labels"], [-100, -100, -100, 4, 5, 9])
        batch = collate_batch(encoded + [{"input_ids": [1, 2], "labels": [-100, 2]}], 0)
        self.assertEqual(batch["labels"][1].tolist(), [-100, 2, -100, -100, -100, -100])
        self.assertEqual(batch["input_ids"].device.type, "cpu")
        with self.assertRaisesRegex(ValueError, "silent truncation"):
            encode_examples(Tokenizer(), [{"prompt": "record", "response": "answer"}], 5)

    def test_cpu_guard_rejects_other_tensor_devices(self):
        import torch
        from cpu_ai_lab.training import assert_cpu_model

        assert_cpu_model(torch.nn.Linear(2, 2, device="cpu"))
        with self.assertRaisesRegex(RuntimeError, "Non-CPU"):
            assert_cpu_model(torch.nn.Linear(2, 2, device="meta"))


if __name__ == "__main__":
    unittest.main()
