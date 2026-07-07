import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.activations.collect_residuals import collect_residual_stats_from_tokens
from src.data.load_text_chunks import load_wikitext_token_chunks
from src.experiments.run_analysis import pearson_corr, topk_overlap
from src.experiments.run_collect_stats import select_model_configs
from src.metrics.input_dependence import (
    classify_cells,
    compute_input_variation_ratio,
    summarize_classes_by_layer,
    summarize_by_layer,
)
from src.models.load_model import load_hooked_model
from src.stats.running_stats import RunningActivationStats


class FakeModel:
    class cfg:
        n_layers = 1
        d_model = 1

    def run_with_cache(self, tokens, names_filter, return_type):
        name = "blocks.0.hook_resid_post"
        self.assert_cache_args = names_filter(name) and return_type is None
        return None, {name: tokens.float().unsqueeze(-1)}


class CoreTests(unittest.TestCase):
    def test_statistics_and_score(self):
        values = torch.tensor([-1.0, 1.0]).reshape(2, 1, 1)
        stats = RunningActivationStats(1, 1, 1)
        stats.update_layer(0, values)

        self.assertEqual(stats.variance().item(), 1.0)
        ivr = compute_input_variation_ratio(stats.variance(), stats.mean_square)
        self.assertAlmostEqual(ivr.item(), 1.0)
        self.assertEqual(
            compute_input_variation_ratio(torch.tensor([1.0001]), torch.ones(1)).item(),
            1.0,
        )
        self.assertAlmostEqual(
            compute_input_variation_ratio(torch.tensor([1.0]), torch.tensor([26.0])).item(),
            1 / 26,
        )

        stats.metadata = {"model": "test"}
        with patch("src.stats.running_stats.torch.save") as save, patch(
            "src.stats.running_stats.Path.replace"
        ):
            stats.save("unused.pt")
        self.assertEqual(save.call_args.args[0]["metadata"], {"model": "test"})

    def test_layer_median_flattens_layer(self):
        torch.manual_seed(0)
        score = torch.randn(2, 4, 6)
        expected = score.flatten(1).median(dim=1).values
        self.assertTrue(torch.equal(summarize_by_layer(score)["layer_median"], expected))

    def test_snapshot_count_matches_filename(self):
        saved = []

        def capture(stats, path, force=False):
            saved.append((Path(path).name, int(stats.count[0, 0, 0])))

        model = FakeModel()
        with patch.object(RunningActivationStats, "save", capture):
            collect_residual_stats_from_tokens(
                model,
                torch.arange(20).reshape(10, 2),
                seq_len=2,
                batch_size=4,
                device="cpu",
                snapshot_points={3},
                snapshot_dir=Path("."),
            )

        self.assertTrue(model.assert_cache_args)
        self.assertEqual(saved, [("run_n3_stats.pt", 3)])

    def test_split_half_counts(self):
        saved = []

        def capture(stats, path, force=False):
            saved.append((Path(path).name, int(stats.count[0, 0, 0])))

        with patch.object(RunningActivationStats, "save", capture):
            stats = collect_residual_stats_from_tokens(
                FakeModel(),
                torch.arange(20).reshape(10, 2),
                seq_len=2,
                batch_size=4,
                device="cpu",
                snapshot_dir=Path("."),
                split_half_seed=0,
            )

        self.assertEqual(int(stats.count[0, 0, 0]), 10)
        self.assertEqual(saved, [("run_split_a_stats.pt", 5), ("run_split_b_stats.pt", 5)])

    def test_chunk_sampling_is_seeded(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False, **kwargs):
                return [ord(character) for character in text]

        rows = [{"text": "abcdefghijklmnopqrstuvwxyz"}]
        samples = []
        for seed in (3, 3, 4):
            with patch("src.data.load_text_chunks.load_dataset", return_value=rows):
                samples.append(
                    load_wikitext_token_chunks(
                        Tokenizer(), seq_len=2, max_chunks=4, sample_seed=seed
                    )
                )

        self.assertEqual(samples[0].shape, (4, 2))
        self.assertTrue(torch.equal(samples[0], samples[1]))
        self.assertFalse(torch.equal(samples[0], samples[2]))
        self.assertFalse(
            torch.equal(samples[0], torch.tensor([[97, 98], [99, 100], [101, 102], [103, 104]]))
        )

        with patch("src.data.load_text_chunks.load_dataset") as load_dataset:
            empty = load_wikitext_token_chunks(Tokenizer(), seq_len=2, max_chunks=0)
        self.assertEqual(empty.shape, (0, 2))
        load_dataset.assert_not_called()

    def test_chunk_tokenization_preserves_joined_context(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False, **kwargs):
                return [999 if part == "\n\n\n" else ord(part) for part in split(text)]

        def split(text):
            parts = []
            while text:
                if text.startswith("\n\n\n"):
                    parts.append("\n\n\n")
                    text = text[3:]
                else:
                    parts.append(text[0])
                    text = text[1:]
            return parts

        def rows():
            for text in ("one\n", "two"):
                yield {"text": text}

        tokenizer = Tokenizer()
        with patch("src.data.load_text_chunks.load_dataset", return_value=rows()):
            chunks = load_wikitext_token_chunks(tokenizer, seq_len=1)

        expected = tokenizer.encode("one\n\n\ntwo")
        self.assertEqual(chunks.flatten().tolist(), expected)

    def test_convergence_helpers_validate_ranges(self):
        values = torch.linspace(-1, 1, 100_000)
        correlation = pearson_corr(values, values)
        self.assertLessEqual(correlation, 1.0)
        self.assertAlmostEqual(correlation, 1.0)
        with self.assertRaisesRegex(ValueError, "positive"):
            topk_overlap(values, values, k=0)

    def test_model_selection(self):
        models = [{"name": "a"}, {"name": "b"}]
        self.assertEqual(select_model_configs(models, ["all"]), models)
        self.assertEqual(select_model_configs(models, ["b"]), [models[1]])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            select_model_configs(models, ["missing"])

    def test_model_loading_requires_safetensors(self):
        with patch("src.models.load_model.HookedTransformer.from_pretrained") as load:
            load_hooked_model("model", device="cpu", revision="commit")

        self.assertTrue(load.call_args.kwargs["use_safetensors"])

    def test_classes_are_meaningful_and_exhaustive(self):
        ivr = torch.tensor([[[0.0, 0.05], [0.2, 0.8]]])
        energy = torch.tensor([[[0.001, 1.0], [1.0, 1.0]]])
        classes = classify_cells(energy, ivr)
        fractions = summarize_classes_by_layer(classes)

        membership = sum(
            classes[name].int()
            for name in (
                "inactive",
                "consistent_active",
                "mixed_active",
                "input_varying_active",
            )
        )
        self.assertTrue(torch.equal(membership, torch.ones_like(membership)))
        self.assertTrue(torch.allclose(sum(fractions.values()), torch.ones(ivr.shape[0])))
        self.assertTrue(all(value.item() == 0.25 for value in fractions.values()))


if __name__ == "__main__":
    unittest.main()
