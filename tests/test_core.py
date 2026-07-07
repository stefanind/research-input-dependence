import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.activations.collect_residuals import (
    collect_activation_stats_from_tokens,
)
from src.data.load_text_chunks import (
    load_or_create_wikitext_raw_windows,
    tokenize_raw_windows,
)
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
        cache = {
            "blocks.0.hook_resid_pre": tokens.float().unsqueeze(-1),
            "blocks.0.hook_resid_post": (tokens.float() + 10).unsqueeze(-1),
            "blocks.0.hook_attn_out": (tokens.float() + 1).unsqueeze(-1),
            "blocks.0.hook_mlp_out": (tokens.float() + 2).unsqueeze(-1),
        }
        self.assert_cache_args = names_filter("blocks.0.hook_resid_post") and return_type is None
        return None, cache


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
            collect_activation_stats_from_tokens(
                model,
                torch.arange(20).reshape(10, 2),
                seq_len=2,
                batch_size=4,
                targets=["resid_post"],
                device="cpu",
                snapshot_points={3},
                snapshot_dir=Path("."),
            )

        self.assertTrue(model.assert_cache_args)
        self.assertEqual(saved, [("resid_post_n3_stats.pt", 3)])

    def test_split_half_counts(self):
        saved = []

        def capture(stats, path, force=False):
            saved.append((Path(path).name, int(stats.count[0, 0, 0])))

        with patch.object(RunningActivationStats, "save", capture):
            stats = collect_activation_stats_from_tokens(
                FakeModel(),
                torch.arange(20).reshape(10, 2),
                seq_len=2,
                batch_size=4,
                targets=["resid_post"],
                device="cpu",
                snapshot_dir=Path("."),
                split_half_seed=0,
            )["resid_post"]

        self.assertEqual(int(stats.count[0, 0, 0]), 10)
        self.assertEqual(
            saved,
            [("resid_post_split_a_stats.pt", 5), ("resid_post_split_b_stats.pt", 5)],
        )

    def test_collects_multiple_targets_and_resid_delta(self):
        stats = collect_activation_stats_from_tokens(
            FakeModel(),
            torch.arange(6).reshape(3, 2),
            seq_len=2,
            batch_size=2,
            targets=["resid_post", "resid_delta", "attn_out", "mlp_out"],
            device="cpu",
        )

        self.assertEqual(set(stats), {"resid_post", "resid_delta", "attn_out", "mlp_out"})
        self.assertAlmostEqual(stats["resid_delta"].mean[0, 0, 0].item(), 10.0)
        self.assertAlmostEqual(stats["attn_out"].mean[0, 0, 0].item(), 3.0)
        self.assertEqual(stats["resid_delta"].metadata["activation_target"], "resid_delta")

    def test_raw_windows_are_cached_and_tokenized(self):
        class Tokenizer:
            def encode(self, text, add_special_tokens=False, **kwargs):
                return [ord(character) for character in text]

        path = Path("unused.jsonl")
        rows = [{"text": "abcdefghij"}, {"text": "klmnopqrst"}]
        with patch("src.data.load_text_chunks.load_dataset", return_value=rows):
            with patch("pathlib.Path.exists", return_value=False), patch(
                "pathlib.Path.open"
            ), patch("pathlib.Path.replace"), patch("pathlib.Path.mkdir"):
                windows = load_or_create_wikitext_raw_windows(
                    path, max_windows=2, window_chars=5
                )

        chunks = tokenize_raw_windows(Tokenizer(), windows, seq_len=2, max_chunks=2)
        self.assertEqual(chunks.shape, (2, 2))

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
