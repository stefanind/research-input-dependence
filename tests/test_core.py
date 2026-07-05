import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.activations.collect_residuals import collect_residual_stats_from_tokens
from src.data.load_text_chunks import load_wikitext_token_chunks
from src.metrics.input_dependence import compute_input_dependence_score, summarize_by_layer
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
        score = compute_input_dependence_score(stats.variance(), stats.mean_square)
        self.assertAlmostEqual(score.item(), 1.0)

    def test_layer_median_flattens_layer(self):
        torch.manual_seed(0)
        score = torch.randn(2, 4, 6)
        expected = score.flatten(1).median(dim=1).values
        self.assertTrue(torch.equal(summarize_by_layer(score)["layer_median"], expected))

    def test_snapshot_count_matches_filename(self):
        saved = []

        def capture(stats, path):
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

    def test_chunk_limit_stops_iteration(self):
        seen = []

        def rows():
            for text in ("abcd", "unused"):
                seen.append(text)
                yield {"text": text}

        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return list(range(len(text)))

        with patch("src.data.load_text_chunks.load_dataset", return_value=rows()):
            chunks = load_wikitext_token_chunks(Tokenizer(), seq_len=2, max_chunks=1)

        self.assertEqual(chunks.shape, (1, 2))
        self.assertEqual(seen, ["abcd"])

        with patch("src.data.load_text_chunks.load_dataset") as load_dataset:
            empty = load_wikitext_token_chunks(Tokenizer(), seq_len=2, max_chunks=0)
        self.assertEqual(empty.shape, (0, 2))
        load_dataset.assert_not_called()


if __name__ == "__main__":
    unittest.main()
