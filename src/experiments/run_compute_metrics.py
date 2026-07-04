import argparse
from pathlib import Path

import torch

from src.metrics.input_dependence import (
    compute_input_dependence_score,
    summarize_by_layer,
    summarize_by_dimension,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/metrics")
    args = parser.parse_args()

    stats = torch.load(args.stats_path, map_location="cpu")

    score = compute_input_dependence_score(
        variance=stats["variance"],
        mean_square=stats["mean_square"],
    )

    layer_summary = summarize_by_layer(score)
    dim_summary = summarize_by_dimension(score)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.stats_path).stem

    torch.save(
        {
            "input_dependence_score": score,
            "layer_summary": layer_summary,
            "dimension_summary": dim_summary,
        },
        output_dir / f"{stem}_metrics.pt",
    )

    print(f"Saved metrics to {output_dir / f'{stem}_metrics.pt'}")


if __name__ == "__main__":
    main()