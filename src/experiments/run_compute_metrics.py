import argparse
from pathlib import Path

import torch
import yaml

from src.experiments.artifacts import artifact_paths, atomic_torch_save
from src.metrics.input_dependence import (
    compute_input_dependence_score,
    classify_energy_variance_buckets,
    summarize_buckets_by_layer,
    summarize_by_layer,
    summarize_by_dimension,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", nargs="+", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    output_dir = Path(args.output_dir) if args.output_dir else artifact_paths(cfg).metrics
    output_dir.mkdir(parents=True, exist_ok=True)

    for stats_path in args.stats_path:
        stats = torch.load(stats_path, map_location="cpu")

        score = compute_input_dependence_score(
            variance=stats["variance"],
            mean_square=stats["mean_square"],
        )

        layer_summary = summarize_by_layer(score)
        dim_summary = summarize_by_dimension(score)
        buckets = classify_energy_variance_buckets(
            stats["variance"], stats["mean_square"], score
        )
        bucket_summary = summarize_buckets_by_layer(buckets)

        stem = Path(stats_path).stem.removesuffix("_stats")
        output_path = output_dir / f"{stem}_metrics.pt"

        atomic_torch_save(
            {
                "input_dependence_score": score,
                "layer_summary": layer_summary,
                "dimension_summary": dim_summary,
                "bucket_summary": bucket_summary,
                "bucket_thresholds": {
                    "ids": buckets["ids_threshold"],
                    "energy": buckets["energy_threshold"],
                },
                "metadata": stats.get("metadata", {}),
            },
            output_path,
            force=args.force,
        )

        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
