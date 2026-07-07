import argparse
from pathlib import Path

import torch
import yaml

from src.experiments.artifacts import artifact_paths, atomic_torch_save
from src.metrics.input_dependence import (
    classify_cells,
    compute_input_variation_ratio,
    summarize_classes_by_layer,
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

        ivr = compute_input_variation_ratio(
            variance=stats["variance"],
            mean_square=stats["mean_square"],
        )

        layer_summary = summarize_by_layer(ivr)
        dim_summary = summarize_by_dimension(ivr)
        classes = classify_cells(
            stats["mean_square"], ivr, **cfg["classification"]
        )
        class_summary = summarize_classes_by_layer(classes)

        stem = Path(stats_path).stem.removesuffix("_stats")
        output_path = output_dir / f"{stem}_metrics.pt"

        atomic_torch_save(
            {
                "mean_activation": stats["mean"],
                "rms_activation": stats["mean_square"].sqrt(),
                "input_variation_ratio": ivr,
                "layer_summary": layer_summary,
                "dimension_summary": dim_summary,
                "class_summary": class_summary,
                "classification": {
                    **cfg["classification"],
                    "energy_reference": classes["energy_reference"],
                    "energy_threshold": classes["energy_threshold"],
                },
                "metadata": stats.get("metadata", {}),
            },
            output_path,
            force=args.force,
        )

        print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
