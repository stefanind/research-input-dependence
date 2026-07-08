import argparse
import subprocess
import sys

import yaml

from src.experiments.artifacts import artifact_paths, ensure_artifact_dirs, ensure_manifest
from src.experiments.run_collect_stats import activation_targets, safe_name, select_model_configs


def run_module(module: str, arguments: list[str]) -> None:
    subprocess.run([sys.executable, "-m", module, *arguments], check=True)


def complete_or_missing(paths) -> str:
    existing = [path.exists() for path in paths]
    if all(existing):
        return "complete"
    if not any(existing):
        return "missing"
    return "partial"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["all"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--clear-hf-cache", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, "r") as f:
        models_cfg = yaml.safe_load(f)

    models = select_model_configs(models_cfg["models"], args.model)
    paths = artifact_paths(cfg)
    ensure_artifact_dirs(paths)
    ensure_manifest(cfg, models_cfg, paths, force=args.force)
    targets = activation_targets(cfg["activations"])
    snapshot_cfg = cfg.get("snapshots", {})
    snapshots = snapshot_cfg.get("points", []) if snapshot_cfg.get("enabled", False) else []
    split_half = cfg.get("reliability", {}).get("split_half", False)

    models_to_collect = []
    for model in models:
        model_key = safe_name(model["name"])
        collection = []
        for target in targets:
            collection.append(paths.stats / model_key / f"{target}_stats.pt")
            collection += [
                paths.stats / model_key / f"{target}_n{n}_stats.pt" for n in snapshots
            ]
            if split_half:
                collection += [
                    paths.stats / model_key / f"{target}_split_{side}_stats.pt"
                    for side in ("a", "b")
                ]

        state = complete_or_missing(collection)
        if args.force or state == "missing":
            models_to_collect.append(model["name"])
        elif state == "partial":
            raise RuntimeError(f"Incomplete collection for {model['name']}; use --force")

    if models_to_collect:
        command = [
            "--model",
            *models_to_collect,
            "--config",
            args.config,
            "--models-config",
            args.models_config,
        ]
        if args.force:
            command.append("--force")
        if args.clear_hf_cache:
            command.append("--clear-hf-cache")
        run_module("src.experiments.run_collect_stats", command)

    for model in models:
        model_key = safe_name(model["name"])
        for target in targets:
            stats_path = paths.stats / model_key / f"{target}_stats.pt"
            metrics_path = paths.metrics / model_key / f"{target}_metrics.pt"
            if args.force or not metrics_path.exists():
                command = ["--stats-path", str(stats_path), "--config", args.config]
                if args.force:
                    command.append("--force")
                run_module("src.experiments.run_compute_metrics", command)

    final_stats = [
        paths.stats / safe_name(model["name"]) / f"{target}_stats.pt"
        for model in models
        for target in targets
    ]
    analysis_outputs = []
    for model in models:
        model_key = safe_name(model["name"])
        for target in targets:
            analysis_outputs.append(paths.figures / model_key / f"{target}_summary.png")
            if len(snapshots) >= 2:
                analysis_outputs.append(
                    paths.tables / model_key / f"{target}_convergence.csv"
                )
            if split_half:
                analysis_outputs.append(
                    paths.tables / model_key / f"{target}_reliability.csv"
                )
    for target in targets:
        analysis_outputs.extend(
            [
                paths.tables / "comparison" / f"{target}_model_layer_summary.csv",
                paths.figures / "comparison" / f"{target}_model_comparison.png",
            ]
        )

    state = complete_or_missing(analysis_outputs)
    if args.force or state == "missing":
        command = [
            "--stats-path",
            *(str(path) for path in final_stats),
            "--config",
            args.config,
        ]
        if args.force:
            command.append("--force")
        run_module("src.experiments.run_analysis", command)
    elif state == "partial":
        raise RuntimeError("Incomplete analysis artifacts; use --force")


if __name__ == "__main__":
    main()
