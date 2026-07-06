import argparse
import subprocess
import sys

import yaml

from src.experiments.artifacts import artifact_paths, ensure_artifact_dirs, ensure_manifest
from src.experiments.run_collect_stats import select_model_configs


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
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, "r") as f:
        models_cfg = yaml.safe_load(f)

    models = select_model_configs(models_cfg["models"], args.model)
    paths = artifact_paths(cfg)
    ensure_artifact_dirs(paths)
    ensure_manifest(cfg, models_cfg, paths, force=args.force)
    hook = cfg["activations"]["hook_name"]
    snapshot_cfg = cfg.get("snapshots", {})
    snapshots = snapshot_cfg.get("points", []) if snapshot_cfg.get("enabled", False) else []
    split_half = cfg.get("reliability", {}).get("split_half", False)

    for model in models:
        stem = f"{model['name'].replace('/', '__')}_{hook}"
        stats_path = paths.stats / f"{stem}_stats.pt"
        collection = [stats_path]
        collection += [paths.stats / f"{stem}_n{n}_stats.pt" for n in snapshots]
        if split_half:
            collection += [paths.stats / f"{stem}_split_{side}_stats.pt" for side in ("a", "b")]

        state = complete_or_missing(collection)
        if args.force or state == "missing":
            command = [
                "--model",
                model["name"],
                "--config",
                args.config,
                "--models-config",
                args.models_config,
            ]
            if args.force:
                command.append("--force")
            run_module("src.experiments.run_collect_stats", command)
        elif state == "partial":
            raise RuntimeError(f"Incomplete collection for {model['name']}; use --force")

        metrics_path = paths.metrics / f"{stem}_metrics.pt"
        if args.force or not metrics_path.exists():
            command = ["--stats-path", str(stats_path), "--config", args.config]
            if args.force:
                command.append("--force")
            run_module("src.experiments.run_compute_metrics", command)

    final_stats = sorted(paths.stats.glob(f"*_{hook}_stats.pt"))
    labels = [path.stem.removesuffix(f"_{hook}_stats") for path in final_stats]
    analysis_outputs = [
        paths.tables / "model_layer_summary.csv",
        paths.figures / "model_comparison.png",
    ]
    for label in labels:
        analysis_outputs.append(paths.figures / f"{label}_summary.png")
        if len(snapshots) >= 2:
            analysis_outputs.append(paths.tables / f"{label}_convergence.csv")
        if split_half:
            analysis_outputs.append(paths.tables / f"{label}_reliability.csv")

    state = complete_or_missing(analysis_outputs)
    if args.force or state == "missing":
        command = ["--stats-path", *(str(path) for path in final_stats), "--config", args.config]
        if args.force:
            command.append("--force")
        run_module("src.experiments.run_analysis", command)
    elif state == "partial":
        raise RuntimeError("Incomplete analysis artifacts; use --force")


if __name__ == "__main__":
    main()
