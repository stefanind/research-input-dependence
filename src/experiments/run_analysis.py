import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import yaml

from src.experiments.artifacts import artifact_paths, atomic_csv_save, require_writable
from src.metrics.input_dependence import (
    classify_energy_variance_buckets,
    compute_input_dependence_score,
    summarize_buckets_by_layer,
    summarize_by_layer,
)


def load_stats(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def compute_ids(stats: dict) -> torch.Tensor:
    variance = stats.get("m2", stats["variance"])
    if "m2" in stats:
        variance = variance / stats["count"].clamp_min(1.0)
    return compute_input_dependence_score(variance, stats["mean_square"])


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double() - a.double().mean()
    b = b.flatten().double() - b.double().mean()
    denominator = a.norm() * b.norm()
    if denominator.item() == 0:
        return float("nan")
    return max(-1.0, min(1.0, (a @ b / denominator).item()))


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int, largest=True) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, a.numel(), b.numel())
    a_indices = set(torch.topk(a.flatten(), k, largest=largest).indices.tolist())
    b_indices = set(torch.topk(b.flatten(), k, largest=largest).indices.tolist())
    return len(a_indices & b_indices) / k


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    union = (a | b).sum().item()
    return float("nan") if union == 0 else (a & b).sum().item() / union


def extract_n(path: Path) -> int:
    match = re.search(r"_n(\d+)", path.name)
    return int(match.group(1)) if match else -1


def save_figure(figure, path: Path, force: bool) -> None:
    require_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    figure.savefig(temporary, dpi=200, format="png")
    temporary.replace(path)
    plt.close(figure)


def convergence_rows(stats_path: Path, final_ids: torch.Tensor, top_k: int) -> list[dict]:
    snapshots = snapshot_paths(stats_path)
    final_layer = final_ids.mean(dim=(1, 2))
    final_dims = final_ids.mean(dim=1)
    rows = []
    for path in snapshots:
        ids = compute_ids(load_stats(path))
        layer = ids.mean(dim=(1, 2))
        dims = ids.mean(dim=1)
        rows.append(
            {
                "snapshot": path.name,
                "n": extract_n(path),
                "layer_corr": pearson_corr(layer, final_layer),
                "all_corr": pearson_corr(ids, final_ids),
                "top_high": topk_overlap(dims, final_dims, top_k),
                "top_low": topk_overlap(dims, final_dims, top_k, False),
                "mean_abs_layer_diff": (layer - final_layer).abs().mean().item(),
            }
        )
    return rows


def snapshot_paths(stats_path: Path) -> list[Path]:
    stem = stats_path.stem.removesuffix("_stats")
    return sorted(stats_path.parent.glob(f"{stem}_n*_stats.pt"), key=extract_n)


def split_paths(stats_path: Path) -> list[Path]:
    stem = stats_path.stem.removesuffix("_stats")
    return [stats_path.parent / f"{stem}_split_{label}_stats.pt" for label in ("a", "b")]


def reliability_rows(stats_path: Path, top_k: int) -> list[dict]:
    paths = split_paths(stats_path)
    if not all(path.exists() for path in paths):
        return []
    split_stats = [load_stats(path) for path in paths]
    ids_a, ids_b = (compute_ids(stats) for stats in split_stats)
    buckets = [
        classify_energy_variance_buckets(stats["variance"], stats["mean_square"], ids)
        for stats, ids in zip(split_stats, (ids_a, ids_b))
    ]
    dims_a, dims_b = ids_a.mean(dim=1), ids_b.mean(dim=1)
    overall = pearson_corr(ids_a, ids_b)
    high = topk_overlap(dims_a, dims_b, top_k)
    low = topk_overlap(dims_a, dims_b, top_k, False)
    return [
        {
            "layer": layer,
            "ids_corr": pearson_corr(ids_a[layer], ids_b[layer]),
            "active_invariant_jaccard": jaccard(
                buckets[0]["active_invariant"][layer],
                buckets[1]["active_invariant"][layer],
            ),
            "overall_ids_corr": overall,
            "top_high_overlap": high,
            "top_low_overlap": low,
        }
        for layer in range(ids_a.shape[0])
    ]


def layer_rows(label: str, stats: dict, ids: torch.Tensor) -> list[dict]:
    summary = summarize_by_layer(ids)
    buckets = classify_energy_variance_buckets(stats["variance"], stats["mean_square"], ids)
    fractions = summarize_buckets_by_layer(buckets)
    layers = ids.shape[0]
    return [
        {
            "model": label,
            "layer": layer,
            "relative_depth": layer / max(layers - 1, 1),
            "mean_ids": summary["layer_mean"][layer].item(),
            "median_ids": summary["layer_median"][layer].item(),
            "ids_threshold": buckets["ids_threshold"][layer].item(),
            "energy_threshold": buckets["energy_threshold"][layer].item(),
            **{name: values[layer].item() for name, values in fractions.items()},
        }
        for layer in range(layers)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats-path", nargs="+", required=True)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    paths = artifact_paths(cfg)
    split_half = cfg.get("reliability", {}).get("split_half", False)

    records = []
    for value in args.stats_path:
        path = Path(value)
        stats = load_stats(path)
        label = stats.get("metadata", {}).get("model", path.stem.removesuffix("_stats"))
        records.append((label, path, stats, compute_ids(stats)))

    planned = [paths.tables / "model_layer_summary.csv", paths.figures / "model_comparison.png"]
    for label, path, _, _ in records:
        planned.append(paths.figures / f"{label}_summary.png")
        if len(snapshot_paths(path)) >= 2:
            planned.append(paths.tables / f"{label}_convergence.csv")
        if split_half and all(split.exists() for split in split_paths(path)):
            planned.append(paths.tables / f"{label}_reliability.csv")
    for path in planned:
        require_writable(path, args.force)

    all_rows = []
    for label, path, stats, ids in records:
        rows = layer_rows(label, stats, ids)
        all_rows.extend(rows)
        convergence = convergence_rows(path, ids, args.top_k)
        if convergence:
            atomic_csv_save(convergence, paths.tables / f"{label}_convergence.csv", args.force)
        reliability = reliability_rows(path, args.top_k) if split_half else []
        if reliability:
            atomic_csv_save(reliability, paths.tables / f"{label}_reliability.csv", args.force)

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        depth = [row["relative_depth"] for row in rows]
        axes[0].plot(depth, [row["mean_ids"] for row in rows], marker="o")
        for name in ("inactive_invariant", "active_invariant", "input_varying_active", "weak_noisy"):
            axes[1].plot(depth, [row[name] for row in rows], label=name)
        axes[0].set(title="Mean IDS", xlabel="Relative depth", ylabel="Mean IDS")
        axes[1].set(title="Bucket fractions", xlabel="Relative depth", ylabel="Fraction")
        axes[1].legend(fontsize=8)
        figure.tight_layout()
        save_figure(figure, paths.figures / f"{label}_summary.png", args.force)

    atomic_csv_save(all_rows, paths.tables / "model_layer_summary.csv", args.force)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, _, _, _ in records:
        rows = [row for row in all_rows if row["model"] == label]
        depth = [row["relative_depth"] for row in rows]
        axes[0].plot(depth, [row["mean_ids"] for row in rows], marker="o", label=label)
        axes[1].plot(depth, [row["active_invariant"] for row in rows], marker="o", label=label)
    axes[0].set(title="Mean IDS", xlabel="Relative depth", ylabel="Mean IDS")
    axes[1].set(title="Active-invariant fraction", xlabel="Relative depth", ylabel="Fraction")
    for axis in axes:
        axis.legend()
    figure.tight_layout()
    save_figure(figure, paths.figures / "model_comparison.png", args.force)
    print(f"Saved analysis to {paths.root}")


if __name__ == "__main__":
    main()
