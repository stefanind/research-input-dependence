import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from src.experiments.artifacts import artifact_paths, atomic_csv_save, require_writable
from src.metrics.input_dependence import (
    classify_cells,
    compute_input_variation_ratio,
    summarize_classes_by_layer,
    summarize_by_layer,
)


def load_stats(path: Path) -> dict:
    return torch.load(path, map_location="cpu")


def compute_ivr(stats: dict) -> torch.Tensor:
    variance = stats.get("m2", stats["variance"])
    if "m2" in stats:
        variance = variance / stats["count"].clamp_min(1.0)
    return compute_input_variation_ratio(variance, stats["mean_square"])


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


def convergence_rows(stats_path: Path, final_ivr: torch.Tensor, top_k: int) -> list[dict]:
    snapshots = snapshot_paths(stats_path)
    final_layer = final_ivr.mean(dim=(1, 2))
    final_dims = final_ivr.mean(dim=1)
    rows = []
    for path in snapshots:
        ivr = compute_ivr(load_stats(path))
        layer = ivr.mean(dim=(1, 2))
        dims = ivr.mean(dim=1)
        rows.append(
            {
                "snapshot": path.name,
                "n": extract_n(path),
                "layer_corr": pearson_corr(layer, final_layer),
                "all_ivr_corr": pearson_corr(ivr, final_ivr),
                "top_varying_overlap": topk_overlap(dims, final_dims, top_k),
                "top_consistent_overlap": topk_overlap(dims, final_dims, top_k, False),
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


def reliability_rows(stats_path: Path, top_k: int, class_cfg: dict) -> list[dict]:
    paths = split_paths(stats_path)
    if not all(path.exists() for path in paths):
        return []
    split_stats = [load_stats(path) for path in paths]
    ivr_a, ivr_b = (compute_ivr(stats) for stats in split_stats)
    classes = [
        classify_cells(stats["mean_square"], ivr, **class_cfg)
        for stats, ivr in zip(split_stats, (ivr_a, ivr_b))
    ]
    dims_a, dims_b = ivr_a.mean(dim=1), ivr_b.mean(dim=1)
    overall = pearson_corr(ivr_a, ivr_b)
    high = topk_overlap(dims_a, dims_b, top_k)
    low = topk_overlap(dims_a, dims_b, top_k, False)
    return [
        {
            "layer": layer,
            "ivr_corr": pearson_corr(ivr_a[layer], ivr_b[layer]),
            "consistent_active_jaccard": jaccard(
                classes[0]["consistent_active"][layer],
                classes[1]["consistent_active"][layer],
            ),
            "overall_ivr_corr": overall,
            "top_varying_overlap": high,
            "top_consistent_overlap": low,
        }
        for layer in range(ivr_a.shape[0])
    ]


def safe_name(name: str) -> str:
    return name.replace("/", "__")


def model_family(model: str) -> str:
    if model == "distilgpt2" or model.startswith("gpt2"):
        return "gpt2"
    if model.startswith("gpt-neo"):
        return "gpt-neo"
    if model.startswith("opt"):
        return "opt"
    if model.startswith("pythia"):
        return "pythia"
    if model.startswith("bloom"):
        return "bloom"
    if model.startswith("olmo"):
        return "olmo"
    if model.startswith("qwen"):
        return "qwen"
    if model.startswith("phi"):
        return "phi"
    if model.startswith("tinystories"):
        return "tinystories"
    return "other"


def natural_key(value: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def model_size_key(model: str) -> float:
    manual_sizes_millions = {
        "distilgpt2": 82,
        "gpt2-small": 124,
        "gpt2-medium": 355,
        "gpt2-large": 774,
        "gpt2-xl": 1558,
        "bloom-1b1": 1100,
        "bloom-1b7": 1700,
        "phi-1": 1300,
        "phi-1_5": 1300,
        "phi-2": 2700,
    }
    if model in manual_sizes_millions:
        return manual_sizes_millions[model]

    matches = re.findall(r"(\d+(?:\.\d+)?)([mb])", model.lower())
    if not matches:
        return float("inf")
    value, unit = matches[-1]
    multiplier = 1000 if unit == "b" else 1
    return float(value) * multiplier


def model_sort_key(model: str) -> tuple:
    family_order = {
        "gpt2": 0,
        "gpt-neo": 1,
        "opt": 2,
        "pythia": 3,
        "bloom": 4,
        "olmo": 5,
        "qwen": 6,
        "phi": 7,
        "tinystories": 8,
        "other": 9,
    }
    return (
        family_order[model_family(model)],
        model_size_key(model),
        "deduped" in model,
        natural_key(model),
    )


def average(rows: list[dict], key: str) -> float:
    return sum(row[key] for row in rows) / len(rows)


def comparison_figure(target: str, all_rows: list[dict]):
    rows_by_model = {}
    for row in all_rows:
        rows_by_model.setdefault(row["model"], []).append(row)

    models = sorted(rows_by_model, key=model_sort_key)
    summary_columns = [
        ("mean IVR", lambda rows: average(rows, "mean_ivr")),
        ("avg consistent", lambda rows: average(rows, "consistent_active")),
        ("max consistent", lambda rows: max(row["consistent_active"] for row in rows)),
        ("avg mixed", lambda rows: average(rows, "mixed_active")),
        ("avg varying", lambda rows: average(rows, "input_varying_active")),
    ]
    summary = np.array(
        [
            [function(rows_by_model[model]) for _, function in summary_columns]
            for model in models
        ]
    )

    depth_points = np.linspace(0, 1, 11)
    depth_summary = np.zeros((len(models), len(depth_points)))
    for model_index, model in enumerate(models):
        rows = sorted(rows_by_model[model], key=lambda row: row["relative_depth"])
        depth_summary[model_index] = np.interp(
            depth_points,
            [row["relative_depth"] for row in rows],
            [row["consistent_active"] for row in rows],
        )

    height = max(8.0, 0.3 * len(models))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, height), width_ratios=(1, 1.4))

    image = axes[0].imshow(summary, aspect="auto", vmin=0, vmax=1, cmap="magma")
    axes[0].set(
        title=f"{target}: model-level summary",
        xticks=range(len(summary_columns)),
        yticks=range(len(models)),
    )
    axes[0].set_xticklabels([name for name, _ in summary_columns], rotation=45, ha="right")
    axes[0].set_yticklabels(models)
    figure.colorbar(image, ax=axes[0], fraction=0.04, pad=0.02)

    depth_image = axes[1].imshow(
        depth_summary,
        aspect="auto",
        vmin=0,
        vmax=max(0.1, depth_summary.max()),
        cmap="viridis",
    )
    axes[1].set(
        title="consistent-active by relative depth",
        xlabel="Relative depth bin",
        xticks=range(len(depth_points)),
        yticks=range(len(models)),
    )
    axes[1].set_xticklabels(
        [f"{depth:.1f}" for depth in depth_points],
        rotation=45,
        ha="right",
    )
    axes[1].set_yticklabels([])
    figure.colorbar(depth_image, ax=axes[1], fraction=0.04, pad=0.02)

    for axis in axes:
        axis.tick_params(axis="y", labelsize=7)
        axis.tick_params(axis="x", labelsize=8)
    figure.tight_layout()
    return figure


def layer_rows(
    model: str,
    target: str,
    stats: dict,
    ivr: torch.Tensor,
    class_cfg: dict,
) -> list[dict]:
    summary = summarize_by_layer(ivr)
    classes = classify_cells(stats["mean_square"], ivr, **class_cfg)
    fractions = summarize_classes_by_layer(classes)
    layers = ivr.shape[0]
    return [
        {
            "model": model,
            "target": target,
            "layer": layer,
            "relative_depth": layer / max(layers - 1, 1),
            "mean_ivr": summary["layer_mean"][layer].item(),
            "median_ivr": summary["layer_median"][layer].item(),
            "ivr_p01": summary["layer_bottom_1pct"][layer].item(),
            "ivr_p05": summary["layer_bottom_5pct"][layer].item(),
            "ivr_p10": summary["layer_bottom_10pct"][layer].item(),
            "median_rms": classes["energy_reference"][layer].sqrt().item(),
            "energy_reference": classes["energy_reference"][layer].item(),
            "inactive_energy_threshold": classes["energy_threshold"][layer].item(),
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
    class_cfg = cfg["classification"]

    records = []
    for value in args.stats_path:
        path = Path(value)
        stats = load_stats(path)
        metadata = stats.get("metadata", {})
        model = metadata.get("model", path.parent.name)
        target = metadata.get("activation_target", path.stem.removesuffix("_stats"))
        records.append((model, target, path, stats, compute_ivr(stats)))

    targets = sorted({target for _, target, _, _, _ in records})
    planned = []
    for model, target, path, _, _ in records:
        model_key = safe_name(model)
        planned.append(paths.figures / model_key / f"{target}_summary.png")
        if len(snapshot_paths(path)) >= 2:
            planned.append(paths.tables / model_key / f"{target}_convergence.csv")
        if split_half and all(split.exists() for split in split_paths(path)):
            planned.append(paths.tables / model_key / f"{target}_reliability.csv")
    for target in targets:
        planned.extend(
            [
                paths.tables / "comparison" / f"{target}_model_layer_summary.csv",
                paths.figures / "comparison" / f"{target}_model_comparison.png",
            ]
        )
    for path in planned:
        require_writable(path, args.force)

    rows_by_target = {target: [] for target in targets}
    for model, target, path, stats, ivr in records:
        model_key = safe_name(model)
        rows = layer_rows(model, target, stats, ivr, class_cfg)
        rows_by_target[target].extend(rows)
        convergence = convergence_rows(path, ivr, args.top_k)
        if convergence:
            atomic_csv_save(
                convergence,
                paths.tables / model_key / f"{target}_convergence.csv",
                args.force,
            )
        reliability = reliability_rows(path, args.top_k, class_cfg) if split_half else []
        if reliability:
            atomic_csv_save(
                reliability,
                paths.tables / model_key / f"{target}_reliability.csv",
                args.force,
            )

        figure, axes = plt.subplots(2, 2, figsize=(12, 8))
        depth = [row["relative_depth"] for row in rows]
        axes[0, 0].plot(depth, [row["mean_ivr"] for row in rows], marker="o")
        for key, percentile in (
            ("ivr_p01", "1%"),
            ("ivr_p05", "5%"),
            ("ivr_p10", "10%"),
        ):
            axes[0, 1].plot(
                depth,
                [row[key] for row in rows],
                marker="o",
                label=percentile,
            )
        axes[1, 0].plot(depth, [row["median_rms"] for row in rows], marker="o")
        for name in ("consistent_active", "mixed_active", "inactive"):
            axes[1, 1].plot(depth, [row[name] for row in rows], marker="o", label=name)
        axes[0, 0].set(
            title="Mean input variation ratio",
            xlabel="Relative depth",
            ylabel="Mean IVR",
        )
        axes[0, 1].set(
            title="Lower-tail IVR quantiles",
            xlabel="Relative depth",
            ylabel="IVR",
        )
        axes[0, 1].set_yscale("symlog", linthresh=1e-3)
        axes[0, 1].legend(title="Percentile", fontsize=8)
        axes[1, 0].set(
            title="Median RMS activity",
            xlabel="Relative depth",
            ylabel="RMS",
        )
        axes[1, 0].set_yscale("log")
        axes[1, 1].set(
            title="Persistent, mixed, and inactive fractions",
            xlabel="Relative depth",
            ylabel="Fraction",
        )
        axes[1, 1].set_yscale("symlog", linthresh=1e-4)
        axes[1, 1].legend(fontsize=8)
        figure.tight_layout()
        save_figure(
            figure,
            paths.figures / model_key / f"{target}_summary.png",
            args.force,
        )

    for target, all_rows in rows_by_target.items():
        atomic_csv_save(
            all_rows,
            paths.tables / "comparison" / f"{target}_model_layer_summary.csv",
            args.force,
        )
        figure = comparison_figure(target, all_rows)
        save_figure(
            figure,
            paths.figures / "comparison" / f"{target}_model_comparison.png",
            args.force,
        )
    print(f"Saved analysis to {paths.root}")


if __name__ == "__main__":
    main()
