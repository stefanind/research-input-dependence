import torch


def compute_input_dependence_score(
    variance: torch.Tensor,
    mean_square: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    score[layer, position, dim] =
        variance_across_inputs / mean_square_activation

    Low score:
        mostly stable across inputs

    High score:
        highly input-varying
    """

    return (variance / (mean_square + eps)).clamp(0.0, 1.0)


def summarize_by_layer(score: torch.Tensor) -> dict[str, torch.Tensor]:
    """
    score shape:
        [layers, seq_len, d_model]
    """

    return {
        "layer_mean": score.mean(dim=(1, 2)),
        "layer_median": score.flatten(1).median(dim=1).values,
        "layer_top_1pct": torch.quantile(score.flatten(1), 0.99, dim=1),
        "layer_bottom_1pct": torch.quantile(score.flatten(1), 0.01, dim=1),
    }


def summarize_by_dimension(score: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        [layers, d_model]

    This averages over positions.
    """

    return score.mean(dim=1)



def classify_energy_variance_buckets(
    variance: torch.Tensor,
    mean_square: torch.Tensor,
    ids: torch.Tensor,
    energy_quantile: float = 0.5,
    ids_quantile: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Classify every cell using per-layer IDS and energy thresholds."""
    if variance.shape != mean_square.shape or ids.shape != mean_square.shape:
        raise ValueError("variance, mean_square, and ids must have matching shapes")
    if ids.ndim != 3:
        raise ValueError("Expected tensors with shape [layers, positions, dimensions]")

    shape = (-1, 1, 1)
    energy_threshold = mean_square.flatten(1).quantile(
        energy_quantile, dim=1
    ).view(shape)
    ids_threshold = ids.flatten(1).quantile(ids_quantile, dim=1).view(shape)

    invariant = ids <= ids_threshold
    active = mean_square >= energy_threshold

    return {
        "energy_threshold": energy_threshold.flatten(),
        "ids_threshold": ids_threshold.flatten(),
        "inactive_invariant": invariant & ~active,
        "active_invariant": invariant & active,
        "input_varying_active": ~invariant & active,
        "weak_noisy": ~invariant & ~active,
    }


def summarize_buckets_by_layer(
    buckets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    names = (
        "inactive_invariant",
        "active_invariant",
        "input_varying_active",
        "weak_noisy",
    )
    return {name: buckets[name].float().mean(dim=(1, 2)) for name in names}
