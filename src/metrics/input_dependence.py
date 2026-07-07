import torch


def compute_input_variation_ratio(
    variance: torch.Tensor,
    mean_square: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the fraction of activation energy due to input variation."""
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
        "layer_bottom_5pct": torch.quantile(score.flatten(1), 0.05, dim=1),
        "layer_bottom_10pct": torch.quantile(score.flatten(1), 0.10, dim=1),
    }


def summarize_by_dimension(score: torch.Tensor) -> torch.Tensor:
    """
    Returns:
        [layers, d_model]

    This averages over positions.
    """

    return score.mean(dim=1)

def classify_cells(
    mean_square: torch.Tensor,
    ivr: torch.Tensor,
    inactive_energy_ratio: float = 0.01,
    consistent_ivr_max: float = 0.1,
    varying_ivr_min: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Classify cells using absolute IVR and layer-relative energy thresholds."""
    if ivr.shape != mean_square.shape:
        raise ValueError("mean_square and ivr must have matching shapes")
    if ivr.ndim != 3:
        raise ValueError("Expected tensors with shape [layers, positions, dimensions]")
    if inactive_energy_ratio < 0:
        raise ValueError("inactive_energy_ratio cannot be negative")
    if not 0 <= consistent_ivr_max < varying_ivr_min <= 1:
        raise ValueError("IVR thresholds must satisfy 0 <= consistent < varying <= 1")

    shape = (-1, 1, 1)
    energy_reference = mean_square.flatten(1).median(dim=1).values.view(shape)
    energy_threshold = inactive_energy_ratio * energy_reference
    inactive = mean_square <= energy_threshold
    active = ~inactive

    return {
        "energy_reference": energy_reference.flatten(),
        "energy_threshold": energy_threshold.flatten(),
        "inactive": inactive,
        "consistent_active": active & (ivr < consistent_ivr_max),
        "mixed_active": active & (ivr >= consistent_ivr_max) & (ivr < varying_ivr_min),
        "input_varying_active": active & (ivr >= varying_ivr_min),
    }


def summarize_classes_by_layer(
    classes: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    names = ("inactive", "consistent_active", "mixed_active", "input_varying_active")
    return {name: classes[name].float().mean(dim=(1, 2)) for name in names}
