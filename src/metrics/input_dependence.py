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

    return variance / (mean_square + eps)


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
    variance,
    mean_square,
    ids,
    energy_quantile: float = 0.5,
    ids_low_quantile: float = 0.25,
    ids_high_quantile: float = 0.75,
):
    energy_threshold = mean_square.quantile(energy_quantile)
    ids_low = ids.quantile(ids_low_quantile)
    ids_high = ids.quantile(ids_high_quantile)

    inactive_invariant = (mean_square < energy_threshold) & (ids <= ids_low)
    active_invariant = (mean_square >= energy_threshold) & (ids <= ids_low)
    input_varying_active = (mean_square >= energy_threshold) & (ids >= ids_high)
    noisy_weak = (mean_square < energy_threshold) & (ids >= ids_high)

    return {
        "energy_threshold": energy_threshold,
        "ids_low": ids_low,
        "ids_high": ids_high,
        "inactive_invariant": inactive_invariant,
        "active_invariant": active_invariant,
        "input_varying_active": input_varying_active,
        "noisy_weak": noisy_weak,
    }
