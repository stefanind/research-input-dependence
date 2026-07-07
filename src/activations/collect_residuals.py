import torch
from tqdm import tqdm

from src.stats.running_stats import RunningActivationStats

HOOK_SUFFIXES = {
    "resid_pre": "hook_resid_pre",
    "resid_mid": "hook_resid_mid",
    "resid_post": "hook_resid_post",
    "attn_out": "hook_attn_out",
    "mlp_out": "hook_mlp_out",
}


def batch_tokens(
    token_chunks: torch.Tensor,
    batch_size: int,
    snapshot_points: set[int] | None = None,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    points = sorted(snapshot_points or set())
    start = 0
    while start < token_chunks.shape[0]:
        stop = min(start + batch_size, token_chunks.shape[0])
        stop = min((point for point in points if start < point < stop), default=stop)
        yield token_chunks[start:stop]
        start = stop


def required_hook_suffixes(targets: list[str]) -> set[str]:
    suffixes = set()
    for target in targets:
        if target == "resid_delta":
            suffixes.update(("hook_resid_pre", "hook_resid_post"))
        elif target in HOOK_SUFFIXES:
            suffixes.add(HOOK_SUFFIXES[target])
        else:
            raise ValueError(f"Unknown activation target: {target}")
    return suffixes


def activation_from_cache(cache, layer_idx: int, target: str) -> torch.Tensor:
    if target == "resid_delta":
        return (
            cache[f"blocks.{layer_idx}.hook_resid_post"]
            - cache[f"blocks.{layer_idx}.hook_resid_pre"]
        )
    return cache[f"blocks.{layer_idx}.{HOOK_SUFFIXES[target]}"]


@torch.no_grad()
def collect_activation_stats_from_tokens(
    model,
    token_chunks: torch.Tensor,
    seq_len: int,
    batch_size: int,
    targets: list[str],
    device: str = "cuda",
    snapshot_points: set[int] | None = None,
    snapshot_dir=None,
    split_half_seed: int | None = None,
    metadata: dict | None = None,
    force: bool = False,
) -> dict[str, RunningActivationStats]:
    if not targets:
        raise ValueError("At least one activation target is required")
    targets = list(dict.fromkeys(targets))
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    stats = {
        target: RunningActivationStats(
            num_layers=num_layers,
            seq_len=seq_len,
            d_model=d_model,
            device="cpu",
            metadata={**(metadata or {}), "activation_target": target, "hook": target},
        )
        for target in targets
    }
    split_stats = None
    split_assignment = None
    if split_half_seed is not None:
        if token_chunks.shape[0] < 2:
            raise ValueError("Split-half reliability requires at least two chunks")
        split_stats = {
            target: (
                RunningActivationStats(
                    num_layers,
                    seq_len,
                    d_model,
                    device="cpu",
                    metadata={
                        **(metadata or {}),
                        "activation_target": target,
                        "hook": target,
                        "split": "a",
                        "split_seed": split_half_seed,
                    },
                ),
                RunningActivationStats(
                    num_layers,
                    seq_len,
                    d_model,
                    device="cpu",
                    metadata={
                        **(metadata or {}),
                        "activation_target": target,
                        "hook": target,
                        "split": "b",
                        "split_seed": split_half_seed,
                    },
                ),
            )
            for target in targets
        }
        generator = torch.Generator().manual_seed(split_half_seed)
        permutation = torch.randperm(token_chunks.shape[0], generator=generator)
        split_assignment = torch.zeros(token_chunks.shape[0], dtype=torch.bool)
        split_assignment[permutation[: token_chunks.shape[0] // 2]] = True

    hook_suffixes = required_hook_suffixes(targets)
    snapshot_points = snapshot_points or set()
    if any(point <= 0 for point in snapshot_points):
        raise ValueError("snapshot points must be positive")
    processed = 0
    saved_snapshots = set()

    for tokens in tqdm(batch_tokens(token_chunks, batch_size, snapshot_points)):
        split_mask = None
        if split_assignment is not None:
            split_mask = split_assignment[processed : processed + tokens.shape[0]]
        tokens = tokens.to(device)

        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: any(suffix in name for suffix in hook_suffixes),
            return_type=None,
        )

        for layer_idx in range(num_layers):
            for target in targets:
                activation_cpu = activation_from_cache(cache, layer_idx, target).cpu()
                stats[target].update_layer(layer_idx, activation_cpu)
                if split_stats is not None:
                    if split_mask.any():
                        split_stats[target][0].update_layer(
                            layer_idx, activation_cpu[split_mask]
                        )
                    if (~split_mask).any():
                        split_stats[target][1].update_layer(
                            layer_idx, activation_cpu[~split_mask]
                        )

        processed += tokens.shape[0]

        for point in sorted(snapshot_points):
            if processed >= point and point not in saved_snapshots:
                if snapshot_dir is not None:
                    for target in targets:
                        snapshot_path = snapshot_dir / f"{target}_n{point}_stats.pt"
                        stats[target].save(str(snapshot_path), force=force)
                        print(f"Saved snapshot: {snapshot_path}")
                saved_snapshots.add(point)

        del cache

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if split_stats is not None and snapshot_dir is not None:
        for target in targets:
            for label, split_stat in zip(("a", "b"), split_stats[target]):
                path = snapshot_dir / f"{target}_split_{label}_stats.pt"
                split_stat.save(str(path), force=force)
                print(f"Saved split-half stats: {path}")

    return stats
