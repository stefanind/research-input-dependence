import torch
from tqdm import tqdm

from src.stats.running_stats import RunningActivationStats


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


@torch.no_grad()
def collect_residual_stats_from_tokens(
    model,
    token_chunks: torch.Tensor,
    seq_len: int,
    batch_size: int,
    hook_type: str = "resid_post",
    device: str = "cuda",
    snapshot_points: set[int] | None = None,
    snapshot_dir=None,
    run_name: str = "run",
    split_half_seed: int | None = None,
    metadata: dict | None = None,
    force: bool = False,
) -> RunningActivationStats:
    num_layers = model.cfg.n_layers
    d_model = model.cfg.d_model

    stats = RunningActivationStats(
        num_layers=num_layers,
        seq_len=seq_len,
        d_model=d_model,
        device="cpu",
        metadata=metadata,
    )
    split_stats = None
    split_assignment = None
    if split_half_seed is not None:
        if token_chunks.shape[0] < 2:
            raise ValueError("Split-half reliability requires at least two chunks")
        split_stats = (
            RunningActivationStats(
                num_layers,
                seq_len,
                d_model,
                device="cpu",
                metadata={**(metadata or {}), "split": "a", "split_seed": split_half_seed},
            ),
            RunningActivationStats(
                num_layers,
                seq_len,
                d_model,
                device="cpu",
                metadata={**(metadata or {}), "split": "b", "split_seed": split_half_seed},
            ),
        )
        generator = torch.Generator().manual_seed(split_half_seed)
        permutation = torch.randperm(token_chunks.shape[0], generator=generator)
        split_assignment = torch.zeros(token_chunks.shape[0], dtype=torch.bool)
        split_assignment[permutation[: token_chunks.shape[0] // 2]] = True

    hook_suffix = {
        "resid_pre": "hook_resid_pre",
        "resid_mid": "hook_resid_mid",
        "resid_post": "hook_resid_post",
    }[hook_type]

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
            names_filter=lambda name: hook_suffix in name,
            return_type=None,
        )

        for layer_idx in range(num_layers):
            hook_name = f"blocks.{layer_idx}.{hook_suffix}"
            resid = cache[hook_name]
            resid_cpu = resid.cpu()
            stats.update_layer(layer_idx, resid_cpu)
            if split_stats is not None:
                if split_mask.any():
                    split_stats[0].update_layer(layer_idx, resid_cpu[split_mask])
                if (~split_mask).any():
                    split_stats[1].update_layer(layer_idx, resid_cpu[~split_mask])

        processed += tokens.shape[0]

        for point in sorted(snapshot_points):
            if processed >= point and point not in saved_snapshots:
                if snapshot_dir is not None:
                    snapshot_path = snapshot_dir / f"{run_name}_n{point}_stats.pt"
                    stats.save(str(snapshot_path), force=force)
                    print(f"Saved snapshot: {snapshot_path}")
                saved_snapshots.add(point)

        del cache

        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    if split_stats is not None and snapshot_dir is not None:
        for label, split_stat in zip(("a", "b"), split_stats):
            path = snapshot_dir / f"{run_name}_split_{label}_stats.pt"
            split_stat.save(str(path), force=force)
            print(f"Saved split-half stats: {path}")

    return stats
