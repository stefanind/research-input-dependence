import torch
from pathlib import Path


class RunningActivationStats:
    """
    Tracks running mean, variance, and mean-square activation.

    Shape convention:
        [num_layers, seq_len, d_model]
    """

    def __init__(
        self,
        num_layers: int,
        seq_len: int,
        d_model: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        metadata: dict | None = None,
    ):
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.d_model = d_model
        self.metadata = dict(metadata or {})

        shape = (num_layers, seq_len, d_model)

        self.count = torch.zeros(shape, device=device, dtype=dtype)
        self.mean = torch.zeros(shape, device=device, dtype=dtype)
        self.m2 = torch.zeros(shape, device=device, dtype=dtype)
        self.mean_square = torch.zeros(shape, device=device, dtype=dtype)

    @torch.no_grad()
    def update_layer(self, layer_idx: int, values: torch.Tensor) -> None:
        """
        values shape:
            [batch, seq_len, d_model]
        """

        if values.ndim != 3:
            raise ValueError(f"Expected [batch, seq_len, d_model], got {values.shape}")

        values = values.detach().to(self.mean.device, dtype=self.mean.dtype)

        batch_size, seq_len, d_model = values.shape

        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")

        if d_model != self.d_model:
            raise ValueError(f"Expected d_model={self.d_model}, got {d_model}")

        old_count = self.count[layer_idx]
        batch_count = torch.full_like(old_count, fill_value=batch_size)

        batch_mean = values.mean(dim=0)
        batch_m2 = ((values - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)
        batch_mean_square = (values ** 2).mean(dim=0)

        new_count = old_count + batch_count
        delta = batch_mean - self.mean[layer_idx]

        new_mean = self.mean[layer_idx] + delta * (batch_count / new_count)

        new_m2 = (
            self.m2[layer_idx]
            + batch_m2
            + (delta ** 2) * old_count * batch_count / new_count.clamp_min(1.0)
        )

        old_weight = old_count / new_count.clamp_min(1.0)
        batch_weight = batch_count / new_count.clamp_min(1.0)

        new_mean_square = (
            old_weight * self.mean_square[layer_idx]
            + batch_weight * batch_mean_square
        )

        self.count[layer_idx] = new_count
        self.mean[layer_idx] = new_mean
        self.m2[layer_idx] = new_m2
        self.mean_square[layer_idx] = new_mean_square

    def variance(self) -> torch.Tensor:
        """Return population variance across the observed inputs."""
        return self.m2 / self.count.clamp_min(1.0)

    def save(self, path: str, force: bool = False) -> None:
        output_path = Path(path)
        if output_path.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite {output_path}; use --force")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        torch.save(
            {
                "count": self.count.cpu(),
                "mean": self.mean.cpu(),
                "m2": self.m2.cpu(),
                "variance": self.variance().cpu(),
                "mean_square": self.mean_square.cpu(),
                "num_layers": self.num_layers,
                "seq_len": self.seq_len,
                "d_model": self.d_model,
                "metadata": self.metadata,
            },
            temporary,
        )
        temporary.replace(output_path)
