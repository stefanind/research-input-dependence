import argparse
import gc

import torch
import yaml

from src.activations.collect_residuals import activation_from_cache, required_hook_suffixes
from src.experiments.run_collect_stats import activation_targets, select_model_configs
from src.models.load_model import clear_hf_model_cache, load_hooked_model


def make_tokens(tokenizer, seq_len: int, device: str) -> torch.Tensor:
    text = (
        "This is a short smoke test for loading the model and caching activations. "
        * 64
    )
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) < seq_len:
        raise ValueError(f"Tokenizer only produced {len(token_ids)} tokens")
    return torch.tensor([token_ids[:seq_len]], dtype=torch.long, device=device)


@torch.no_grad()
def smoke_test_model(model_cfg: dict, targets: list[str], seq_len: int, device: str) -> None:
    if model_cfg["library"] != "transformer_lens":
        raise ValueError(f"Unsupported model library: {model_cfg['library']}")
    if seq_len > model_cfg["max_seq_len"]:
        raise ValueError(f"seq_len exceeds the limit for {model_cfg['name']}")

    model = tokens = cache = None
    try:
        model = load_hooked_model(
            model_cfg["hf_name"],
            device=device,
            revision=model_cfg.get("revision"),
            use_safetensors=model_cfg.get("use_safetensors", True),
        )
        tokens = make_tokens(model.tokenizer, seq_len, device)
        hook_suffixes = required_hook_suffixes(targets)
        _, cache = model.run_with_cache(
            tokens,
            names_filter=lambda name: any(suffix in name for suffix in hook_suffixes),
            return_type=None,
        )

        for layer_idx in range(model.cfg.n_layers):
            for target in targets:
                activation = activation_from_cache(cache, layer_idx, target)
                expected = (1, seq_len, model.cfg.d_model)
                if tuple(activation.shape) != expected:
                    raise ValueError(
                        f"{target} layer {layer_idx} has shape "
                        f"{tuple(activation.shape)}, expected {expected}"
                    )
    finally:
        del cache, tokens, model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", default=["all"])
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clear-hf-cache", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    with open(args.models_config, "r") as f:
        models_cfg = yaml.safe_load(f)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    targets = activation_targets(cfg["activations"])
    models = select_model_configs(models_cfg["models"], args.model)

    failures = []
    for model_cfg in models:
        name = model_cfg["name"]
        print(f"\nSmoke testing {name} on {device}")
        if args.clear_hf_cache:
            for path in clear_hf_model_cache(model_cfg["hf_name"]):
                print(f"Cleared pre-existing HF cache: {path}")
        try:
            smoke_test_model(model_cfg, targets, args.seq_len, device)
            print(f"OK: {name}")
        except Exception as error:
            failures.append((name, error))
            print(f"FAILED: {name}: {error}")
        if args.clear_hf_cache:
            for path in clear_hf_model_cache(model_cfg["hf_name"]):
                print(f"Cleared HF cache: {path}")

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise SystemExit(f"Smoke test failed for: {names}")


if __name__ == "__main__":
    main()
