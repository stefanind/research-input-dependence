import argparse
from pathlib import Path

import torch
import yaml

from src.models.load_model import load_hooked_model
from src.data.load_text_chunks import load_wikitext_token_chunks
from src.activations.collect_residuals import collect_residual_stats_from_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    act_cfg = cfg["activations"]
    out_cfg = cfg["outputs"]
    snapshot_cfg = cfg.get("snapshots", {"enabled": False, "points": []})

    with open(args.models_config, "r") as f:
        models = yaml.safe_load(f)["models"]

    model_cfg = next((model for model in models if model["name"] == args.model), None)
    if model_cfg is None:
        raise ValueError(f"Unknown model: {args.model}")
    if model_cfg["library"] != "transformer_lens":
        raise ValueError(f"Unsupported model library: {model_cfg['library']}")
    if exp_cfg["seq_len"] > model_cfg["max_seq_len"]:
        raise ValueError(f"seq_len exceeds the limit for {args.model}")

    device = exp_cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_hooked_model(
        model_cfg["hf_name"],
        device=device,
    )

    if data_cfg["source"] == "wikitext":
        token_chunks = load_wikitext_token_chunks(
            tokenizer=model.tokenizer,
            seq_len=exp_cfg["seq_len"],
            max_chunks=exp_cfg["num_chunks"],
            split=data_cfg.get("split", "train"),
        )
    else:
        raise ValueError(f"Unknown data source: {data_cfg['source']}")

    output_dir = Path(out_cfg["stats_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_model_name = args.model.replace("/", "__")
    run_name = f"{exp_cfg['name']}_{safe_model_name}_{act_cfg['hook_name']}"

    stats = collect_residual_stats_from_tokens(
        model=model,
        token_chunks=token_chunks,
        seq_len=exp_cfg["seq_len"],
        batch_size=exp_cfg["batch_size"],
        hook_type=act_cfg["hook_name"],
        device=device,
        snapshot_points=set(snapshot_cfg.get("points", [])) if snapshot_cfg.get("enabled", False) else set(),
        snapshot_dir=output_dir,
        run_name=run_name,
    )

    output_path = output_dir / f"{run_name}_stats.pt"
    stats.save(str(output_path))

    print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()
