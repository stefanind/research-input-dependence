import argparse
from pathlib import Path

import yaml

from src.models.load_model import load_hooked_model
from src.data.load_text_chunks import load_wikitext_token_chunks
from src.activations.collect_residuals import collect_residual_stats_from_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    act_cfg = cfg["activations"]
    out_cfg = cfg["outputs"]
    snapshot_cfg = cfg.get("snapshots", {"enabled": False, "points": []})

    model = load_hooked_model(
        args.model,
        device=exp_cfg["device"],
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

    stats = collect_residual_stats_from_tokens(
        model=model,
        token_chunks=token_chunks,
        seq_len=exp_cfg["seq_len"],
        batch_size=exp_cfg["batch_size"],
        hook_type=act_cfg["hook_name"],
        device=exp_cfg["device"],
        snapshot_points=set(snapshot_cfg.get("points", [])) if snapshot_cfg.get("enabled", False) else set(),
        snapshot_dir=output_dir,
        run_name=f"{safe_model_name}_{act_cfg['hook_name']}",
    )

    output_path = output_dir / f"{safe_model_name}_{act_cfg['hook_name']}_stats.pt"
    stats.save(str(output_path))

    print(f"Saved stats to {output_path}")


if __name__ == "__main__":
    main()