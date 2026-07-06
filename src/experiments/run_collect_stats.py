import argparse
import gc

import torch
import yaml

from src.experiments.artifacts import (
    artifact_paths,
    ensure_artifact_dirs,
    ensure_manifest,
    require_writable,
)
from src.models.load_model import load_hooked_model
from src.data.load_text_chunks import load_wikitext_token_chunks
from src.activations.collect_residuals import collect_residual_stats_from_tokens


def select_model_configs(models: list[dict], names: list[str]) -> list[dict]:
    if names == ["all"]:
        return models
    if "all" in names:
        raise ValueError("Use 'all' by itself")

    by_name = {model["name"]: model for model in models}
    unknown = [name for name in names if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}")
    return [by_name[name] for name in names]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", nargs="+", required=True)
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--models-config", type=str, default="configs/models.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    exp_cfg = cfg["experiment"]
    data_cfg = cfg["data"]
    act_cfg = cfg["activations"]
    snapshot_cfg = cfg.get("snapshots", {"enabled": False, "points": []})
    reliability_cfg = cfg.get("reliability", {"split_half": False})

    with open(args.models_config, "r") as f:
        models_cfg = yaml.safe_load(f)
    models = models_cfg["models"]

    selected_models = select_model_configs(models, args.model)
    if data_cfg["source"] != "wikitext":
        raise ValueError(f"Unknown data source: {data_cfg['source']}")
    for model_cfg in selected_models:
        if model_cfg["library"] != "transformer_lens":
            raise ValueError(f"Unsupported model library: {model_cfg['library']}")
        if exp_cfg["seq_len"] > model_cfg["max_seq_len"]:
            raise ValueError(f"seq_len exceeds the limit for {model_cfg['name']}")
    if any(
        point > exp_cfg["num_chunks"] for point in snapshot_cfg.get("points", [])
    ):
        raise ValueError("snapshot points cannot exceed num_chunks")

    device = exp_cfg["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    paths = artifact_paths(cfg)
    ensure_artifact_dirs(paths)
    ensure_manifest(cfg, models_cfg, paths, force=args.force)

    for model_cfg in selected_models:
        model_name = model_cfg["name"]
        safe_model_name = model_name.replace("/", "__")
        run_name = f"{safe_model_name}_{act_cfg['hook_name']}"
        planned = [paths.stats / f"{run_name}_stats.pt"]
        if snapshot_cfg.get("enabled", False):
            planned.extend(
                paths.stats / f"{run_name}_n{point}_stats.pt"
                for point in snapshot_cfg.get("points", [])
            )
        if reliability_cfg.get("split_half", False):
            planned.extend(
                paths.stats / f"{run_name}_split_{label}_stats.pt"
                for label in ("a", "b")
            )
        for output_path in planned:
            require_writable(output_path, args.force)

        print(f"\nCollecting {model_name} on {device}")
        model = load_hooked_model(
            model_cfg["hf_name"],
            device=device,
            revision=model_cfg.get("revision"),
        )

        token_chunks = load_wikitext_token_chunks(
            tokenizer=model.tokenizer,
            seq_len=exp_cfg["seq_len"],
            max_chunks=exp_cfg["num_chunks"],
            split=data_cfg.get("split", "train"),
            revision=data_cfg.get("revision"),
        )

        stats = collect_residual_stats_from_tokens(
            model=model,
            token_chunks=token_chunks,
            seq_len=exp_cfg["seq_len"],
            batch_size=exp_cfg["batch_size"],
            hook_type=act_cfg["hook_name"],
            device=device,
            snapshot_points=set(snapshot_cfg.get("points", [])) if snapshot_cfg.get("enabled", False) else set(),
            snapshot_dir=paths.stats,
            run_name=run_name,
            split_half_seed=(
                reliability_cfg.get("seed", 0)
                if reliability_cfg.get("split_half", False)
                else None
            ),
            metadata={
                "experiment": exp_cfg["name"],
                "model": model_name,
                "hf_model": model_cfg["hf_name"],
                "model_revision": model_cfg.get("revision"),
                "data_source": data_cfg["source"],
                "data_split": data_cfg.get("split", "train"),
                "data_revision": data_cfg.get("revision"),
                "hook": act_cfg["hook_name"],
                "num_chunks": token_chunks.shape[0],
            },
            force=args.force,
        )

        output_path = paths.stats / f"{run_name}_stats.pt"
        stats.save(str(output_path), force=args.force)
        print(f"Saved stats to {output_path}")

        del stats, token_chunks, model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
