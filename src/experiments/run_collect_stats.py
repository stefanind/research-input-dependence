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
from src.data.load_text_chunks import (
    load_or_create_wikitext_raw_windows,
    tokenize_raw_windows,
)
from src.activations.collect_residuals import collect_activation_stats_from_tokens


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


def activation_targets(act_cfg: dict) -> list[str]:
    targets = act_cfg.get("targets")
    if targets is None:
        targets = [act_cfg["hook_name"]]
    targets = list(dict.fromkeys(targets))
    if not targets:
        raise ValueError("At least one activation target is required")
    return targets


def safe_name(name: str) -> str:
    return name.replace("/", "__")


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
    targets = activation_targets(act_cfg)

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
    data_tag = safe_name(
        f"{data_cfg['source']}_{data_cfg.get('split', 'train')}_"
        f"{data_cfg.get('revision') or 'default'}"
    )
    raw_windows_path = (
        paths.data
        / (
            f"raw_windows_{data_tag}_seed{data_cfg.get('sample_seed', 0)}"
            f"_n{exp_cfg['num_chunks']}.jsonl"
        )
    )
    raw_windows = load_or_create_wikitext_raw_windows(
        raw_windows_path,
        max_windows=exp_cfg["num_chunks"],
        split=data_cfg.get("split", "train"),
        revision=data_cfg.get("revision"),
        sample_seed=data_cfg.get("sample_seed", 0),
    )

    for model_cfg in selected_models:
        model_name = model_cfg["name"]
        model_stats_dir = paths.stats / safe_name(model_name)
        planned = []
        for target in targets:
            planned.append(model_stats_dir / f"{target}_stats.pt")
            if snapshot_cfg.get("enabled", False):
                planned.extend(
                    model_stats_dir / f"{target}_n{point}_stats.pt"
                    for point in snapshot_cfg.get("points", [])
                )
            if reliability_cfg.get("split_half", False):
                planned.extend(
                    model_stats_dir / f"{target}_split_{label}_stats.pt"
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

        token_chunks = tokenize_raw_windows(
            tokenizer=model.tokenizer,
            windows=raw_windows,
            seq_len=exp_cfg["seq_len"],
            max_chunks=exp_cfg["num_chunks"],
        )

        stats_by_target = collect_activation_stats_from_tokens(
            model=model,
            token_chunks=token_chunks,
            seq_len=exp_cfg["seq_len"],
            batch_size=exp_cfg["batch_size"],
            targets=targets,
            device=device,
            snapshot_points=(
                set(snapshot_cfg.get("points", []))
                if snapshot_cfg.get("enabled", False)
                else set()
            ),
            snapshot_dir=model_stats_dir,
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
                "sample_seed": data_cfg.get("sample_seed", 0),
                "raw_windows_path": str(raw_windows_path),
                "seq_len": exp_cfg["seq_len"],
                "num_chunks": token_chunks.shape[0],
            },
            force=args.force,
        )

        for target, stats in stats_by_target.items():
            output_path = model_stats_dir / f"{target}_stats.pt"
            stats.save(str(output_path), force=args.force)
            print(f"Saved stats to {output_path}")

        del stats_by_target, token_chunks, model
        gc.collect()
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
