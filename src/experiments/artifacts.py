import csv
import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    stats: Path
    metrics: Path
    tables: Path
    figures: Path
    manifest: Path


def artifact_paths(cfg: dict) -> ArtifactPaths:
    root = Path(cfg["outputs"].get("root", "results")) / cfg["experiment"]["name"]
    return ArtifactPaths(
        root=root,
        stats=root / "stats",
        metrics=root / "metrics",
        tables=root / "tables",
        figures=root / "figures",
        manifest=root / "manifest.json",
    )


def ensure_artifact_dirs(paths: ArtifactPaths) -> None:
    for path in (paths.stats, paths.metrics, paths.tables, paths.figures):
        path.mkdir(parents=True, exist_ok=True)


def require_writable(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; use --force")


def atomic_json_save(data: dict, path: Path, force: bool = False) -> None:
    require_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_csv_save(rows: list[dict], path: Path, force: bool = False) -> None:
    if not rows:
        raise ValueError("Cannot save an empty CSV")
    require_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_torch_save(data: dict, path: Path, force: bool = False) -> None:
    require_writable(path, force)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def build_manifest(cfg: dict, models_cfg: dict) -> dict:
    packages = {}
    for package in (
        "torch",
        "transformer-lens",
        "transformers",
        "datasets",
        "numpy",
        "matplotlib",
        "pyyaml",
        "tqdm",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit = None
        git_dirty = None

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "git_dirty": git_dirty,
        "config": cfg,
        "models": models_cfg,
        "versions": {
            **packages,
            "python": sys.version,
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        },
        "schema": {"stats": 2, "metrics": 3, "classification": 3},
    }


def ensure_manifest(
    cfg: dict,
    models_cfg: dict,
    paths: ArtifactPaths,
    force: bool = False,
) -> None:
    if paths.manifest.exists() and not force:
        existing = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing.get("config") != cfg or existing.get("models") != models_cfg:
            raise ValueError(
                f"Existing manifest disagrees with current config: {paths.manifest}"
            )
        return
    atomic_json_save(build_manifest(cfg, models_cfg), paths.manifest, force=force)
