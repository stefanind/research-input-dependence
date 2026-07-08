import shutil
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_official_model_name


def load_hooked_model(
    model_name: str,
    device: str = "cuda",
    revision: str | None = None,
    use_safetensors: bool = True,
) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        revision=revision,
        use_safetensors=use_safetensors,
    )

    model.eval()
    return model


def clear_hf_model_cache(model_name: str) -> list[Path]:
    cache_root = Path(HF_HUB_CACHE).resolve()
    repo_ids = {model_name}
    try:
        repo_ids.add(get_official_model_name(model_name))
    except ValueError:
        pass

    removed = []
    for repo_id in repo_ids:
        cache_path = (
            cache_root / repo_folder_name(repo_id=repo_id, repo_type="model")
        ).resolve()
        if not cache_path.exists():
            continue
        if cache_path != cache_root and cache_root not in cache_path.parents:
            raise RuntimeError(f"Refusing to remove cache outside {cache_root}")
        shutil.rmtree(cache_path)
        removed.append(cache_path)
    return removed
