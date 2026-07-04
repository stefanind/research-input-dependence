from pathlib import Path


def load_prompts(path: str, max_prompts: int | None = None) -> list[str]:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")

    prompts = []
    with path_obj.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                prompts.append(text)

            if max_prompts is not None and len(prompts) >= max_prompts:
                break
            

    return prompts