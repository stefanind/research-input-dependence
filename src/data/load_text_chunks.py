import json
import random
from pathlib import Path

from datasets import load_dataset
import torch


def load_or_create_wikitext_raw_windows(
    path: Path,
    max_windows: int,
    split: str = "train",
    revision: str | None = None,
    sample_seed: int = 0,
    window_chars: int = 4096,
) -> list[str]:
    if max_windows < 0:
        raise ValueError("max_windows cannot be negative")
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line)["text"] for line in f]

    windows = sample_wikitext_raw_windows(
        max_windows=max_windows,
        split=split,
        revision=revision,
        sample_seed=sample_seed,
        window_chars=window_chars,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as f:
        for index, text in enumerate(windows):
            f.write(json.dumps({"id": index, "text": text}) + "\n")
    temporary.replace(path)
    return windows


def sample_wikitext_raw_windows(
    max_windows: int,
    split: str = "train",
    revision: str | None = None,
    sample_seed: int = 0,
    window_chars: int = 4096,
) -> list[str]:
    if max_windows == 0:
        return []

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        revision=revision,
    )
    rng = random.Random(sample_seed)
    windows = []
    buffer = ""
    seen = 0

    def consume(text: str) -> None:
        nonlocal buffer, seen
        buffer += text
        while len(buffer) >= window_chars:
            seen += 1
            window = buffer[:window_chars]
            buffer = buffer[window_chars:]
            if len(windows) < max_windows:
                windows.append(window)
            else:
                index = rng.randrange(seen)
                if index < max_windows:
                    windows[index] = window

    started = False
    for row in dataset:
        text = row["text"]
        if not text.strip():
            continue
        consume(("\n\n" if started else "") + text)
        started = True

    if not windows:
        raise ValueError("No raw text windows were created.")
    rng.shuffle(windows)
    return windows


def tokenize_raw_windows(
    tokenizer,
    windows: list[str],
    seq_len: int,
    max_chunks: int,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if max_chunks < 0:
        raise ValueError("max_chunks cannot be negative")
    if max_chunks == 0:
        return torch.empty((0, seq_len), dtype=torch.long)

    chunks = []
    for text in windows:
        token_ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
        if len(token_ids) >= seq_len:
            chunks.append(token_ids[:seq_len])
        if len(chunks) >= max_chunks:
            break

    if len(chunks) < max_chunks:
        raise ValueError(
            f"Only created {len(chunks)} chunks from {len(windows)} raw windows."
        )
    return torch.tensor(chunks, dtype=torch.long)
