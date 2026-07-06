import random

from datasets import load_dataset
import torch


def load_wikitext_token_chunks(
    tokenizer,
    seq_len: int,
    max_chunks: int | None = None,
    split: str = "train",
    revision: str | None = None,
    sample_seed: int = 0,
) -> torch.Tensor:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if max_chunks is not None and max_chunks < 0:
        raise ValueError("max_chunks cannot be negative")
    if max_chunks == 0:
        return torch.empty((0, seq_len), dtype=torch.long)

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        revision=revision,
    )

    rng = random.Random(sample_seed)
    chunks = []
    token_remainder = []
    parts = []
    text_length = 0
    seen = 0
    started = False

    def consume_text(text: str) -> None:
        nonlocal chunks, token_remainder, seen
        token_ids = token_remainder + tokenizer.encode(
            text, add_special_tokens=False, verbose=False
        )
        stop = len(token_ids) - len(token_ids) % seq_len
        for start in range(0, stop, seq_len):
            seen += 1
            if max_chunks is None or len(chunks) < max_chunks:
                chunks.append(token_ids[start : start + seq_len])
            else:
                index = rng.randrange(seen)
                if index < max_chunks:
                    chunks[index] = token_ids[start : start + seq_len]
        token_remainder = token_ids[stop:]

    for row in dataset:
        text = row["text"]
        if not text.strip():
            continue

        if started:
            parts.append("\n\n")
            text_length += 2
        parts.append(text)
        text_length += len(text)
        started = True

        if text_length >= 1_000_000:
            consume_text("".join(parts))
            parts = []
            text_length = 0

    if parts:
        consume_text("".join(parts))
    if not chunks:
        raise ValueError("No chunks were created. Check seq_len or dataset loading.")

    if max_chunks is not None:
        rng.shuffle(chunks)
    return torch.tensor(chunks, dtype=torch.long)
