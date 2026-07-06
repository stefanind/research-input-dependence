from datasets import load_dataset
import torch


def load_wikitext_token_chunks(
    tokenizer,
    seq_len: int,
    max_chunks: int | None = None,
    split: str = "train",
    revision: str | None = None,
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

    parts = []
    text_length = 0
    target_tokens = max_chunks * seq_len if max_chunks is not None else None
    next_check = target_tokens

    for row in dataset:
        text = row["text"]
        if not text.strip():
            continue

        if parts:
            parts.append("\n\n")
            text_length += 2
        parts.append(text)
        text_length += len(text)

        if next_check is not None and text_length >= next_check:
            token_ids = tokenizer.encode("".join(parts), add_special_tokens=False)
            if len(token_ids) > target_tokens:
                return torch.tensor(token_ids[:target_tokens], dtype=torch.long).view(
                    max_chunks, seq_len
                )
            next_check = max(text_length + 1, next_check * 2)

    token_ids = tokenizer.encode("".join(parts), add_special_tokens=False)
    num_chunks = len(token_ids) // seq_len
    if max_chunks is not None:
        num_chunks = min(num_chunks, max_chunks)
    if num_chunks == 0:
        raise ValueError("No chunks were created. Check seq_len or dataset loading.")

    return torch.tensor(token_ids[: num_chunks * seq_len], dtype=torch.long).view(
        num_chunks, seq_len
    )
