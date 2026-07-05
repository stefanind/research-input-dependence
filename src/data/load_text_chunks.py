from datasets import load_dataset
import torch


def load_wikitext_token_chunks(
    tokenizer,
    seq_len: int,
    max_chunks: int | None = None,
    split: str = "train",
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
    )

    chunks = []
    pending = []
    first_text = True
    for row in dataset:
        text = row["text"]
        if not text.strip():
            continue

        prefix = "" if first_text else "\n\n"
        pending.extend(tokenizer.encode(prefix + text, add_special_tokens=False))
        first_text = False

        while len(pending) >= seq_len:
            chunks.append(pending[:seq_len])
            del pending[:seq_len]
            if max_chunks is not None and len(chunks) >= max_chunks:
                return torch.tensor(chunks, dtype=torch.long)

    if not chunks:
        raise ValueError("No chunks were created. Check seq_len or dataset loading.")

    return torch.tensor(chunks, dtype=torch.long)
