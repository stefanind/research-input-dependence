from datasets import load_dataset
import torch


def load_wikitext_token_chunks(
    tokenizer,
    seq_len: int,
    max_chunks: int | None = None,
    split: str = "train",
) -> torch.Tensor:
    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
    )

    texts = [row["text"] for row in dataset if row["text"].strip()]
    full_text = "\n\n".join(texts)

    token_ids = tokenizer.encode(full_text)

    chunks = []
    for start in range(0, len(token_ids) - seq_len + 1, seq_len):
        chunk = token_ids[start : start + seq_len]
        chunks.append(chunk)

        if max_chunks is not None and len(chunks) >= max_chunks:
            break

    if not chunks:
        raise ValueError("No chunks were created. Check seq_len or dataset loading.")

    return torch.tensor(chunks, dtype=torch.long)