from transformer_lens import HookedTransformer


def load_hooked_model(
    model_name: str,
    device: str = "cuda",
    revision: str | None = None,
) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        revision=revision,
        use_safetensors=True,
    )

    model.eval()
    return model
