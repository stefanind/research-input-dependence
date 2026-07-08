# Residual input variation

For a fixed layer `l`, token position `p`, residual dimension `d`, and
activation target, over `N` input chunks:

```text
mean     = mean(h)
variance = mean((h - mean) ^ 2)
energy   = mean(h ^ 2) = mean ^ 2 + variance
IVR      = variance / (energy + epsilon)
```

The means use denominator `N` (no `N-1` sample correction). The input
variation ratio (IVR) is the fraction of activation energy attributable to
input-to-input variation. Low IVR indicates a persistent mean; high IVR
indicates input-varying activity.

Classification is secondary to the continuous IVR and energy measurements:

- `inactive`: energy is at most 1% of the layer median energy
- `consistent_active`: active and IVR < 0.1
- `mixed_active`: active and 0.1 <= IVR < 0.5
- `input_varying_active`: active and IVR >= 0.5

These measurements describe variation over the sampled input distribution;
they do not establish universal invariance.

The default targets are:

- `resid_post`: residual state after the block
- `resid_delta`: net block write, `resid_post - resid_pre`
- `attn_out`: attention write into the residual stream
- `mlp_out`: MLP write into the residual stream

The same energy/IVR classification is applied to every target.

## RunPod setup

Target image:
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

```bash
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The check should report Torch 2.4, CUDA 12.4, and `True`.

Before spending a full run, smoke-test model loading and activation hooks:

```bash
python -m src.experiments.smoke_test_models --model all
```

If disk space is tight, delete each model from the Hugging Face cache after
testing it:

```bash
python -m src.experiments.smoke_test_models --model all --clear-hf-cache
```

## Run the experiment

Run commands from the repository root. The default experiment uses 10,000
seeded, randomly sampled WikiText-103 raw text windows. Each model tokenizes
the same raw windows with its own tokenizer and keeps fixed-length, unpadded
chunks of 128 tokens.

The pipeline runs collection, metrics, per-model plots, and cross-model
comparison. Completed artifacts are skipped. Convergence snapshots and
split-half reliability are disabled by default because 10,000 chunks has
already been checked; turn them on only for validation runs.

```bash
python -m src.experiments.run_pipeline \
  --model gpt2-small pythia-160m opt-125m
```

Add `pythia-410m` for the optional scale check, or use `--model all`. Use
`--force` only when intentionally replacing every artifact for the selected
models.

Use `--clear-hf-cache` if the RunPod volume cannot hold all downloaded model
weights at once. This redownloads models on future runs.

Artifacts are written beneath the configured experiment name:

```text
results/residual_input_dependence_v002/
  manifest.json
  data/
    raw_windows_wikitext_train_b08601e_seed44_n10000.jsonl
  stats/<model>/
    <target>_stats.pt
  metrics/<model>/
    <target>_metrics.pt
  tables/<model>/
    <target>_convergence.csv
    <target>_reliability.csv
  tables/comparison/
    <target>_model_layer_summary.csv
  figures/<model>/
    <target>_summary.png
  figures/comparison/
    <target>_model_comparison.png
```

The manifest records configurations, pinned model and dataset revisions,
relevant package/CUDA versions, schema versions, timestamp, and Git commit.
Writes are atomic. Complete stages are skipped, partial stages stop with an
error, and `--force` intentionally rebuilds the selected experiment outputs.
Commit the code before the final run; the manifest records a dirty worktree.

Cross-model plots use relative layer depth. Tokenizers differ, so comparisons
are distribution-level rather than aligned-token or aligned-dimension claims.

## Individual stages

The three stages also remain independently runnable:

```bash
python -m src.experiments.run_collect_stats --help
python -m src.experiments.run_compute_metrics --help
python -m src.experiments.run_analysis --help
```

## Test

```bash
python -m unittest discover -s tests
```
