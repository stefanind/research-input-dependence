# Residual input variation

For a fixed layer `l`, token position `p`, and residual dimension `d`, over
`N` input chunks:

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

## RunPod setup

Target image:
`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.

```bash
python -m pip install -e .
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The check should report Torch 2.4, CUDA 12.4, and `True`.

## Run the experiment

Run commands from the repository root. The default experiment uses 10,000
seeded, randomly sampled, unpadded WikiText-103 chunks of 128 tokens each.

The pipeline runs collection, metrics, convergence, split-half reliability,
per-model plots, and cross-model comparison. Completed artifacts are skipped.

```bash
python -m src.experiments.run_pipeline \
  --model gpt2-small pythia-160m opt-125m
```

Add `pythia-410m` for the optional scale check, or use `--model all`. Use
`--force` only when intentionally replacing every artifact for the selected
models.

Artifacts are written beneath the configured experiment name:

```text
results/residual_input_dependence_v001/
  manifest.json
  stats/       # large tensors; ignored by Git
  metrics/     # derived tensors; ignored by Git
  tables/      # convergence, reliability, and comparison CSVs
  figures/     # per-model and cross-model plots
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
