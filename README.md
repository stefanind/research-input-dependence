# Residual input dependence

Measures how much each residual-stream coordinate varies across WikiText inputs.
For each layer, token position, and residual dimension, the score is
`population variance / mean-square activation` and lies between zero and one.

## Run

```powershell
python -m pip install -e .
python -m src.experiments.run_collect_stats --model gpt2-small
python -m src.experiments.run_compute_metrics --stats-path results/stats/residual_input_dependence_v001_gpt2-small_resid_post_stats.pt
```

Models are registered in `configs/models.yaml`; experiment size, device, hooks,
and snapshot points are set in `configs/experiment.yaml`. The `auto` device uses
CUDA when available and otherwise falls back to CPU.

## Test

```powershell
python -m unittest discover -s tests
```
