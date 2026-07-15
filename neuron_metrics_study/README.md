# Neuron metrics overlap study

Self-contained study package comparing **distance correlation (DC)**, **probability (KOS)**, and **MIG** under three tuned overlap regimes, with a full **beta sweep 0.0–1.0** (step 0.1).

## Fixed simulation settings

| Parameter | Value |
|-----------|-------|
| N | 1000 |
| M | 2000 |
| K | 300 |
| `alpha_mode` | exponential |
| `overlap_threshold` | None (all pairs) |
| `size_strategy` | dirichlet (1–200) |
| seed | 12345 |

## Overlap regimes (tuned for target mean overlap)

| Regime | Target mean | Realized (seed=12345) | Key parameters |
|--------|-------------|------------------------|----------------|
| low_overlap | ≈ 0 | mean **0.010**, med 0.000 | skew=50, ceiling=0.05, dirichlet=1.5 |
| medium_overlap | ≈ 0.2 | mean **0.204**, med 0.041 | skew=0.2, floor=0.02, dirichlet=14 |
| high_overlap | ≈ 0.4 | mean **0.402**, med 0.300 | skew=0.001, floor=0.35, dirichlet=30, **size_min=10** |

High overlap uses a larger `size_min` so concepts are not size-1 (which would force overlap ∈ {0, 1} and cap the mean).

## Layout

```
neuron_metrics_study/
  shared/           # config, dataset cache, overlap plots
  data/             # cached X per regime + overlap distribution figure
  distance_corr/    # metric CSV + runner
  probability/
  mig/
  aggregate/        # combined DC + KOS + MIG plots
  run_all.py        # end-to-end orchestrator
```

## Run

From the repository root (requires `neuron_simulation/` on `PYTHONPATH` via runners):

```bash
cd neuron_metrics_study
../venv_vega/bin/python run_all.py --run-name v2_mean_020_040 --force-data
```

Previous results at the top level (`data/`, `*/outputs/`) are left untouched. Named runs go under `runs/<run-name>/`.

Options:

```bash
../venv_vega/bin/python run_all.py --force-data      # regenerate cached datasets (same run folder)
../venv_vega/bin/python run_all.py --run-name v2_mean_020_040 --force-data  # new run folder
../venv_vega/bin/python run_all.py --metrics-only mig
../venv_vega/bin/python run_all.py --skip-metrics    # aggregate only
```

## Outputs

- `data/overlap_regime_distributions.png` — histograms with **mean** and **median** overlap per regime
- `{metric}/outputs/*_overlap_sweep.csv` — all regimes × all betas
- `aggregate/outputs/01_combined_metrics_{regime}_beta_dual.png`
- `aggregate/outputs/02_combined_metrics_{regime}_beta_shared_yscale.png`
- `aggregate/outputs/03_combined_metrics_by_overlap_dual.png`
- `aggregate/outputs/04_combined_metrics_by_overlap_shared_yscale.png`

Depends on the parent repo’s `neuron_simulation/` modules (`data_generation`, `neuron_activation_simulation`, etc.).
