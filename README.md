# Evaluation of Neuron Interpretability

This repository contains two independent workflows, each with its own environment:

1. **Data & neuron-activation simulation** (`data_generation.py`,
   `neuron_activation_simulation.py`, `probability_metrics_simulation.py`,
   `distance_corr_simulation.py`, and their `test_*` runners). Runs on a lightweight
   Python virtual environment (`.venv`) built from `requirements.txt`.
2. **The Vega model** (`vega/`). The upstream [VEGA repository](https://github.com/LucasESBS/vega)
   is **not** bundled in this repo (`vega/` holds an empty `PLACEHOLDER` file on GitHub).
   Clone it locally before training or running sweeps. Vega runs in a dedicated Python 3.7
   conda environment (`venv_vega`) built from `requirements_vega.txt` (PyTorch 1.5.1,
   scanpy 1.5.1, ...).


---

## 1. Simulation environment (`.venv`)

Used to generate the synthetic data, simulate neuron activations, and compute the
probability and distance-correlation metrics.

### Create the environment

```bash
# From the repository root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

To update the environment after `requirements.txt` changes, with `.venv` activated:

```bash
pip install -r requirements.txt
```

### Run the simulations

```bash
# Activate first
source .venv/bin/activate

# Generate a structured dataset and visualize it
python data_generation.py

# Simulate neuron activations and the beta-impact summary
python neuron_activation_simulation.py

# Probability metric studies (beta / overlap / M / K / threshold sweeps)
python test_metrics_simulation.py --run-name prob_v1

# Distance-correlation metric studies (beta / M / K / overlap sweeps)
python test_distance_corr_simulation.py --run-name dcorr_v1

# MIG metric studies (beta / M / K / overlap sweeps; same N/M/K defaults)
python test_neuron_activation/test_mig_simulation.py --run-name mig_v1

# Use exponential alpha weighting instead of uniform:
python test_distance_corr_simulation.py --alpha-mode exponential --run-name dcorr_v1_exp

```

Each run writes its plots and CSVs to a `plots_<run-name>/` folder along with a
`run_parameters.txt` describing the configuration.

---

## 2. Vega environment (`venv_vega`)

Used to run the Vega model and the interpretability scripts (`vega_interpretability_simulation.py`,
`run_vega_fcn_sweep.py`, …). Requires the upstream VEGA source under `vega/` plus a
Python 3.7 conda environment. Training scripts expect pathway definitions at
`vega/vega/data/reactomes.gmt`.

### Clone upstream VEGA (required once)

After cloning **this** repository, `vega/` contains only an empty `PLACEHOLDER` file.
Replace it with the official VEGA clone:

```bash
# From the repository root
rm vega/PLACEHOLDER
git clone https://github.com/LucasESBS/vega.git vega
```

If `vega/` already exists and is not empty (e.g. you cloned VEGA there before), skip
the steps above.

If you previously cloned VEGA with `git clone` inside `vega/`, that folder contains
a nested `.git` directory. Remove it so this repository can track the empty
`PLACEHOLDER` marker (your local VEGA files stay on disk; they remain git-ignored):

```bash
rm -rf vega/.git
```

### Create the environment

Requires conda (Miniconda/Anaconda) for Python 3.7; pinned packages are installed with
pip. This works the same on **Linux and macOS** (CPU build).

```bash
# From the repository root
conda create -n venv_vega python=3.7 -y
conda activate venv_vega
pip install --upgrade pip
pip install -r requirements_vega.txt
```

> **Apple Silicon (M1/M2/M3) only.** These old versions have no arm64 wheels, so create
> the env under the Intel (`osx-64`) subdir via Rosetta:
>
> ```bash
> CONDA_SUBDIR=osx-64 conda create -n venv_vega python=3.7 -y
> conda activate venv_vega
> conda config --env --set subdir osx-64
> pip install --upgrade pip
> pip install -r requirements_vega.txt
> ```

If you prefer pure `venv` (no conda) you still need a Python 3.7 interpreter available:

```bash
python3.7 -m venv venv_vega
source venv_vega/bin/activate
pip install --upgrade pip
pip install -r requirements_vega.txt
```

Or use the helper script (macOS / Linux):

```bash
./create_vega_env.sh notlinux   # macOS / Apple Silicon (Rosetta, CPU-only)
./create_vega_env.sh linux      # Linux (NVIDIA GPU optional)
```

If `conda activate venv_vega` fails but a folder `venv_vega/` exists at the repo root,
activate by path instead:

```bash
conda activate "$(pwd)/venv_vega"
# or: source venv_vega/bin/activate
```

### Activate and use

```bash
conda activate venv_vega

# Optional: register the environment as a Jupyter kernel
python -m ipykernel install --user --name venv_vega --display-name "Python (venv_vega)"
```

To update the environment after editing `requirements_vega.txt` (with it activated):

```bash
pip install -r requirements_vega.txt
```

To remove it:

```bash
conda deactivate
conda env remove -n venv_vega
```

### VEGA fully-connected neuron sweep (`vega_fcn_sweep`)

Train VEGA2 on PBMC 8K across 11 fully-connected-neuron fractions (`fcn_000` … `fcn_100`,
step 10%), then compute interpretability metrics on the **test set only** (inference mode).

Metrics (via `vega_simulation/vega_fcn_metrics.py`):

- **Distance correlation** (`distance_corr`) — same logic as `vega_usage/distances_metrics.py`
- **Reduction-score probability** (`reduction_score_probability`) — overlap threshold **0.5**
  by default (exclude pairs with overlap ≥ threshold)

```bash
# Activate venv_vega first (see above)
cd test_vega_simulation

# Full sweep: train 11 models, evaluate metrics, aggregate plots
python run_vega_fcn_sweep.py --train --eval --plot

# Evaluate + plot only (checkpoints already in vega_fcn_sweep/)
python run_vega_fcn_sweep.py --eval --plot

# Subset of fractions (e.g. sparse, half, dense)
python run_vega_fcn_sweep.py --eval --plot --percents 0,50,100

# Skip training when metrics.json + checkpoint already exist
python run_vega_fcn_sweep.py --train --eval --plot --skip-existing

# Quick smoke test (3 fractions, 5 epochs, 20 pathways)
python run_vega_fcn_sweep.py --train --eval --plot --quick

# Custom overlap threshold for probability metric
python run_vega_fcn_sweep.py --eval --plot --overlap-threshold 0.5
```

**Outputs**

Per fraction (`vega_fcn_sweep/fcn_XXX/`): trained model, training plots, `metrics.json`,
`interpretability_metrics.csv` (per-pathway `distance_corr` and `reduction_score_probability`).

Aggregate (`vega_fcn_sweep/aggregate/`): `per_neuron_metrics.csv`,
`01_distance_corr_by_fcn.png`, `02_reduction_score_probability_by_fcn.png`,
combined metric plots, `run_parameters.txt`.

**MIG metric** (gp.prerank enrichment + latent MI; requires `gseapy` in `venv_vega`):

```bash
# Shared pathway enrichment once, then MIG for all fcn_* folders
python test_vega_simulation/run_vega_mig_sweep.py --force-recompute-enrichment

# Resume MIG only (enrichment already cached under shared_mig_enrichment/)
python test_vega_simulation/run_vega_mig_sweep.py --skip-existing

# Subset or aggregate-only
python test_vega_simulation/run_vega_mig_sweep.py --percents 0,50,100 --skip-existing
python test_vega_simulation/run_vega_mig_sweep.py --plot

# Single run folder
python vega_simulation/vega_mig_metrics.py --run-dir vega_fcn_sweep/fcn_000
```

MIG aggregate: `vega_fcn_sweep/mig_aggregate/` (`mig_summaries.csv`, `01_mig_by_fcn.png`).
After MIG is computed, re-run `python run_vega_fcn_sweep.py --plot` to refresh combined
plots that overlay distance corr, probability, and MIG.

**Sparse VEGA2 baseline** (single PBMC run, `fully_connected_neuron_fraction=0`):

```bash
python test_vega_simulation/test_vega_sparse_pbmc.py
python test_vega_simulation/test_vega_sparse_pbmc.py --compare-distance-corr-only --output-dir vega_sparse_test
```

Outputs default to `vega_sparse_test/` (model, performance metrics, interpretability CSVs,
distance-corr comparison vs original `vega_usage` code).
