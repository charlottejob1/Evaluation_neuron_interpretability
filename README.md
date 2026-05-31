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

Keep the two environments separate: the simulation code targets a modern Python while
Vega needs an older, pinned stack.

> **Cross-platform note.** `requirements_vega.txt` is both **Linux- and macOS-friendly**:
> the original spec's `cudatoolkit=10.2` (NVIDIA/CUDA, unavailable on macOS) is dropped
> in favor of the **CPU PyTorch build**, which runs on both OSes. On Apple Silicon
> (arm64) these old pinned versions have no native wheels — see the Apple Silicon note
> below to build the env under `osx-64` (Rosetta).

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
# Use exponential alpha weighting instead of uniform:
python test_distance_corr_simulation.py --alpha-mode exponential --run-name dcorr_v1_exp
# Faster reduced sweeps:
python test_distance_corr_simulation.py --quick --run-name dcorr_quick
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

To pin a specific release instead of `main`, clone then check out a tag, for example:

```bash
cd vega
git checkout v1.0.0   # use the tag documented in your experiment notes, if any
cd ..
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
