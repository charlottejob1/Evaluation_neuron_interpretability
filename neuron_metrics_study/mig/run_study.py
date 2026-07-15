#!/usr/bin/env python
"""MIG overlap-regime study (full beta sweep 0–1)."""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_STUDY)
_SIM = os.path.join(_REPO, "neuron_simulation")
for p in (_STUDY, _SIM):
    if p not in sys.path:
        sys.path.insert(0, p)

from mig_metrics_simulation import compute_mig, mig_rows_from_result
from neuron_activation_simulation import simulate_neuron_activations
from shared.config import (
    ALPHA_EXP_SCALE,
    ALPHA_MODE,
    BETA_SWEEP,
    FIXED_K,
    FIXED_M,
    FIXED_N_EXAMPLES,
    SEED,
)
from shared.overlap_viz import plot_overlap_regime_distributions
from shared.paths import add_run_name_arg, apply_run_name_from_args, metric_output_dir
from shared.regime_data import build_all_regime_caches, load_regime_cache


def run_study(force_data: bool = False) -> pd.DataFrame:
    out_dir = metric_output_dir("mig")
    os.makedirs(out_dir, exist_ok=True)
    cached = build_all_regime_caches(force=force_data)
    plot_overlap_regime_distributions(cached)

    rows = []
    for item in cached:
        name = item["name"]
        X, concept_activities, concept_map, meta = load_regime_cache(name)
        for beta in BETA_SWEEP:
            print("[mig] %s beta=%.1f" % (name, beta))
            Z, _, _ = simulate_neuron_activations(
                X=X,
                concept_map=concept_map,
                beta=beta,
                alpha_mode=ALPHA_MODE,
                alpha_exp_scale=ALPHA_EXP_SCALE,
            )
            result = compute_mig(
                concept_activities,
                Z,
                n_bins=20,
                random_state=SEED,
                n_jobs=-1,
            )
            result["beta"] = beta
            result["overlap_profile"] = name
            result["n_examples"] = FIXED_N_EXAMPLES
            result["n_variables"] = FIXED_M
            result["n_concepts"] = FIXED_K
            batch = mig_rows_from_result(result)
            for row in batch:
                row["mean_overlap"] = meta["overlap_stats"]["mean_overlap"]
                row["median_overlap"] = meta["overlap_stats"]["median_overlap"]
            rows.extend(batch)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "mig_overlap_sweep.csv")
    df.to_csv(csv_path, index=False)
    print("Saved: %s" % csv_path)
    return df


def main():
    p = argparse.ArgumentParser(description="MIG overlap-regime metric study.")
    add_run_name_arg(p)
    p.add_argument("--force-data", action="store_true", help="Regenerate cached regime datasets.")
    args = p.parse_args()
    apply_run_name_from_args(args)
    run_study(force_data=args.force_data)


if __name__ == "__main__":
    main()
