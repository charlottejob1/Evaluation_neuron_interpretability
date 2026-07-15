#!/usr/bin/env python
"""Distance-correlation overlap-regime study (full beta sweep 0–1)."""

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

from distance_corr_simulation import (
    compute_distance_corr_per_concept,
    distance_corr_rows,
    _concept_keys_in_order,
)
from neuron_activation_simulation import simulate_neuron_activations
from shared.config import (
    ALPHA_EXP_SCALE,
    ALPHA_MODE,
    BETA_SWEEP,
    FIXED_K,
    FIXED_M,
    FIXED_N_EXAMPLES,
)
from shared.overlap_viz import plot_overlap_regime_distributions
from shared.paths import add_run_name_arg, apply_run_name_from_args, metric_output_dir
from shared.regime_data import build_all_regime_caches, load_regime_cache


def run_study(force_data: bool = False) -> pd.DataFrame:
    out_dir = metric_output_dir("distance_corr")
    os.makedirs(out_dir, exist_ok=True)
    cached = build_all_regime_caches(force=force_data)
    plot_overlap_regime_distributions(cached)

    rows = []
    for item in cached:
        name = item["name"]
        X, _, concept_map, meta = load_regime_cache(name)
        concept_keys = _concept_keys_in_order(concept_map)
        for beta in BETA_SWEEP:
            print("[distance_corr] %s beta=%.1f" % (name, beta))
            Z, _, _ = simulate_neuron_activations(
                X=X,
                concept_map=concept_map,
                beta=beta,
                alpha_mode=ALPHA_MODE,
                alpha_exp_scale=ALPHA_EXP_SCALE,
            )
            results = compute_distance_corr_per_concept(X, concept_map, Z, concept_keys)
            batch = distance_corr_rows(results, beta, name)
            for row in batch:
                row["n_variables"] = FIXED_M
                row["n_concepts"] = FIXED_K
                row["mean_overlap"] = meta["overlap_stats"]["mean_overlap"]
                row["median_overlap"] = meta["overlap_stats"]["median_overlap"]
            rows.extend(batch)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "distance_corr_overlap_sweep.csv")
    df.to_csv(csv_path, index=False)
    print("Saved: %s" % csv_path)
    return df


def main():
    p = argparse.ArgumentParser(description="Distance-corr overlap-regime metric study.")
    add_run_name_arg(p)
    p.add_argument("--force-data", action="store_true", help="Regenerate cached regime datasets.")
    args = p.parse_args()
    apply_run_name_from_args(args)
    run_study(force_data=args.force_data)


if __name__ == "__main__":
    main()
