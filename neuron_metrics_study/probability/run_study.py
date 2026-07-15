#!/usr/bin/env python
"""Probability (KOS) overlap-regime study (full beta sweep 0–1, no overlap threshold)."""

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

from probability_metrics_simulation import (
    compute_reduction_tensor,
    probability_rows_from_reduction,
)
from shared.config import (
    ALPHA_EXP_SCALE,
    ALPHA_MODE,
    BETA_SWEEP,
    FIXED_K,
    FIXED_M,
    OVERLAP_THRESHOLD,
)
from shared.overlap_viz import plot_overlap_regime_distributions
from shared.paths import add_run_name_arg, apply_run_name_from_args, metric_output_dir
from shared.regime_data import build_all_regime_caches, load_regime_cache


def run_study(force_data: bool = False) -> pd.DataFrame:
    out_dir = metric_output_dir("probability")
    os.makedirs(out_dir, exist_ok=True)
    cached = build_all_regime_caches(force=force_data)
    plot_overlap_regime_distributions(cached)

    rows = []
    for item in cached:
        name = item["name"]
        X, _, concept_map, meta = load_regime_cache(name)
        for beta in BETA_SWEEP:
            print("[probability] %s beta=%.1f" % (name, beta))
            reduction_tensor, _ = compute_reduction_tensor(
                X=X,
                concept_map=concept_map,
                beta=beta,
                alpha_mode=ALPHA_MODE,
                alpha_exp_scale=ALPHA_EXP_SCALE,
            )
            batch = probability_rows_from_reduction(
                reduction_tensor=reduction_tensor,
                concept_map=concept_map,
                n_variables=FIXED_M,
                beta=beta,
                overlap_profile=name,
                overlap_threshold=OVERLAP_THRESHOLD,
            )
            for row in batch:
                row["n_variables"] = FIXED_M
                row["n_concepts"] = FIXED_K
                row["mean_overlap"] = meta["overlap_stats"]["mean_overlap"]
                row["median_overlap"] = meta["overlap_stats"]["median_overlap"]
            rows.extend(batch)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "probabilities_overlap_sweep.csv")
    df.to_csv(csv_path, index=False)
    print("Saved: %s" % csv_path)
    return df


def main():
    p = argparse.ArgumentParser(description="Probability overlap-regime metric study.")
    add_run_name_arg(p)
    p.add_argument("--force-data", action="store_true", help="Regenerate cached regime datasets.")
    args = p.parse_args()
    apply_run_name_from_args(args)
    run_study(force_data=args.force_data)


if __name__ == "__main__":
    main()
