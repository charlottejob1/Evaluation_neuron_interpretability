#!/usr/bin/env python3
"""
Validate VEGA interpretability metrics on the sparse decoder (fcn=0).

In-memory only (no saved intermediates). Uses the vega_usage pipeline:
  to_latent, inhibition perturbation, overlap_threshold=0.5, mask-based overlap.

Expected: mean reduction-score probability > 0.6.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "vega_simulation"))

from vega_fcn_metrics import (  # noqa: E402
    DEFAULT_OVERLAP_THRESHOLD,
    evaluate_vega2_interpretability,
    load_trained_vega2,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="results_vega_pbmc_fcn0.00")
    p.add_argument("--data-dir", default="pbmc_data")
    p.add_argument("--max-pathways", type=int, default=50)
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isfile(os.path.join(run_dir, "metrics.json")):
        run_dir = os.path.abspath(os.path.join("vega_fcn_sweep", "fcn_000"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Run dir: %s | device: %s" % (run_dir, device))

    model, ctx, meta = load_trained_vega2(run_dir, data_dir=args.data_dir, device=device)
    df = evaluate_vega2_interpretability(
        model,
        ctx["adata_test"],
        ctx["pathway_dict"],
        ctx["list_pathways"],
        ctx["pathway_mask"],
        overlap_threshold=DEFAULT_OVERLAP_THRESHOLD,
        max_pathways=args.max_pathways,
    )

    prob = df["reduction_score_probability"].dropna()
    dist = df["distance_corr"].dropna()
    print("\nSparse decoder (fcn=%.0f%%, n_fc=%d)" % (
        meta["fully_connected_neuron_fraction"] * 100,
        meta["n_fully_connected_neurons_selected"],
    ))
    print("Pathways evaluated: %d (test set, max=%d)" % (len(df), args.max_pathways))
    print("Overlap threshold: %.1f" % DEFAULT_OVERLAP_THRESHOLD)
    print("Mean reduction-score probability: %.4f" % prob.mean())
    print("Median reduction-score probability: %.4f" % prob.median())
    print("Mean distance correlation: %.4f" % dist.mean())
    print("\nTarget (vega_usage): mean probability > 0.6")
    print("PASS" if prob.mean() > 0.6 else "BELOW TARGET")


if __name__ == "__main__":
    main()
