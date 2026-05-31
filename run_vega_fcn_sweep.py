"""
Train and evaluate VEGA2 across fully-connected-neuron fractions 0%–100% (step 10%).

Creates a sweep repository::

    vega_fcn_sweep/
      fcn_000/   # 0% selected → original sparse decoder
      fcn_010/
      ...
      fcn_100/   # 100% → fully connected linear decoder
      aggregate/
        per_neuron_metrics.csv
        01_distance_corr_by_fcn.png
        02_reduction_score_probability_by_fcn.png
        run_parameters.txt

Each sub-folder holds one full PBMC training run (model, metrics, plots).
After training, each saved model is evaluated in inference mode on the **test set
only** using the metrics defined in ``probability_metrics.py`` and
``distances_metrics.py`` (via ``vega_fcn_metrics.py``).

Example (full sweep, PBMC 8K hyperparameters):

  python run_vega_fcn_sweep.py --train --eval --plot

Evaluate + plot only (models already trained):

  python run_vega_fcn_sweep.py --eval --plot

Quick smoke test (2 fractions, 5 epochs, 20 pathways):

  python run_vega_fcn_sweep.py --train --eval --plot --quick
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional

_REPO = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_REPO, "test_vega_simulation"))
sys.path.insert(0, os.path.join(_REPO, "vega_simulation"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_vega_pbmc import run_pbmc_training
from vega_fcn_metrics import DEFAULT_OVERLAP_THRESHOLD, evaluate_run_directory


SWEEP_ROOT = "vega_fcn_sweep"
FCN_PERCENTS = list(range(0, 101, 10))  # 0, 10, ..., 100

# PBMC 8K tuned hyperparameters (VEGA2 beta = KL weight)
PBMC8K_HYPERPARAMS = {
    "n_epochs": 1000,
    "learning_rate": 0.00062187,
    "batch_size": 256,
    "kld_weight": 0.000233864,
    "dropout": 0.0452273,
    "train_patience": 25,
    "test_patience": 25,
    "n_top_genes": 2000,
    "train_size": 0.9,
    "seed": 42,
}


def fcn_dir_name(percent: int) -> str:
    return "fcn_%03d" % percent


def fcn_fraction(percent: int) -> float:
    return percent / 100.0


def run_dir_for_percent(sweep_root: str, percent: int) -> str:
    return os.path.abspath(os.path.join(sweep_root, fcn_dir_name(percent)))


def legend_label(n_selected: int, n_pathway_nodes: int, percent: int) -> str:
    if percent == 0:
        suffix = "sparse decoder"
    elif percent == 100:
        suffix = "dense decoder"
    else:
        suffix = "%d%% selected" % percent
    return "%d / %d FC neurons (%s)" % (n_selected, n_pathway_nodes, suffix)


def train_one(
    percent: int,
    sweep_root: str,
    *,
    data_dir: str,
    gmt_path: str,
    hyperparams: dict,
    device: torch.device,
    skip_existing: bool,
) -> str:
    out_dir = run_dir_for_percent(sweep_root, percent)
    model_glob = os.path.join(out_dir, "vega2_pbmc_fcn*.pt")
    if skip_existing and os.path.isfile(os.path.join(out_dir, "metrics.json")):
        import glob

        if glob.glob(model_glob):
            print("[skip train] %s already has a checkpoint" % out_dir)
            return out_dir

    print("\n" + "=" * 60)
    print("Training fcn_%03d (%d%% fully connected neurons)" % (percent, percent))
    print("Output: %s" % out_dir)
    print("=" * 60)

    run_pbmc_training(
        data_dir=data_dir,
        gmt_path=gmt_path,
        fully_connected_neuron_fraction=fcn_fraction(percent),
        output_dir=out_dir,
        device=device,
        **hyperparams,
    )
    return out_dir


def evaluate_one(
    run_dir: str,
    *,
    data_dir: str,
    overlap_threshold: float,
    max_pathways: Optional[int],
    device: torch.device,
) -> pd.DataFrame:
    print("Evaluating interpretability metrics (test set only): %s" % run_dir)
    df = evaluate_run_directory(
        run_dir,
        data_dir=data_dir,
        overlap_threshold=overlap_threshold,
        max_pathways=max_pathways,
        device=device,
        save_csv=True,
    )
    print("Saved %d pathway metrics for %s" % (len(df), run_dir))
    return df


def _blue_boxplot(
    data_by_group: List[np.ndarray],
    positions: List[int],
    xticklabels: List[str],
    ylabel: str,
    title: str,
    out_path: str,
    ylim: Optional[tuple] = None,
    reference_y: Optional[float] = None,
):
    plt.figure(figsize=(max(14, 1.2 * len(positions)), 6))
    bp = plt.boxplot(
        data_by_group,
        positions=positions,
        widths=0.65,
        patch_artist=True,
        showfliers=False,
    )
    for box in bp["boxes"]:
        box.set(facecolor="steelblue", edgecolor="navy", alpha=0.75)
    for element in ("whiskers", "caps", "medians"):
        for item in bp[element]:
            item.set(color="navy")
    means = [
        float(np.nanmean(d)) if len(d) > 0 else float("nan")
        for d in data_by_group
    ]
    plt.plot(
        positions,
        means,
        color="red",
        marker="o",
        linewidth=2,
        markersize=6,
        label="Mean",
        zorder=3,
    )
    plt.xticks(positions, xticklabels, rotation=45, ha="right")
    if ylim is not None:
        plt.ylim(*ylim)
    if reference_y is not None:
        plt.axhline(
            reference_y,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label="y = %.1f" % reference_y,
            zorder=2,
        )
    plt.xlabel("Fully connected neurons selected")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot: %s" % out_path)


def plot_sweep_metrics(combined: pd.DataFrame, aggregate_dir: str):
    os.makedirs(aggregate_dir, exist_ok=True)

    # One box per FC-neuron count (11 thresholds: 0, 67, …, 674)
    group_col = "n_fully_connected_neurons_selected"
    groups = (
        combined.groupby(group_col, sort=True)
        .agg(
            n_fc_neurons=(group_col, "first"),
            n_pathway_nodes=("n_pathway_nodes", "first"),
            fcn_fraction=("fully_connected_neuron_fraction", "first"),
        )
        .reset_index()
    )
    groups["fcn_percent"] = (groups["fcn_fraction"] * 100).round(0).astype(int)

    x_labels = [
        legend_label(int(r.n_fc_neurons), int(r.n_pathway_nodes), int(r.fcn_percent))
        for _, r in groups.iterrows()
    ]
    positions = list(range(len(groups)))

    # --- Distance correlation ---
    dist_col = "distance_corr"
    dist_data = [
        combined.loc[
            combined["n_fully_connected_neurons_selected"] == n, dist_col
        ].dropna().values
        for n in groups["n_fully_connected_neurons_selected"]
    ]
    all_dist = combined[dist_col].dropna().values
    ylim_dist = (-0.05, 1.0) if len(all_dist) == 0 else (
        min(-0.05, float(np.nanmin(all_dist)) - 0.05),
        1.0,
    )
    _blue_boxplot(
        dist_data,
        positions,
        x_labels,
        "Distance correlation (per neuron)",
        "Distance metric vs fully connected neuron selection\n"
        "(each box = one threshold; 0 FC = sparse decoder, max FC = dense decoder)",
        os.path.join(aggregate_dir, "01_distance_corr_by_fcn.png"),
        ylim=ylim_dist,
        reference_y=0.0,
    )

    # --- Reduction-score probability ---
    prob_col = "reduction_score_probability"
    prob_data = [
        combined.loc[
            combined["n_fully_connected_neurons_selected"] == n, prob_col
        ].dropna().values
        for n in groups["n_fully_connected_neurons_selected"]
    ]
    all_prob = combined[prob_col].dropna().values
    ylim_prob = (0.0, 1.05) if len(all_prob) == 0 else (
        max(0.0, float(np.nanmin(all_prob)) - 0.05),
        min(1.05, float(np.nanmax(all_prob)) + 0.05),
    )
    _blue_boxplot(
        prob_data,
        positions,
        x_labels,
        "Reduction score probability (per neuron)",
        "Probability metric vs fully connected neuron selection\n"
        "(each box = one threshold; 0 FC = sparse decoder, max FC = dense decoder)",
        os.path.join(aggregate_dir, "02_reduction_score_probability_by_fcn.png"),
        ylim=ylim_prob,
        reference_y=0.5,
    )


def write_run_parameters(path: str, args, hyperparams: dict, combined: Optional[pd.DataFrame]):
    with open(path, "w", encoding="utf-8") as f:
        f.write("VEGA2 fully-connected neuron sweep\n")
        f.write("Generated (UTC): %s\n\n" % datetime.now(timezone.utc).isoformat())
        f.write("Sweep root: %s\n" % os.path.abspath(args.sweep_root))
        f.write("FCN percents: %s\n\n" % ",".join(str(p) for p in FCN_PERCENTS))
        f.write("Training hyperparameters\n")
        f.write("-" * 40 + "\n")
        for k, v in hyperparams.items():
            f.write("%s: %s\n" % (k, v))
        f.write("\nEvaluation (test set only)\n")
        f.write("-" * 40 + "\n")
        f.write("overlap_threshold: %s\n" % args.overlap_threshold)
        f.write("data_dir: %s\n" % args.data_dir)
        f.write("gmt_path: %s\n" % args.gmt_path)
        if combined is not None:
            f.write("\nAggregate rows: %d\n" % len(combined))
            f.write(
                "FC neuron counts: %s\n"
                % ", ".join(
                    str(x)
                    for x in sorted(combined["n_fully_connected_neurons_selected"].unique())
                )
            )


def parse_args():
    p = argparse.ArgumentParser(
        description="Train/evaluate VEGA2 across 11 fully-connected-neuron fractions."
    )
    p.add_argument("--sweep-root", type=str, default=SWEEP_ROOT)
    p.add_argument("--data-dir", type=str, default="pbmc_data")
    p.add_argument("--gmt-path", type=str, default="vega/vega/data/reactomes.gmt")
    p.add_argument("--train", action="store_true", help="Train all 11 configurations.")
    p.add_argument("--eval", action="store_true", help="Run interpretability metrics.")
    p.add_argument("--plot", action="store_true", help="Aggregate CSVs and plot boxplots.")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training when metrics.json + checkpoint exist.",
    )
    p.add_argument("--overlap-threshold", type=float, default=DEFAULT_OVERLAP_THRESHOLD)
    p.add_argument(
        "--max-pathways",
        type=int,
        default=None,
        help="Limit pathways for faster debugging.",
    )
    p.add_argument(
        "--percents",
        type=str,
        default=None,
        help="Comma-separated subset of percents (e.g. 0,50,100). Default: all 11.",
    )
    p.add_argument("--quick", action="store_true", help="Smoke test settings.")
    return p.parse_args()


def main():
    args = parse_args()
    if not (args.train or args.eval or args.plot):
        print("Nothing to do. Pass --train, --eval, and/or --plot.")
        return

    percents = FCN_PERCENTS
    if args.percents:
        percents = [int(x.strip()) for x in args.percents.split(",") if x.strip()]

    hyperparams = dict(PBMC8K_HYPERPARAMS)
    if args.quick:
        hyperparams["n_epochs"] = 5
        hyperparams["train_patience"] = 2
        hyperparams["test_patience"] = 2
        if args.max_pathways is None:
            args.max_pathways = 20
        if len(percents) == 11:
            percents = [0, 50, 100]
        print("[quick] Reduced epochs, patience, and fraction list.")

    os.makedirs(args.sweep_root, exist_ok=True)
    aggregate_dir = os.path.join(args.sweep_root, "aggregate")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    if args.train:
        for pct in percents:
            train_one(
                pct,
                args.sweep_root,
                data_dir=args.data_dir,
                gmt_path=args.gmt_path,
                hyperparams=hyperparams,
                device=device,
                skip_existing=args.skip_existing,
            )

    combined: Optional[pd.DataFrame] = None
    if args.eval:
        frames: List[pd.DataFrame] = []
        for pct in percents:
            run_dir = run_dir_for_percent(args.sweep_root, pct)
            if not os.path.isdir(run_dir):
                print("[warn] missing run dir: %s" % run_dir)
                continue
            frames.append(
                evaluate_one(
                    run_dir,
                    data_dir=args.data_dir,
                    overlap_threshold=args.overlap_threshold,
                    max_pathways=args.max_pathways,
                    device=device,
                )
            )
        if frames:
            combined = pd.concat(frames, ignore_index=True)
            os.makedirs(aggregate_dir, exist_ok=True)
            agg_csv = os.path.join(aggregate_dir, "per_neuron_metrics.csv")
            combined.to_csv(agg_csv, index=False)
            print("Saved aggregate CSV: %s" % agg_csv)

    if args.plot:
        agg_csv = os.path.join(aggregate_dir, "per_neuron_metrics.csv")
        if combined is None:
            if not os.path.isfile(agg_csv):
                print("No aggregate CSV at %s — run --eval first." % agg_csv)
                return
            combined = pd.read_csv(agg_csv)
        plot_sweep_metrics(combined, aggregate_dir)

    param_path = os.path.join(aggregate_dir, "run_parameters.txt")
    os.makedirs(aggregate_dir, exist_ok=True)
    write_run_parameters(param_path, args, hyperparams, combined)
    print("\nDone. Sweep root: %s" % os.path.abspath(args.sweep_root))


if __name__ == "__main__":
    main()
