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
        03_combined_metrics_shared_yscale.png
        04_combined_metrics_dual_yscale.png
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

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "test_vega_simulation"))
sys.path.insert(0, os.path.join(_REPO, "vega_simulation"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_vega_pbmc import run_pbmc_training
from vega_fcn_metrics import (
    DEFAULT_OVERLAP_THRESHOLD,
    evaluate_run_directory,
    unrescale_reduction_score_probability,
)

PROBABILITY_YLIM = (0.45, 1.05)
PROBABILITY_YLABEL = "P(self reduction > other)"

LEGEND_DC = "DC"
LEGEND_KOS = "KOS"
LEGEND_MIG = "MIG"

# Combined shared-yscale plot (03_combined_metrics_shared_yscale.png).
# Full resolution (699×411 px @ 150 dpi), typography aligned with neuron fig 04.
COMBINED_SHARED_DPI = 150
COMBINED_SHARED_FIGSIZE = (700 / COMBINED_SHARED_DPI, 315 / COMBINED_SHARED_DPI)
COMBINED_AXIS_LABEL_FONTSIZE = 6
COMBINED_TICK_LABEL_FONTSIZE = 5
COMBINED_LEGEND_FONTSIZE = 5
COMBINED_TITLE_FONTSIZE = 7
COMBINED_SHARED_LINE_WIDTH = 1.0
COMBINED_SHARED_MARKER_SIZE = 3
COMBINED_SHARED_REF_LINE_WIDTH = 1.0
COMBINED_SPINE_LINE_WIDTH = 0.8
COMBINED_SHARED_SUBPLOT_MARGINS = dict(left=0.14, right=0.99, top=0.91, bottom=0.19)

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


def combined_plot_xlabel(n_selected: int, _n_pathway_nodes: int, percent: int) -> str:
    """X tick label for combined metric plots, e.g. ``10%``."""
    return "%d%%" % percent


COMBINED_XLABEL = "Percentage of Fully-Connected Neurons (gamma)"
COMBINED_SHARED_TITLE = "VEGA"


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


def _fcn_groups_layout(combined: pd.DataFrame):
    """Shared x-axis layout: 11 FC thresholds, legend labels, positions."""
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
    fc_counts = groups[group_col].tolist()
    return group_col, groups, x_labels, positions, fc_counts


def _means_by_fc_count(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    fc_counts: List[int],
) -> np.ndarray:
    return np.array(
        [
            float(np.nanmean(df.loc[df[group_col] == n, value_col].dropna().values))
            for n in fc_counts
        ],
        dtype=float,
    )


def _load_mig_means_by_fc(sweep_root: str, fc_counts: List[int]) -> Optional[np.ndarray]:
    group_col = "n_fully_connected_neurons_selected"
    per_concept_path = os.path.join(
        sweep_root, "mig_aggregate", "per_concept_mig_all.csv"
    )
    if os.path.isfile(per_concept_path):
        mig_df = pd.read_csv(per_concept_path)
        if "per_concept_mig" in mig_df.columns and group_col in mig_df.columns:
            return _means_by_fc_count(mig_df, group_col, "per_concept_mig", fc_counts)

    summaries_path = os.path.join(sweep_root, "mig_aggregate", "mig_summaries.csv")
    if not os.path.isfile(summaries_path):
        return None
    summ = pd.read_csv(summaries_path).sort_values("fcn_percent")
    if len(summ) != len(fc_counts):
        return None
    mig_col = "mig" if "mig" in summ.columns else "per_concept_mig_mean"
    if mig_col not in summ.columns:
        return None
    return summ[mig_col].to_numpy(dtype=float)


def _mig_adapted_ylim(mig_means: np.ndarray) -> tuple:
    valid = mig_means[np.isfinite(mig_means)]
    if len(valid) == 0:
        return (0.0, 1.0)
    span = float(np.max(valid) - np.min(valid))
    pad = max(0.15 * span, 0.00025)
    return (float(np.min(valid)) - pad, float(np.max(valid)) + pad)


def _add_y_reference_lines(ax) -> None:
    """Dotted black line at y=0.5 (chance) with label beside the right edge."""
    ax.axhline(0.5, color="black", linestyle=":", linewidth=COMBINED_SHARED_REF_LINE_WIDTH, zorder=2)
    ax.text(
        1.01,
        0.5,
        "0.5",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        color="black",
        fontsize=COMBINED_TICK_LABEL_FONTSIZE,
        clip_on=False,
    )


def _style_compact_legend(ax, fontsize: int) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    ax.legend(
        handles,
        labels,
        loc="upper right",
        fontsize=fontsize,
        ncol=len(labels),
        handlelength=1.0,
        handletextpad=0.35,
        borderpad=0.25,
        labelspacing=0.2,
        columnspacing=0.6,
        markerscale=0.7,
    )


def _style_combined_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(COMBINED_SPINE_LINE_WIDTH)
    ax.tick_params(axis="both", width=COMBINED_SPINE_LINE_WIDTH, length=3)


def _style_combined_axes(ax, *, legend_loc: str = "best") -> None:
    ax.tick_params(axis="both", labelsize=COMBINED_TICK_LABEL_FONTSIZE)
    ax.xaxis.label.set_fontsize(COMBINED_AXIS_LABEL_FONTSIZE)
    ax.yaxis.label.set_fontsize(COMBINED_AXIS_LABEL_FONTSIZE)
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=COMBINED_TITLE_FONTSIZE)
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(COMBINED_LEGEND_FONTSIZE)
    elif legend_loc:
        ax.legend(loc=legend_loc, fontsize=COMBINED_LEGEND_FONTSIZE)


def plot_combined_metrics_by_fcn(
    combined: pd.DataFrame,
    sweep_root: str,
    aggregate_dir: str,
) -> None:
    """Three mean curves (distance corr, probability, MIG) vs FC selection."""
    os.makedirs(aggregate_dir, exist_ok=True)
    group_col, groups, _x_labels, positions, fc_counts = _fcn_groups_layout(combined)
    x_labels = [
        combined_plot_xlabel(int(r.n_fc_neurons), int(r.n_pathway_nodes), int(r.fcn_percent))
        for _, r in groups.iterrows()
    ]

    dist_means = _means_by_fc_count(combined, group_col, "distance_corr", fc_counts)
    prob_means = _means_by_fc_count(
        combined, group_col, "reduction_score_probability", fc_counts
    )
    mig_means = _load_mig_means_by_fc(sweep_root, fc_counts)
    if mig_means is None:
        print(
            "[plot] No MIG aggregate at %s/mig_aggregate/ — skipping combined plots."
            % sweep_root
        )
        return

    title_base = "Interpretability metrics vs fully connected neuron selection"
    line_kw = dict(linewidth=2, markersize=6, zorder=3)
    shared_line_kw = dict(
        linewidth=COMBINED_SHARED_LINE_WIDTH,
        markersize=COMBINED_SHARED_MARKER_SIZE,
        zorder=3,
    )

    # --- Shared y-axis (0–1, same as distance-corr / probability plots) ---
    fig, ax = plt.subplots(figsize=COMBINED_SHARED_FIGSIZE)
    ax.plot(
        positions,
        dist_means,
        color="steelblue",
        marker="o",
        label=LEGEND_DC,
        **shared_line_kw,
    )
    ax.plot(
        positions,
        prob_means,
        color="darkorange",
        marker="s",
        label=LEGEND_KOS,
        **shared_line_kw,
    )
    ax.plot(
        positions,
        mig_means,
        color="red",
        marker="^",
        label=LEGEND_MIG,
        **shared_line_kw,
    )
    ax.set_xticks(positions, x_labels)
    ax.set_ylim(0.0, 1.0)
    _add_y_reference_lines(ax)
    ax.set_xlabel(COMBINED_XLABEL, labelpad=1)
    ax.set_ylabel("Metric value")
    ax.set_title(COMBINED_SHARED_TITLE, pad=2)
    ax.legend(loc="upper right")
    _style_combined_axes(ax, legend_loc="")
    _style_combined_spines(ax)
    _style_compact_legend(ax, COMBINED_LEGEND_FONTSIZE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.subplots_adjust(**COMBINED_SHARED_SUBPLOT_MARGINS)
    shared_path = os.path.join(aggregate_dir, "03_combined_metrics_shared_yscale.png")
    fig.savefig(shared_path, dpi=COMBINED_SHARED_DPI)
    plt.close(fig)
    print("Saved plot: %s" % shared_path)

    # --- Dual y-axis: left = dist + prob; right = MIG (adapted) ---
    dual_figsize = (max(14, 1.2 * len(positions)), 6)
    fig, ax_left = plt.subplots(figsize=dual_figsize)
    ax_left.plot(
        positions,
        dist_means,
        color="steelblue",
        marker="o",
        label=LEGEND_DC,
        **line_kw,
    )
    ax_left.plot(
        positions,
        prob_means,
        color="darkorange",
        marker="s",
        label=LEGEND_KOS,
        **line_kw,
    )
    left_vals = np.concatenate(
        [dist_means[np.isfinite(dist_means)], prob_means[np.isfinite(prob_means)]]
    )
    if len(left_vals) == 0:
        ax_left.set_ylim(0.0, 1.0)
    else:
        ax_left.set_ylim(
            max(0.0, float(np.nanmin(left_vals)) - 0.05),
            min(1.05, float(np.nanmax(left_vals)) + 0.05),
        )

    ax_right = ax_left.twinx()
    ax_right.plot(
        positions,
        mig_means,
        color="red",
        marker="^",
        label=LEGEND_MIG,
        **line_kw,
    )
    ax_right.set_ylim(_mig_adapted_ylim(mig_means))
    ax_right.set_ylabel(LEGEND_MIG, color="red")
    ax_right.tick_params(axis="y", labelcolor="red")

    ax_left.set_xticks(positions, x_labels)
    ax_left.set_xlabel(COMBINED_XLABEL)
    ax_left.set_ylabel("%s / %s" % (LEGEND_DC, LEGEND_KOS))
    ax_left.set_title(title_base + "\n(dual y-axis: left = dist & prob, right = MIG)")
    _add_y_reference_lines(ax_left)
    ax_left.grid(True, axis="y", alpha=0.3)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")

    fig.subplots_adjust(right=0.88)
    fig.tight_layout()
    dual_path = os.path.join(aggregate_dir, "04_combined_metrics_dual_yscale.png")
    fig.savefig(dual_path, dpi=150)
    plt.close(fig)
    print("Saved plot: %s" % dual_path)


def unrescale_probability_csv(path: str) -> bool:
    """Revert 2*(p-0.5) on reduction_score_probability in one CSV."""
    if not os.path.isfile(path):
        return False
    df = pd.read_csv(path)
    col = "reduction_score_probability"
    if col not in df.columns:
        return False
    mask = df[col].notna()
    df.loc[mask, col] = df.loc[mask, col].astype(float).map(
        unrescale_reduction_score_probability
    )
    df.to_csv(path, index=False)
    return True


def unrescale_sweep_probability_csvs(sweep_root: str) -> None:
    """Unrescale all interpretability_metrics.csv and aggregate per_neuron_metrics.csv."""
    agg = os.path.join(sweep_root, "aggregate", "per_neuron_metrics.csv")
    if unrescale_probability_csv(agg):
        print("Unrescaled: %s" % agg)
    for pct in FCN_PERCENTS:
        path = os.path.join(
            sweep_root, fcn_dir_name(pct), "interpretability_metrics.csv"
        )
        if unrescale_probability_csv(path):
            print("Unrescaled: %s" % path)


def plot_sweep_metrics(combined: pd.DataFrame, aggregate_dir: str, sweep_root: str = SWEEP_ROOT):
    os.makedirs(aggregate_dir, exist_ok=True)

    # One box per FC-neuron count (11 thresholds: 0, 67, …, 674)
    _group_col, groups, x_labels, positions, _fc_counts = _fcn_groups_layout(combined)

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
    ylim_prob = PROBABILITY_YLIM if len(all_prob) == 0 else (
        max(PROBABILITY_YLIM[0], float(np.nanmin(all_prob)) - 0.05),
        min(PROBABILITY_YLIM[1], float(np.nanmax(all_prob)) + 0.05),
    )
    _blue_boxplot(
        prob_data,
        positions,
        x_labels,
        PROBABILITY_YLABEL + " (per pathway neuron)",
        "Probability metric vs fully connected neuron selection\n"
        "(each box = one threshold; 0 FC = sparse decoder, max FC = dense decoder)",
        os.path.join(aggregate_dir, "02_reduction_score_probability_by_fcn.png"),
        ylim=ylim_prob,
        reference_y=0.5,
    )

    plot_combined_metrics_by_fcn(combined, sweep_root, aggregate_dir)


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
        "--unrescale-csvs",
        action="store_true",
        help="Revert 2*(p-0.5) on reduction_score_probability in sweep CSVs, then exit.",
    )
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
    if not (args.train or args.eval or args.plot or args.unrescale_csvs):
        print("Nothing to do. Pass --train, --eval, --plot, and/or --unrescale-csvs.")
        return

    if args.unrescale_csvs:
        unrescale_sweep_probability_csvs(args.sweep_root)
        print("Done (unrescale-csvs).")
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
        plot_sweep_metrics(combined, aggregate_dir, sweep_root=args.sweep_root)

    param_path = os.path.join(aggregate_dir, "run_parameters.txt")
    os.makedirs(aggregate_dir, exist_ok=True)
    write_run_parameters(param_path, args, hyperparams, combined)
    print("\nDone. Sweep root: %s" % os.path.abspath(args.sweep_root))


if __name__ == "__main__":
    main()
