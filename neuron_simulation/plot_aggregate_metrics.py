"""
Combine neuron-simulation sweeps into aggregate plots (3 mean curves per figure).

Beta sweeps (base overlap):
  distance_corr_beta_sweep.csv, probabilities_beta_sweep.csv, mig_beta_sweep.csv

Overlap sweeps (low / medium / high regimes, fixed betas 0–0.9):
  distance_corr_overlap_sweep.csv, probabilities_overlap_sweep.csv, mig_overlap_sweep.csv
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_DCORR_DIR = "plots_dcorr_v3_exp"
DEFAULT_PROB_DIR = "plots_trial_v7_exp"
DEFAULT_MIG_DIR = "plots_mig_v1_exp"
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "aggregate")

LEGEND_DC = "DC"
LEGEND_KOS = "KOS"
LEGEND_MIG = "MIG"

# Match vega_fcn_sweep/aggregate/03_combined_metrics_shared_yscale.png (14×6 in @ 150 dpi → 2100×900 px).
COMBINED_SHARED_FIGSIZE = (14, 6)
COMBINED_SHARED_DPI = 150

# Fixed simulation dimensions (matches test_neuron_activation sweeps).
DEFAULT_N_EXAMPLES = 1000
DEFAULT_M_VARIABLES = 2000
DEFAULT_K_CONCEPTS = 300

OVERLAP_REGIME_ORDER = ["low_overlap", "medium_overlap", "high_overlap"]
OVERLAP_REGIME_LABELS = {
    "low_overlap": "Low overlap",
    "medium_overlap": "Medium overlap",
    "high_overlap": "High overlap",
}

def _add_y_reference_lines(ax) -> None:
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.5, zorder=2)
    ax.text(
        1.01,
        0.5,
        "0.5",
        transform=ax.get_yaxis_transform(),
        va="center",
        ha="left",
        color="black",
        fontsize=9,
        clip_on=False,
    )


def _mig_adapted_ylim(mig_means: np.ndarray) -> Tuple[float, float]:
    valid = mig_means[np.isfinite(mig_means)]
    if len(valid) == 0:
        return (0.0, 1.0)
    span = float(np.max(valid) - np.min(valid))
    pad = max(0.15 * span, 0.01)
    return (float(np.min(valid)) - pad, float(np.max(valid)) + pad)


def _means_per_beta(
    df: pd.DataFrame,
    value_col: str,
    betas: List[float],
    *,
    overlap_profile: Optional[str] = "base",
) -> np.ndarray:
    sub = df
    if overlap_profile is not None and "overlap_profile" in sub.columns:
        sub = sub.loc[sub["overlap_profile"] == overlap_profile]
    return np.array(
        [
            float(np.nanmean(sub.loc[sub["beta"] == b, value_col].dropna().values))
            for b in betas
        ],
        dtype=float,
    )


def _means_per_regime_beta_grid(
    df: pd.DataFrame,
    value_col: str,
    regimes: List[str],
    betas: List[float],
) -> np.ndarray:
    """Flat array: for each regime, means at each beta (regime-major order)."""
    return np.array(
        [
            float(
                np.nanmean(
                    df.loc[
                        (df["overlap_profile"] == regime) & (df["beta"] == beta),
                        value_col,
                    ]
                    .dropna()
                    .values
                )
            )
            for regime in regimes
            for beta in betas
        ],
        dtype=float,
    )


def _overlap_grouped_x_layout(
    regimes: List[str],
    betas: List[float],
) -> Tuple[List[int], List[str], List[float], List[float]]:
    """Positions, beta tick labels, regime label centers, regime separator x."""
    positions: List[int] = []
    x_labels: List[str] = []
    regime_centers: List[float] = []
    separators: List[float] = []
    pos = 0
    for regime_idx, _regime in enumerate(regimes):
        start = pos
        for beta in betas:
            positions.append(pos)
            x_labels.append("%.1f" % beta)
            pos += 1
        regime_centers.append((start + pos - 1) / 2.0)
        if regime_idx < len(regimes) - 1:
            separators.append(pos - 0.5)
    return positions, x_labels, regime_centers, separators


def _mig_column(df: pd.DataFrame) -> str:
    # per_concept_mig: mean over factors (matches 01_mig_by_beta boxplot rows)
    # mig_dataset: paper Eq. (6) dataset MIG (constant within each beta)
    for col in ("per_concept_mig", "per_neuron_mig", "mig_dataset"):
        if col in df.columns:
            return col
    raise ValueError(
        "MIG CSV must contain per_concept_mig, per_neuron_mig, or mig_dataset"
    )


def load_beta_means(
    dcorr_csv: str,
    prob_csv: str,
    mig_csv: str,
    *,
    overlap_profile: str = "base",
) -> Tuple[List[float], np.ndarray, np.ndarray, np.ndarray, str]:
    dcorr_df = pd.read_csv(dcorr_csv)
    prob_df = pd.read_csv(prob_csv)
    mig_df = pd.read_csv(mig_csv)
    mig_col = _mig_column(mig_df)

    betas = sorted(
        set(dcorr_df["beta"].unique())
        & set(prob_df["beta"].unique())
        & set(mig_df["beta"].unique())
    )
    if not betas:
        raise ValueError("No common beta values across the three CSV files.")

    dist_means = _means_per_beta(
        dcorr_df, "distance_corr", betas, overlap_profile=overlap_profile
    )
    prob_means = _means_per_beta(
        prob_df, "probability", betas, overlap_profile=overlap_profile
    )
    mig_means = _means_per_beta(
        mig_df, mig_col, betas, overlap_profile=overlap_profile
    )
    return betas, dist_means, prob_means, mig_means, mig_col


def _title_suffix(
    n_examples: Optional[int] = None,
    n_variables: Optional[int] = None,
    n_concepts: Optional[int] = None,
    *,
    context: Optional[str] = None,
) -> str:
    parts: List[str] = []
    if n_examples is not None:
        parts.append("N=%d" % n_examples)
    if n_variables is not None:
        parts.append("M=%d" % n_variables)
    if n_concepts is not None:
        parts.append("K=%d" % n_concepts)
    if context:
        parts.append(context)
    if not parts:
        return ""
    return " (%s)" % ", ".join(parts)


def _plot_three_mean_curves(ax, positions, dist_means, prob_means, mig_means) -> None:
    line_kw = dict(linewidth=2, markersize=6, zorder=3)
    ax.plot(
        positions,
        dist_means,
        color="steelblue",
        marker="o",
        label=LEGEND_DC,
        **line_kw,
    )
    ax.plot(
        positions,
        prob_means,
        color="darkorange",
        marker="s",
        label=LEGEND_KOS,
        **line_kw,
    )
    ax.plot(
        positions,
        mig_means,
        color="red",
        marker="^",
        label=LEGEND_MIG,
        **line_kw,
    )


def _plot_overlap_regime_segments(
    ax,
    positions: List[int],
    values: np.ndarray,
    regimes: List[str],
    betas: List[float],
    *,
    color: str,
    marker: str,
    label: str,
) -> None:
    """One line segment per overlap regime (no lines across regimes)."""
    line_kw = dict(linewidth=2, markersize=6, zorder=3, color=color)
    n_beta = len(betas)
    for r_idx in range(len(regimes)):
        start = r_idx * n_beta
        end = start + n_beta
        ax.plot(
            positions[start:end],
            values[start:end],
            marker=marker,
            label=label if r_idx == 0 else "_nolegend_",
            **line_kw,
        )


def plot_combined_metrics_shared(
    x_values: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    xlabel: str = "beta",
    title_line1: str = "Neuron simulation: interpretability metrics vs beta",
    title_line2: str = "",
) -> None:
    """Single y-axis 0–1 (all three metrics on the same scale)."""
    positions = list(range(len(x_values)))
    x_labels = ["%.1f" % b for b in x_values]

    fig, ax = plt.subplots(figsize=COMBINED_SHARED_FIGSIZE)
    _plot_three_mean_curves(ax, positions, dist_means, prob_means, mig_means)
    ax.set_xticks(positions, x_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    _add_y_reference_lines(ax)
    ax.set_title(title_line1 if not title_line2 else "%s\n%s" % (title_line1, title_line2))
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.88)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=COMBINED_SHARED_DPI)
    plt.close(fig)
    print("Saved plot: %s" % out_path)


def plot_combined_metrics_dual(
    x_values: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    xlabel: str = "beta",
    title_line1: str = "Neuron simulation: interpretability metrics vs beta",
    title_line2: str = "(dual y-axis)",
) -> None:
    positions = list(range(len(x_values)))
    x_labels = ["%.1f" % b for b in x_values]
    figsize = (max(12, 1.0 * len(positions)), 6)

    fig, ax_left = plt.subplots(figsize=figsize)
    line_kw = dict(linewidth=2, markersize=6, zorder=3)
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
        linewidth=2,
        markersize=6,
        zorder=3,
    )
    ax_right.set_ylim(_mig_adapted_ylim(mig_means))
    ax_right.set_ylabel(LEGEND_MIG, color="red")
    ax_right.tick_params(axis="y", labelcolor="red")

    ax_left.set_xticks(positions, x_labels)
    ax_left.set_xlabel(xlabel)
    ax_left.set_ylabel(
        "Distance correlation (mean) / Reduction probability score (mean)"
    )
    ax_left.set_title("%s\n%s" % (title_line1, title_line2))
    _add_y_reference_lines(ax_left)
    ax_left.grid(True, axis="y", alpha=0.3)

    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")

    fig.subplots_adjust(right=0.88)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved plot: %s" % out_path)


def plot_combined_metrics_by_beta_shared(
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    n_examples: Optional[int] = None,
    n_variables: Optional[int] = None,
    n_concepts: Optional[int] = None,
) -> None:
    plot_combined_metrics_shared(
        betas,
        dist_means,
        prob_means,
        mig_means,
        out_path,
        title_line2=_title_suffix(
            n_examples, n_variables, n_concepts, context="base overlap"
        ),
    )


def plot_combined_metrics_by_beta_dual(
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    n_examples: Optional[int] = None,
    n_variables: Optional[int] = None,
    n_concepts: Optional[int] = None,
) -> None:
    suffix = _title_suffix(n_examples, n_variables, n_concepts, context="base overlap")
    plot_combined_metrics_dual(
        betas,
        dist_means,
        prob_means,
        mig_means,
        out_path,
        title_line2="%s\n(dual y-axis)" % suffix.strip() if suffix else "(dual y-axis)",
    )


def write_means_csv(
    out_csv: str,
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    mig_col: str,
    *,
    overlap_profile: Optional[str] = None,
) -> None:
    data = {
        "beta": betas,
        "distance_corr_mean": dist_means,
        "probability_mean": prob_means,
        "mig_mean": mig_means,
        "mig_source_column": mig_col,
    }
    if overlap_profile is not None:
        data["overlap_profile"] = overlap_profile
    pd.DataFrame(data).to_csv(out_csv, index=False)
    print("Saved CSV: %s" % out_csv)


def load_overlap_sweep_grid(
    dcorr_csv: str,
    prob_csv: str,
    mig_csv: str,
) -> Tuple[List[float], List[str], np.ndarray, np.ndarray, np.ndarray, str]:
    dcorr_df = pd.read_csv(dcorr_csv)
    prob_df = pd.read_csv(prob_csv)
    mig_df = pd.read_csv(mig_csv)
    mig_col = _mig_column(mig_df)

    regimes = [r for r in OVERLAP_REGIME_ORDER if r in dcorr_df["overlap_profile"].unique()]
    betas = sorted(
        set(dcorr_df["beta"].unique())
        & set(prob_df["beta"].unique())
        & set(mig_df["beta"].unique())
    )
    if not regimes or not betas:
        raise ValueError("No overlap regimes or betas in overlap sweep CSVs.")

    dist_means = _means_per_regime_beta_grid(dcorr_df, "distance_corr", regimes, betas)
    prob_means = _means_per_regime_beta_grid(prob_df, "probability", regimes, betas)
    mig_means = _means_per_regime_beta_grid(mig_df, mig_col, regimes, betas)
    return betas, regimes, dist_means, prob_means, mig_means, mig_col


def _decorate_overlap_xaxis(
    ax,
    positions: List[int],
    x_labels: List[str],
    regime_centers: List[float],
    separators: List[float],
    regimes: List[str],
) -> None:
    ax.set_xticks(positions, x_labels)
    ax.set_xlabel("beta (within each overlap regime)")
    for sep in separators:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    for center, regime in zip(regime_centers, regimes):
        ax.text(
            center,
            -0.14,
            OVERLAP_REGIME_LABELS.get(regime, regime),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=10,
        )


def plot_combined_metrics_by_overlap_dual(
    regimes: List[str],
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    n_examples: Optional[int] = None,
    n_variables: Optional[int] = None,
    n_concepts: Optional[int] = None,
) -> None:
    positions, x_labels, regime_centers, separators = _overlap_grouped_x_layout(
        regimes, betas
    )
    figsize = (max(14, 1.2 * len(positions)), 6)
    dim_suffix = _title_suffix(n_examples, n_variables, n_concepts)

    fig, ax_left = plt.subplots(figsize=figsize)
    _plot_overlap_regime_segments(
        ax_left, positions, dist_means, regimes, betas,
        color="steelblue", marker="o", label=LEGEND_DC,
    )
    _plot_overlap_regime_segments(
        ax_left, positions, prob_means, regimes, betas,
        color="darkorange", marker="s", label=LEGEND_KOS,
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
    _plot_overlap_regime_segments(
        ax_right, positions, mig_means, regimes, betas,
        color="red", marker="^", label=LEGEND_MIG,
    )
    ax_right.set_ylim(_mig_adapted_ylim(mig_means))
    ax_right.set_ylabel(LEGEND_MIG, color="red")
    ax_right.tick_params(axis="y", labelcolor="red")

    _decorate_overlap_xaxis(
        ax_left, positions, x_labels, regime_centers, separators, regimes
    )
    ax_left.set_ylabel("%s / %s" % (LEGEND_DC, LEGEND_KOS))
    ax_left.set_title(
        "Neuron simulation: interpretability metrics vs overlap regime%s\n"
        "(dual y-axis)" % dim_suffix
    )
    _add_y_reference_lines(ax_left)
    ax_left.grid(True, axis="y", alpha=0.3)
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    fig.subplots_adjust(right=0.88, bottom=0.18)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved plot: %s" % out_path)


def plot_combined_metrics_by_overlap_shared(
    regimes: List[str],
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_path: str,
    *,
    n_examples: Optional[int] = None,
    n_variables: Optional[int] = None,
    n_concepts: Optional[int] = None,
) -> None:
    positions, x_labels, regime_centers, separators = _overlap_grouped_x_layout(
        regimes, betas
    )
    figsize = (max(14, 1.2 * len(positions)), 6)
    dim_suffix = _title_suffix(n_examples, n_variables, n_concepts)

    fig, ax = plt.subplots(figsize=figsize)
    _plot_overlap_regime_segments(
        ax, positions, dist_means, regimes, betas,
        color="steelblue", marker="o", label=LEGEND_DC,
    )
    _plot_overlap_regime_segments(
        ax, positions, prob_means, regimes, betas,
        color="darkorange", marker="s", label=LEGEND_KOS,
    )
    _plot_overlap_regime_segments(
        ax, positions, mig_means, regimes, betas,
        color="red", marker="^", label=LEGEND_MIG,
    )
    _decorate_overlap_xaxis(
        ax, positions, x_labels, regime_centers, separators, regimes
    )
    ax.set_ylim(0.0, 1.0)
    _add_y_reference_lines(ax)
    ax.set_ylabel("Metric value")
    ax.set_title(
        "Neuron simulation: interpretability metrics vs overlap regime%s"
        % dim_suffix
    )
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(right=0.88, bottom=0.18)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved plot: %s" % out_path)


def plot_overlap_aggregate(
    dcorr_csv: str,
    prob_csv: str,
    mig_csv: str,
    out_dir: str,
    *,
    n_examples: Optional[int],
    n_variables: Optional[int],
    n_concepts: Optional[int],
    mig_col: str,
) -> pd.DataFrame:
    """Single figure: x = 3 overlap regimes (grouped), 3 metric curves over beta means."""
    betas, regimes, dist_means, prob_means, mig_means, _ = load_overlap_sweep_grid(
        dcorr_csv, prob_csv, mig_csv
    )
    plot_combined_metrics_by_overlap_dual(
        regimes,
        betas,
        dist_means,
        prob_means,
        mig_means,
        os.path.join(out_dir, "03_combined_metrics_by_overlap_dual.png"),
        n_examples=n_examples,
        n_variables=n_variables,
        n_concepts=n_concepts,
    )
    plot_combined_metrics_by_overlap_shared(
        regimes,
        betas,
        dist_means,
        prob_means,
        mig_means,
        os.path.join(out_dir, "04_combined_metrics_by_overlap_shared_yscale.png"),
        n_examples=n_examples,
        n_variables=n_variables,
        n_concepts=n_concepts,
    )

    rows = []
    for regime in regimes:
        for i, beta in enumerate(betas):
            idx = regimes.index(regime) * len(betas) + i
            rows.append(
                {
                    "overlap_profile": regime,
                    "beta": beta,
                    "distance_corr_mean": dist_means[idx],
                    "probability_mean": prob_means[idx],
                    "mig_mean": mig_means[idx],
                    "mig_source_column": mig_col,
                }
            )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Combined neuron-simulation metrics plots (beta + overlap sweeps)."
    )
    p.add_argument("--dcorr-dir", default=DEFAULT_DCORR_DIR)
    p.add_argument("--prob-dir", default=DEFAULT_PROB_DIR)
    p.add_argument("--mig-dir", default=DEFAULT_MIG_DIR)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--overlap-profile",
        default="base",
        help="Filter rows to this overlap profile (default: base).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dcorr_csv = os.path.join(args.dcorr_dir, "distance_corr_beta_sweep.csv")
    prob_csv = os.path.join(args.prob_dir, "probabilities_beta_sweep.csv")
    mig_csv = os.path.join(args.mig_dir, "mig_beta_sweep.csv")

    for path in (dcorr_csv, prob_csv, mig_csv):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    os.makedirs(args.out_dir, exist_ok=True)
    betas, dist_means, prob_means, mig_means, mig_col = load_beta_means(
        dcorr_csv,
        prob_csv,
        mig_csv,
        overlap_profile=args.overlap_profile,
    )
    mig_df = pd.read_csv(mig_csv)
    n_variables = (
        int(mig_df["n_variables"].iloc[0])
        if "n_variables" in mig_df.columns
        else DEFAULT_M_VARIABLES
    )
    n_concepts = (
        int(mig_df["n_concepts"].iloc[0])
        if "n_concepts" in mig_df.columns
        else DEFAULT_K_CONCEPTS
    )
    n_examples = DEFAULT_N_EXAMPLES

    plot_combined_metrics_by_beta_dual(
        betas,
        dist_means,
        prob_means,
        mig_means,
        os.path.join(args.out_dir, "01_combined_metrics_by_beta_dual.png"),
        n_examples=n_examples,
        n_variables=n_variables,
        n_concepts=n_concepts,
    )
    plot_combined_metrics_by_beta_shared(
        betas,
        dist_means,
        prob_means,
        mig_means,
        os.path.join(args.out_dir, "02_combined_metrics_by_beta_shared_yscale.png"),
        n_examples=n_examples,
        n_variables=n_variables,
        n_concepts=n_concepts,
    )

    out_csv = os.path.join(args.out_dir, "combined_metrics_beta_means.csv")
    write_means_csv(out_csv, betas, dist_means, prob_means, mig_means, mig_col)

    dcorr_overlap = os.path.join(args.dcorr_dir, "distance_corr_overlap_sweep.csv")
    prob_overlap = os.path.join(args.prob_dir, "probabilities_overlap_sweep.csv")
    mig_overlap = os.path.join(args.mig_dir, "mig_overlap_sweep.csv")
    overlap_rows = None
    if all(os.path.isfile(p) for p in (dcorr_overlap, prob_overlap, mig_overlap)):
        overlap_rows = plot_overlap_aggregate(
            dcorr_overlap,
            prob_overlap,
            mig_overlap,
            args.out_dir,
            n_examples=n_examples,
            n_variables=n_variables,
            n_concepts=n_concepts,
            mig_col=mig_col,
        )
        overlap_csv = os.path.join(args.out_dir, "combined_metrics_overlap_means.csv")
        overlap_rows.to_csv(overlap_csv, index=False)
        print("Saved CSV: %s" % overlap_csv)
    else:
        print(
            "[plot] Overlap sweep CSVs missing — skipping overlap combined plots."
        )

    param_path = os.path.join(args.out_dir, "run_parameters.txt")
    with open(param_path, "w", encoding="utf-8") as f:
        f.write("Neuron simulation — combined metrics aggregate\n")
        f.write("Generated (UTC): %s\n\n" % datetime.now(timezone.utc).isoformat())
        f.write("distance_corr_csv (beta): %s\n" % os.path.abspath(dcorr_csv))
        f.write("probability_csv (beta): %s\n" % os.path.abspath(prob_csv))
        f.write("mig_csv (beta): %s\n" % os.path.abspath(mig_csv))
        f.write("mig_column: %s\n" % mig_col)
        f.write("overlap_profile (beta sweep): %s\n" % args.overlap_profile)
        f.write("beta_values (beta sweep): %s\n" % ",".join("%.1f" % b for b in betas))
        f.write("\nOverlap sweep outputs (per regime: low, medium, high):\n")
        f.write("  distance_corr: %s\n" % os.path.abspath(dcorr_overlap))
        f.write("  probability: %s\n" % os.path.abspath(prob_overlap))
        f.write("  mig: %s\n" % os.path.abspath(mig_overlap))
        if overlap_rows is not None:
            f.write(
                "  overlap_regimes: %s\n"
                % ",".join(OVERLAP_REGIME_ORDER)
            )
            f.write(
                "  overlap_betas: %s\n"
                % ",".join(
                    "%.1f"
                    % b
                    for b in sorted(overlap_rows["beta"].unique())
                )
            )
    print("Wrote %s" % param_path)


if __name__ == "__main__":
    main()
