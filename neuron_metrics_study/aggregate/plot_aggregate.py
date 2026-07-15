"""
Aggregate combined metric plots for neuron_metrics_study.

- One combined beta plot per overlap regime (title shows mean overlap).
- One overlap-regime figure with all betas 0–1 (shared y-scale + dual y-scale).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO = os.path.dirname(_STUDY)
if _STUDY not in sys.path:
    sys.path.insert(0, _STUDY)

from shared.config import (
    BETA_SWEEP,
    COMBINED_AXIS_LABEL_FONTSIZE,
    COMBINED_LEGEND_FONTSIZE,
    COMBINED_REGIME_LABEL_FONTSIZE,
    COMBINED_SHARED_DPI,
    COMBINED_SHARED_FIGSIZE,
    OVERLAP_SHARED_FIGSIZE,
    BETA_SHARED_FIGSIZE,
    DEFAULT_VEGA_SWEEP_ROOT,
    MERGED_NEURON_VEGA_FIGSIZE,
    MERGED_VEGA_SECTION_LABEL,
    MERGED_VEGA_SUBTITLE,
    MERGED_NEURON_PANEL_TITLES,
    MERGED_PANEL_LETTERS,
    MERGED_VEGA_XLABEL,
    MERGED_AXIS_LABEL_FONTSIZE,
    MERGED_TICK_LABEL_FONTSIZE,
    MERGED_TITLE_FONTSIZE,
    MERGED_REGIME_LABEL_FONTSIZE,
    MERGED_PARAM_LABEL_FONTSIZE,
    MERGED_LEGEND_FONTSIZE,
    MERGED_LEGEND_HANDLELENGTH,
    MERGED_LEGEND_SYMBOL_SCALE,
    MERGED_LINE_WIDTH,
    MERGED_MARKER_SIZE,
    MERGED_REF_LINE_WIDTH,
    MERGED_FONT_SCALE,
    MERGED_SIZE_SCALE,
    MERGED_XLABEL_Y,
    MERGED_SECTION_LABEL_Y,
    MERGED_PANEL_LETTER_Y,
    MERGED_XLIM_PAD_LEFT,
    MERGED_XLIM_PAD_RIGHT,
    MERGED_SUBPLOT_MARGINS,
    OVERLAP_SHARED_SUBPLOT_MARGINS,
    OVERLAP_SHARED_TITLE_PAD,
    COMBINED_TICK_LABEL_FONTSIZE,
    COMBINED_TITLE_FONTSIZE,
    FIXED_K,
    FIXED_M,
    FIXED_N_EXAMPLES,
    LEGEND_DC,
    LEGEND_KOS,
    LEGEND_MIG,
    OVERLAP_REGIME_LABELS,
    OVERLAP_REGIME_ORDER,
    OVERLAP_SHARED_AXIS_LABEL_FONTSIZE,
    OVERLAP_SHARED_LEGEND_FONTSIZE,
    OVERLAP_SHARED_LINE_WIDTH,
    OVERLAP_SHARED_MARKER_SIZE,
    OVERLAP_SHARED_REGIME_LABEL_FONTSIZE,
    OVERLAP_SHARED_REGIME_LABEL_Y,
    OVERLAP_SHARED_XLABEL_Y,
    OVERLAP_SHARED_XLIM_PAD_LEFT,
    OVERLAP_SHARED_XLIM_PAD_RIGHT,
    OVERLAP_SHARED_FIG04_TITLE,
    OVERLAP_SHARED_TICK_LABEL_FONTSIZE,
    OVERLAP_SHARED_TITLE_FONTSIZE,
)


from shared.paths import (
    add_run_name_arg,
    aggregate_output_dir,
    apply_run_name_from_args,
    metric_output_dir,
    run_root,
)


def _mig_column(df: pd.DataFrame) -> str:
    for col in ("per_concept_mig", "per_neuron_mig", "mig"):
        if col in df.columns:
            return col
    raise ValueError("No MIG column in dataframe")


def _add_y_reference_lines(ax, *, ref_fontsize: Optional[int] = None) -> None:
    fs = ref_fontsize if ref_fontsize is not None else COMBINED_TICK_LABEL_FONTSIZE
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.5, zorder=2)
    ax.text(
        1.01, 0.5, "0.5",
        transform=ax.get_yaxis_transform(),
        va="center", ha="left", color="black", fontsize=fs, clip_on=False,
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


def _style_combined_axes(
    ax,
    *,
    legend_loc: str = "best",
    axis_label_fontsize: Optional[int] = None,
    tick_label_fontsize: Optional[int] = None,
    legend_fontsize: Optional[int] = None,
    title_fontsize: Optional[int] = None,
) -> None:
    axis_fs = axis_label_fontsize if axis_label_fontsize is not None else COMBINED_AXIS_LABEL_FONTSIZE
    tick_fs = tick_label_fontsize if tick_label_fontsize is not None else COMBINED_TICK_LABEL_FONTSIZE
    legend_fs = legend_fontsize if legend_fontsize is not None else COMBINED_LEGEND_FONTSIZE
    title_fs = title_fontsize if title_fontsize is not None else COMBINED_TITLE_FONTSIZE
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.xaxis.label.set_fontsize(axis_fs)
    ax.yaxis.label.set_fontsize(axis_fs)
    title = ax.get_title()
    if title:
        ax.set_title(title, fontsize=title_fs)
    leg = ax.get_legend()
    if leg is not None:
        for text in leg.get_texts():
            text.set_fontsize(legend_fs)
    elif legend_loc:
        ax.legend(loc=legend_loc, fontsize=legend_fs)


def _mig_adapted_ylim(mig_means: np.ndarray) -> Tuple[float, float]:
    valid = mig_means[np.isfinite(mig_means)]
    if len(valid) == 0:
        return (0.0, 1.0)
    span = float(np.max(valid) - np.min(valid))
    pad = max(0.15 * span, 0.01)
    return (float(np.min(valid)) - pad, float(np.max(valid)) + pad)


def _means_per_beta(
    df: pd.DataFrame, value_col: str, betas: List[float], overlap_profile: str
) -> np.ndarray:
    sub = df.loc[df["overlap_profile"] == overlap_profile]
    return np.array(
        [
            float(np.nanmean(sub.loc[sub["beta"] == b, value_col].values))
            if len(sub.loc[sub["beta"] == b, value_col].dropna())
            else float("nan")
            for b in betas
        ],
        dtype=float,
    )


def _means_per_regime_beta_grid(
    df: pd.DataFrame, value_col: str, regimes: List[str], betas: List[float]
) -> np.ndarray:
    out = []
    for regime in regimes:
        out.extend(_means_per_beta(df, value_col, betas, regime))
    return np.array(out, dtype=float)


def _overlap_grouped_x_layout(regimes: List[str], betas: List[float]):
    positions = list(range(len(regimes) * len(betas)))
    x_labels = []
    regime_centers = []
    separators = []
    n_beta = len(betas)
    for r_idx, regime in enumerate(regimes):
        start = r_idx * n_beta
        for b in betas:
            x_labels.append("%.1f" % b)
        regime_centers.append(start + (n_beta - 1) / 2.0)
        if r_idx < len(regimes) - 1:
            separators.append(start + n_beta - 0.5)
    return positions, x_labels, regime_centers, separators


def _plot_three_mean_curves(ax, positions, dist_means, prob_means, mig_means) -> None:
    kw = dict(linewidth=2, markersize=6, zorder=3)
    ax.plot(positions, dist_means, color="steelblue", marker="o", label=LEGEND_DC, **kw)
    ax.plot(positions, prob_means, color="darkorange", marker="s", label=LEGEND_KOS, **kw)
    ax.plot(positions, mig_means, color="red", marker="^", label=LEGEND_MIG, **kw)


def _plot_overlap_regime_segments(
    ax, positions, values, regimes, betas, *, color, marker, label, linewidth=2, markersize=6
):
    n_beta = len(betas)
    kw = dict(linewidth=linewidth, markersize=markersize, zorder=3, color=color)
    for r_idx in range(len(regimes)):
        start = r_idx * n_beta
        end = start + n_beta
        ax.plot(
            positions[start:end], values[start:end], marker=marker,
            label=label if r_idx == 0 else "_nolegend_", **kw,
        )


def _decorate_overlap_xaxis(
    ax, positions, x_labels, regime_centers, separators, regimes, *,
    regime_label_fontsize=None, regime_label_y=-0.16, show_beta_xlabel=True,
):
    fs = regime_label_fontsize if regime_label_fontsize is not None else COMBINED_REGIME_LABEL_FONTSIZE
    ax.set_xticks(positions, x_labels)
    if show_beta_xlabel:
        ax.set_xlabel("beta")
    for sep in separators:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    for center, regime in zip(regime_centers, regimes):
        item = cached_stats.get(regime, {})
        mean_o = item.get("mean_overlap", float("nan"))
        med_o = item.get("median_overlap", float("nan"))
        label = OVERLAP_REGIME_LABELS.get(regime, regime)
        ax.text(
            center, regime_label_y,
            "%s\nmean=%.3f, med=%.3f" % (label, mean_o, med_o),
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=fs,
        )


def _title_dims() -> str:
    return "N=%d, M=%d, K=%d" % (FIXED_N_EXAMPLES, FIXED_M, FIXED_K)


def plot_combined_by_regime_beta(
    regime: str,
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    mean_overlap: float,
    median_overlap: float,
    out_dir: str,
) -> None:
    positions = list(range(len(betas)))
    x_labels = ["%.1f" % b for b in betas]
    title_base = (
        "Neuron simulation: interpretability metrics vs beta\n"
        "%s — mean overlap=%.3f, median overlap=%.3f"
        % (_title_dims(), mean_overlap, median_overlap)
    )

    # Shared y-scale
    fig, ax = plt.subplots(figsize=BETA_SHARED_FIGSIZE)
    _plot_three_mean_curves(ax, positions, dist_means, prob_means, mig_means)
    ax.set_xticks(positions, x_labels)
    ax.set_xlabel("beta")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0.0, 1.0)
    _add_y_reference_lines(ax)
    ax.set_title(title_base)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    shared_path = os.path.join(
        out_dir, "02_combined_metrics_%s_beta_shared_yscale.png" % regime
    )
    fig.savefig(shared_path, dpi=COMBINED_SHARED_DPI)
    plt.close(fig)
    print("Saved: %s" % shared_path)

    # Dual y-scale
    fig, ax_left = plt.subplots(figsize=(max(12, len(betas)), 6))
    kw = dict(linewidth=2, markersize=6, zorder=3)
    ax_left.plot(positions, dist_means, color="steelblue", marker="o", label=LEGEND_DC, **kw)
    ax_left.plot(positions, prob_means, color="darkorange", marker="s", label=LEGEND_KOS, **kw)
    left_vals = np.concatenate([
        dist_means[np.isfinite(dist_means)], prob_means[np.isfinite(prob_means)]
    ])
    ax_left.set_ylim(
        max(0.0, float(np.nanmin(left_vals)) - 0.05) if len(left_vals) else 0.0,
        min(1.05, float(np.nanmax(left_vals)) + 0.05) if len(left_vals) else 1.0,
    )
    ax_right = ax_left.twinx()
    ax_right.plot(positions, mig_means, color="red", marker="^", label=LEGEND_MIG, **kw)
    ax_right.set_ylim(_mig_adapted_ylim(mig_means))
    ax_right.set_ylabel(LEGEND_MIG, color="red")
    ax_right.tick_params(axis="y", labelcolor="red")
    ax_left.set_xticks(positions, x_labels)
    ax_left.set_xlabel("beta")
    ax_left.set_ylabel("%s / %s" % (LEGEND_DC, LEGEND_KOS))
    ax_left.set_title(title_base + "\n(dual y-axis)")
    _add_y_reference_lines(ax_left)
    ax_left.grid(True, axis="y", alpha=0.3)
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    fig.tight_layout()
    dual_path = os.path.join(out_dir, "01_combined_metrics_%s_beta_dual.png" % regime)
    fig.savefig(dual_path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % dual_path)


def plot_combined_by_overlap_shared(
    regimes: List[str],
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_dir: str,
) -> None:
    positions, x_labels, regime_centers, separators = _overlap_grouped_x_layout(regimes, betas)
    fig, ax = plt.subplots(figsize=OVERLAP_SHARED_FIGSIZE)
    _plot_overlap_regime_segments(
        ax, positions, dist_means, regimes, betas,
        color="steelblue", marker="o", label=LEGEND_DC,
        linewidth=OVERLAP_SHARED_LINE_WIDTH, markersize=OVERLAP_SHARED_MARKER_SIZE,
    )
    _plot_overlap_regime_segments(
        ax, positions, prob_means, regimes, betas,
        color="darkorange", marker="s", label=LEGEND_KOS,
        linewidth=OVERLAP_SHARED_LINE_WIDTH, markersize=OVERLAP_SHARED_MARKER_SIZE,
    )
    _plot_overlap_regime_segments(
        ax, positions, mig_means, regimes, betas,
        color="red", marker="^", label=LEGEND_MIG,
        linewidth=OVERLAP_SHARED_LINE_WIDTH, markersize=OVERLAP_SHARED_MARKER_SIZE,
    )
    _decorate_overlap_xaxis(
        ax, positions, x_labels, regime_centers, separators, regimes,
        regime_label_fontsize=OVERLAP_SHARED_REGIME_LABEL_FONTSIZE,
        regime_label_y=OVERLAP_SHARED_REGIME_LABEL_Y,
        show_beta_xlabel=False,
    )
    for center in regime_centers:
        ax.text(
            center, OVERLAP_SHARED_XLABEL_Y, "beta",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=OVERLAP_SHARED_AXIS_LABEL_FONTSIZE,
        )
    ax.set_xlim(-OVERLAP_SHARED_XLIM_PAD_LEFT, positions[-1] + OVERLAP_SHARED_XLIM_PAD_RIGHT)
    ax.margins(x=0)
    ax.set_ylim(0.0, 1.0)
    _add_y_reference_lines(ax, ref_fontsize=OVERLAP_SHARED_TICK_LABEL_FONTSIZE)
    ax.set_ylabel("Metric value", fontsize=OVERLAP_SHARED_AXIS_LABEL_FONTSIZE)
    ax.set_title(OVERLAP_SHARED_FIG04_TITLE, pad=OVERLAP_SHARED_TITLE_PAD)
    ax.legend(loc="upper right")
    _style_combined_axes(
        ax,
        legend_loc="",
        axis_label_fontsize=OVERLAP_SHARED_AXIS_LABEL_FONTSIZE,
        tick_label_fontsize=OVERLAP_SHARED_TICK_LABEL_FONTSIZE,
        title_fontsize=OVERLAP_SHARED_TITLE_FONTSIZE,
    )
    _style_compact_legend(ax, OVERLAP_SHARED_LEGEND_FONTSIZE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(**OVERLAP_SHARED_SUBPLOT_MARGINS)
    path = os.path.join(out_dir, "04_combined_metrics_by_overlap_shared_yscale.png")
    fig.savefig(path, dpi=COMBINED_SHARED_DPI)
    plt.close(fig)
    print("Saved: %s" % path)


def _load_vega_fc_means(sweep_root: str):
    if _REPO not in sys.path:
        sys.path.insert(0, _REPO)
    sys.path.insert(0, os.path.join(_REPO, "test_vega_simulation"))
    from run_vega_fcn_sweep import (
        _fcn_groups_layout,
        _load_mig_means_by_fc,
        _means_by_fc_count,
        combined_plot_xlabel,
    )

    combined_path = os.path.join(sweep_root, "aggregate", "per_neuron_metrics.csv")
    combined = pd.read_csv(combined_path)
    group_col, groups, _x_labels, positions, fc_counts = _fcn_groups_layout(combined)
    x_labels = [
        combined_plot_xlabel(int(r.n_fc_neurons), int(r.n_pathway_nodes), int(r.fcn_percent))
        for _, r in groups.iterrows()
    ]
    dist_means = _means_by_fc_count(combined, group_col, "distance_corr", fc_counts)
    prob_means = _means_by_fc_count(combined, group_col, "reduction_score_probability", fc_counts)
    mig_means = _load_mig_means_by_fc(sweep_root, fc_counts)
    if mig_means is None:
        raise FileNotFoundError("Missing MIG aggregate under %s/mig_aggregate/" % sweep_root)
    return x_labels, dist_means, prob_means, mig_means


def plot_combined_neuron_vega_merged(
    regimes: List[str],
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_dir: str,
) -> None:
    """Neuron overlap sweep with VEGA FC section appended after high overlap (fig 05)."""
    n_beta = len(betas)
    n_neuron = len(regimes) * n_beta
    vega_xlabels, dist_v, prob_v, mig_v = _load_vega_fc_means(
        os.path.join(_REPO, DEFAULT_VEGA_SWEEP_ROOT)
    )
    n_vega = len(vega_xlabels)
    vega_start = n_neuron + 1
    vega_positions = list(range(vega_start, vega_start + n_vega))
    vega_center = vega_start + (n_vega - 1) / 2.0

    positions_n, x_labels_n, regime_centers, separators = _overlap_grouped_x_layout(regimes, betas)
    all_positions = positions_n + vega_positions
    all_x_labels = x_labels_n + vega_xlabels

    fig, ax = plt.subplots(figsize=MERGED_NEURON_VEGA_FIGSIZE)
    line_kw = dict(linewidth=MERGED_LINE_WIDTH, markersize=MERGED_MARKER_SIZE, zorder=3)

    _plot_overlap_regime_segments(
        ax, positions_n, dist_means, regimes, betas,
        color="steelblue", marker="o", label=LEGEND_DC,
        linewidth=MERGED_LINE_WIDTH, markersize=MERGED_MARKER_SIZE,
    )
    _plot_overlap_regime_segments(
        ax, positions_n, prob_means, regimes, betas,
        color="darkorange", marker="s", label=LEGEND_KOS,
        linewidth=MERGED_LINE_WIDTH, markersize=MERGED_MARKER_SIZE,
    )
    _plot_overlap_regime_segments(
        ax, positions_n, mig_means, regimes, betas,
        color="red", marker="^", label=LEGEND_MIG,
        linewidth=MERGED_LINE_WIDTH, markersize=MERGED_MARKER_SIZE,
    )
    ax.plot(vega_positions, dist_v, color="steelblue", marker="o", label="_nolegend_", **line_kw)
    ax.plot(vega_positions, prob_v, color="darkorange", marker="s", label="_nolegend_", **line_kw)
    ax.plot(vega_positions, mig_v, color="red", marker="^", label="_nolegend_", **line_kw)

    ax.set_xticks(all_positions, all_x_labels)
    ax.set_xlim(-MERGED_XLIM_PAD_LEFT, all_positions[-1] + MERGED_XLIM_PAD_RIGHT)
    ax.margins(x=0)
    for sep in separators:
        ax.axvline(sep, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)
    ax.axvline(n_neuron + 0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, zorder=1)

    fs = MERGED_REGIME_LABEL_FONTSIZE
    for i, (center, regime) in enumerate(zip(regime_centers, regimes)):
        item = cached_stats.get(regime, {})
        mean_o = item.get("mean_overlap", float("nan"))
        med_o = item.get("median_overlap", float("nan"))
        label = OVERLAP_REGIME_LABELS.get(regime, regime)
        letter = MERGED_PANEL_LETTERS[i]
        ax.text(
            center, MERGED_SECTION_LABEL_Y,
            "%s\nmean=%.3f, med=%.3f" % (label, mean_o, med_o),
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=fs,
        )
        ax.text(
            center, MERGED_PANEL_LETTER_Y, letter,
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=MERGED_TITLE_FONTSIZE,
        )
    ax.text(
        vega_center, MERGED_SECTION_LABEL_Y, MERGED_VEGA_SECTION_LABEL,
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=fs,
    )
    ax.text(
        vega_center, MERGED_PANEL_LETTER_Y, MERGED_PANEL_LETTERS[3],
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=MERGED_TITLE_FONTSIZE,
    )

    for center, regime in zip(regime_centers, regimes):
        ax.text(
            center, 1.04, MERGED_NEURON_PANEL_TITLES.get(regime, regime),
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=MERGED_TITLE_FONTSIZE,
        )
    ax.text(
        vega_center, 1.04, MERGED_VEGA_SUBTITLE,
        transform=ax.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=MERGED_TITLE_FONTSIZE,
    )
    for center, regime in zip(regime_centers, regimes):
        ax.text(
            center, MERGED_XLABEL_Y, "beta",
            transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=MERGED_PARAM_LABEL_FONTSIZE,
        )
    ax.text(
        vega_center, MERGED_XLABEL_Y, MERGED_VEGA_XLABEL,
        transform=ax.get_xaxis_transform(),
        ha="center", va="top", fontsize=MERGED_PARAM_LABEL_FONTSIZE,
    )

    ax.set_ylim(0.0, 1.0)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=MERGED_REF_LINE_WIDTH, zorder=2)
    ax.text(
        1.01, 0.5, "0.5",
        transform=ax.get_yaxis_transform(),
        va="center", ha="left", color="black", fontsize=MERGED_TICK_LABEL_FONTSIZE, clip_on=False,
    )
    ax.set_ylabel("Metric value", fontsize=MERGED_AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis="both", labelsize=MERGED_TICK_LABEL_FONTSIZE)
    handles, labels = ax.get_legend_handles_labels()
    leg = fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        bbox_transform=fig.transFigure,
        fontsize=MERGED_LEGEND_FONTSIZE,
        ncol=len(labels),
        handlelength=MERGED_LEGEND_HANDLELENGTH,
        handletextpad=0.35 * MERGED_SIZE_SCALE,
        borderpad=0.2 * MERGED_SIZE_SCALE,
        labelspacing=0.2 * MERGED_SIZE_SCALE,
        columnspacing=0.7 * MERGED_SIZE_SCALE,
        markerscale=1.0,
        frameon=False,
    )
    for handle in leg.legendHandles:
        handle.set_linewidth(MERGED_LINE_WIDTH * MERGED_LEGEND_SYMBOL_SCALE)
        handle.set_markersize(MERGED_MARKER_SIZE * MERGED_LEGEND_SYMBOL_SCALE)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.grid(True, axis="y", alpha=0.3)
    fig.subplots_adjust(**MERGED_SUBPLOT_MARGINS)
    path = os.path.join(out_dir, "05_combined_neuron_vega_merged.png")
    fig.savefig(path, dpi=COMBINED_SHARED_DPI)
    dup_path = os.path.join(_REPO, DEFAULT_VEGA_SWEEP_ROOT, "aggregate", "05_combined_neuron_vega_merged.png")
    os.makedirs(os.path.dirname(dup_path), exist_ok=True)
    shutil.copy2(path, dup_path)
    plt.close(fig)
    print("Saved: %s" % path)
    print("Saved: %s" % dup_path)


def plot_combined_by_overlap_dual(
    regimes: List[str],
    betas: List[float],
    dist_means: np.ndarray,
    prob_means: np.ndarray,
    mig_means: np.ndarray,
    out_dir: str,
) -> None:
    positions, x_labels, regime_centers, separators = _overlap_grouped_x_layout(regimes, betas)
    fig, ax_left = plt.subplots(figsize=(max(14, 1.2 * len(positions)), 6))
    _plot_overlap_regime_segments(ax_left, positions, dist_means, regimes, betas, color="steelblue", marker="o", label=LEGEND_DC)
    _plot_overlap_regime_segments(ax_left, positions, prob_means, regimes, betas, color="darkorange", marker="s", label=LEGEND_KOS)
    left_vals = np.concatenate([
        dist_means[np.isfinite(dist_means)], prob_means[np.isfinite(prob_means)]
    ])
    ax_left.set_ylim(
        max(0.0, float(np.nanmin(left_vals)) - 0.05) if len(left_vals) else 0.0,
        min(1.05, float(np.nanmax(left_vals)) + 0.05) if len(left_vals) else 1.0,
    )
    ax_right = ax_left.twinx()
    _plot_overlap_regime_segments(ax_right, positions, mig_means, regimes, betas, color="red", marker="^", label=LEGEND_MIG)
    ax_right.set_ylim(_mig_adapted_ylim(mig_means))
    ax_right.set_ylabel(LEGEND_MIG, color="red")
    ax_right.tick_params(axis="y", labelcolor="red")
    _decorate_overlap_xaxis(ax_left, positions, x_labels, regime_centers, separators, regimes)
    ax_left.set_ylabel("%s / %s" % (LEGEND_DC, LEGEND_KOS))
    ax_left.set_title(
        "Neuron simulation: interpretability metrics vs overlap regime\n"
        "%s — full beta sweep (dual y-axis)" % _title_dims()
    )
    _add_y_reference_lines(ax_left)
    ax_left.grid(True, axis="y", alpha=0.3)
    lines_l, labels_l = ax_left.get_legend_handles_labels()
    lines_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_l + lines_r, labels_l + labels_r, loc="best")
    fig.subplots_adjust(bottom=0.22)
    fig.tight_layout()
    path = os.path.join(out_dir, "03_combined_metrics_by_overlap_dual.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("Saved: %s" % path)


cached_stats = {}


def main():
    global cached_stats
    p = argparse.ArgumentParser(description="Aggregate neuron_metrics_study plots.")
    add_run_name_arg(p)
    args = p.parse_args()
    apply_run_name_from_args(args)

    out_dir = aggregate_output_dir()
    os.makedirs(out_dir, exist_ok=True)

    dcorr_csv = os.path.join(
        metric_output_dir("distance_corr"), "distance_corr_overlap_sweep.csv"
    )
    prob_csv = os.path.join(
        metric_output_dir("probability"), "probabilities_overlap_sweep.csv"
    )
    mig_csv = os.path.join(metric_output_dir("mig"), "mig_overlap_sweep.csv")

    dcorr_df = pd.read_csv(dcorr_csv)
    prob_df = pd.read_csv(prob_csv)
    mig_df = pd.read_csv(mig_csv)
    mig_col = _mig_column(mig_df)

    betas = [b for b in BETA_SWEEP if b in set(dcorr_df["beta"]) & set(prob_df["beta"]) & set(mig_df["beta"])]
    regimes = [r for r in OVERLAP_REGIME_ORDER if r in dcorr_df["overlap_profile"].unique()]

    for regime in regimes:
        sub = dcorr_df.loc[dcorr_df["overlap_profile"] == regime]
        if "mean_overlap" in sub.columns and len(sub):
            cached_stats[regime] = {
                "mean_overlap": float(sub["mean_overlap"].iloc[0]),
                "median_overlap": float(sub["median_overlap"].iloc[0]),
            }

    # Per-regime beta combined plots
    mean_rows = []
    for regime in regimes:
        stats = cached_stats.get(regime, {"mean_overlap": float("nan"), "median_overlap": float("nan")})
        dist_m = _means_per_beta(dcorr_df, "distance_corr", betas, regime)
        prob_m = _means_per_beta(prob_df, "probability", betas, regime)
        mig_m = _means_per_beta(mig_df, mig_col, betas, regime)
        plot_combined_by_regime_beta(
            regime, betas, dist_m, prob_m, mig_m,
            stats["mean_overlap"], stats["median_overlap"], out_dir,
        )
        for i, beta in enumerate(betas):
            mean_rows.append({
                "overlap_profile": regime,
                "beta": beta,
                "mean_overlap": stats["mean_overlap"],
                "median_overlap": stats["median_overlap"],
                "distance_corr_mean": dist_m[i],
                "probability_mean": prob_m[i],
                "mig_mean": mig_m[i],
                "mig_source_column": mig_col,
            })

    pd.DataFrame(mean_rows).to_csv(
        os.path.join(out_dir, "combined_metrics_overlap_beta_means.csv"), index=False
    )

    dist_grid = _means_per_regime_beta_grid(dcorr_df, "distance_corr", regimes, betas)
    prob_grid = _means_per_regime_beta_grid(prob_df, "probability", regimes, betas)
    mig_grid = _means_per_regime_beta_grid(mig_df, mig_col, regimes, betas)

    plot_combined_by_overlap_dual(regimes, betas, dist_grid, prob_grid, mig_grid, out_dir)
    plot_combined_by_overlap_shared(regimes, betas, dist_grid, prob_grid, mig_grid, out_dir)
    plot_combined_neuron_vega_merged(regimes, betas, dist_grid, prob_grid, mig_grid, out_dir)

    with open(os.path.join(out_dir, "run_parameters.txt"), "w", encoding="utf-8") as f:
        f.write("neuron_metrics_study aggregate\n")
        f.write("Run root: %s\n" % run_root())
        f.write("Generated (UTC): %s\n\n" % datetime.now(timezone.utc).isoformat())
        f.write("N=%d, M=%d, K=%d\n" % (FIXED_N_EXAMPLES, FIXED_M, FIXED_K))
        f.write("alpha_mode: exponential\n")
        f.write("overlap_threshold: None\n")
        f.write("beta_sweep: %s\n\n" % ",".join("%.1f" % b for b in betas))
        for regime, stats in cached_stats.items():
            f.write("%s: mean_overlap=%.4f, median_overlap=%.4f\n" % (
                regime, stats["mean_overlap"], stats["median_overlap"],
            ))

    print("Done. Outputs in: %s" % out_dir)


if __name__ == "__main__":
    main()
