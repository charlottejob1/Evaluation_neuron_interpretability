"""
Plot mutual-information heatmaps (neurons × pathway concepts) from VEGA MIG outputs.

Writes per run_dir:
  mig_metrics/mi_heatmap_full.png            — full 674×674, native neuron/concept order
  mig_metrics/mi_heatmap_clustered.png       — full matrix + hierarchical clustering
  mig_metrics/mi_heatmap_top40.png           — top 40×40 by max row/column MI (labeled)
  mig_metrics/mi_heatmap_diagonal_only.png   — 674×674, only matched neuron–concept pairs
  mig_metrics/mi_diagonal_strip.png          — 674×1 strip of matched-pair MI (native order)
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

FCN_PERCENTS = list(range(0, 101, 10))


def fcn_dir_name(percent: int) -> str:
    return "fcn_%03d" % percent


def _load_mi_run(mig_dir: str) -> tuple:
    """Return (mi_matrix, pathway_names) from a mig_metrics folder."""
    mi_path = os.path.join(mig_dir, "mi_matrix.npy")
    csv_path = os.path.join(mig_dir, "mig_per_neuron.csv")
    if not (os.path.isfile(mi_path) and os.path.isfile(csv_path)):
        raise FileNotFoundError(mi_path)

    mi = np.load(mi_path)
    df = pd.read_csv(csv_path)
    names = df["pathway"].astype(str).tolist()
    if mi.shape[0] != len(names) or mi.shape[1] != mi.shape[0]:
        raise ValueError(
            "mi_matrix shape %s incompatible with %d pathway names in %s"
            % (mi.shape, len(names), csv_path)
        )
    return mi, names


def _title_suffix(fcn_percent: Optional[int]) -> str:
    pct_label = "%d%%" % fcn_percent if fcn_percent is not None else "?"
    return "fcn_%03d, %s FC neurons" % (
        fcn_percent if fcn_percent is not None else -1,
        pct_label,
    )


def plot_diagonal_mi_for_run(
    mig_dir: str,
    *,
    fcn_percent: Optional[int] = None,
    dpi: int = 150,
    skip_existing: bool = False,
) -> bool:
    """
    Plot MI only for matched neuron–concept pairs (matrix diagonal).

    Row/column ``i`` are the same Reactome pathway after VEGA alignment; off-diagonal
    cells are masked out.
    """
    path_matrix = os.path.join(mig_dir, "mi_heatmap_diagonal_only.png")
    path_strip = os.path.join(mig_dir, "mi_diagonal_strip.png")

    if skip_existing and os.path.isfile(path_matrix) and os.path.isfile(path_strip):
        return False

    try:
        mi, _names = _load_mi_run(mig_dir)
    except (FileNotFoundError, ValueError):
        return False

    diag = np.diag(mi)
    title_suffix = _title_suffix(fcn_percent)
    sns.set_theme(style="white", context="notebook")

    if not (skip_existing and os.path.isfile(path_matrix)):
        diag_only = np.full_like(mi, np.nan, dtype=float)
        np.fill_diagonal(diag_only, diag)
        cmap = plt.cm.get_cmap("viridis").copy()
        cmap.set_bad(color="#f0f0f0")

        fig, ax = plt.subplots(figsize=(12, 11))
        sns.heatmap(
            diag_only,
            cmap=cmap,
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            cbar_kws={"label": "Mutual information (matched pairs)"},
        )
        ax.set_xlabel("Pathway enrichment (concept)")
        ax.set_ylabel("Latent neuron")
        ax.set_title(
            "Matched-pair MI only (diagonal, native order; %s)" % title_suffix,
            fontsize=11,
        )
        fig.savefig(path_matrix, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not (skip_existing and os.path.isfile(path_strip)):
        strip = diag.reshape(-1, 1)
        fig, ax = plt.subplots(figsize=(4, 12))
        sns.heatmap(
            strip,
            cmap="viridis",
            xticklabels=["Matched pair"],
            yticklabels=False,
            ax=ax,
            cbar_kws={"label": "Mutual information"},
        )
        ax.set_ylabel("Neuron / concept index (native order)")
        ax.set_title("Diagonal MI strip (%s)" % title_suffix, fontsize=10)
        fig.savefig(path_strip, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return True


def plot_mi_heatmaps_for_run(
    mig_dir: str,
    *,
    fcn_percent: Optional[int] = None,
    top_n: int = 40,
    dpi: int = 150,
    skip_existing: bool = False,
    only_full: bool = False,
) -> bool:
    """
    Generate clustered and top-N MI heatmaps for one ``fcn_XXX/mig_metrics`` folder.

    Returns True if plots were written, False if skipped or inputs missing.
    """
    path_full = os.path.join(mig_dir, "mi_heatmap_full.png")
    path_clustered = os.path.join(mig_dir, "mi_heatmap_clustered.png")
    path_top = os.path.join(mig_dir, "mi_heatmap_top%d.png" % top_n)

    outputs = [path_full] if only_full else [path_full, path_clustered, path_top]
    if skip_existing and all(os.path.isfile(p) for p in outputs):
        return False

    mi, names = _load_mi_run(mig_dir)
    title_suffix = _title_suffix(fcn_percent)

    sns.set_theme(style="white", context="notebook")

    if not (skip_existing and os.path.isfile(path_full)):
        fig, ax = plt.subplots(figsize=(12, 11))
        sns.heatmap(
            mi,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            ax=ax,
            cbar_kws={"label": "Mutual information"},
        )
        ax.set_xlabel("Pathway enrichment (concept)")
        ax.set_ylabel("Latent neuron")
        ax.set_title(
            "MI: neurons × concepts, native order (%s)" % title_suffix,
            fontsize=11,
        )
        fig.savefig(path_full, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    if not only_full and not (skip_existing and os.path.isfile(path_clustered)):
        g = sns.clustermap(
            mi,
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
            figsize=(12, 11),
            dendrogram_ratio=0.08,
            cbar_pos=(0.02, 0.8, 0.03, 0.15),
            cbar_kws={"label": "Mutual information"},
        )
        g.fig.suptitle(
            "MI: VEGA latent neurons vs pathway enrichment (%s)" % title_suffix,
            y=1.02,
            fontsize=11,
        )
        g.savefig(path_clustered, dpi=dpi, bbox_inches="tight")
        plt.close("all")

    if only_full or (skip_existing and os.path.isfile(path_top)):
        return True

    n = min(top_n, mi.shape[0], mi.shape[1])
    row_strength = mi.max(axis=1)
    col_strength = mi.max(axis=0)
    top_rows = np.argsort(row_strength)[-n:][::-1]
    top_cols = np.argsort(col_strength)[-n:][::-1]
    sub = mi[np.ix_(top_rows, top_cols)]
    row_labels = [names[i].replace("REACTOME_", "")[:45] for i in top_rows]
    col_labels = [names[i].replace("REACTOME_", "")[:45] for i in top_cols]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sub,
        xticklabels=col_labels,
        yticklabels=row_labels,
        cmap="viridis",
        ax=ax,
        cbar_kws={"label": "Mutual information"},
    )
    ax.set_xlabel("Pathway enrichment (concept)")
    ax.set_ylabel("Latent neuron")
    ax.set_title("Top-%d MI by max row/column MI (%s)" % (n, title_suffix))
    plt.xticks(rotation=90, ha="center", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    fig.savefig(path_top, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot MI heatmaps for VEGA fcn_* MIG runs.")
    p.add_argument("--sweep-root", type=str, default="vega_fcn_sweep")
    p.add_argument(
        "--percents",
        type=str,
        default=None,
        help="Comma-separated FC percents (default: 0,10,...,100).",
    )
    p.add_argument("--top-n", type=int, default=40, help="Submatrix size for labeled heatmap.")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--only-full",
        action="store_true",
        help="Only write mi_heatmap_full.png (no cluster / top-N plots).",
    )
    p.add_argument(
        "--only-diagonal",
        action="store_true",
        help="Only write matched-pair (diagonal) MI heatmaps.",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip PNGs that already exist.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    percents = FCN_PERCENTS
    if args.percents:
        percents = [int(x.strip()) for x in args.percents.split(",") if x.strip()]

    n_ok = 0
    for pct in percents:
        mig_dir = os.path.join(args.sweep_root, fcn_dir_name(pct), "mig_metrics")
        if not os.path.isdir(mig_dir):
            print("[skip] missing %s" % mig_dir)
            continue
        try:
            if args.only_diagonal:
                wrote = plot_diagonal_mi_for_run(
                    mig_dir,
                    fcn_percent=pct,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                )
            else:
                wrote = plot_mi_heatmaps_for_run(
                    mig_dir,
                    fcn_percent=pct,
                    top_n=args.top_n,
                    dpi=args.dpi,
                    skip_existing=args.skip_existing,
                    only_full=args.only_full,
                )
                if wrote and not args.only_full:
                    plot_diagonal_mi_for_run(
                        mig_dir,
                        fcn_percent=pct,
                        dpi=args.dpi,
                        skip_existing=args.skip_existing,
                    )
        except (ValueError, FileNotFoundError) as exc:
            print("[error] %s: %s" % (fcn_dir_name(pct), exc))
            continue
        if wrote:
            if args.only_diagonal:
                print(
                    "[ok] %s → mi_heatmap_diagonal_only.png, mi_diagonal_strip.png"
                    % fcn_dir_name(pct)
                )
            elif args.only_full:
                print("[ok] %s → mi_heatmap_full.png" % fcn_dir_name(pct))
            else:
                print(
                    "[ok] %s → mi_heatmap_full.png, mi_heatmap_clustered.png, "
                    "mi_heatmap_top%d.png, diagonal plots"
                    % (fcn_dir_name(pct), args.top_n)
                )
            n_ok += 1
        else:
            print("[skip] %s (no mi_matrix or --skip-existing)" % fcn_dir_name(pct))

    print("Done. Wrote heatmaps for %d / %d fractions." % (n_ok, len(percents)))


if __name__ == "__main__":
    main()
