"""Plot realized overlap distributions with mean and median annotations."""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.config import (
    FIXED_K,
    FIXED_M,
    OVERLAP_REGIME_DEFINITIONS,
    OVERLAP_REGIME_ORDER,
)
from shared.paths import data_dir


def plot_overlap_regime_distributions(cached: List[Dict[str, object]]) -> str:
    cache_by_name = {item["name"]: item for item in cached}
    panel_order = [n for n in OVERLAP_REGIME_ORDER if n in cache_by_name]
    colors = {
        "low_overlap": "steelblue",
        "medium_overlap": "darkorange",
        "high_overlap": "firebrick",
    }

    fig, axes = plt.subplots(
        1,
        len(panel_order),
        figsize=(6.5 * len(panel_order), 6.5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    records = []

    for ax, name in zip(axes, panel_order):
        item = cache_by_name[name]
        off_diag = item["off_diag"]
        stats = item["overlap_stats"]
        mean_o = stats["mean_overlap"]
        med_o = stats["median_overlap"]

        for v in off_diag:
            records.append({"overlap_regime": name, "pair_overlap": float(v)})

        sns.histplot(
            x=off_diag,
            bins=40,
            stat="density",
            kde=True,
            ax=ax,
            color=colors.get(name, "gray"),
        )
        ax.set_xlabel("Directional overlap |Ci ∩ Cj| / |Ci|")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.set_title(
            "%s\nmean=%.3f, median=%.3f"
            % (name.replace("_", " ").title(), mean_o, med_o),
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlim(0.0, 1.0)
        ax.axvline(mean_o, color="black", linestyle="--", linewidth=1.2, label="mean")
        ax.axvline(med_o, color="gray", linestyle=":", linewidth=1.2, label="median")

        definition = OVERLAP_REGIME_DEFINITIONS.get(name, "")
        regime = item["regime"]
        ann = (
            "M=%d, K=%d\nskew=%s, ceiling=%s\ndirichlet=%s"
            % (
                FIXED_M,
                FIXED_K,
                regime["overlap_skew"],
                regime["overlap_ceiling"],
                regime["dirichlet_total_assignments_factor"],
            )
        )
        text = "%s\n\n%s" % (definition, ann)
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    fig.suptitle(
        "Realized pairwise overlap distributions per regime\n"
        "(fixed M=%d, K=%d; title shows mean and median overlap)"
        % (FIXED_M, FIXED_K),
        fontsize=13,
    )
    out_path = os.path.join(data_dir(), "overlap_regime_distributions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: %s" % out_path)

    csv_path = os.path.join(data_dir(), "overlap_regime_pairwise_values.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    print("Saved: %s" % csv_path)
    return out_path
