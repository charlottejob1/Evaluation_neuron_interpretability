"""Regenerate probability study PNGs from existing probabilities*.csv (no simulation)."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIXED_M = 2000
FIXED_K = 300
PROBABILITY_YLIM = (0.45, 1.05)
PROBABILITY_YLABEL = "P(self reduction > other)"


def _plot_beta(df: pd.DataFrame, out_dir: str) -> None:
    betas = sorted(df["beta"].unique())
    data = [df.loc[df["beta"] == b, "probability"].values for b in betas]
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(data, positions=range(len(betas)), widths=0.6, patch_artist=True, showfliers=False)
    for box in bp["boxes"]:
        box.set(facecolor="steelblue", edgecolor="navy", alpha=0.7)
    plt.xticks(range(len(betas)), [f"{b:.1f}" for b in betas])
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel("beta")
    plt.ylabel(PROBABILITY_YLABEL)
    plt.title("Influence of beta on probability distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p = os.path.join(out_dir, "01_probabilities_by_beta.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)


def _plot_overlap(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="overlap_profile",
        y="probability",
        hue="beta",
        order=["low_overlap", "medium_overlap", "high_overlap"],
        showfliers=False,
    )
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel(f"Overlap regime (M={FIXED_M}, K={FIXED_K} fixed)")
    plt.ylabel(PROBABILITY_YLABEL)
    plt.title("Impact of overlap regime on probabilities at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p = os.path.join(out_dir, "02b_probabilities_by_overlap.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)


def _plot_M(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="n_variables", y="probability", hue="beta", showfliers=False)
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel("M (number of variables)")
    plt.ylabel(PROBABILITY_YLABEL)
    plt.title("Impact of M on probabilities at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p = os.path.join(out_dir, "03_probabilities_by_M.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)


def _plot_K(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="n_concepts", y="probability", hue="beta", showfliers=False)
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel("K (number of concepts); concept size scales as ~M/K to fix overlap density")
    plt.ylabel(PROBABILITY_YLABEL)
    plt.title("Impact of K on probabilities at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p = os.path.join(out_dir, "04_probabilities_by_K.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)


def _plot_threshold(df: pd.DataFrame, out_dir: str) -> None:
    mean_df = (
        df.groupby(["beta", "overlap_threshold"], as_index=False)["probability"]
        .mean()
        .rename(columns={"probability": "mean_probability"})
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=mean_df, x="overlap_threshold", y="mean_probability", hue="beta", marker="o")
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel("Overlap threshold (include pairs with overlap(observed, perturbated) < threshold)")
    plt.ylabel("Mean " + PROBABILITY_YLABEL)
    plt.title("Impact of overlap threshold on specificity (wide overlap spectrum)")
    plt.tight_layout()
    p = os.path.join(out_dir, "05_mean_probability_vs_overlap_threshold.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)

    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df, x="overlap_threshold", y="probability", hue="beta", showfliers=False)
    plt.ylim(*PROBABILITY_YLIM)
    plt.xlabel("Overlap threshold")
    plt.ylabel(PROBABILITY_YLABEL)
    plt.title("Probability distributions vs overlap threshold and beta")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    p = os.path.join(out_dir, "06_probability_distributions_threshold_beta.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)

    pair_counts = (
        df.groupby("overlap_threshold")["perturbated_concept"].count().reset_index(name="n_pairs")
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pair_counts, x="overlap_threshold", y="n_pairs", color="steelblue")
    plt.xlabel("Overlap threshold")
    plt.ylabel("Number of (perturbed, observed) pairs included")
    plt.title("Pair count retained (overlap < threshold, wide overlap spectrum)")
    plt.tight_layout()
    p = os.path.join(out_dir, "07_pair_count_vs_overlap_threshold.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved:", p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="plots_trial_v7_exp")
    args = p.parse_args()
    out_dir = os.path.abspath(args.output_dir)
    sns.set_theme(style="whitegrid")
    jobs = [
        ("probabilities_beta_sweep.csv", _plot_beta),
        ("probabilities_overlap_sweep.csv", _plot_overlap),
        ("probabilities_M_sweep.csv", _plot_M),
        ("probabilities_K_sweep.csv", _plot_K),
        ("probabilities_overlap_threshold_sweep.csv", _plot_threshold),
    ]
    for csv_name, fn in jobs:
        path = os.path.join(out_dir, csv_name)
        if not os.path.isfile(path):
            print("Skip missing:", path)
            continue
        fn(pd.read_csv(path), out_dir)


if __name__ == "__main__":
    main()
