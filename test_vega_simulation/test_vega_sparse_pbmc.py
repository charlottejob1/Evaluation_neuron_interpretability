"""
End-to-end test: train original VEGA2 (sparse decoder) on preprocessed PBMC,
then measure interpretability on the held-out test set.

Step 1 — Training (``train_vega_pbmc.run_pbmc_training``):
  Same hyperparameters as ``test_vega_simulation.py`` with
  ``fully_connected_neuron_fraction=0`` (sparse Reactome-masked architecture).

Step 2 — Interpretability (``vega_fcn_metrics.evaluate_run_directory``):
  Trained model in inference mode on the **test set only**; per-pathway
  ``distance_corr`` and ``reduction_score_probability`` (overlap threshold 0.5).

Outputs (default repository ``vega_sparse_test/``):
  vega2_pbmc_fcn0.00.pt          — saved checkpoint
  metrics.json, training plots   — from training step
  run.txt                        — full setup and summary
  performance_metrics.csv        — train/test reconstruction metrics
  interpretability_metrics.csv   — per-pathway interpretability metrics
  interpretability_means.txt     — mean distance_corr and probability
  distance_corr_comparison.csv   — new vs original distance_corr per pathway
  distance_corr_comparison.txt   — summary of the distance_corr comparison

Examples::

  python test_vega_sparse_pbmc.py
  python test_vega_sparse_pbmc.py --quick
  python test_vega_sparse_pbmc.py --eval --output-dir vega_sparse_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_REPO, "test_vega_simulation"))
sys.path.insert(0, os.path.join(_REPO, "vega_simulation"))

from test_vega_simulation import DEFAULTS, resolve_output_dir
from train_vega_pbmc import run_pbmc_training
from vega_fcn_metrics import (
    DEFAULT_OVERLAP_THRESHOLD,
    compare_distance_corr_run_directory,
    evaluate_run_directory,
)

DEFAULT_OUTPUT_DIR = "vega_sparse_test"
SPARSE_FCN_FRACTION = 0.0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Train sparse VEGA2 on PBMC, then evaluate distance_corr and "
            "reduction_score_probability on the test set."
        )
    )
    p.add_argument("--data-dir", type=str, default=DEFAULTS["data_dir"])
    p.add_argument("--gmt-path", type=str, default=DEFAULTS["gmt_path"])
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Test repository root (default: %s)." % DEFAULT_OUTPUT_DIR,
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Shortcut: saves to results_<run-name>/ (overrides --output-dir).",
    )
    p.add_argument(
        "--overlap-threshold",
        type=float,
        default=DEFAULT_OVERLAP_THRESHOLD,
        help="Overlap threshold for reduction_score_probability (default: 0.5).",
    )
    p.add_argument(
        "--max-pathways",
        type=int,
        default=None,
        help="Limit pathways for faster debugging.",
    )
    p.add_argument("--train", action="store_true", help="Run step 1 (training).")
    p.add_argument("--eval", action="store_true", help="Run step 2 (interpretability).")
    p.add_argument(
        "--skip-existing-train",
        action="store_true",
        help="Skip training when metrics.json and checkpoint already exist.",
    )
    p.add_argument(
        "--skip-distance-corr-compare",
        action="store_true",
        help="Skip new vs original distance_corr comparison step.",
    )
    p.add_argument(
        "--compare-distance-corr-only",
        action="store_true",
        help="Only run distance_corr comparison (requires trained model).",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer epochs, patience, and pathways for a fast smoke test.",
    )
    return p.parse_args()


def _training_hyperparams(args) -> dict:
    hp = {
        "n_top_genes": DEFAULTS["n_top_genes"],
        "train_size": DEFAULTS["train_size"],
        "batch_size": DEFAULTS["batch_size"],
        "learning_rate": DEFAULTS["learning_rate"],
        "n_epochs": DEFAULTS["n_epochs"],
        "train_patience": DEFAULTS["train_patience"],
        "test_patience": DEFAULTS["test_patience"],
        "kld_weight": DEFAULTS["kld_weight"],
        "dropout": DEFAULTS["dropout"],
        "seed": DEFAULTS["seed"],
    }
    if args.quick:
        hp["n_epochs"] = 20
        hp["train_patience"] = 5
        hp["test_patience"] = 5
    return hp


def _resolve_output_dir(args) -> str:
    if args.output_dir is not None:
        return os.path.abspath(args.output_dir)
    if args.run_name is not None and str(args.run_name).strip():
        return resolve_output_dir(None, args.run_name, SPARSE_FCN_FRACTION)
    return os.path.abspath(DEFAULT_OUTPUT_DIR)


def _checkpoint_exists(out_dir: str) -> bool:
    if not os.path.isfile(os.path.join(out_dir, "metrics.json")):
        return False
    for name in os.listdir(out_dir):
        if name.startswith("vega2_pbmc_fcn") and name.endswith(".pt"):
            return True
    return False


def save_performance_metrics_csv(metrics: dict, path: str) -> None:
    rows = []
    for split in ("train", "test"):
        split_metrics = metrics.get(split, {})
        row = {"split": split}
        row.update(split_metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def compute_interpretability_means(df: pd.DataFrame) -> dict:
    dist = df["distance_corr"].dropna().astype(float)
    prob = df["reduction_score_probability"].dropna().astype(float)
    return {
        "distance_corr_mean": float(dist.mean()) if len(dist) else float("nan"),
        "distance_corr_n": int(len(dist)),
        "reduction_score_probability_mean": float(prob.mean()) if len(prob) else float("nan"),
        "reduction_score_probability_n": int(len(prob)),
        "n_pathways_total": int(len(df)),
    }


def write_interpretability_means_txt(path: str, means: dict, overlap_threshold: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("VEGA2 sparse PBMC — interpretability means (test set only)\n")
        f.write("=" * 60 + "\n")
        f.write("Generated (UTC): %s\n\n" % datetime.now(timezone.utc).isoformat())
        f.write("eval_split: test\n")
        f.write("overlap_threshold: %.4f\n\n" % overlap_threshold)
        f.write("distance_corr\n")
        f.write("-" * 40 + "\n")
        f.write("mean: %.6f\n" % means["distance_corr_mean"])
        f.write("n_valid_pathways: %d / %d\n\n"
                % (means["distance_corr_n"], means["n_pathways_total"]))
        f.write("reduction_score_probability\n")
        f.write("-" * 40 + "\n")
        f.write("mean: %.6f\n" % means["reduction_score_probability_mean"])
        f.write("n_valid_pathways: %d / %d\n"
                % (means["reduction_score_probability_n"], means["n_pathways_total"]))


def write_run_txt(
    path: str,
    *,
    out_dir: str,
    args,
    hyperparams: dict,
    metrics: Optional[dict],
    interpretability_means: Optional[dict],
    distance_corr_summary: Optional[dict],
    model_path: Optional[str],
    steps_run: list,
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("VEGA2 sparse PBMC end-to-end test run\n")
        f.write("=" * 60 + "\n")
        f.write("Generated (UTC): %s\n" % datetime.now(timezone.utc).isoformat())
        f.write("Output directory: %s\n" % out_dir)
        f.write("Steps executed: %s\n\n" % ", ".join(steps_run))

        f.write("Architecture\n")
        f.write("-" * 40 + "\n")
        f.write("model: VEGA2 (original sparse Reactome-masked decoder)\n")
        f.write("fully_connected_neuron_fraction: %.4f\n\n" % SPARSE_FCN_FRACTION)

        f.write("Data\n")
        f.write("-" * 40 + "\n")
        f.write("data_dir: %s\n" % os.path.abspath(args.data_dir))
        f.write("gmt_path: %s\n" % args.gmt_path)
        f.write("preprocessing: top %d highly variable genes\n" % hyperparams["n_top_genes"])
        f.write("train_size (first split): %.4f\n\n" % hyperparams["train_size"])

        f.write("Training hyperparameters (from test_vega_simulation.py)\n")
        f.write("-" * 40 + "\n")
        for key, value in hyperparams.items():
            f.write("%s: %s\n" % (key, value))
        f.write("\n")

        f.write("Interpretability evaluation\n")
        f.write("-" * 40 + "\n")
        f.write("eval_split: test\n")
        f.write("inference_mode: True\n")
        f.write("metrics: distance_corr, reduction_score_probability\n")
        f.write("overlap_threshold: %.4f\n" % args.overlap_threshold)
        if args.max_pathways is not None:
            f.write("max_pathways: %d\n" % args.max_pathways)
        f.write("\n")

        if metrics is not None:
            f.write("Performance metrics (reconstruction)\n")
            f.write("-" * 40 + "\n")
            for split in ("train", "test"):
                sm = metrics[split]
                f.write(
                    "%s  MSE=%.6f | Pearson overall=%.6f | per-cell=%.6f | n_cells=%d\n"
                    % (
                        split.upper(),
                        sm["mse"],
                        sm["pearson_overall"],
                        sm["pearson_per_cell_mean"],
                        sm["n_cells"],
                    )
                )
            f.write("saved: performance_metrics.csv, metrics.json\n\n")

        if model_path is not None:
            f.write("Saved model: %s\n\n" % model_path)

        if interpretability_means is not None:
            f.write("Interpretability means (test set)\n")
            f.write("-" * 40 + "\n")
            f.write("distance_corr mean: %.6f (n=%d)\n"
                    % (interpretability_means["distance_corr_mean"],
                       interpretability_means["distance_corr_n"]))
            f.write("reduction_score_probability mean: %.6f (n=%d)\n"
                    % (interpretability_means["reduction_score_probability_mean"],
                       interpretability_means["reduction_score_probability_n"]))
            f.write("saved: interpretability_metrics.csv, interpretability_means.txt\n\n")

        if distance_corr_summary is not None:
            f.write("Distance correlation comparison (new vs vega_usage original)\n")
            f.write("-" * 40 + "\n")
            f.write(
                "exact match: %d / %d pathways | mean |diff|=%.6e | max |diff|=%.6e\n"
                % (
                    distance_corr_summary["n_exact_match"],
                    distance_corr_summary["n_pathways"],
                    distance_corr_summary["mean_abs_diff"],
                    distance_corr_summary["max_abs_diff"],
                )
            )
            f.write(
                "mean distance_corr (new): %.6f | mean (original): %.6f\n"
                % (
                    distance_corr_summary["mean_distance_corr_new"],
                    distance_corr_summary["mean_distance_corr_original"],
                )
            )
            f.write("saved: distance_corr_comparison.csv, distance_corr_comparison.txt\n\n")

        f.write("Output files\n")
        f.write("-" * 40 + "\n")
        f.write("run.txt\n")
        f.write("performance_metrics.csv\n")
        f.write("interpretability_metrics.csv\n")
        f.write("interpretability_means.txt\n")
        f.write("distance_corr_comparison.csv\n")
        f.write("distance_corr_comparison.txt\n")
        f.write("metrics.json\n")
        f.write("training_info.txt\n")
        f.write("vega2_pbmc_fcn0.00.pt\n")
        f.write("01_fully_connected_neuron_mask.png\n")
        f.write("02_training_loss.png\n")
        f.write("03_reconstruction_train.png\n")
        f.write("04_reconstruction_test.png\n")


def run_training_step(args, out_dir: str, device: torch.device, hyperparams: dict) -> dict:
    if args.skip_existing_train and _checkpoint_exists(out_dir):
        print("[skip train] checkpoint already exists in %s" % out_dir)
        metrics_path = os.path.join(out_dir, "metrics.json")
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    print("\n" + "=" * 60)
    print("Step 1 — Train sparse VEGA2 on PBMC")
    print("Output: %s" % out_dir)
    print("=" * 60)

    result = run_pbmc_training(
        data_dir=args.data_dir,
        gmt_path=args.gmt_path,
        fully_connected_neuron_fraction=SPARSE_FCN_FRACTION,
        output_dir=out_dir,
        device=device,
        **hyperparams,
    )

    metrics = result["metrics"]
    perf_csv = os.path.join(out_dir, "performance_metrics.csv")
    save_performance_metrics_csv(metrics, perf_csv)
    print("Saved performance metrics CSV: %s" % perf_csv)
    return metrics


def run_distance_corr_comparison_step(
    args,
    out_dir: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("Step 3 — Compare distance_corr (new vs vega_usage original)")
    print("Run directory: %s" % out_dir)
    print("=" * 60)

    max_pathways = args.max_pathways
    if args.quick and max_pathways is None:
        max_pathways = 20

    return compare_distance_corr_run_directory(
        out_dir,
        data_dir=args.data_dir,
        max_pathways=max_pathways,
        device=device,
        save_csv=True,
        verbose=True,
    )


def run_interpretability_step(
    args,
    out_dir: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict]:
    print("\n" + "=" * 60)
    print("Step 2 — Interpretability on test set (inference mode)")
    print("Run directory: %s" % out_dir)
    print("=" * 60)

    max_pathways = args.max_pathways
    if args.quick and max_pathways is None:
        max_pathways = 20

    df = evaluate_run_directory(
        out_dir,
        data_dir=args.data_dir,
        overlap_threshold=args.overlap_threshold,
        max_pathways=max_pathways,
        device=device,
        save_csv=True,
    )
    means = compute_interpretability_means(df)
    means_path = os.path.join(out_dir, "interpretability_means.txt")
    write_interpretability_means_txt(means_path, means, args.overlap_threshold)
    print("Saved interpretability metrics CSV: %s"
          % os.path.join(out_dir, "interpretability_metrics.csv"))
    print("Saved interpretability means: %s" % means_path)
    print(
        "Mean distance_corr=%.4f (n=%d) | mean probability=%.4f (n=%d)"
        % (
            means["distance_corr_mean"],
            means["distance_corr_n"],
            means["reduction_score_probability_mean"],
            means["reduction_score_probability_n"],
        )
    )
    return df, means


def main():
    args = parse_args()
    do_compare_only = args.compare_distance_corr_only
    do_train = (args.train or not args.eval) and not do_compare_only
    do_eval = (args.eval or not args.train) and not do_compare_only
    do_compare = not args.skip_distance_corr_compare and (
        do_eval or do_compare_only
    )

    if args.quick and args.max_pathways is None:
        print("[quick] Reduced epochs, patience; interpretability limited to 20 pathways.")

    out_dir = _resolve_output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyperparams = _training_hyperparams(args)

    print("Output directory: %s" % out_dir)
    print("Device: %s" % device)
    print("Sparse architecture (fully_connected_neuron_fraction): %.2f" % SPARSE_FCN_FRACTION)

    steps_run = []
    metrics: Optional[dict] = None
    interpretability_means: Optional[dict] = None
    distance_corr_summary: Optional[dict] = None
    model_path: Optional[str] = None

    if do_train:
        metrics = run_training_step(args, out_dir, device, hyperparams)
        steps_run.append("train")
        fcn_tag = "%.2f" % SPARSE_FCN_FRACTION
        model_path = os.path.join(out_dir, "vega2_pbmc_fcn%s.pt" % fcn_tag)
    elif os.path.isfile(os.path.join(out_dir, "metrics.json")):
        with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
            metrics = json.load(f)
        for name in os.listdir(out_dir):
            if name.startswith("vega2_pbmc_fcn") and name.endswith(".pt"):
                model_path = os.path.join(out_dir, name)
                break

    if do_eval:
        if not _checkpoint_exists(out_dir):
            raise FileNotFoundError(
                "No trained model in %s — run training first (pass --train or omit --eval)."
                % out_dir
            )
        _, interpretability_means = run_interpretability_step(args, out_dir, device)
        steps_run.append("interpretability")

        if metrics is None and os.path.isfile(os.path.join(out_dir, "metrics.json")):
            with open(os.path.join(out_dir, "metrics.json"), encoding="utf-8") as f:
                metrics = json.load(f)
        if metrics is not None:
            perf_csv = os.path.join(out_dir, "performance_metrics.csv")
            save_performance_metrics_csv(metrics, perf_csv)

    if do_compare:
        if not _checkpoint_exists(out_dir):
            raise FileNotFoundError(
                "No trained model in %s — run training first."
                % out_dir
            )
        _, distance_corr_summary = run_distance_corr_comparison_step(
            args, out_dir, device
        )
        steps_run.append("distance_corr_compare")

    run_txt_path = os.path.join(out_dir, "run.txt")
    write_run_txt(
        run_txt_path,
        out_dir=out_dir,
        args=args,
        hyperparams=hyperparams,
        metrics=metrics,
        interpretability_means=interpretability_means,
        distance_corr_summary=distance_corr_summary,
        model_path=model_path,
        steps_run=steps_run,
    )
    print("\nSaved run summary: %s" % run_txt_path)
    print("Done. All outputs in: %s" % out_dir)


if __name__ == "__main__":
    main()
