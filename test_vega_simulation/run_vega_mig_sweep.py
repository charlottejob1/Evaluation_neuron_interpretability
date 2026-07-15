"""
Run VEGA MIG (gp.prerank + latent) on the full test set for each fcn_* sweep folder.

Default: all 849 test cells (no --max-cells subsample), all fractions
fcn_000 … fcn_100 in ``vega_fcn_sweep/``.

Outputs per run::

    vega_fcn_sweep/fcn_XXX/mig_metrics/
      mig_summary.json
      mig_per_concept.csv
      mig_per_neuron.csv
      mi_matrix.npy
      pathway_enrichment_scores.npy
      latent_activations.npy

Aggregate::

    vega_fcn_sweep/mig_aggregate/
      mig_summaries.csv
      per_concept_mig_all.csv
      01_mig_by_fcn.png
      run_parameters.txt

Example (full test set, all 11 FC settings):

  python test_vega_simulation/run_vega_mig_sweep.py

Subset / resume:

  python test_vega_simulation/run_vega_mig_sweep.py --percents 0,50,100 --skip-existing
  python test_vega_simulation/run_vega_mig_sweep.py --plot   # aggregate only

Pathway enrichment (gp.prerank on all 2000 genes/cell) is computed **once** and
reused for all fcn_* runs (see ``vega_fcn_sweep/shared_mig_enrichment/``).
Each fraction only runs latent + parallel MI (``--n-jobs -1``).

Runtime note: the shared prerank pass is heavy (849 cells × 2000 genes); use
``--enrich-jobs -1`` and optionally ``--permutation-num`` to tune speed.
"""

from __future__ import annotations

import argparse
import json
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

from run_vega_fcn_sweep import (
    FCN_PERCENTS,
    fcn_dir_name,
    fcn_fraction,
    legend_label,
)
from train_vega_pbmc import create_vega_test_eval_context, load_pbmc_8k
from vega_fcn_metrics import find_model_checkpoint
from vega_mig_metrics import (
    ENRICHMENT_METHOD,
    compute_or_load_shared_pathway_enrichment,
    evaluate_vega_mig_run,
    pathway_names_from_list,
    shared_enrichment_dir,
    subsample_adata,
)

SWEEP_ROOT = "vega_fcn_sweep"
AGGREGATE_DIR = "mig_aggregate"


def run_dir_for_percent(sweep_root: str, percent: int) -> str:
    return os.path.join(sweep_root, fcn_dir_name(percent))


def mig_summary_path(run_dir: str) -> str:
    return os.path.join(run_dir, "mig_metrics", "mig_summary.json")


def mig_summary_is_current(
    run_dir: str,
    *,
    n_cells: int,
    n_genes: int,
) -> bool:
    """True if existing mig_summary matches the requested prerank / cell settings."""
    path = mig_summary_path(run_dir)
    if not os.path.isfile(path):
        return False
    with open(path, encoding="utf-8") as f:
        row = json.load(f)
    if int(row.get("n_cells", -1)) != n_cells:
        return False
    if row.get("enrichment_method", "") != ENRICHMENT_METHOD:
        return False
    if int(row.get("n_genes", -1)) != n_genes:
        return False
    return True


def evaluate_one_mig(
    run_dir: str,
    *,
    data_dir: str,
    gmt_path: str,
    max_pathways: Optional[int],
    max_cells: Optional[int],
    pathway_scores,
    pathway_order: List[str],
    enrichment_source: str,
    n_jobs: int,
    device,
    verbose: bool,
) -> dict:
    """MIG for one VEGA run using shared pathway enrichment scores."""
    return evaluate_vega_mig_run(
        run_dir,
        data_dir=data_dir,
        gmt_path=gmt_path,
        max_cells=max_cells,
        max_pathways=max_pathways,
        pathway_scores=pathway_scores,
        pathway_order=pathway_order,
        pathway_enrichment_source=enrichment_source,
        n_jobs=n_jobs,
        device=device,
        verbose=verbose,
    )


def build_shared_enrichment_for_sweep(
    args: argparse.Namespace,
    *,
    force_recompute: bool = False,
    verbose: bool = True,
) -> tuple:
    """One gp.prerank pass on the test split; same scores for every fcn_* model."""
    adata, column_labels_name = load_pbmc_8k(args.data_dir)
    eval_seed = 42
    ctx = create_vega_test_eval_context(
        adata=adata,
        pathway_file=args.gmt_path,
        column_labels_name=column_labels_name,
        n_top_genes=2000,
        train_size=0.9,
        random_seed=eval_seed,
    )
    adata_test = ctx["adata_test"]
    adata_eval, _ = subsample_adata(adata_test, args.max_cells, seed=eval_seed)
    pathway_names = pathway_names_from_list(ctx["list_pathways"])

    shared_dir = shared_enrichment_dir(args.sweep_root)
    scores, order = compute_or_load_shared_pathway_enrichment(
        adata_eval,
        pathway_names,
        args.gmt_path,
        shared_dir,
        permutation_num=args.permutation_num,
        enrich_n_jobs=args.enrich_jobs,
        prerank_seed=eval_seed,
        force_recompute=force_recompute,
        verbose=verbose,
    )
    source = os.path.abspath(shared_dir)
    return adata_eval, scores, order, source


def load_summary_row(run_dir: str) -> Optional[dict]:
    path = mig_summary_path(run_dir)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        row = json.load(f)
    row["run_dir"] = os.path.abspath(run_dir)
    return row


def aggregate_summaries(sweep_root: str, percents: List[int]) -> pd.DataFrame:
    rows = []
    for pct in percents:
        run_dir = run_dir_for_percent(sweep_root, pct)
        row = load_summary_row(run_dir)
        if row is None:
            print("[aggregate] missing mig_summary: %s" % run_dir)
            continue
        row["fcn_percent"] = pct
        row["fcn_dir"] = fcn_dir_name(pct)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_per_concept(sweep_root: str, percents: List[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for pct in percents:
        run_dir = run_dir_for_percent(sweep_root, pct)
        csv_path = os.path.join(run_dir, "mig_metrics", "mig_per_concept.csv")
        if not os.path.isfile(csv_path):
            print("[aggregate] missing %s" % csv_path)
            continue
        df = pd.read_csv(csv_path)
        summary = load_summary_row(run_dir) or {}
        df["fcn_percent"] = pct
        df["fcn_dir"] = fcn_dir_name(pct)
        df["fully_connected_neuron_fraction"] = summary.get(
            "fully_connected_neuron_fraction", fcn_fraction(pct)
        )
        df["mig_dataset"] = summary.get("mig", np.nan)
        df["n_cells"] = summary.get("n_cells", np.nan)
        metrics_path = os.path.join(run_dir, "metrics.json")
        if os.path.isfile(metrics_path):
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
            df["n_fully_connected_neurons_selected"] = metrics.get(
                "n_fully_connected_neurons_selected",
                int(round(fcn_fraction(pct) * 674)),
            )
            df["n_pathway_nodes"] = metrics.get("n_pathway_nodes", 674)
        else:
            df["n_fully_connected_neurons_selected"] = int(round(fcn_fraction(pct) * 674))
            df["n_pathway_nodes"] = 674
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_mig_by_fcn(per_concept_df: pd.DataFrame, out_path: str) -> None:
    """
    Mean MIG per FC threshold only (red line), same axes/labels as distance-corr plot.
    """
    if per_concept_df.empty or "per_concept_mig" not in per_concept_df.columns:
        print("[plot] No per-concept MIG data; skipping.")
        return

    group_col = "n_fully_connected_neurons_selected"
    groups = (
        per_concept_df.groupby(group_col, sort=True)
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

    mig_col = "per_concept_mig"
    means = np.array(
        [
            float(np.nanmean(
                per_concept_df.loc[per_concept_df[group_col] == n, mig_col].dropna().values
            ))
            for n in groups[group_col]
        ],
        dtype=float,
    )
    valid_means = means[np.isfinite(means)]
    if len(valid_means) == 0:
        ylim_mig = (0.0, 1.0)
    else:
        span = float(np.max(valid_means) - np.min(valid_means))
        pad = max(0.15 * span, 0.00025)
        ylim_mig = (float(np.min(valid_means)) - pad, float(np.max(valid_means)) + pad)

    plt.figure(figsize=(max(14, 1.2 * len(positions)), 6))
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
    plt.xticks(positions, x_labels, rotation=45, ha="right")
    plt.ylim(ylim_mig)
    plt.xlabel("Fully connected neurons selected")
    plt.ylabel("MIG score per pathway (paper Eq. 6)")
    plt.title(
        "MIG vs fully connected neuron selection\n"
        "(mean over pathways; 0 FC = sparse decoder, max FC = dense decoder)"
    )
    plt.legend(loc="best")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print("Saved plot: %s" % out_path)


def write_run_parameters(
    path: str,
    args: argparse.Namespace,
    summaries: Optional[pd.DataFrame],
) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("VEGA MIG sweep (full test set)\n")
        f.write("=" * 60 + "\n")
        f.write("Generated (UTC): %s\n" % datetime.now(timezone.utc).isoformat())
        f.write("Sweep root: %s\n" % os.path.abspath(args.sweep_root))
        f.write("FCN percents: %s\n" % ",".join(str(p) for p in args._percents))
        f.write(
            "max_cells: %s\n"
            % ("all test" if args.max_cells is None else args.max_cells)
        )
        f.write("data_dir: %s\n" % args.data_dir)
        f.write("gmt_path: %s\n" % args.gmt_path)
        f.write("enrichment_method: %s\n" % ENRICHMENT_METHOD)
        f.write("n_genes (ranked per cell): all adata vars (typically 2000)\n")
        f.write("permutation_num: %d\n" % args.permutation_num)
        f.write("enrich_jobs (parallel prerank): %d\n" % args.enrich_jobs)
        f.write("n_jobs (parallel MI): %d\n" % args.n_jobs)
        f.write("mig_definition: paper Eq. (6), mean over factors\n")
        f.write(
            "shared_enrichment_dir: %s\n"
            % os.path.join(os.path.abspath(args.sweep_root), "shared_mig_enrichment")
        )
        f.write("shared_enrichment: enabled (one gp.prerank for all fcn_*)\n")
        if summaries is not None and not summaries.empty:
            f.write("\nDataset MIG by fraction\n")
            f.write("-" * 40 + "\n")
            for _, r in summaries.sort_values("fcn_percent").iterrows():
                f.write(
                    "  fcn_%03d: mig=%.6f n_cells=%s\n"
                    % (int(r["fcn_percent"]), float(r["mig"]), r.get("n_cells", ""))
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MIG on full VEGA test set for each vega_fcn_sweep fraction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sweep-root", type=str, default=SWEEP_ROOT)
    p.add_argument("--data-dir", type=str, default="pbmc_data")
    p.add_argument("--gmt-path", type=str, default="vega/vega/data/reactomes.gmt")
    p.add_argument(
        "--permutation-num",
        type=int,
        default=1000,
        help="Permutations per gp.prerank call (shared enrichment pass).",
    )
    p.add_argument(
        "--enrich-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-cell gp.prerank (-1 = all CPUs, 1 = serial).",
    )
    p.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-neuron MI (-1 = all CPUs).",
    )
    p.add_argument(
        "--percents",
        type=str,
        default=None,
        help="Comma-separated FC percents (default: 0,10,...,100).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip run if mig_metrics/mig_summary.json already exists.",
    )
    p.add_argument(
        "--max-pathways",
        type=int,
        default=None,
        help="Debug: limit pathways passed to MIG.",
    )
    p.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip per-run MIG (only aggregate/plot from existing mig_metrics/).",
    )
    p.add_argument(
        "--no-aggregate",
        action="store_true",
        help="Skip writing mig_aggregate/ CSVs.",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip 01_mig_by_fcn.png boxplot.",
    )
    p.add_argument("-q", "--quiet", action="store_true", help="Less per-cell enrich logging.")
    p.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Subsample test cells (applies to shared enrich + all fractions).",
    )
    p.add_argument(
        "--force-recompute-enrichment",
        action="store_true",
        help="Ignore cached shared_mig_enrichment and rerun gp.prerank.",
    )
    p.add_argument(
        "--no-shared-enrichment",
        action="store_true",
        help="Run gp.prerank separately for each fcn_* (slow; old behaviour).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    do_eval = not args.no_eval
    do_aggregate = not args.no_aggregate
    do_plot = not args.no_plot

    percents = FCN_PERCENTS
    if args.percents:
        percents = [int(x.strip()) for x in args.percents.split(",") if x.strip()]
    args._percents = percents

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)
    print(
        "Cells: %s. Fractions: %s"
        % ("all test" if args.max_cells is None else args.max_cells, percents)
    )

    os.makedirs(args.sweep_root, exist_ok=True)
    aggregate_dir = os.path.join(args.sweep_root, AGGREGATE_DIR)

    pathway_scores = None
    pathway_order: List[str] = []
    enrichment_source = "per_run"
    n_cells_expected = 0
    n_genes_expected = 0

    if do_eval and not args.no_shared_enrichment:
        adata_eval, pathway_scores, pathway_order, enrichment_source = (
            build_shared_enrichment_for_sweep(
                args,
                force_recompute=args.force_recompute_enrichment,
                verbose=not args.quiet,
            )
        )
        n_cells_expected = int(pathway_scores.shape[0])
        n_genes_expected = int(adata_eval.n_vars)
        print(
            "[shared enrichment] %d cells × %d pathways (%d genes/cell, gp.prerank)"
            % (n_cells_expected, pathway_scores.shape[1], n_genes_expected),
            flush=True,
        )
    elif do_eval:
        print("[warn] --no-shared-enrichment: gp.prerank will run for every fcn_* folder.")

    if do_eval:
        for pct in percents:
            run_dir = run_dir_for_percent(args.sweep_root, pct)
            if not os.path.isdir(run_dir):
                print("[warn] missing: %s" % run_dir)
                continue
            try:
                find_model_checkpoint(run_dir)
            except FileNotFoundError as exc:
                print("[warn] %s — skip" % exc)
                continue

            if args.skip_existing:
                if n_cells_expected > 0 and mig_summary_is_current(
                    run_dir,
                    n_cells=n_cells_expected,
                    n_genes=n_genes_expected,
                ):
                    print(
                        "[skip] %s (mig_summary matches %d cells, prerank n_genes=%d)"
                        % (fcn_dir_name(pct), n_cells_expected, n_genes_expected)
                    )
                    continue
                if n_cells_expected == 0 and os.path.isfile(mig_summary_path(run_dir)):
                    print("[skip] %s (mig_summary.json exists)" % fcn_dir_name(pct))
                    continue

            print("\n=== MIG %s (FC %d%%) ===" % (fcn_dir_name(pct), pct))
            evaluate_one_mig(
                run_dir,
                data_dir=args.data_dir,
                gmt_path=args.gmt_path,
                max_pathways=args.max_pathways,
                max_cells=args.max_cells,
                pathway_scores=pathway_scores,
                pathway_order=pathway_order,
                enrichment_source=enrichment_source,
                n_jobs=args.n_jobs,
                device=device,
                verbose=not args.quiet,
            )

    summaries = aggregate_summaries(args.sweep_root, percents) if do_aggregate else None
    per_concept = aggregate_per_concept(args.sweep_root, percents) if do_aggregate else None

    if do_aggregate and summaries is not None and not summaries.empty:
        os.makedirs(aggregate_dir, exist_ok=True)
        sum_path = os.path.join(aggregate_dir, "mig_summaries.csv")
        summaries.sort_values("fcn_percent").to_csv(sum_path, index=False)
        print("Saved: %s" % sum_path)

    if do_aggregate and per_concept is not None and not per_concept.empty:
        os.makedirs(aggregate_dir, exist_ok=True)
        pc_path = os.path.join(aggregate_dir, "per_concept_mig_all.csv")
        per_concept.to_csv(pc_path, index=False)
        print("Saved: %s" % pc_path)

    if do_plot and per_concept is not None and not per_concept.empty:
        os.makedirs(aggregate_dir, exist_ok=True)
        plot_mig_by_fcn(
            per_concept,
            os.path.join(aggregate_dir, "01_mig_by_fcn.png"),
        )

    if do_aggregate:
        os.makedirs(aggregate_dir, exist_ok=True)
        write_run_parameters(
            os.path.join(aggregate_dir, "run_parameters.txt"),
            args,
            summaries,
        )

    print("\nDone. Per-run outputs: <sweep>/fcn_XXX/mig_metrics/")
    print("Aggregate: %s" % os.path.abspath(aggregate_dir))


if __name__ == "__main__":
    main()
