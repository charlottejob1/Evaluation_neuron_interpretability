"""
MIG between VEGA latent neurons and Reactome pathway enrichment scores.

Pipeline
--------
1. Extract latent neuron activations ``z`` from a trained VEGA2 model (``to_latent``).
2. For each cell, rank all dataset genes by that cell's expression and run
   ``gseapy.prerank`` against the Reactome ``.gmt`` gene sets (no top-gene cutoff).
3. Pathway score matrix ``P`` (cells × pathways): NES per GMT term.
4. Compute paper MIG (Eq. 6) via ``compute_mig``: per-factor gap over neurons,
   normalized by H(v_k), with prerank enrichment as v_k and latent neurons as z_j.

Enrichment is expression-only and cached once under ``shared_mig_enrichment/`` for
all VEGA ``fcn_*`` models.

Requires: ``gseapy`` (``pip install gseapy``) in the VEGA environment.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_NEURON_SIM = os.path.join(_REPO_ROOT, "neuron_simulation")
_TRAIN_DIR = os.path.join(_REPO_ROOT, "test_vega_simulation")
for _p in (_REPO_ROOT, _NEURON_SIM, _TRAIN_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mig_metrics_simulation import compute_mig
from train_vega_pbmc import adata_to_array
from vega_fcn_metrics import (
    embeddings_dataframe,
    extract_latent_embeddings,
    load_trained_vega2,
)

DEFAULT_GMT = os.path.join(_REPO_ROOT, "vega", "vega", "data", "reactomes.gmt")
SHARED_ENRICHMENT_SUBDIR = "shared_mig_enrichment"
ENRICHMENT_METHOD = "prerank"
DEFAULT_PERMUTATION_NUM = 1000


def _require_gseapy():
    try:
        import gseapy as gp  # noqa: WPS433
    except ImportError as exc:
        raise ImportError(
            "gseapy is required for pathway enrichment. Install with: pip install gseapy"
        ) from exc
    return gp


def pathway_names_from_list(list_pathways: Sequence[str]) -> List[str]:
    """Reactome pathway names only (exclude VEGA UNANNOTATED latent nodes)."""
    return [p for p in list_pathways if not str(p).startswith("UNANNOTATED")]


def cell_expression_ranking(
    expression_row: np.ndarray,
    gene_names: Sequence[str],
) -> pd.Series:
    """
    Pre-ranked gene list for ``gp.prerank``: all genes, highest expression first.
    """
    x = np.asarray(expression_row, dtype=float).ravel()
    if x.size != len(gene_names):
        raise ValueError("expression_row length must match gene_names.")
    rnk = pd.Series(x, index=[str(g) for g in gene_names])
    return rnk.sort_values(ascending=False)


def prerank_one_cell(
    rnk: pd.Series,
    gmt_path: str,
    *,
    permutation_num: int = DEFAULT_PERMUTATION_NUM,
    min_size: int = 15,
    max_size: int = 500,
    seed: Optional[int] = None,
    threads: int = 1,
) -> pd.DataFrame:
    """
    Run ``gseapy.prerank`` for one cell's expression ranking vs Reactome GMT.

    Returns a DataFrame with at least columns ``Term`` and ``NES``.
    """
    gp = _require_gseapy()
    if rnk.empty:
        return pd.DataFrame(columns=["Term", "NES"])

    pre = gp.prerank(
        rnk=rnk,
        gene_sets=os.path.abspath(gmt_path),
        outdir=None,
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        ascending=False,
        threads=threads,
        no_plot=True,
        seed=seed if seed is not None else 123,
        verbose=False,
    )

    if hasattr(pre, "res2d") and pre.res2d is not None:
        df = pre.res2d.copy()
    elif hasattr(pre, "results") and pre.results is not None:
        df = pre.results.copy()
    else:
        return pd.DataFrame(columns=["Term", "NES"])

    if "Term" not in df.columns:
        for alt in ("term", "NAME"):
            if alt in df.columns:
                df = df.rename(columns={alt: "Term"})
                break

    return df


def scores_from_gsea_table(
    gsea_df: pd.DataFrame,
    pathway_names: Sequence[str],
    *,
    score_column: str = "NES",
) -> np.ndarray:
    """Map GSEA/prerank output to a fixed pathway order; missing terms → 0."""
    if gsea_df.empty or "Term" not in gsea_df.columns:
        return np.zeros(len(pathway_names), dtype=float)

    col = score_column
    if col not in gsea_df.columns:
        for alt in ("NES", "ES", "Combined Score", "FDR q-val"):
            if alt in gsea_df.columns:
                col = alt
                break
        else:
            col = gsea_df.columns[-1]

    if col in ("FDR q-val", "NOM p-val", "Adjusted P-value"):
        p = np.clip(gsea_df[col].astype(float), 1e-300, 1.0)
        values = -np.log10(p)
    else:
        values = gsea_df[col].astype(float)

    lookup = dict(zip(gsea_df["Term"].astype(str), values))
    return np.array([float(lookup.get(name, 0.0)) for name in pathway_names], dtype=float)


def _prerank_scores_for_cell(
    cell_index: int,
    expression_row: np.ndarray,
    gene_names: Sequence[str],
    gmt_path: str,
    pathway_names: Sequence[str],
    *,
    permutation_num: int,
    prerank_seed: Optional[int],
) -> Tuple[int, np.ndarray]:
    """One cell row of the pathway score matrix (for parallel execution)."""
    rnk = cell_expression_ranking(expression_row, gene_names)
    cell_seed = None if prerank_seed is None else int(prerank_seed) + cell_index
    gsea_df = prerank_one_cell(
        rnk,
        gmt_path,
        permutation_num=permutation_num,
        seed=cell_seed,
        threads=1,
    )
    return cell_index, scores_from_gsea_table(gsea_df, pathway_names)


def shared_enrichment_dir(sweep_root: str) -> str:
    return os.path.join(sweep_root, SHARED_ENRICHMENT_SUBDIR)


def _shared_enrichment_paths(
    shared_dir: str,
    n_cells: int,
    n_genes: int,
    *,
    method: str = ENRICHMENT_METHOD,
) -> Dict[str, str]:
    tag = "cells%d_%s_n%d" % (n_cells, method, n_genes)
    return {
        "scores": os.path.join(shared_dir, "pathway_enrichment_%s.npy" % tag),
        "pathway_names": os.path.join(shared_dir, "pathway_names_%s.json" % tag),
        "meta": os.path.join(shared_dir, "enrichment_meta_%s.json" % tag),
    }


def load_shared_pathway_enrichment(
    shared_dir: str,
    n_cells: int,
    n_genes: int,
    *,
    method: str = ENRICHMENT_METHOD,
) -> Optional[Tuple[np.ndarray, List[str], dict]]:
    """Load cached prerank matrix if present and consistent."""
    paths = _shared_enrichment_paths(shared_dir, n_cells, n_genes, method=method)
    if not (os.path.isfile(paths["scores"]) and os.path.isfile(paths["pathway_names"])):
        return None
    with open(paths["pathway_names"], encoding="utf-8") as f:
        pathway_names = json.load(f)
    scores = np.load(paths["scores"])
    meta = {}
    if os.path.isfile(paths["meta"]):
        with open(paths["meta"], encoding="utf-8") as f:
            meta = json.load(f)
    if scores.shape[0] != n_cells or scores.shape[1] != len(pathway_names):
        return None
    return scores, pathway_names, meta


def save_shared_pathway_enrichment(
    shared_dir: str,
    scores: np.ndarray,
    pathway_names: List[str],
    *,
    n_cells: int,
    n_genes: int,
    gmt_path: str,
    method: str = ENRICHMENT_METHOD,
    permutation_num: int = DEFAULT_PERMUTATION_NUM,
    extra_meta: Optional[dict] = None,
) -> Dict[str, str]:
    os.makedirs(shared_dir, exist_ok=True)
    paths = _shared_enrichment_paths(shared_dir, n_cells, n_genes, method=method)
    np.save(paths["scores"], scores)
    with open(paths["pathway_names"], "w", encoding="utf-8") as f:
        json.dump(pathway_names, f, indent=2)
    meta = {
        "n_cells": n_cells,
        "n_pathways": len(pathway_names),
        "n_genes": n_genes,
        "enrichment_method": method,
        "permutation_num": permutation_num,
        "gmt_path": os.path.abspath(gmt_path),
        "note": "Expression-only gp.prerank (NES) scores; reusable across all VEGA fcn_* models.",
    }
    if extra_meta:
        meta.update(extra_meta)
    with open(paths["meta"], "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return paths


def compute_or_load_shared_pathway_enrichment(
    adata,
    pathway_names: Sequence[str],
    gmt_path: str,
    shared_dir: str,
    *,
    permutation_num: int = DEFAULT_PERMUTATION_NUM,
    enrich_n_jobs: int = 1,
    prerank_seed: Optional[int] = None,
    force_recompute: bool = False,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    gp.prerank scores for ``adata`` (cells × pathways), computed once per shared_dir.

    Pathway enrichment depends only on expression, not on VEGA weights.
    """
    n_cells = int(adata.n_obs)
    n_genes = int(adata.n_vars)
    if not force_recompute:
        cached = load_shared_pathway_enrichment(shared_dir, n_cells, n_genes)
        if cached is not None:
            scores, names, meta = cached
            if meta.get("enrichment_method") == ENRICHMENT_METHOD:
                if verbose:
                    print(
                        "[shared enrichment] Loaded cache (%d cells × %d pathways, %d genes) from %s"
                        % (n_cells, len(names), n_genes, shared_dir),
                        flush=True,
                    )
                return scores, names

    if verbose:
        print(
            "[shared enrichment] Computing gp.prerank for %d cells × %d genes (saved under %s)..."
            % (n_cells, n_genes, shared_dir),
            flush=True,
        )
    scores, order = compute_pathway_enrichment_matrix(
        adata,
        gmt_path=gmt_path,
        pathway_names=pathway_names,
        permutation_num=permutation_num,
        enrich_n_jobs=enrich_n_jobs,
        prerank_seed=prerank_seed,
        verbose=verbose,
    )
    save_shared_pathway_enrichment(
        shared_dir,
        scores,
        order,
        n_cells=n_cells,
        n_genes=n_genes,
        gmt_path=gmt_path,
        permutation_num=permutation_num,
    )
    if verbose:
        print("[shared enrichment] Saved cache.", flush=True)
    return scores, order


def subsample_adata(adata, max_cells: Optional[int], seed: Optional[int] = None):
    """Return (adata_sub, row_indices) for paired latent / enrichment computation."""
    n = adata.n_obs
    if max_cells is None or max_cells >= n:
        return adata, np.arange(n, dtype=int)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(n, size=max_cells, replace=False))
    return adata[idx].copy(), idx


def compute_pathway_enrichment_matrix(
    adata,
    gmt_path: str,
    pathway_names: Sequence[str],
    *,
    permutation_num: int = DEFAULT_PERMUTATION_NUM,
    enrich_n_jobs: int = 1,
    prerank_seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Per-cell pathway enrichment scores from ``gp.prerank`` (cells × pathways).

    For each cell ``n``:
      - rank all ``adata.n_vars`` genes by expression in that cell
      - ``gp.prerank(rnk, gene_sets=reactome.gmt)``
      - score for pathway ``k`` = NES for term ``pathway_names[k]``

    Returns:
      scores: (N, K_pathways)
      pathway_names: list of K pathway names (column order)
    """
    gmt_path = os.path.abspath(gmt_path)
    if not os.path.isfile(gmt_path):
        raise FileNotFoundError("GMT file not found: %s" % gmt_path)

    pathway_names = list(pathway_names)
    X = adata_to_array(adata)
    gene_names = list(adata.var_names)
    n_cells = X.shape[0]

    scores = np.zeros((n_cells, len(pathway_names)), dtype=float)

    if enrich_n_jobs == 1:
        for i in range(n_cells):
            if verbose and (i == 0 or (i + 1) % 25 == 0 or i + 1 == n_cells):
                print("  gp.prerank cell %d / %d" % (i + 1, n_cells), flush=True)
            _, row = _prerank_scores_for_cell(
                i,
                X[i],
                gene_names,
                gmt_path,
                pathway_names,
                permutation_num=permutation_num,
                prerank_seed=prerank_seed,
            )
            scores[i, :] = row
        return scores, pathway_names

    if enrich_n_jobs < 0:
        enrich_n_jobs = os.cpu_count() or 1

    from joblib import Parallel, delayed

    if verbose:
        print(
            "  gp.prerank parallel: %d cells, %d workers" % (n_cells, enrich_n_jobs),
            flush=True,
        )
    results = Parallel(n_jobs=enrich_n_jobs, prefer="processes")(
        delayed(_prerank_scores_for_cell)(
            i,
            X[i],
            gene_names,
            gmt_path,
            pathway_names,
            permutation_num=permutation_num,
            prerank_seed=prerank_seed,
        )
        for i in range(n_cells)
    )
    for i, row in results:
        scores[i, :] = row

    return scores, pathway_names


def align_latent_to_pathways(
    latent: np.ndarray,
    list_pathways: Sequence[str],
    pathway_names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Select latent columns that match ``pathway_names`` (same order as enrichment).
    """
    names = pathway_names_from_list(list_pathways)
    name_to_col = {p: i for i, p in enumerate(names)}
    if latent.shape[1] > len(names):
        latent = latent[:, : len(names)]

    cols = []
    kept = []
    for p in pathway_names:
        j = name_to_col.get(p)
        if j is None:
            continue
        cols.append(j)
        kept.append(p)

    if len(cols) == 0:
        raise ValueError("No overlap between latent pathway names and enrichment pathways.")

    return latent[:, cols], kept


def compute_vega_mig(
    model,
    adata,
    pathway_dict: Dict[str, List[str]],
    list_pathways: Sequence[str],
    *,
    gmt_path: str = DEFAULT_GMT,
    permutation_num: int = DEFAULT_PERMUTATION_NUM,
    enrich_n_jobs: int = 1,
    max_cells: Optional[int] = None,
    pathway_scores: Optional[np.ndarray] = None,
    pathway_order: Optional[Sequence[str]] = None,
    n_bins: int = 20,
    random_state: Optional[int] = None,
    n_neighbors: int = 3,
    n_jobs: int = 1,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    MIG between VEGA latent neurons and GSEApy Reactome prerank scores.

    If ``pathway_scores`` and ``pathway_order`` are provided (e.g. from
    ``compute_or_load_shared_pathway_enrichment``), gp.prerank is skipped.

    Returns the same dict as ``compute_mig``, plus VEGA-specific fields.
    """
    pathway_names = pathway_names_from_list(list_pathways)
    adata_eval, _ = subsample_adata(adata, max_cells, seed=random_state)

    if verbose:
        print(
            "Extracting latent embeddings (VEGA to_latent) on %d cells..."
            % adata_eval.n_obs,
            flush=True,
        )
    latent = extract_latent_embeddings(model, adata_eval)

    if pathway_scores is not None and pathway_order is not None:
        if pathway_scores.shape[0] != adata_eval.n_obs:
            raise ValueError(
                "pathway_scores rows (%d) must match adata cells (%d)."
                % (pathway_scores.shape[0], adata_eval.n_obs)
            )
        pathway_order = list(pathway_order)
        if verbose:
            print(
                "Using precomputed pathway enrichment (%d cells × %d pathways)."
                % pathway_scores.shape,
                flush=True,
            )
    else:
        if verbose:
            print(
                "Computing per-cell pathway enrichment (gp.prerank × %d cells)..."
                % adata_eval.n_obs,
                flush=True,
            )
        pathway_scores, pathway_order = compute_pathway_enrichment_matrix(
            adata_eval,
            gmt_path=gmt_path,
            pathway_names=pathway_names,
            permutation_num=permutation_num,
            enrich_n_jobs=enrich_n_jobs,
            prerank_seed=random_state,
            verbose=verbose,
        )

    z_aligned, shared_names = align_latent_to_pathways(
        latent, list_pathways, pathway_order
    )
    name_to_score_col = {p: i for i, p in enumerate(pathway_order)}
    score_cols = [name_to_score_col[p] for p in shared_names]
    score_aligned = pathway_scores[:, score_cols]

    if verbose:
        nj = n_jobs if n_jobs > 0 else "all CPUs"
        print(
            "Computing MIG on aligned matrices: %d cells, %d pathways/neurons (n_jobs=%s)"
            % (score_aligned.shape[0], len(shared_names), nj),
            flush=True,
        )

    mig_out = compute_mig(
        score_aligned,
        z_aligned,
        n_bins=n_bins,
        random_state=random_state,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )

    mig_out["pathway_names"] = shared_names
    mig_out["pathway_enrichment_scores"] = score_aligned
    mig_out["latent_activations"] = z_aligned
    mig_out["gmt_path"] = os.path.abspath(gmt_path)
    mig_out["n_cells"] = int(score_aligned.shape[0])
    mig_out["n_genes"] = int(adata_eval.n_vars)
    mig_out["enrichment_method"] = ENRICHMENT_METHOD
    mig_out["pathway_dict"] = pathway_dict
    return mig_out


def evaluate_vega_mig_run(
    run_dir: str,
    *,
    data_dir: str = "pbmc_data",
    gmt_path: str = DEFAULT_GMT,
    max_cells: Optional[int] = None,
    max_pathways: Optional[int] = None,
    save_dir: Optional[str] = None,
    pathway_scores: Optional[np.ndarray] = None,
    pathway_order: Optional[Sequence[str]] = None,
    pathway_enrichment_source: Optional[str] = None,
    n_jobs: int = 1,
    device=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """
    Load a trained VEGA2 checkpoint and compute MIG on the test split.
    """
    model, eval_ctx, meta = load_trained_vega2(
        run_dir, data_dir=data_dir, device=device
    )
    gmt_path = meta.get("hyperparameters", {}).get("gmt_path", gmt_path)

    list_pathways = eval_ctx["list_pathways"]
    if max_pathways is not None:
        trimmed = pathway_names_from_list(list_pathways)[:max_pathways]
        list_pathways = trimmed + [
            p for p in list_pathways if str(p).startswith("UNANNOTATED")
        ]

    seed = int(meta.get("hyperparameters", {}).get("seed", 42))
    result = compute_vega_mig(
        model,
        eval_ctx["adata_test"],
        eval_ctx["pathway_dict"],
        list_pathways,
        gmt_path=gmt_path,
        max_cells=max_cells,
        pathway_scores=pathway_scores,
        pathway_order=pathway_order,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    result["run_dir"] = os.path.abspath(run_dir)
    result["fully_connected_neuron_fraction"] = meta.get("fully_connected_neuron_fraction")
    result["eval_split"] = "test"

    out_dir = save_dir or os.path.join(run_dir, "mig_metrics")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "mig": result["mig"],
        "n_cells": result["n_cells"],
        "n_pathways": len(result["pathway_names"]),
        "gmt_path": result["gmt_path"],
        "run_dir": result["run_dir"],
        "fully_connected_neuron_fraction": result["fully_connected_neuron_fraction"],
        "per_concept_mig_mean": float(np.nanmean(result["per_concept_mig"])),
        "per_concept_mig_median": float(np.nanmedian(result["per_concept_mig"])),
        "mig_definition": "paper_eq6_mean_over_factors",
        "pathway_enrichment_source": pathway_enrichment_source or "per_run",
        "enrichment_method": result.get("enrichment_method", ENRICHMENT_METHOD),
        "n_genes": int(result.get("n_genes", 0)),
        "n_jobs_mi": n_jobs,
    }
    with open(os.path.join(out_dir, "mig_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pathway_names = list(result["pathway_names"])
    concept_entropies = np.asarray(result["concept_entropies"], dtype=float)
    top_neuron_idx = np.asarray(result["top_neuron_per_concept"], dtype=int)
    pd.DataFrame(
        {
            "pathway": pathway_names,
            "per_concept_mig": result["per_concept_mig"],
            "top_neuron_index": top_neuron_idx,
            "top_neuron_pathway": [pathway_names[i] for i in top_neuron_idx],
            "concept_entropy": concept_entropies,
        }
    ).to_csv(os.path.join(out_dir, "mig_per_concept.csv"), index=False)

    top_concept_idx = np.asarray(result["top_concept_per_neuron"], dtype=int)
    pd.DataFrame(
        {
            "pathway": pathway_names,
            "per_neuron_mig": result["per_neuron_mig"],
            "top_pathway_index": top_concept_idx,
            "matched_pathway_entropy": concept_entropies,
            "top_pathway_entropy": concept_entropies[top_concept_idx],
        }
    ).to_csv(os.path.join(out_dir, "mig_per_neuron.csv"), index=False)

    np.save(os.path.join(out_dir, "mi_matrix.npy"), result["mi_matrix"])
    np.save(
        os.path.join(out_dir, "pathway_enrichment_scores.npy"),
        result["pathway_enrichment_scores"],
    )
    np.save(os.path.join(out_dir, "latent_activations.npy"), result["latent_activations"])

    if verbose:
        print("Saved MIG outputs to %s" % out_dir)
        print("  dataset MIG = %.6f" % result["mig"])

    result["save_dir"] = out_dir
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MIG for VEGA latent neurons vs Reactome gp.prerank pathway scores.",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="VEGA training run directory (metrics.json + vega2_pbmc_fcn*.pt).",
    )
    parser.add_argument("--data-dir", type=str, default="pbmc_data")
    parser.add_argument("--gmt-path", type=str, default=DEFAULT_GMT)
    parser.add_argument(
        "--max-cells",
        type=int,
        default=None,
        help="Subsample cells for gp.prerank (each cell is one prerank call).",
    )
    parser.add_argument(
        "--max-pathways",
        type=int,
        default=None,
        help="Use only the first N Reactome pathways (debug / faster runs).",
    )
    parser.add_argument(
        "--enrich-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-cell gp.prerank (-1 = all CPUs, 1 = serial).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers for per-neuron MI (-1 = all CPUs, 1 = serial).",
    )
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_vega_mig_run(
        args.run_dir,
        data_dir=args.data_dir,
        gmt_path=args.gmt_path,
        max_cells=args.max_cells,
        max_pathways=args.max_pathways,
        n_jobs=args.n_jobs,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
