"""
Interpretability metrics for trained VEGA2 models on PBMC.

Follows the ``vega_usage/`` workflow:
  - latent embeddings via ``model.to_latent`` (as in ``vega_perturbation_copy.py``)
  - inhibition perturbation (pathway genes set to 0)
  - reduction scores + probabilities from ``vega_usage/probability_metrics.py``
  - overlap matrix from the pathway **mask** (``vega_utils_copy.build_overlap_matrix_Vega``)
  - distance correlation from ``distances_metrics.compute_distance_corr_one_pathway_one_dim``
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_TRAIN_DIR = os.path.join(_REPO_ROOT, "test_vega_simulation")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from train_vega_pbmc import (
    adata_to_array,
    apply_fully_connected_neuron_fraction_to_mask,
    create_vega_test_eval_context,
    load_pathway_context,
    load_pbmc_8k,
)
from vega_interpretability_simulation import VEGA2

DEFAULT_OVERLAP_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_run_metadata(run_dir: str) -> dict:
    path = os.path.join(run_dir, "metrics.json")
    if not os.path.isfile(path):
        raise FileNotFoundError("Missing metrics.json in %s" % run_dir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_training_var_names(run_dir: str) -> Optional[List[str]]:
    """Gene order used at training time (``training_var_names.json``), if saved."""
    path = os.path.join(run_dir, "training_var_names.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        names = json.load(f)
    return list(names) if names else None


def _align_adata_to_training_genes(adata, training_var_names: List[str]):
    """Subset and order genes to match the trained VEGA2 checkpoint."""
    missing = [g for g in training_var_names if g not in adata.var_names]
    if missing:
        raise ValueError(
            "%d training genes missing from eval adata (e.g. %s). "
            "Use the same PBMC preprocessing / data release as training."
            % (len(missing), missing[:3])
        )
    return adata[:, training_var_names].copy()


def find_model_checkpoint(run_dir: str) -> str:
    for name in os.listdir(run_dir):
        if name.startswith("vega2_pbmc_fcn") and name.endswith(".pt"):
            return os.path.join(run_dir, name)
    raise FileNotFoundError("No vega2_pbmc_fcn*.pt checkpoint in %s" % run_dir)


def load_trained_vega2(
    run_dir: str,
    *,
    data_dir: str = "pbmc_data",
    gmt_path: str = "vega/vega/data/reactomes.gmt",
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> Tuple[VEGA2, dict, dict]:
    meta = load_run_metadata(run_dir)
    hp = meta["hyperparameters"]
    fcn = float(meta["fully_connected_neuron_fraction"])
    seed = int(hp.get("seed", seed))
    gmt_path = hp.get("gmt_path", gmt_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    expected_n_genes = int(meta.get("n_genes", hp.get("n_top_genes", 2000)))

    adata, column_labels_name = load_pbmc_8k(data_dir)
    training_var_names = _load_training_var_names(run_dir)
    eval_ctx = create_vega_test_eval_context(
        adata=adata,
        pathway_file=gmt_path,
        column_labels_name=column_labels_name,
        n_top_genes=int(hp.get("n_top_genes", 2000)),
        train_size=float(hp.get("train_size", 0.9)),
        random_seed=seed,
    )
    mask, _ = apply_fully_connected_neuron_fraction_to_mask(
        eval_ctx["pathway_mask"],
        fully_connected_neuron_fraction=fcn,
        add_nodes=1,
        seed=seed,
    )

    model_path = find_model_checkpoint(run_dir)
    model = VEGA2(
        pathway_mask=mask,
        positive_decoder=True,
        device=device,
        beta=float(hp.get("kld_weight", 1e-4)),
        dropout=float(hp.get("dropout", 0.1)),
        save_path=model_path,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, eval_ctx, meta


# ---------------------------------------------------------------------------
# Inference (vega_usage/vega_perturbation_copy.py)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_latent_embeddings(model: VEGA2, adata) -> np.ndarray:
    """Sampled latent z via ``to_latent`` (matches original VEGA pipeline)."""
    X = torch.tensor(adata_to_array(adata), dtype=torch.float32, device=model.dev)
    model.eval()
    return model.to_latent(X).cpu().numpy()


def embeddings_dataframe(mu: np.ndarray, list_pathways: List[str]) -> pd.DataFrame:
    return pd.DataFrame(mu, columns=list_pathways[: mu.shape[1]])


def perturb_pathway_inhibition(
    adata,
    pathway_dict: Dict[str, List[str]],
    pathway: str,
):
    """Inhibition perturbation: zero expression of pathway genes (one_vs_all)."""
    genes = [g for g in pathway_dict.get(pathway, []) if g in adata.var_names]
    if not genes:
        return adata.copy()
    ad = adata.copy()
    X = adata_to_array(ad)
    name_to_idx = {g: i for i, g in enumerate(ad.var_names)}
    for g in genes:
        X[:, name_to_idx[g]] = 0.0
    ad.X = X
    return ad


def build_overlap_matrix_from_mask(
    pathway_mask: np.ndarray,
    adata,
    list_pathways: List[str],
) -> pd.DataFrame:
    """Same logic as ``vega_utils_copy.build_overlap_matrix_Vega``."""
    df = pd.DataFrame(pathway_mask, columns=list_pathways, index=adata.var_names)
    pathway_dict_dataset = {
        name: df.index[df[name] == 1].tolist() for name in df.columns
    }
    rows = []
    for pathway_selected in list_pathways:
        genes1 = [g for g in pathway_dict_dataset[pathway_selected] if g in adata.var_names]
        for pathway_compared, genes in pathway_dict_dataset.items():
            genes2 = [g for g in genes if g in adata.var_names]
            intersection = set(genes1) & set(genes2)
            rows.append(
                {
                    "Pathway Selected": pathway_selected,
                    "Compared Pathway": pathway_compared,
                    "Genes Overlap": len(intersection),
                    "Overlap Proportion": (
                        len(intersection) / len(genes1) if len(genes1) > 0 else 0.0
                    ),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distance metric (distances_metrics.py)
# ---------------------------------------------------------------------------

DISTANCE_CORR_MATCH_ATOL = 1e-9


def compute_distance_corr_one_pathway_original(
    pathway_selected: str,
    adata,
    embeddings_original: pd.DataFrame,
    pathway_dict: Dict[str, List[str]],
) -> Optional[float]:
    """
    Port of ``vega_usage/distances_metrics.compute_distance_corr_one_pathway_one_dim``.

    Uses ``adata.X.toarray()`` and ``pathway_dict[pathway_selected]`` exactly as in
    the original script (no heatmaps, no stdout).
    """
    if pathway_selected not in pathway_dict:
        return None

    list_genes_pathway = pathway_dict[pathway_selected]
    if len(list_genes_pathway) == 0:
        return None

    df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    df_pathway = df[[gene for gene in list_genes_pathway if gene in adata.var_names]]
    if df_pathway.empty:
        return None

    if pathway_selected not in embeddings_original.columns:
        return None

    dist_matrix_pathway = pairwise_distances(df_pathway.values, metric="euclidean")
    dist_matrix_neuron = pairwise_distances(
        embeddings_original[pathway_selected].values.reshape(-1, 1),
        metric="euclidean",
    )
    triu_idx = np.triu_indices_from(dist_matrix_pathway, k=1)
    vec1 = dist_matrix_pathway[triu_idx]
    vec2 = dist_matrix_neuron[triu_idx]
    corr, _ = pearsonr(vec1, vec2)
    if corr is None or (isinstance(corr, float) and np.isnan(corr)):
        return None
    return float(corr)


def compute_distance_corr_one_pathway(
    pathway: str,
    adata,
    embeddings_original: pd.DataFrame,
    pathway_dict: Dict[str, List[str]],
) -> Optional[float]:
    list_genes = [g for g in pathway_dict.get(pathway, []) if g in adata.var_names]
    if len(list_genes) == 0 or pathway not in embeddings_original.columns:
        return None

    gene_idx = [list(adata.var_names).index(g) for g in list_genes]
    X_path = adata_to_array(adata)[:, gene_idx]
    dist_input = pairwise_distances(X_path, metric="euclidean")
    dist_neuron = pairwise_distances(
        embeddings_original[pathway].values.reshape(-1, 1), metric="euclidean"
    )
    triu_idx = np.triu_indices_from(dist_input, k=1)
    vec1 = dist_input[triu_idx]
    vec2 = dist_neuron[triu_idx]
    if np.std(vec1) == 0.0 or np.std(vec2) == 0.0:
        return None
    corr, _ = pearsonr(vec1, vec2)
    return float(corr)


# ---------------------------------------------------------------------------
# Probability (vectorized, same logic as vega_usage/probability_metrics.py)
# ---------------------------------------------------------------------------

def unrescale_reduction_score_probability(p: Optional[float]) -> Optional[float]:
    """Inverse of 2*(p-0.5): map rescaled values back to raw P in [0.5, 1]."""
    if p is None:
        return None
    if isinstance(p, float) and np.isnan(p):
        return p
    return float(float(p) / 2.0 + 0.5)


def _build_overlap_lookup(overlap_matrix: pd.DataFrame) -> dict:
    """Map (pathway_selected, compared_pathway) -> overlap proportion."""
    lookup = {}
    for _, row in overlap_matrix.iterrows():
        key = (row["Pathway Selected"], row["Compared Pathway"])
        lookup[key] = float(row["Overlap Proportion"])
    return lookup


def overall_proba_pathway_vectorized(
    pathway_perturbated: str,
    list_pathways: List[str],
    embeddings_original: pd.DataFrame,
    embeddings_perturbated: pd.DataFrame,
    overlap_lookup: dict,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
) -> Optional[float]:
    """
    Fast equivalent of ``overall_proba_one_pathway_perturbated`` with
    ``proba_impact_pathway_perturbation2`` and ``reduction_score``.
    """
    cols = [p for p in list_pathways if p in embeddings_original.columns]
    idx = {p: i for i, p in enumerate(cols)}
    if pathway_perturbated not in idx:
        return None

    orig = embeddings_original[cols].to_numpy()
    pert = embeddings_perturbated[cols].to_numpy()
    reduction = np.abs(orig - pert)
    i_pert = idx[pathway_perturbated]
    self_red = reduction[:, i_pert]

    probas = []
    for pathway_compared in list_pathways:
        if pathway_compared not in idx:
            continue
        overlap_score = overlap_lookup.get((pathway_compared, pathway_perturbated))
        if overlap_score is None or overlap_score >= overlap_threshold:
            continue
        other_red = reduction[:, idx[pathway_compared]]
        probas.append(float(np.mean(self_red > other_red)))

    if not probas:
        return None
    return float(np.mean(probas))


def evaluate_vega2_interpretability(
    model: VEGA2,
    adata_eval,
    pathway_dict: Dict[str, List[str]],
    list_pathways: List[str],
    pathway_mask: np.ndarray,
    *,
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    max_pathways: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Per-neuron metrics on the test set using the original VEGA interpretability pipeline.
    """
    pathways = list_pathways[:-1]  # exclude UNANNOTATED, as in step4 notebook
    if max_pathways is not None:
        pathways = pathways[:max_pathways]

    overlap_matrix = build_overlap_matrix_from_mask(pathway_mask, adata_eval, list_pathways)
    overlap_lookup = _build_overlap_lookup(overlap_matrix)
    embeddings_original = embeddings_dataframe(
        extract_latent_embeddings(model, adata_eval), list_pathways
    )

    records = []
    for i, pathway in enumerate(pathways):
        if verbose and (i + 1) % 25 == 0:
            print("  perturbation %d / %d" % (i + 1, len(pathways)), flush=True)

        dist_corr = compute_distance_corr_one_pathway(
            pathway, adata_eval, embeddings_original, pathway_dict
        )

        adata_pert = perturb_pathway_inhibition(adata_eval, pathway_dict, pathway)
        emb_pert = embeddings_dataframe(
            extract_latent_embeddings(model, adata_pert), list_pathways
        )
        overall_proba = overall_proba_pathway_vectorized(
            pathway,
            list_pathways,
            embeddings_original,
            emb_pert,
            overlap_lookup,
            overlap_threshold=overlap_threshold,
        )

        records.append(
            {
                "pathway": pathway,
                "distance_corr": dist_corr,
                "reduction_score_probability": overall_proba,
            }
        )

    return pd.DataFrame(records)


def _distance_corr_values_match(
    new_val: Optional[float],
    original_val: Optional[float],
    *,
    atol: float = DISTANCE_CORR_MATCH_ATOL,
) -> bool:
    if new_val is None and original_val is None:
        return True
    if new_val is None or original_val is None:
        return False
    return bool(np.isclose(new_val, original_val, rtol=0.0, atol=atol))


def compare_distance_corr_implementations(
    model: VEGA2,
    adata_eval,
    pathway_dict: Dict[str, List[str]],
    list_pathways: List[str],
    *,
    max_pathways: Optional[int] = None,
    saved_metrics_path: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run distance correlation with both implementations on the same latent draw.

    Returns a per-pathway comparison table. If ``saved_metrics_path`` points to
    ``interpretability_metrics.csv``, includes the previously saved ``distance_corr``
    column for cross-check.
    """
    pathways = list_pathways[:-1]
    if max_pathways is not None:
        pathways = pathways[:max_pathways]

    embeddings_original = embeddings_dataframe(
        extract_latent_embeddings(model, adata_eval), list_pathways
    )

    records = []
    for i, pathway in enumerate(pathways):
        if verbose and (i + 1) % 25 == 0:
            print("  distance_corr compare %d / %d" % (i + 1, len(pathways)), flush=True)

        new_val = compute_distance_corr_one_pathway(
            pathway, adata_eval, embeddings_original, pathway_dict
        )
        original_val = compute_distance_corr_one_pathway_original(
            pathway, adata_eval, embeddings_original, pathway_dict
        )
        abs_diff = None
        if new_val is not None and original_val is not None:
            abs_diff = float(abs(new_val - original_val))

        records.append(
            {
                "pathway": pathway,
                "distance_corr_new": new_val,
                "distance_corr_original": original_val,
                "abs_diff": abs_diff,
                "match": _distance_corr_values_match(new_val, original_val),
            }
        )

    df = pd.DataFrame(records)

    if saved_metrics_path and os.path.isfile(saved_metrics_path):
        saved = pd.read_csv(saved_metrics_path)
        if "pathway" in saved.columns and "distance_corr" in saved.columns:
            saved = saved[["pathway", "distance_corr"]].rename(
                columns={"distance_corr": "distance_corr_saved"}
            )
            df = df.merge(saved, on="pathway", how="left")
            df["saved_matches_new"] = [
                _distance_corr_values_match(n, s)
                for n, s in zip(df["distance_corr_new"], df["distance_corr_saved"])
            ]

    return df


def summarize_distance_corr_comparison(df: pd.DataFrame) -> dict:
    both_valid = df["distance_corr_new"].notna() & df["distance_corr_original"].notna()
    diffs = df.loc[both_valid, "abs_diff"].astype(float)
    summary = {
        "n_pathways": int(len(df)),
        "n_both_valid": int(both_valid.sum()),
        "n_exact_match": int(df["match"].sum()),
        "n_mismatch": int((~df["match"]).sum()),
        "mean_abs_diff": float(diffs.mean()) if len(diffs) else float("nan"),
        "max_abs_diff": float(diffs.max()) if len(diffs) else float("nan"),
        "mean_distance_corr_new": float(
            df["distance_corr_new"].dropna().astype(float).mean()
        )
        if df["distance_corr_new"].notna().any()
        else float("nan"),
        "mean_distance_corr_original": float(
            df["distance_corr_original"].dropna().astype(float).mean()
        )
        if df["distance_corr_original"].notna().any()
        else float("nan"),
    }
    if "distance_corr_saved" in df.columns:
        saved_valid = df["distance_corr_saved"].notna() & df["distance_corr_new"].notna()
        summary["n_saved_matches_new"] = int(df.loc[saved_valid, "saved_matches_new"].sum())
        summary["n_saved_mismatch_new"] = int(
            saved_valid.sum() - summary["n_saved_matches_new"]
        )
    return summary


def write_distance_corr_comparison_txt(path: str, summary: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Distance correlation — new vs original implementation\n")
        f.write("=" * 60 + "\n")
        f.write("new: vega_simulation/vega_fcn_metrics.compute_distance_corr_one_pathway\n")
        f.write(
            "original: vega_usage/distances_metrics.compute_distance_corr_one_pathway_one_dim\n\n"
        )
        f.write("Pathways compared: %d\n" % summary["n_pathways"])
        f.write("Both implementations valid: %d\n" % summary["n_both_valid"])
        f.write("Exact match (atol=%.0e): %d\n" % (DISTANCE_CORR_MATCH_ATOL, summary["n_exact_match"]))
        f.write("Mismatch: %d\n\n" % summary["n_mismatch"])
        f.write("mean distance_corr (new): %.6f\n" % summary["mean_distance_corr_new"])
        f.write("mean distance_corr (original): %.6f\n" % summary["mean_distance_corr_original"])
        f.write("mean |new - original|: %.6e\n" % summary["mean_abs_diff"])
        f.write("max |new - original|: %.6e\n" % summary["max_abs_diff"])
        if "n_saved_matches_new" in summary:
            f.write("\nCross-check vs interpretability_metrics.csv distance_corr:\n")
            f.write("  saved matches new recompute: %d\n" % summary["n_saved_matches_new"])
            f.write("  saved differs from new recompute: %d\n" % summary["n_saved_mismatch_new"])
            f.write(
                "  (saved column may differ if CSV was produced with a different latent draw)\n"
            )


def compare_distance_corr_run_directory(
    run_dir: str,
    *,
    data_dir: str = "pbmc_data",
    max_pathways: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_csv: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Compare distance_corr implementations on the test set for one run folder."""
    run_dir = os.path.abspath(run_dir)
    model, eval_ctx, meta = load_trained_vega2(
        run_dir, data_dir=data_dir, device=device
    )
    saved_metrics_path = os.path.join(run_dir, "interpretability_metrics.csv")
    df = compare_distance_corr_implementations(
        model,
        eval_ctx["adata_test"],
        eval_ctx["pathway_dict"],
        eval_ctx["list_pathways"],
        max_pathways=max_pathways,
        saved_metrics_path=saved_metrics_path
        if os.path.isfile(saved_metrics_path)
        else None,
        verbose=verbose,
    )
    df["eval_split"] = "test"
    df["fully_connected_neuron_fraction"] = meta["fully_connected_neuron_fraction"]
    summary = summarize_distance_corr_comparison(df)

    if save_csv:
        csv_path = os.path.join(run_dir, "distance_corr_comparison.csv")
        df.to_csv(csv_path, index=False)
        txt_path = os.path.join(run_dir, "distance_corr_comparison.txt")
        write_distance_corr_comparison_txt(txt_path, summary)
        if verbose:
            print("Saved distance_corr comparison CSV: %s" % csv_path)
            print("Saved distance_corr comparison summary: %s" % txt_path)
            print(
                "Distance corr match: %d / %d pathways (mean |diff|=%.2e)"
                % (
                    summary["n_exact_match"],
                    summary["n_pathways"],
                    summary["mean_abs_diff"],
                )
            )

    return df, summary


def evaluate_run_directory(
    run_dir: str,
    *,
    data_dir: str = "pbmc_data",
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    max_pathways: Optional[int] = None,
    device: Optional[torch.device] = None,
    save_csv: bool = True,
) -> pd.DataFrame:
    """Load checkpoint and compute per-neuron metrics on test set only."""
    model, eval_ctx, meta = load_trained_vega2(
        run_dir, data_dir=data_dir, device=device
    )
    df = evaluate_vega2_interpretability(
        model,
        eval_ctx["adata_test"],
        eval_ctx["pathway_dict"],
        eval_ctx["list_pathways"],
        eval_ctx["pathway_mask"],
        overlap_threshold=overlap_threshold,
        max_pathways=max_pathways,
    )
    df["eval_split"] = "test"
    df["fully_connected_neuron_fraction"] = meta["fully_connected_neuron_fraction"]
    df["n_fully_connected_neurons_selected"] = meta["n_fully_connected_neurons_selected"]
    df["n_pathway_nodes"] = meta["n_pathway_nodes"]
    df["overlap_threshold"] = overlap_threshold
    df["run_dir"] = os.path.abspath(run_dir)

    if save_csv:
        out_csv = os.path.join(run_dir, "interpretability_metrics.csv")
        df.to_csv(out_csv, index=False)
    return df
