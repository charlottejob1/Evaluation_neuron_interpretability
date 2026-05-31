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
    load_pbmc_8k,
)
from vega_interpretability_simulation import VEGA2

# Default from vega_compute_probas_step4_copy.ipynb
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

    adata, column_labels_name = load_pbmc_8k(data_dir)
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
