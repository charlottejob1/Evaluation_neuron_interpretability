"""
Distance-correlation metric on simulated data.

For each concept i, we measure how well the geometry of the input variables linked
to concept i is preserved in the corresponding neuron i activation. Concretely, we
compute the Pearson correlation between:

  - the pairwise (Euclidean) distance matrix of X reduced to the variables linked to
    concept i, i.e. X[:, g(i)], and
  - the pairwise (Euclidean) distance matrix of neuron i, i.e. the activation z_i
    simulated by `simulate_neuron_activations`.

This produces one correlation value per concept (per neuron), analogous to
`compute_distance_corr_one_pathway_one_dim` in `distances_metrics.py` but applied to
the simulated dataset X and the simulated neuron activations Z.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances

from data_generation import generate_structured_dataset
from neuron_activation_simulation import simulate_neuron_activations


def _concept_keys_in_order(concept_map: Dict[str, List[int]]) -> List[str]:
    return sorted(concept_map.keys(), key=lambda k: int(k.split("_")[-1]))


def compute_distance_corr_per_concept(
    X: np.ndarray,
    concept_map: Dict[str, List[int]],
    Z: np.ndarray,
    concept_keys: Optional[List[str]] = None,
) -> List[Tuple[str, float]]:
    """
    Computes one Pearson distance correlation per concept/neuron.

    For each concept i with variables g(i):
      d_input(i)  = pairwise distances of X[:, g(i)]   (input reduced to concept i)
      d_neuron(i) = pairwise distances of Z[:, i]      (activation of neuron i)
      corr(i)     = pearson( upper_triangle(d_input), upper_triangle(d_neuron) )

    Args:
      X: (N, M) simulated dataset.
      concept_map: mapping concept_key -> list of variable indices.
      Z: (N, K) simulated neuron activations (one neuron per concept, same order).
      concept_keys: optional explicit concept ordering; defaults to numeric order.

    Returns:
      List of (concept_key, corr). corr is NaN when the concept has no variables or
      when one of the distance vectors has zero variance (correlation undefined).
    """
    if concept_keys is None:
        concept_keys = _concept_keys_in_order(concept_map)

    N = X.shape[0]
    triu_idx = np.triu_indices(N, k=1)  # k=1 excludes the diagonal

    results: List[Tuple[str, float]] = []
    for idx, key in enumerate(concept_keys):
        vars_idx = concept_map[key]
        if len(vars_idx) == 0:
            results.append((key, float("nan")))
            continue

        dist_input = pairwise_distances(X[:, vars_idx], metric="euclidean")
        dist_neuron = pairwise_distances(Z[:, idx].reshape(-1, 1), metric="euclidean")

        vec_input = dist_input[triu_idx]
        vec_neuron = dist_neuron[triu_idx]

        if np.std(vec_input) == 0.0 or np.std(vec_neuron) == 0.0:
            corr = float("nan")
        else:
            corr, _ = pearsonr(vec_input, vec_neuron)
        results.append((key, float(corr)))

    return results


def distance_corr_rows(
    results: List[Tuple[str, float]],
    beta: float,
    overlap_profile: str,
) -> List[Dict[str, object]]:
    """Turns per-concept correlations into tidy rows (NaNs dropped)."""
    rows: List[Dict[str, object]] = []
    for key, corr in results:
        if corr is None or np.isnan(corr):
            continue
        rows.append(
            {
                "beta": beta,
                "overlap_profile": overlap_profile,
                "concept": key,
                "distance_corr": float(corr),
            }
        )
    return rows


def run_distance_corr_pipeline(
    n_examples: int,
    n_variables: int,
    n_concepts: int,
    beta: float,
    overlap_profile: Dict[str, object],
    *,
    size_strategy: str = "range",
    max_concept_size: int = 200,
    size_min: int = 1,
    size_max: int = 200,
    dirichlet_total_assignments_factor: float = 1.8,
    variable_mean_strategy: str = "mean",
    alpha_mode: str = "uniform",
    alpha_exp_scale: float = 1.0,
    seed: int = None,
) -> List[Dict[str, object]]:
    """
    Full pipeline for one configuration: generate X, simulate neuron activations Z,
    then compute the per-concept distance correlation between the input reduced to
    each concept and the corresponding neuron activation.
    """
    profile_name = str(overlap_profile["name"])
    X, _, concept_map = generate_structured_dataset(
        n_examples=n_examples,
        n_variables=n_variables,
        n_concepts=n_concepts,
        overlap_skew=float(overlap_profile["overlap_skew"]),
        max_concept_size=max_concept_size,
        size_strategy=size_strategy,
        size_range=(size_min, size_max),
        overlap_floor=float(overlap_profile["overlap_floor"]),
        overlap_ceiling=float(overlap_profile["overlap_ceiling"]),
        overlap_convergence_power=float(overlap_profile["overlap_convergence_power"]),
        overlap_reference_concepts=int(overlap_profile["overlap_reference_concepts"]),
        dirichlet_total_assignments_factor=dirichlet_total_assignments_factor,
        variable_mean_strategy=variable_mean_strategy,
        seed=seed,
    )

    concept_keys = _concept_keys_in_order(concept_map)
    Z, _, _ = simulate_neuron_activations(
        X=X,
        concept_map=concept_map,
        beta=beta,
        alpha_mode=alpha_mode,
        alpha_exp_scale=alpha_exp_scale,
    )

    results = compute_distance_corr_per_concept(X, concept_map, Z, concept_keys)
    return distance_corr_rows(results, beta, profile_name)
