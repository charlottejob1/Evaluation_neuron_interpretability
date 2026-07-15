"""
Mutual Information Gap (MIG) between neurons z_j and concepts v_k.

Inputs:
  - concept values v_k: per-sample concept signal (N × K), e.g. mean concept
    activity μ_c in simulation or gp.prerank enrichment scores in VEGA.
  - neuron activations z_j (N × K_n).

MI matrix entry [j, k] ≈ I(z_j; v_k) via ``sklearn.feature_selection.mutual_info_regression``.

Paper MIG (Eq. 6): for each factor k,
  j(k) = argmax_j I(z_j; v_k),
  score_k = (I(z_{j(k)}; v_k) - max_{j≠j(k)} I(z_j; v_k)) / H(v_k),
  MIG = (1/K) Σ_k score_k.

``mig_per_neuron`` is kept as an auxiliary neuron-centric view only.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.feature_selection import mutual_info_regression

from data_generation import generate_structured_dataset
from neuron_activation_simulation import simulate_neuron_activations


def concept_entropy(
    concept_values: np.ndarray,
    n_bins: int = 20,
) -> float:
    """
    Shannon entropy of a 1D mean concept-activity signal μ_c (histogram estimate).
    """
    x = np.asarray(concept_values, dtype=float).ravel()
    if x.size == 0:
        return float("nan")
    counts, _ = np.histogram(x, bins=n_bins)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / np.sum(counts)
    return float(scipy_entropy(p))


def compute_concept_entropies(
    concept_mean_activities: np.ndarray,
    n_bins: int = 20,
) -> np.ndarray:
    """
    Per-concept entropies H(μ_c_k) for matrix ``concept_mean_activities`` (N, K).
    """
    K = concept_mean_activities.shape[1]
    return np.array(
        [
            concept_entropy(concept_mean_activities[:, k], n_bins=n_bins)
            for k in range(K)
        ],
        dtype=float,
    )


def _mi_row_for_neuron(
    j: int,
    mu_c: np.ndarray,
    z_j: np.ndarray,
    *,
    n_neighbors_eff: int,
    random_state: Optional[int],
) -> Tuple[int, np.ndarray]:
    """One neuron row of the MI matrix (for parallel execution)."""
    rs = None if random_state is None else int(random_state) + j
    row = mutual_info_regression(
        mu_c,
        z_j,
        n_neighbors=n_neighbors_eff,
        random_state=rs,
    )
    return j, row


def compute_mutual_info_matrix(
    concept_mean_activities: np.ndarray,
    neuron_activations: np.ndarray,
    *,
    random_state: Optional[int] = None,
    n_neighbors: int = 3,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Mutual information I(z_j; μ_c_k) for each neuron j and concept k.

    Uses ``mutual_info_regression`` with columns μ_c as features and each z_j
    as the target.

    Args:
      concept_mean_activities: (N, K) mean concept activities μ_c from data gen.
      neuron_activations: (N, K_n) simulated neuron activations z.
      random_state: passed to sklearn for reproducibility.
      n_neighbors: k-NN parameter for MI estimation.
      n_jobs: parallel workers for per-neuron MI (1 = serial; -1 = all CPUs).

    Returns:
      mi_matrix: (K_n, K) with rows = neurons, columns = concepts.
    """
    mu_c = np.asarray(concept_mean_activities, dtype=float)
    z = np.asarray(neuron_activations, dtype=float)
    if mu_c.ndim != 2 or z.ndim != 2:
        raise ValueError("concept_mean_activities and neuron_activations must be 2D.")
    if mu_c.shape[0] != z.shape[0]:
        raise ValueError("μ_c and z must have the same number of examples (rows).")

    n_neurons = z.shape[1]
    n_concepts = mu_c.shape[1]
    n_samples = mu_c.shape[0]
    # sklearn uses n_neighbors + 1 points per radius; need n_samples > n_neighbors
    n_neighbors_eff = min(n_neighbors, max(1, n_samples - 1))
    mi_matrix = np.zeros((n_neurons, n_concepts), dtype=float)

    if n_jobs == 1:
        for j in range(n_neurons):
            _, row = _mi_row_for_neuron(
                j,
                mu_c,
                z[:, j],
                n_neighbors_eff=n_neighbors_eff,
                random_state=random_state,
            )
            mi_matrix[j, :] = row
        return mi_matrix

    if n_jobs < 0:
        n_jobs = os.cpu_count() or 1

    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_mi_row_for_neuron)(
            j,
            mu_c,
            z[:, j],
            n_neighbors_eff=n_neighbors_eff,
            random_state=random_state,
        )
        for j in range(n_neurons)
    )
    for j, row in results:
        mi_matrix[j, :] = row

    return mi_matrix


def mig_per_concept(
    mi_matrix: np.ndarray,
    concept_entropies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Paper MIG terms (Eq. 6): one score per factor v_k.

    For factor k:
      j(k) = argmax_j I(z_j; v_k)
      score_k = (I(z_{j(k)}; v_k) - max_{j≠j(k)} I(z_j; v_k)) / H(v_k)

    Returns:
      per_concept_scores: (K,)
      top_neuron_indices: (K,) j(k)
      second_neuron_indices: (K,) neuron with second-highest MI to v_k
    """
    mi_matrix = np.asarray(mi_matrix, dtype=float)
    concept_entropies = np.asarray(concept_entropies, dtype=float)
    n_neurons, n_concepts = mi_matrix.shape

    if n_neurons < 2:
        raise ValueError("MIG requires at least 2 neurons (latent variables).")
    if n_concepts < 1:
        raise ValueError("MIG requires at least 1 concept.")

    per_concept = np.full(n_concepts, np.nan, dtype=float)
    top_neuron = np.zeros(n_concepts, dtype=int)
    second_neuron = np.zeros(n_concepts, dtype=int)

    for k in range(n_concepts):
        col = mi_matrix[:, k]
        top_two = np.argsort(col)[-2:][::-1]
        j_top, j_second = int(top_two[0]), int(top_two[1])
        top_neuron[k] = j_top
        second_neuron[k] = j_second

        gap = col[j_top] - col[j_second]
        h_k = concept_entropies[k]
        if h_k <= 0.0 or not np.isfinite(h_k):
            per_concept[k] = 0.0
        else:
            per_concept[k] = gap / h_k

    return per_concept, top_neuron, second_neuron


def mig_per_neuron(
    mi_matrix: np.ndarray,
    concept_entropies: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Auxiliary neuron-centric gap (not the paper's MIG aggregate).

    For neuron j:
      gap_j = MI(top concept) - MI(second concept)
      score_j = gap_j / H(top concept)

    Returns:
      per_neuron_scores: (K_n,)
      top_concept_indices: (K_n,) index of the highest-MI concept per neuron.
    """
    mi_matrix = np.asarray(mi_matrix, dtype=float)
    concept_entropies = np.asarray(concept_entropies, dtype=float)
    n_neurons, n_concepts = mi_matrix.shape

    if n_concepts < 2:
        raise ValueError("MIG requires at least 2 concepts.")

    per_neuron = np.full(n_neurons, np.nan, dtype=float)
    top_indices = np.zeros(n_neurons, dtype=int)

    for j in range(n_neurons):
        row = mi_matrix[j, :]
        top_two = np.argsort(row)[-2:][::-1]
        top_idx, second_idx = int(top_two[0]), int(top_two[1])
        top_indices[j] = top_idx

        gap = row[top_idx] - row[second_idx]
        h_top = concept_entropies[top_idx]
        if h_top <= 0.0 or not np.isfinite(h_top):
            per_neuron[j] = 0.0
        else:
            per_neuron[j] = gap / h_top

    return per_neuron, top_indices


def compute_mig(
    concept_mean_activities: np.ndarray,
    neuron_activations: np.ndarray,
    *,
    n_bins: int = 20,
    random_state: Optional[int] = None,
    n_neighbors: int = 3,
    n_jobs: int = 1,
) -> Dict[str, object]:
    """
    Full MIG computation for concept values v_k and neuron activations z_j.

    Returns a dict with:
      - mi_matrix: (n_neurons, n_concepts)
      - concept_entropies: (n_concepts,) H(v_k)
      - per_concept_mig: (n_concepts,) paper Eq. (6) per factor
      - top_neuron_per_concept: (n_concepts,) j(k) = argmax_j I(z_j; v_k)
      - second_neuron_per_concept: (n_concepts,)
      - mig: (1/K) Σ_k per_concept_mig[k]  (paper dataset MIG)
      - per_neuron_mig, top_concept_per_neuron: auxiliary neuron-centric view
    """
    mi_matrix = compute_mutual_info_matrix(
        concept_mean_activities,
        neuron_activations,
        random_state=random_state,
        n_neighbors=n_neighbors,
        n_jobs=n_jobs,
    )
    concept_entropies = compute_concept_entropies(
        concept_mean_activities, n_bins=n_bins
    )
    per_concept, top_neuron, second_neuron = mig_per_concept(
        mi_matrix, concept_entropies
    )
    per_neuron, top_concept = mig_per_neuron(mi_matrix, concept_entropies)

    valid = np.isfinite(per_concept)
    mig = float(np.mean(per_concept[valid])) if np.any(valid) else float("nan")

    return {
        "mi_matrix": mi_matrix,
        "concept_entropies": concept_entropies,
        "per_concept_mig": per_concept,
        "top_neuron_per_concept": top_neuron,
        "second_neuron_per_concept": second_neuron,
        "mig": mig,
        "per_neuron_mig": per_neuron,
        "top_concept_per_neuron": top_concept,
    }


def run_mig_pipeline(
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
    seed: Optional[int] = None,
    n_bins: int = 20,
    random_state: Optional[int] = None,
) -> Dict[str, object]:
    """
    Generate structured X, simulate Z, compute MIG on (μ_c, z).

    μ_c is ``concept_activities`` from data generation; z is ``Z`` from
    ``simulate_neuron_activations``.
    """
    X, concept_activities, concept_map = generate_structured_dataset(
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

    Z, _, _ = simulate_neuron_activations(
        X=X,
        concept_map=concept_map,
        beta=beta,
        alpha_mode=alpha_mode,
        alpha_exp_scale=alpha_exp_scale,
    )

    result = compute_mig(
        concept_activities,
        Z,
        n_bins=n_bins,
        random_state=random_state if random_state is not None else seed,
    )
    result["beta"] = beta
    result["overlap_profile"] = str(overlap_profile["name"])
    result["n_examples"] = n_examples
    result["n_variables"] = n_variables
    result["n_concepts"] = n_concepts
    return result


def mig_rows_from_result(result: Dict[str, object]) -> List[Dict[str, object]]:
    """
    Tidy rows for sweep plots: one row per concept (paper per-factor MIG), plus dataset mean.
    """
    beta = float(result["beta"])
    overlap_profile = str(result["overlap_profile"])
    mig_dataset = float(result["mig"])
    n_variables = int(result.get("n_variables", 0))
    n_concepts = int(result["n_concepts"])
    per_concept = np.asarray(result["per_concept_mig"], dtype=float)

    rows: List[Dict[str, object]] = []
    for k, score in enumerate(per_concept):
        if not np.isfinite(score):
            continue
        row: Dict[str, object] = {
            "beta": beta,
            "overlap_profile": overlap_profile,
            "concept": f"concept_{k}",
            "per_concept_mig": float(score),
            "mig_dataset": mig_dataset,
            "n_concepts": n_concepts,
        }
        if n_variables > 0:
            row["n_variables"] = n_variables
        rows.append(row)
    return rows


if __name__ == "__main__":
    # Small example on generated X to verify the pipeline runs end-to-end.
    overlap_profile = {
        "name": "example",
        "overlap_skew": 0.8,
        "overlap_floor": 0.0,
        "overlap_ceiling": 1.0,
        "overlap_convergence_power": 0.4,
        "overlap_reference_concepts": 50,
    }

    out = run_mig_pipeline(
        n_examples=2000,
        n_variables=500,
        n_concepts=30,
        beta=0.3,
        overlap_profile=overlap_profile,
        seed=42,
        random_state=42,
    )

    mi = out["mi_matrix"]
    print("MIG example on generated structured data")
    print(f"  beta={out['beta']}, n_examples={out['n_examples']}, n_concepts={out['n_concepts']}")
    print(f"  MI matrix shape (neurons x concepts): {mi.shape}")
    print(f"  MI range: [{mi.min():.6f}, {mi.max():.6f}]")
    print(f"  per-concept MIG (first 5): {out['per_concept_mig'][:5]}")
    print(f"  dataset MIG (paper, mean over factors): {out['mig']:.6f}")
