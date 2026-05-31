import argparse
import csv
import json
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_generation import generate_structured_dataset
from neuron_activation_simulation import simulate_neuron_activations


DEFAULTS = {
    "n_examples": 10000,
    "n_variables": 2000,
    "n_concepts": 300,
    "size_strategy": "range",
    "max_concept_size": 200,
    "size_min": 1,
    "size_max": 200,
    "overlap_skew": 0.8,
    "overlap_floor": 0.0,
    "overlap_ceiling": 1.0,
    "overlap_convergence_power": 0.4,
    "overlap_reference_concepts": 300,
    "dirichlet_total_assignments_factor": 1.8,
    "variable_mean_strategy": "mean",
    "seed": 12345,
    "beta_values": "0.0,0.2,0.4,0.6,0.8,1.0",
    # Format per profile: name|overlap_skew|overlap_floor|overlap_ceiling|overlap_convergence_power|overlap_reference_concepts
    "overlap_profiles": "base|0.8|0.0|1.0|0.4|100",
    "alpha_mode": "uniform",
    "alpha_exp_scale": 1.0,
    "save_metrics_path": "metrics_simulation.json",
    "save_reduction_tensor_path": "reduction_tensors.npz",
    "save_probability_csv_path": "probability_scores.csv",
    "save_probability_plot_path": "probability_distributions_by_beta_overlap.png",
}


def _concept_keys_in_order(concept_map: Dict[str, List[int]]) -> List[str]:
    return sorted(concept_map.keys(), key=lambda k: int(k.split("_")[-1]))


def simulate_concept_perturbation(
    X: np.ndarray,
    concept_map: Dict[str, List[int]],
    concept_to_perturb: str,
    beta: float,
    alpha_mode: str,
    alpha_exp_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates one concept perturbation by setting all variables in concept_to_perturb to 0,
    then computes neuron activations with the equation-based simulator.

    Returns:
      X_perturbed: (N, M) perturbed input matrix
      z_perturbed: (N, K) neuron activations after perturbation
    """
    if concept_to_perturb not in concept_map:
        raise ValueError(f"Unknown concept key: {concept_to_perturb}")

    X_perturbed = X.copy()
    vars_idx = concept_map[concept_to_perturb]
    if len(vars_idx) > 0:
        X_perturbed[:, vars_idx] = 0.0

    z_perturbed, _, _ = simulate_neuron_activations(
        X=X_perturbed,
        concept_map=concept_map,
        beta=beta,
        alpha_mode=alpha_mode,
        alpha_exp_scale=alpha_exp_scale,
    )
    return X_perturbed, z_perturbed


def compute_reduction_tensor(
    X: np.ndarray,
    concept_map: Dict[str, List[int]],
    beta: float,
    alpha_mode: str,
    alpha_exp_scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes reduction tensor R with shape (K, N, K):
      R[i, n, m] = | z_m^(base)(n) - z_m^(perturb concept i)(n) |

    Returns:
      reduction_tensor: (K, N, K)
      baseline_z: (N, K)
    """
    keys = _concept_keys_in_order(concept_map)
    K = len(keys)
    N = X.shape[0]

    baseline_z, _, _ = simulate_neuron_activations(
        X=X,
        concept_map=concept_map,
        beta=beta,
        alpha_mode=alpha_mode,
        alpha_exp_scale=alpha_exp_scale,
    )

    reduction_tensor = np.zeros((K, N, K), dtype=float)

    for i, key_i in enumerate(keys):
        _, z_pert = simulate_concept_perturbation(
            X=X,
            concept_map=concept_map,
            concept_to_perturb=key_i,
            beta=beta,
            alpha_mode=alpha_mode,
            alpha_exp_scale=alpha_exp_scale,
        )
        reduction_tensor[i, :, :] = np.abs(baseline_z - z_pert)

    return reduction_tensor, baseline_z


def compute_concept_overlap_matrix(
    concept_map: Dict[str, List[int]],
    n_variables: int = None,
) -> np.ndarray:
    """
    Directional concept overlap ratios O[i, j] = |g(i) ∩ g(j)| / |g(i)|.

    The numerator is the number of shared variables; the denominator is the size of
    the *observed* concept i (the row index). Overlap is therefore asymmetric:
    O[i, j] measures how much of concept i is contained in concept j. For example,
    if concepts share 2 variables and |g(i)| = 2, |g(j)| = 10, then O[i, j] = 1.0
    (i fully contained in j) while O[j, i] = 0.2.

    Values lie in [0, 1]: 0 = disjoint, 1 = observed concept fully contained in the
    other. Diagonal entries are 1.0.

    n_variables is accepted for backward compatibility but is no longer used.
    """
    keys = _concept_keys_in_order(concept_map)
    sets_i = [set(concept_map[k]) for k in keys]
    K = len(keys)
    O = np.zeros((K, K), dtype=float)
    for i in range(K):
        for j in range(K):
            if i == j:
                O[i, j] = 1.0
            else:
                denom = len(sets_i[i])
                O[i, j] = (len(sets_i[i] & sets_i[j]) / denom) if denom > 0 else 0.0
    return O


def compute_probability_self_more_affected(
    reduction_tensor: np.ndarray,
    overlap_matrix: np.ndarray = None,
    overlap_threshold: Optional[float] = None,
) -> np.ndarray:
    """
    For each perturbation i and each neuron j != i, compute:
      P_i,j = P_n( R[i,n,i] > R[i,n,j] )

    If overlap_threshold is set (and overlap_matrix is provided), only pairs where the
    observed concept j is not highly contained in the perturbated concept i, i.e.
      overlap(observed=j, perturbated=i) = O[j, i] < overlap_threshold
    are computed; other off-diagonal entries remain NaN.
    If overlap_threshold is None, all off-diagonal pairs are included.

    Returns matrix P with shape (K, K), diagonal = NaN.
    """
    K = reduction_tensor.shape[0]
    P = np.full((K, K), np.nan, dtype=float)
    for i in range(K):
        self_effect = reduction_tensor[i, :, i]  # (N,)
        for j in range(K):
            if i == j:
                continue
            if (
                overlap_matrix is not None
                and overlap_threshold is not None
                and overlap_matrix[j, i] >= overlap_threshold
            ):
                continue
            other_effect = reduction_tensor[i, :, j]
            P[i, j] = float(np.mean(self_effect > other_effect))
    return P


def summarize_metrics(reduction_tensor: np.ndarray, prob_matrix: np.ndarray) -> Dict[str, object]:
    """
    Produces summary metrics for quick interpretation.
    """
    K = reduction_tensor.shape[0]
    mean_reduction = np.mean(reduction_tensor, axis=1)  # (K, K), averaged over examples

    self_mean = np.array([mean_reduction[i, i] for i in range(K)], dtype=float)
    others_mean = np.array(
        [
            np.mean(np.delete(mean_reduction[i, :], i))
            for i in range(K)
        ],
        dtype=float,
    )
    ratio_self_vs_others = self_mean / (others_mean + 1e-12)

    per_concept_prob_self_vs_others = np.array(
        [
            np.nanmean(np.delete(prob_matrix[i, :], i))
            for i in range(K)
        ],
        dtype=float,
    )

    return {
        "mean_self_reduction": float(np.mean(self_mean)),
        "mean_other_reduction": float(np.mean(others_mean)),
        "mean_self_over_others_ratio": float(np.mean(ratio_self_vs_others)),
        "mean_prob_self_more_affected_than_others": float(
            np.mean(per_concept_prob_self_vs_others)
        ),
        "per_concept_mean_prob_self_more_affected_than_others": per_concept_prob_self_vs_others.tolist(),
    }


def _parse_beta_values(raw: str) -> List[float]:
    betas = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(betas) == 0:
        raise ValueError("beta_values cannot be empty.")
    for b in betas:
        if not (0.0 <= b <= 1.0):
            raise ValueError("All beta values must be in [0, 1].")
    return sorted(betas)


def _parse_overlap_profiles(raw: str) -> List[Dict[str, object]]:
    """
    Parses profile list from:
      name|skew|floor|ceiling|convergence_power|reference_concepts[, ...]
    """
    profiles: List[Dict[str, object]] = []
    chunks = [c.strip() for c in raw.split(",") if c.strip()]
    if not chunks:
        raise ValueError("overlap_profiles cannot be empty.")

    for chunk in chunks:
        parts = [p.strip() for p in chunk.split("|")]
        if len(parts) != 6:
            raise ValueError(
                "Each overlap profile must have 6 fields: "
                "name|skew|floor|ceiling|convergence_power|reference_concepts"
            )
        name = parts[0]
        profile = {
            "name": name,
            "overlap_skew": float(parts[1]),
            "overlap_floor": float(parts[2]),
            "overlap_ceiling": float(parts[3]),
            "overlap_convergence_power": float(parts[4]),
            "overlap_reference_concepts": int(float(parts[5])),
        }
        profiles.append(profile)

    return profiles


def run_probability_pipeline(
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
    overlap_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    """
    Full pipeline for one configuration: generate data, perturb all concepts,
    compute reduction scores and pairwise probabilities.
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
    overlap_matrix = (
        compute_concept_overlap_matrix(concept_map, n_variables)
        if overlap_threshold is not None
        else None
    )
    reduction_tensor, _ = compute_reduction_tensor(
        X=X,
        concept_map=concept_map,
        beta=beta,
        alpha_mode=alpha_mode,
        alpha_exp_scale=alpha_exp_scale,
    )
    prob_matrix = compute_probability_self_more_affected(
        reduction_tensor,
        overlap_matrix=overlap_matrix,
        overlap_threshold=overlap_threshold,
    )
    return probability_matrix_to_rows(
        concept_keys,
        prob_matrix,
        beta,
        profile_name,
        overlap_matrix=overlap_matrix,
        overlap_threshold=overlap_threshold,
    )


def probability_matrix_to_rows(
    concept_keys: List[str],
    prob_matrix: np.ndarray,
    beta: float,
    overlap_profile: str,
    overlap_matrix: np.ndarray = None,
    overlap_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    K = len(concept_keys)
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            p = prob_matrix[i, j]
            if np.isnan(p):
                continue
            row = {
                "beta": beta,
                "overlap_profile": overlap_profile,
                "perturbated_concept": concept_keys[i],
                "observed_concept": concept_keys[j],
                "probability": float(p),
            }
            if overlap_threshold is not None:
                row["overlap_threshold"] = overlap_threshold
            if overlap_matrix is not None:
                # Directional overlap of the observed concept (j) within the
                # perturbated concept (i): O[j, i] = |g(j) ∩ g(i)| / |g(j)|.
                row["pair_overlap"] = float(overlap_matrix[j, i])
            rows.append(row)
    return rows


def probability_rows_from_reduction(
    reduction_tensor: np.ndarray,
    concept_map: Dict[str, List[int]],
    n_variables: int,
    beta: float,
    overlap_profile: str,
    overlap_threshold: float,
) -> List[Dict[str, object]]:
    """Compute probability rows from an existing reduction tensor (one beta, many thresholds)."""
    concept_keys = _concept_keys_in_order(concept_map)
    overlap_matrix = compute_concept_overlap_matrix(concept_map, n_variables)
    prob_matrix = compute_probability_self_more_affected(
        reduction_tensor,
        overlap_matrix=overlap_matrix,
        overlap_threshold=overlap_threshold,
    )
    return probability_matrix_to_rows(
        concept_keys,
        prob_matrix,
        beta,
        overlap_profile,
        overlap_matrix=overlap_matrix,
        overlap_threshold=overlap_threshold,
    )


def save_probability_rows_csv(rows: List[Dict[str, object]], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "overlap_profile",
                "overlap_threshold",
                "pair_overlap",
                "beta",
                "perturbated_concept",
                "observed_concept",
                "probability",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_probability_distributions_by_beta(rows: List[Dict[str, object]], path: str) -> None:
    group_to_values: Dict[str, List[float]] = {}
    for row in rows:
        profile = str(row["overlap_profile"])
        beta = float(row["beta"])
        p = float(row["probability"])
        key = f"{profile} | b={beta:.2f}"
        group_to_values.setdefault(key, []).append(p)

    labels = sorted(group_to_values.keys())
    data = [group_to_values[k] for k in labels]

    plt.figure(figsize=(max(12, 0.9 * len(labels)), 6))
    plt.boxplot(data, positions=np.arange(len(labels)), showfliers=False)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.xlabel("overlap profile and beta")
    plt.ylabel("Probability")
    plt.title("Distribution of probability scores by overlap profile and beta")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Metrics for concept knockout impact on equation-based neuron activations."
    )

    parser.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"])
    parser.add_argument("--n-variables", type=int, default=DEFAULTS["n_variables"])
    parser.add_argument("--n-concepts", type=int, default=DEFAULTS["n_concepts"])
    parser.add_argument(
        "--size-strategy",
        choices=["dirichlet", "fixed", "range"],
        default=DEFAULTS["size_strategy"],
    )
    parser.add_argument("--max-concept-size", type=int, default=DEFAULTS["max_concept_size"])
    parser.add_argument("--size-min", type=int, default=DEFAULTS["size_min"])
    parser.add_argument("--size-max", type=int, default=DEFAULTS["size_max"])
    parser.add_argument("--overlap-skew", type=float, default=DEFAULTS["overlap_skew"])
    parser.add_argument("--overlap-floor", type=float, default=DEFAULTS["overlap_floor"])
    parser.add_argument("--overlap-ceiling", type=float, default=DEFAULTS["overlap_ceiling"])
    parser.add_argument(
        "--overlap-convergence-power",
        type=float,
        default=DEFAULTS["overlap_convergence_power"],
    )
    parser.add_argument(
        "--overlap-reference-concepts",
        type=int,
        default=DEFAULTS["overlap_reference_concepts"],
    )
    parser.add_argument(
        "--dirichlet-total-assignments-factor",
        type=float,
        default=DEFAULTS["dirichlet_total_assignments_factor"],
    )
    parser.add_argument(
        "--variable-mean-strategy",
        choices=["mean", "max"],
        default=DEFAULTS["variable_mean_strategy"],
    )
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    parser.add_argument(
        "--beta-values",
        type=str,
        default=DEFAULTS["beta_values"],
        help="Comma-separated beta values for full perturbation/probability pipeline.",
    )
    parser.add_argument(
        "--overlap-profiles",
        type=str,
        default=DEFAULTS["overlap_profiles"],
        help=(
            "Comma-separated overlap profiles, each as "
            "name|skew|floor|ceiling|convergence_power|reference_concepts"
        ),
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["uniform", "exponential"],
        default=DEFAULTS["alpha_mode"],
    )
    parser.add_argument("--alpha-exp-scale", type=float, default=DEFAULTS["alpha_exp_scale"])

    parser.add_argument("--save-metrics-path", type=str, default=DEFAULTS["save_metrics_path"])
    parser.add_argument(
        "--save-reduction-tensor-path",
        type=str,
        default=DEFAULTS["save_reduction_tensor_path"],
    )
    parser.add_argument(
        "--save-probability-csv-path",
        type=str,
        default=DEFAULTS["save_probability_csv_path"],
    )
    parser.add_argument(
        "--save-probability-plot-path",
        type=str,
        default=DEFAULTS["save_probability_plot_path"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    betas = _parse_beta_values(args.beta_values)
    overlap_profiles = _parse_overlap_profiles(args.overlap_profiles)

    all_probability_rows: List[Dict[str, object]] = []
    summaries_by_profile_beta: Dict[str, Dict[str, Dict[str, object]]] = {}
    reduction_tensors_to_save: Dict[str, np.ndarray] = {}
    baseline_shape = None

    for profile in overlap_profiles:
        profile_name = str(profile["name"])
        X, _, concept_map = generate_structured_dataset(
            n_examples=args.n_examples,
            n_variables=args.n_variables,
            n_concepts=args.n_concepts,
            overlap_skew=float(profile["overlap_skew"]),
            max_concept_size=args.max_concept_size,
            size_strategy=args.size_strategy,
            size_range=(args.size_min, args.size_max),
            overlap_floor=float(profile["overlap_floor"]),
            overlap_ceiling=float(profile["overlap_ceiling"]),
            overlap_convergence_power=float(profile["overlap_convergence_power"]),
            overlap_reference_concepts=int(profile["overlap_reference_concepts"]),
            dirichlet_total_assignments_factor=args.dirichlet_total_assignments_factor,
            variable_mean_strategy=args.variable_mean_strategy,
            seed=args.seed,
        )
        concept_keys = _concept_keys_in_order(concept_map)
        summaries_by_profile_beta[profile_name] = {}

        for beta in betas:
            reduction_tensor, baseline_z = compute_reduction_tensor(
                X=X,
                concept_map=concept_map,
                beta=beta,
                alpha_mode=args.alpha_mode,
                alpha_exp_scale=args.alpha_exp_scale,
            )
            prob_matrix = compute_probability_self_more_affected(reduction_tensor)
            summary = summarize_metrics(reduction_tensor, prob_matrix)
            rows = probability_matrix_to_rows(concept_keys, prob_matrix, beta, profile_name)

            all_probability_rows.extend(rows)
            summaries_by_profile_beta[profile_name][f"{beta:.6f}"] = summary
            reduction_tensors_to_save[f"{profile_name}__beta_{beta:.6f}"] = reduction_tensor
            baseline_shape = list(baseline_z.shape)

    np.savez(args.save_reduction_tensor_path, **reduction_tensors_to_save)
    save_probability_rows_csv(all_probability_rows, args.save_probability_csv_path)
    plot_probability_distributions_by_beta(all_probability_rows, args.save_probability_plot_path)

    payload = {
        "config": {
            "n_examples": args.n_examples,
            "n_variables": args.n_variables,
            "n_concepts": args.n_concepts,
            "size_strategy": args.size_strategy,
            "beta_values": betas,
            "overlap_profiles": overlap_profiles,
            "alpha_mode": args.alpha_mode,
            "alpha_exp_scale": args.alpha_exp_scale,
            "seed": args.seed,
        },
        "summary_by_profile_beta": summaries_by_profile_beta,
        "csv_columns": [
            "overlap_profile",
            "beta",
            "perturbated_concept",
            "observed_concept",
            "probability",
        ],
        "baseline_activation_shape": baseline_shape,
        "reduction_tensor_shape": [
            len(concept_keys),
            args.n_examples,
            len(concept_keys),
        ],
    }

    with open(args.save_metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Baseline activations shape: {tuple(baseline_shape)}")
    print(
        "Reduction tensor shape per beta: "
        f"({len(concept_keys)}, {args.n_examples}, {len(concept_keys)}) "
        "(perturbation, example, neuron)"
    )
    print(f"Saved metrics to: {args.save_metrics_path}")
    print(f"Saved reduction tensors (.npz) to: {args.save_reduction_tensor_path}")
    print(f"Saved probability CSV to: {args.save_probability_csv_path}")
    print(f"Saved probability distribution plot to: {args.save_probability_plot_path}")


if __name__ == "__main__":
    main()
