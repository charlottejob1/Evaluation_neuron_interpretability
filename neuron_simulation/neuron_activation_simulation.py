import argparse
import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from data_generation import generate_structured_dataset


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
    "overlap_reference_concepts": 100,
    "dirichlet_total_assignments_factor": 1.8,
    "variable_mean_strategy": "mean",
    "seed": 12345,
    "beta": 0.3,
    "alpha_mode": "uniform",
    "alpha_exp_scale": 1.0,
    "save_z_path": "neuron_activations.npy",
    "save_alpha_path": "alpha_weights.npy",
    "save_concept_map_path": "concept_mapping_for_neurons.json",
    "beta_values": "0.0,0.1,0.3,0.5,0.7,0.9,1.0",
    "beta_impact_plot_path": "beta_impact_summary.png",
}


def _concept_keys_in_order(concept_map: Dict[str, List[int]]) -> List[str]:
    # Sort by numeric suffix of keys like concept_0, concept_1, ...
    return sorted(concept_map.keys(), key=lambda k: int(k.split("_")[-1]))


def compute_concept_sums(X: np.ndarray, concept_map: Dict[str, List[int]]) -> np.ndarray:
    """
    Returns S with shape (N, K), where S[:, i] = sum(X[:, j] for j in g(i)).
    """
    keys = _concept_keys_in_order(concept_map)
    N = X.shape[0]
    K = len(keys)
    S = np.zeros((N, K), dtype=float)
    for idx, key in enumerate(keys):
        vars_idx = concept_map[key]
        if len(vars_idx) == 0:
            continue
        S[:, idx] = np.sum(X[:, vars_idx], axis=1)
    return S


def build_alpha_matrix(
    n_concepts: int,
    mode: str,
    exp_scale: float,
) -> np.ndarray:
    """
    Builds A with shape (K, K), where A[i, k] = alpha_{i,k}.
    Enforces alpha_{i,i} = 0 (global influence excludes concept i itself).
    """
    if mode not in {"uniform", "exponential"}:
        raise ValueError("alpha_mode must be one of: uniform, exponential")
    if exp_scale <= 0:
        raise ValueError("alpha_exp_scale must be > 0")

    K = n_concepts
    A = np.zeros((K, K), dtype=float)

    for i in range(K):
        idx = [k for k in range(K) if k != i]

        if len(idx) == 0:
            A[i, i] = 1.0
            continue

        if mode == "uniform":
            w = np.ones(len(idx), dtype=float) / len(idx)
        else:
            w_raw = np.random.exponential(scale=exp_scale, size=len(idx))
            w = w_raw / np.sum(w_raw)

        A[i, idx] = w

    return A


def simulate_neuron_activations(
    X: np.ndarray,
    concept_map: Dict[str, List[int]],
    beta: float,
    alpha_mode: str = "uniform",
    alpha_exp_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates neuron activations Z for K neurons (one per concept):

      z_i = (1 - beta) * S_i + beta * sum_{k != i}(alpha_{i,k} * S_k),

    where S_i = sum_j X_j for j in g(i).

    Returns:
      Z: (N, K) neuron activations
      S: (N, K) concept sums
      A: (K, K) alpha matrix
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    S = compute_concept_sums(X, concept_map)  # (N, K)
    K = S.shape[1]
    A = build_alpha_matrix(
        n_concepts=K,
        mode=alpha_mode,
        exp_scale=alpha_exp_scale,
    )  # (K, K)

    # Global influence term for each neuron i and example n: (S @ A^T)[n, i]
    global_influence = S @ A.T
    Z = (1.0 - beta) * S + beta * global_influence
    return Z, S, A


def _parse_beta_values(raw: str) -> List[float]:
    betas = [float(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if len(betas) == 0:
        raise ValueError("beta_values cannot be empty.")
    for b in betas:
        if not (0.0 <= b <= 1.0):
            raise ValueError("All beta values must be in [0, 1].")
    return sorted(betas)


def visualize_beta_impact(
    S: np.ndarray,
    alpha_matrix: np.ndarray,
    betas: List[float],
    save_path: str,
) -> None:
    """
    For each beta, compute Z per example and concept:
      Z(beta) = (1-beta)*S + beta*(S @ A^T)
    Then aggregate over examples for each concept and visualize.
    """
    global_influence = S @ alpha_matrix.T
    K = S.shape[1]
    B = len(betas)

    mean_z = np.zeros((B, K), dtype=float)
    std_z = np.zeros((B, K), dtype=float)
    mean_abs_z = np.zeros((B, K), dtype=float)

    for b_idx, beta in enumerate(betas):
        Z_beta = (1.0 - beta) * S + beta * global_influence
        mean_z[b_idx, :] = np.mean(Z_beta, axis=0)
        std_z[b_idx, :] = np.std(Z_beta, axis=0)
        mean_abs_z[b_idx, :] = np.mean(np.abs(Z_beta), axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1) Global aggregated effect across concepts.
    axes[0, 0].plot(betas, np.mean(mean_abs_z, axis=1), marker="o", color="navy")
    axes[0, 0].set_title("Mean |z_i| across concepts vs beta")
    axes[0, 0].set_xlabel("beta")
    axes[0, 0].set_ylabel("Average over concepts")
    axes[0, 0].grid(alpha=0.3)

    # 2) Concept-wise mean activation aggregated over examples.
    im1 = axes[0, 1].imshow(mean_z, aspect="auto", cmap="coolwarm")
    axes[0, 1].set_title("Mean z_i over examples (per concept)")
    axes[0, 1].set_xlabel("Concept index i")
    axes[0, 1].set_ylabel("beta index")
    axes[0, 1].set_yticks(np.arange(B))
    axes[0, 1].set_yticklabels([f"{b:.2f}" for b in betas])
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # 3) Concept-wise std aggregated over examples.
    im2 = axes[1, 0].imshow(std_z, aspect="auto", cmap="viridis")
    axes[1, 0].set_title("Std of z_i over examples (per concept)")
    axes[1, 0].set_xlabel("Concept index i")
    axes[1, 0].set_ylabel("beta index")
    axes[1, 0].set_yticks(np.arange(B))
    axes[1, 0].set_yticklabels([f"{b:.2f}" for b in betas])
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # 4) Spread across concepts as beta changes.
    axes[1, 1].plot(betas, np.mean(std_z, axis=1), marker="o", color="darkgreen")
    axes[1, 1].set_title("Mean std(z_i) across concepts vs beta")
    axes[1, 1].set_xlabel("beta")
    axes[1, 1].set_ylabel("Average std over concepts")
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Beta impact visualization saved to: {save_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate neuron activations from concept-linked variables."
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

    parser.add_argument("--beta", type=float, default=DEFAULTS["beta"])
    parser.add_argument(
        "--beta-values",
        type=str,
        default=DEFAULTS["beta_values"],
        help="Comma-separated beta values for sweep visualization.",
    )
    parser.add_argument(
        "--alpha-mode",
        choices=["uniform", "exponential"],
        default=DEFAULTS["alpha_mode"],
    )
    parser.add_argument("--alpha-exp-scale", type=float, default=DEFAULTS["alpha_exp_scale"])
    parser.add_argument("--save-z-path", type=str, default=DEFAULTS["save_z_path"])
    parser.add_argument("--save-alpha-path", type=str, default=DEFAULTS["save_alpha_path"])
    parser.add_argument(
        "--beta-impact-plot-path",
        type=str,
        default=DEFAULTS["beta_impact_plot_path"],
    )
    parser.add_argument(
        "--save-concept-map-path",
        type=str,
        default=DEFAULTS["save_concept_map_path"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    X, _, concept_map = generate_structured_dataset(
        n_examples=args.n_examples,
        n_variables=args.n_variables,
        n_concepts=args.n_concepts,
        overlap_skew=args.overlap_skew,
        max_concept_size=args.max_concept_size,
        size_strategy=args.size_strategy,
        size_range=(args.size_min, args.size_max),
        overlap_floor=args.overlap_floor,
        overlap_ceiling=args.overlap_ceiling,
        overlap_convergence_power=args.overlap_convergence_power,
        overlap_reference_concepts=args.overlap_reference_concepts,
        dirichlet_total_assignments_factor=args.dirichlet_total_assignments_factor,
        variable_mean_strategy=args.variable_mean_strategy,
        seed=args.seed,
    )

    Z, S, A = simulate_neuron_activations(
        X=X,
        concept_map=concept_map,
        beta=args.beta,
        alpha_mode=args.alpha_mode,
        alpha_exp_scale=args.alpha_exp_scale,
    )

    # Sweep beta with constant alpha_{i,k} = 1/(K-1), k != i, and aggregate over examples.
    betas = _parse_beta_values(args.beta_values)
    A_uniform = build_alpha_matrix(n_concepts=S.shape[1], mode="uniform", exp_scale=1.0)
    visualize_beta_impact(
        S=S,
        alpha_matrix=A_uniform,
        betas=betas,
        save_path=args.beta_impact_plot_path,
    )

    np.save(args.save_z_path, Z)
    np.save(args.save_alpha_path, A)
    with open(args.save_concept_map_path, "w", encoding="utf-8") as f:
        json.dump(concept_map, f, indent=2)

    print(f"X shape: {X.shape}")
    print(f"S shape (concept sums): {S.shape}")
    print(f"Z shape (neuron activations): {Z.shape}")
    print(f"Alpha shape: {A.shape}")
    print(f"Saved Z to: {args.save_z_path}")
    print(f"Saved alpha weights to: {args.save_alpha_path}")
    print(f"Saved concept mapping to: {args.save_concept_map_path}")


if __name__ == "__main__":
    main()
