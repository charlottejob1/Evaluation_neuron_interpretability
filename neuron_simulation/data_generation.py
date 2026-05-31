import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import csv
import json


def _sample_truncated_exponential(
    n_samples: int,
    lam: float,
    low: float = 0.0,
    high: float = 1.0
) -> np.ndarray:
    """
    Samples from a truncated exponential distribution on [low, high].
    Density is proportional to exp(-lam * x), so values are concentrated near low.
    """
    if lam <= 0:
        raise ValueError("lam must be > 0.")
    if not (0.0 <= low < high <= 1.0):
        raise ValueError("low/high must satisfy 0 <= low < high <= 1.")

    span = high - low
    u = np.random.uniform(0.0, 1.0, size=n_samples)
    # Inverse CDF for exponential on [0, span], then shifted by low.
    scaled = -np.log(1.0 - u * (1.0 - np.exp(-lam * span))) / lam
    return low + scaled

def generate_concept_activity(n: int):
    """
    Generates a NumPy array of n samples from a standard normal distribution.
    """
    return np.random.standard_normal(n)

def generate_structured_dataset(
    n_examples: int, 
    n_variables: int, 
    n_concepts: int, 
    overlap_skew: float = 15.0,
    max_concept_size: int = 200,
    size_strategy: str = "dirichlet",
    fixed_size: int = 10,
    size_range: Tuple[int, int] = (1, 200),
    overlap_floor: float = 0.0,
    overlap_ceiling: float = 1.0,
    overlap_convergence_power: float = 0.5,
    overlap_reference_concepts: int = 100,
    dirichlet_total_assignments_factor: float = 1.8,
    variable_mean_strategy: str = "mean",
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]]]:
    """
    Generates a dataset X and concept activities based on concept-variable mappings.
    
    Args:
        n_examples: Number of rows (N).
        n_variables: Number of variables (M).
        n_concepts: Number of concepts (K).
        overlap_skew: Base lambda for negative-exponential overlap sampling. Higher values
            push more pairs toward low overlap (few pairs with high overlap).
        max_concept_size: Upper bound on concept size.
        size_strategy: "dirichlet" (partitions all variables), "fixed" (constant size), or "range" (uniform random).
        fixed_size: Number of variables per concept if strategy is "fixed".
        size_range: (min, max) variables per concept if strategy is "range".
        overlap_floor: Minimum overlap ratio in [0, 1]. Overlap is defined as
            |Ci ∩ Cj| / max(|Ci|, |Cj|).
        overlap_ceiling: Maximum overlap ratio in [0, 1].
        overlap_convergence_power: Controls how quickly overlap converges toward low values
            as n_concepts grows.
        overlap_reference_concepts: Reference concept count for normalized convergence scaling.
            Effective lambda = overlap_skew * ((n_concepts / overlap_reference_concepts) ** power).
        dirichlet_total_assignments_factor: Only used with size_strategy="dirichlet".
            Sets target total concept memberships as factor * n_variables to leave room
            for overlap creation (must be >= 1.0).
        variable_mean_strategy: How to compute Gaussian mean for each variable from linked
            concept activities. One of {"mean", "max"}.
        seed: Optional random seed for reproducible generation.

    Returns:
        X: The generated dataset of shape (n_examples, n_variables).
        concept_activities: Matrix of shape (n_examples, n_concepts).
        concept_map: Dictionary mapping concept names (e.g., "concept_0") to variable indices.
    """
    if seed is not None:
        np.random.seed(seed)

    concept_map = {i: set() for i in range(n_concepts)}

    # 1. Decide concept sizes (enforced bounds)
    size_low, size_high = size_range
    size_low = max(1, size_low)
    size_high = min(size_high, n_variables, max_concept_size)
    if size_low > size_high:
        raise ValueError("Invalid size_range after bounds enforcement.")

    if size_strategy == "fixed":
        target_sizes = [int(np.clip(fixed_size, size_low, size_high))] * n_concepts
    elif size_strategy == "dirichlet":
        if dirichlet_total_assignments_factor < 1.0:
            raise ValueError("dirichlet_total_assignments_factor must be >= 1.0")
        raw = np.random.dirichlet([0.3] * n_concepts)
        target_sizes = [size_low] * n_concepts
        total_target_memberships = int(round(dirichlet_total_assignments_factor * n_variables))
        max_total_memberships = n_concepts * size_high
        total_target_memberships = max(sum(target_sizes), min(total_target_memberships, max_total_memberships))
        remaining = max(0, total_target_memberships - sum(target_sizes))
        if remaining > 0:
            extra = np.random.multinomial(remaining, raw)
            target_sizes = [
                int(min(size_high, base + extra_i))
                for base, extra_i in zip(target_sizes, extra)
            ]
    elif size_strategy == "range":
        # Prefer diverse concept sizes. If possible, sample without replacement.
        width = size_high - size_low + 1
        if n_concepts <= width:
            sampled = np.random.choice(
                np.arange(size_low, size_high + 1), size=n_concepts, replace=False
            )
            np.random.shuffle(sampled)
            target_sizes = [int(s) for s in sampled]
        else:
            target_sizes = [
                int(np.random.randint(size_low, size_high + 1))
                for _ in range(n_concepts)
            ]
    else:
        raise ValueError("size_strategy must be one of: dirichlet, fixed, range")

    # If total concept capacity < n_variables, the surplus variables stay unassigned
    # (background-noise variables, mean 0). This keeps the model valid when concepts
    # are small relative to M (e.g. low-K sweeps with small concept sizes).
    total_capacity = sum(target_sizes)
    if total_capacity < n_variables:
        print(
            f"[generate] Note: total concept capacity ({total_capacity}) < n_variables "
            f"({n_variables}); {n_variables - total_capacity} variables will be background noise."
        )

    # 2. Seed each concept with one variable, then assign variables while capacity remains.
    remaining_capacity = target_sizes.copy()
    all_vars = np.arange(n_variables)
    np.random.shuffle(all_vars)
    for c_idx in range(n_concepts):
        v = int(all_vars[c_idx % n_variables])
        concept_map[c_idx].add(v)
        remaining_capacity[c_idx] -= 1

    # Assign remaining variables to a concept (weighted by remaining capacity).
    # Stop once all capacity is used; leftover variables remain as noise.
    assigned = set()
    for c in concept_map.values():
        assigned.update(c)
    for v in range(n_variables):
        if v in assigned:
            continue
        positive = np.array([max(0, cap) for cap in remaining_capacity], dtype=float)
        if positive.sum() == 0:
            break
        probs = positive / positive.sum()
        c_pick = int(np.random.choice(n_concepts, p=probs))
        if v not in concept_map[c_pick]:
            concept_map[c_pick].add(int(v))
            remaining_capacity[c_pick] -= 1
            assigned.add(v)

    # 3. Build pairwise target overlaps as |Ci ∩ Cj| / max(|Ci|, |Cj|) with a
    #    negative-exponential law (most pairs low, few pairs high). Values in [0, 1].
    if n_concepts > 1:
        n_pairs = n_concepts * (n_concepts - 1) // 2
        if overlap_reference_concepts <= 0:
            raise ValueError("overlap_reference_concepts must be > 0")
        concept_ratio = max(1e-8, n_concepts / overlap_reference_concepts)
        effective_lambda = overlap_skew * (concept_ratio ** overlap_convergence_power)
        pair_targets = _sample_truncated_exponential(
            n_samples=n_pairs,
            lam=effective_lambda,
            low=overlap_floor,
            high=overlap_ceiling
        )

        pair_idx = 0
        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                # Overlap ratio is relative to the larger of the two concepts.
                denom = max(target_sizes[i], target_sizes[j])
                target_count = int(np.round(pair_targets[pair_idx] * denom))
                pair_idx += 1

                # Intersection cannot exceed the smaller concept.
                target_count = min(target_count, target_sizes[i], target_sizes[j])
                if target_count <= 0:
                    continue

                set_i, set_j = concept_map[i], concept_map[j]
                current = len(set_i & set_j)
                needed = target_count - current
                if needed <= 0:
                    continue

                # First, copy variables from i to j and j to i while capacities allow.
                if needed > 0 and remaining_capacity[j] > 0:
                    pool_i = list(set_i - set_j)
                    if pool_i:
                        n_add = min(needed, remaining_capacity[j], len(pool_i))
                        chosen = np.random.choice(pool_i, size=n_add, replace=False)
                        set_j.update(int(v) for v in chosen)
                        remaining_capacity[j] -= n_add
                        needed -= n_add

                if needed > 0 and remaining_capacity[i] > 0:
                    pool_j = list(set_j - set_i)
                    if pool_j:
                        n_add = min(needed, remaining_capacity[i], len(pool_j))
                        chosen = np.random.choice(pool_j, size=n_add, replace=False)
                        set_i.update(int(v) for v in chosen)
                        remaining_capacity[i] -= n_add
                        needed -= n_add

                # If still needed, add same new variables to both concepts (consumes both capacities).
                while needed > 0 and remaining_capacity[i] > 0 and remaining_capacity[j] > 0:
                    union_ij = set_i | set_j
                    candidates = list(set(range(n_variables)) - union_ij)
                    if not candidates:
                        break
                    v_new = int(np.random.choice(candidates))
                    set_i.add(v_new)
                    set_j.add(v_new)
                    remaining_capacity[i] -= 1
                    remaining_capacity[j] -= 1
                    needed -= 1

    # 4. Fill remaining capacities to match target concept sizes.
    for c_idx in range(n_concepts):
        cap = remaining_capacity[c_idx]
        if cap <= 0:
            continue
        available = list(set(range(n_variables)) - concept_map[c_idx])
        if not available:
            continue
        n_add = min(cap, len(available))
        chosen = np.random.choice(available, size=n_add, replace=False)
        concept_map[c_idx].update(int(v) for v in chosen)

    # 3. Generate Concept Activities: C ~ N(0, 1)
    concept_activities = np.random.standard_normal((n_examples, n_concepts))

    # 4. Generate Dataset X
    # xi values ~ N(mean=mean(activities of associated concepts), std=sigma_i)
    X = np.zeros((n_examples, n_variables))
    
    # Reverse map: which concepts influence which variable?
    var_to_concepts = {v: [] for v in range(n_variables)}
    for c_idx, vars_in_c in concept_map.items():
        for v in vars_in_c:
            var_to_concepts[v].append(c_idx)

    if variable_mean_strategy not in {"mean", "max"}:
        raise ValueError("variable_mean_strategy must be one of: mean, max")

    for v in range(n_variables):
        relevant_concepts = var_to_concepts[v]
        
        if not relevant_concepts:
            # If a variable is not linked to any concept, generate as pure noise centered at 0.
            mean_activity = np.zeros(n_examples)
        else:
            linked_activities = concept_activities[:, relevant_concepts]
            if variable_mean_strategy == "mean":
                # Use the mean activation across concepts linked to this variable.
                mean_activity = np.mean(linked_activities, axis=1)
            else:
                # Use the max activation across concepts linked to this variable.
                mean_activity = np.max(linked_activities, axis=1)
        
        # Sample X from N(mean_activity, sigma_i) where sigma_i ~ Uniform(0, 1)
        sigma_v = np.random.uniform(0.0, 1.0)
        X[:, v] = np.random.normal(loc=mean_activity, scale=sigma_v)

    concept_map_out = {f"concept_{k}": sorted(list(vs)) for k, vs in concept_map.items()}
    return X, concept_activities, concept_map_out

def visualize_results(
    X: np.ndarray,
    activities: np.ndarray,
    concept_map: Dict[str, List[int]],
    n_variables: int,
    save_path: str = "generation_summary.png",
):
    """
    Visualizes the data distribution, concept activities, and the overlap matrix.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Dataset X Distribution
    axes[0, 0].hist(X.flatten(), bins=100, color='royalblue', alpha=0.7)
    axes[0, 0].set_title("Distribution of Dataset X (All Variables)")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Frequency")

    # 2. Concept Activities Distribution
    axes[0, 1].hist(activities.flatten(), bins=100, color='seagreen', alpha=0.7)
    axes[0, 1].set_title("Distribution of Concept Activities")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Frequency")

    # 3. Overlap Decay Curve (Rank-Size Plot)
    # Directional overlap: overlap(i, j) = |Ci ∩ Cj| / |Ci|, normalized by the size of
    # the observed concept i. It is asymmetric, so each ordered pair (i, j) and (j, i)
    # contributes its own value.
    concept_keys = list(concept_map.keys())
    n_concepts = len(concept_keys)
    concept_sets = [set(concept_map[k]) for k in concept_keys]
    overlap_values = []
    for i in range(n_concepts):
        len_i = len(concept_sets[i])
        for j in range(n_concepts):
            if i == j:
                continue
            inter = len(concept_sets[i] & concept_sets[j])
            overlap_values.append(inter / len_i if len_i > 0 else 0.0)

    # 3. Continuous Overlap Decay Curve (Rank-Size Plot)
    # Sort values descending to map "Number of Pairs" (Rank) to "Overlap Ratio"
    sorted_overlaps = np.sort(overlap_values)[::-1]
    # X-axis represents the cumulative number of pairs (1, 2, 3...)
    pair_counts = np.arange(1, len(sorted_overlaps) + 1)

    axes[1, 0].plot(pair_counts, sorted_overlaps, color='orchid', linewidth=3, alpha=0.8)
    axes[1, 0].fill_between(pair_counts, sorted_overlaps, color='orchid', alpha=0.2)
    
    axes[1, 0].set_xscale('log')
    axes[1, 0].set_title("Overlap Decay (Rank-Size Plot)")
    axes[1, 0].set_xlabel("Number of Pairs (Log Scale)")
    axes[1, 0].set_ylabel("Directional Overlap Ratio (|Ci ∩ Cj| / |Ci|)")
    # Headroom above 1.0 so the top of the curve (overlap == 1.0) is visible.
    axes[1, 0].set_ylim(0, 1.05)
    axes[1, 0].grid(True, which="both", ls="-", alpha=0.2)

    # 4. Concept Sizes Distribution
    concept_sizes = [len(vars_list) for vars_list in concept_map.values()]
    axes[1, 1].hist(concept_sizes, bins=30, color='indianred', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Distribution of Concept Sizes")
    axes[1, 1].set_xlabel("Number of Variables in Concept")
    axes[1, 1].set_ylabel("Frequency (Nb of Concepts)")
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Summary visualization saved to: {save_path}")

def save_concept_table_csv(concept_map: Dict[str, List[int]], filename: str = "concept_mapping.json"):
    """Saves concept mapping as a dictionary JSON file."""
    with open(filename, mode="w", encoding="utf-8") as file:
        json.dump(concept_map, file, indent=2)
    print(f"Concept mapping dictionary saved to: {filename}")

def plot_normal_distribution(n_samples: int = 10000, save_path: str = "distribution_plot.png"):
    """Generates samples and plots a histogram of the distribution."""
    data = generate_concept_activity(n_samples)

    plt.figure(figsize=(10, 6))
    # density=True creates a probability density plot instead of raw counts
    plt.hist(data, bins=100, density=True, color='skyblue', edgecolor='black', alpha=0.7)

    plt.title(f"Standard Normal Distribution (n={n_samples})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid(axis='y', alpha=0.3)
    
    plt.savefig(save_path)
    print(f"Plot saved successfully to: {save_path}")

if __name__ == "__main__":
    # Example parameters for data generation
    N_EXAMPLES, N_VARIABLES, N_CONCEPTS, OVERLAP_SKEW = 1000, 2000, 100, 15.0
    N_CLASSES = 10 # Example number of classes

    # Example 1: Fixed size (e.g., all concepts have 5 variables)
    # X, activities, mapping = generate_structured_dataset(
    #     N_EXAMPLES, N_VARIABLES, N_CONCEPTS, size_strategy="fixed", fixed_size=5
    # )

    # Example 2: Range size (e.g., concepts have between 1 and 200 variables)
    X, activities, mapping = generate_structured_dataset(
        N_EXAMPLES, N_VARIABLES, N_CONCEPTS, size_strategy="range", size_range=(1, 200)
    )

    print(f"Generated Dataset X shape: {X.shape}")
    print(f"Generated Concept Activities shape: {activities.shape}")
    
    # Save the concept mapping table to CSV
    save_concept_table_csv(mapping)

    # Run the comprehensive visualization
    visualize_results(X, activities, mapping, N_VARIABLES)