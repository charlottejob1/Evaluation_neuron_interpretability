import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import csv

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
    max_concept_size: int = 500,
    size_strategy: str = "dirichlet",
    fixed_size: int = 10,
    size_range: Tuple[int, int] = (1, 200)
) -> Tuple[np.ndarray, np.ndarray, Dict[int, List[int]]]:
    """
    Generates a dataset X and concept activities based on concept-variable mappings.
    
    Args:
        n_examples: Number of rows (N).
        n_variables: Number of variables (M).
        n_concepts: Number of concepts (K).
        overlap_skew: Skewness of the overlap distribution (Beta distribution beta parameter).
        max_concept_size: The strict limit on the number of variables per concept.
        size_strategy: "dirichlet" (partitions all variables), "fixed" (constant size), or "range" (uniform random).
        fixed_size: Number of variables per concept if strategy is "fixed".
        size_range: (min, max) variables per concept if strategy is "range".

    Returns:
        X: The generated dataset of shape (n_examples, n_variables).
        concept_activities: Matrix of shape (n_examples, n_concepts).
        concept_map: Dictionary mapping concept index to list of variable indices.
    """
    concept_map = {i: [] for i in range(n_concepts)}
    
    # 1. Determine Initial Concept Sizes and Members
    if size_strategy == "dirichlet":
        # Use a Dirichlet distribution to generate skewed proportions for concept sizes
        # alpha < 1 (0.1) ensures a few concepts get many variables and many get few.
        # This strategy ensures every variable belongs to at least one concept.
        proportions = np.random.dirichlet([0.1] * n_concepts)
        assignments = np.random.choice(n_concepts, size=n_variables, p=proportions)
        
        for v_idx, c_idx in enumerate(assignments):
            concept_map[c_idx].append(int(v_idx))
            
        # Ensure no concept is empty
        for c_idx in range(n_concepts):
            if not concept_map[c_idx]:
                largest_c = max(concept_map, key=lambda k: len(concept_map[k]))
                stolen_var = concept_map[largest_c].pop()
                concept_map[c_idx].append(stolen_var)

    elif size_strategy == "fixed":
        # Each concept gets exactly 'fixed_size' variables
        for i in range(n_concepts):
            size = min(fixed_size, n_variables)
            # Pick 'size' unique variables for this concept
            vars_in_c = np.random.choice(n_variables, size=size, replace=False)
            concept_map[i] = [int(v) for v in vars_in_c]
            
    elif size_strategy == "range":
        # Each concept gets a random size in [min, max]
        low, high = size_range
        for i in range(n_concepts):
            size = np.random.randint(low, high + 1)
            size = min(size, n_variables)
            vars_in_c = np.random.choice(n_variables, size=size, replace=False)
            concept_map[i] = [int(v) for v in vars_in_c]

    # 2. Implement Overlap (Pairwise logic with Saturation Control)
    if n_concepts > 1:
        for i in range(n_concepts):
            for j in range(i + 1, n_concepts):
                # Sample a target overlap ratio for this pair
                target_ratio = np.random.beta(1, overlap_skew)
                # Scale the target overlap count relative to the concept capacity
                target_count = int(target_ratio * max_concept_size)
                
                set_i, set_j = set(concept_map[i]), set(concept_map[j])
                intersection = set_i & set_j
                
                if len(intersection) < target_count:
                    needed = target_count - len(intersection)
                    # Safety check: Ensure concept j has room to grow
                    room_in_j = max_concept_size - len(set_j)
                    if room_in_j <= 0:
                        continue

                    # Prioritize variables already in concept i to join concept j
                    pool = list(set_i - set_j)
                    if len(pool) < needed:
                        global_pool = list(set(range(n_variables)) - (set_i | set_j))
                        pool.extend(global_pool)
                    
                    # Cap the number of shared variables by both 'needed' and 'room_in_j'
                    to_share = np.random.choice(pool, size=min(len(pool), needed, room_in_j), replace=False)
                    concept_map[j].extend([int(v) for v in to_share])

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

    for v in range(n_variables):
        relevant_concepts = var_to_concepts[v]
        
        if not relevant_concepts:
            # If a variable is not linked to any concept, generate as pure noise centered at 0.
            mean_activity = np.zeros(n_examples)
        else:
            # Calculate the mean activation across concepts linked to this variable
            mean_activity = np.mean(concept_activities[:, relevant_concepts], axis=1)
        
        # Sample X from N(mean_activity, sigma_i) where sigma_i ~ Uniform(0, 1)
        sigma_v = np.random.uniform(0.0, 1.0)
        X[:, v] = np.random.normal(loc=mean_activity, scale=sigma_v)

    return X, concept_activities, concept_map

def visualize_results(X: np.ndarray, activities: np.ndarray, concept_map: Dict[int, List[int]], n_variables: int):
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
    n_concepts = len(concept_map)
    overlap_values = []
    for i in range(n_concepts):
        for j in range(i + 1, n_concepts):
            intersection = set(concept_map[i]).intersection(set(concept_map[j]))
            overlap_values.append(len(intersection) / n_variables)

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
    axes[1, 0].set_ylabel("Overlap Ratio (|Ci ∩ Cj| / M)")
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].grid(True, which="both", ls="-", alpha=0.2)

    # 4. Concept Sizes Distribution
    concept_sizes = [len(vars_list) for vars_list in concept_map.values()]
    axes[1, 1].hist(concept_sizes, bins=30, color='indianred', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title("Distribution of Concept Sizes")
    axes[1, 1].set_xlabel("Number of Variables in Concept")
    axes[1, 1].set_ylabel("Frequency (Nb of Concepts)")
    axes[1, 1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig("generation_summary.png")
    print("Summary visualization saved to: generation_summary.png")

def save_concept_table_csv(concept_map: Dict[int, List[int]], filename: str = "concept_mapping.csv"):
    """Saves the concept mapping to a CSV file."""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Concept ID", "Nb Variables", "Variables"])
        for c_id, variables in sorted(concept_map.items()):
            vars_str = " ".join(map(str, sorted(variables)))
            writer.writerow([c_id, len(variables), vars_str])
    print(f"Concept mapping saved to: {filename}")

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