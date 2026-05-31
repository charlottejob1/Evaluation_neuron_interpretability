import argparse
import os

import numpy as np

from data_generation import (
    generate_structured_dataset,
    save_concept_table_csv,
    visualize_results,
)

DEFAULTS = {
    "n_examples": 1000,
    "n_variables": 2000,
    "n_concepts": 300,
    # Dirichlet sizing: right-skewed concept sizes (most small, a few up to
    # max_concept_size), with total membership density capped at the factor below.
    # This allows large concepts (up to 200 vars) without the dense packing that a
    # uniform "range" strategy would cause at M=2000, K=300.
    "size_strategy": "dirichlet",
    # Directional overlap = |Ci ∩ Cj| / |Ci| in [0, 1] (normalized by the observed
    # concept's size, so overlap(i, j) != overlap(j, i)). Generation samples symmetric
    # target overlaps with a negative-exponential law (most pairs low, few high).
    # NOTE: size_min=1 forces a residual high-overlap tail, because a size-1 concept's
    # directional overlap is necessarily 0 or 1 whenever its single variable is shared.
    "max_concept_size": 200,
    "fixed_size": 10,
    "size_min": 1,
    "size_max": 200,
    # Strong negative-exponential overlap: high skew + low ceiling so most pairs share
    # few/no variables and very few (large) pairs reach high overlap.
    "overlap_skew": 30.0,
    "overlap_floor": 0.0,
    "overlap_ceiling": 0.2,
    "overlap_convergence_power": 0.4,
    "overlap_reference_concepts": 300,
    "dirichlet_total_assignments_factor": 1.5,
    "variable_mean_strategy": "mean",
    "seed": 12345,
    # Folder for all outputs (override via CLI --output-dir or --run-name).
    "output_dir": "data_generation_output",
    "concept_mapping_name": "concept_mapping.json",
    "summary_plot_name": "generation_summary.png",
    "save_dataset_arrays": True,
}


def resolve_output_dir(output_dir: str = None, run_name: str = None) -> str:
    """
    Resolve output folder:
      - --output-dir <path>  -> use exactly this folder name/path
      - --run-name <name>    -> use data_<name>/
      - otherwise            -> DEFAULTS['output_dir']
    """
    if output_dir is not None:
        return output_dir
    if run_name is not None and str(run_name).strip():
        return f"data_{str(run_name).strip()}"
    return DEFAULTS["output_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run data generation with configurable parameters.",
        epilog=(
            "Examples:\n"
            "  python test_data_generation.py --run-name trial_v1\n"
            "  python test_data_generation.py --output-dir results/data_exp_2025_05_28\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"], help="Number of examples (N).")
    parser.add_argument("--n-variables", type=int, default=DEFAULTS["n_variables"], help="Number of variables (M).")
    parser.add_argument("--n-concepts", type=int, default=DEFAULTS["n_concepts"], help="Number of concepts (K).")

    parser.add_argument(
        "--size-strategy",
        choices=["dirichlet", "fixed", "range"],
        default=DEFAULTS["size_strategy"],
        help="Concept size strategy.",
    )
    parser.add_argument(
        "--max-concept-size",
        type=int,
        default=DEFAULTS["max_concept_size"],
        help="Global upper bound on concept size.",
    )
    parser.add_argument(
        "--fixed-size",
        type=int,
        default=DEFAULTS["fixed_size"],
        help="Concept size when --size-strategy fixed.",
    )
    parser.add_argument(
        "--size-min",
        type=int,
        default=DEFAULTS["size_min"],
        help="Minimum concept size when --size-strategy range.",
    )
    parser.add_argument(
        "--size-max",
        type=int,
        default=DEFAULTS["size_max"],
        help="Maximum concept size when --size-strategy range.",
    )

    parser.add_argument(
        "--overlap-skew",
        type=float,
        default=DEFAULTS["overlap_skew"],
        help="Base lambda for negative-exponential overlap distribution.",
    )
    parser.add_argument(
        "--overlap-floor",
        type=float,
        default=DEFAULTS["overlap_floor"],
        help="Minimum overlap ratio.",
    )
    parser.add_argument(
        "--overlap-ceiling",
        type=float,
        default=DEFAULTS["overlap_ceiling"],
        help="Maximum overlap ratio.",
    )
    parser.add_argument(
        "--overlap-convergence-power",
        type=float,
        default=DEFAULTS["overlap_convergence_power"],
        help="Controls convergence toward low overlap as K grows.",
    )
    parser.add_argument(
        "--overlap-reference-concepts",
        type=int,
        default=DEFAULTS["overlap_reference_concepts"],
        help="Reference K used to normalize overlap convergence scaling.",
    )
    parser.add_argument(
        "--dirichlet-total-assignments-factor",
        type=float,
        default=DEFAULTS["dirichlet_total_assignments_factor"],
        help="For dirichlet only: total memberships factor relative to M (>=1).",
    )
    parser.add_argument(
        "--variable-mean-strategy",
        choices=["mean", "max"],
        default=DEFAULTS["variable_mean_strategy"],
        help="How to compute Gaussian mean for variables from linked concept activities.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULTS["seed"],
        help="Random seed for reproducible generation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where all outputs are saved. "
            f"If omitted, uses DEFAULTS['output_dir'] ({DEFAULTS['output_dir']!r}) "
            "unless --run-name is set."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Shortcut for output folder: saves to data_<run-name>/ (e.g. trial_v1 -> data_trial_v1/).",
    )
    parser.add_argument(
        "--concept-mapping-name",
        type=str,
        default=DEFAULTS["concept_mapping_name"],
        help="Concept mapping JSON filename inside the output directory.",
    )
    parser.add_argument(
        "--summary-plot-name",
        type=str,
        default=DEFAULTS["summary_plot_name"],
        help="Summary plot PNG filename inside the output directory.",
    )
    parser.add_argument(
        "--no-save-arrays",
        action="store_true",
        help="Do not save dataset_X.npy and concept_activities.npy.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = os.path.abspath(resolve_output_dir(args.output_dir, args.run_name))
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    X, activities, mapping = generate_structured_dataset(
        n_examples=args.n_examples,
        n_variables=args.n_variables,
        n_concepts=args.n_concepts,
        overlap_skew=args.overlap_skew,
        max_concept_size=args.max_concept_size,
        size_strategy=args.size_strategy,
        fixed_size=args.fixed_size,
        size_range=(args.size_min, args.size_max),
        overlap_floor=args.overlap_floor,
        overlap_ceiling=args.overlap_ceiling,
        overlap_convergence_power=args.overlap_convergence_power,
        overlap_reference_concepts=args.overlap_reference_concepts,
        dirichlet_total_assignments_factor=args.dirichlet_total_assignments_factor,
        variable_mean_strategy=args.variable_mean_strategy,
        seed=args.seed,
    )

    print(f"Generated Dataset X shape: {X.shape}")
    print(f"Generated Concept Activities shape: {activities.shape}")

    concept_mapping_path = os.path.join(
        args.output_dir, os.path.basename(args.concept_mapping_name)
    )
    summary_plot_path = os.path.join(
        args.output_dir, os.path.basename(args.summary_plot_name)
    )

    save_concept_table_csv(mapping, filename=concept_mapping_path)
    visualize_results(
        X, activities, mapping, args.n_variables, save_path=summary_plot_path
    )

    if not args.no_save_arrays:
        x_path = os.path.join(args.output_dir, "dataset_X.npy")
        activities_path = os.path.join(args.output_dir, "concept_activities.npy")
        np.save(x_path, X)
        np.save(activities_path, activities)
        print(f"Saved dataset X to: {x_path}")
        print(f"Saved concept activities to: {activities_path}")


if __name__ == "__main__":
    main()
