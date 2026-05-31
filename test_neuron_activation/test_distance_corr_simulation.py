"""
End-to-end distance-correlation studies on simulated data (one PNG per study).

For each concept i we compute the Pearson correlation between the pairwise distance
matrix of the input X reduced to the variables of concept i and the pairwise distance
matrix of neuron i's activation (simulated by `simulate_neuron_activations`). This
yields one correlation value per concept, mirroring
`compute_distance_corr_one_pathway_one_dim` in `distances_metrics.py`.

Fixed simulation dimensions (always applied in main()):
  N (examples) = 1000
  M (variables) when not swept = 2000
  K (concepts) when not swept = 300

Studies:
  1) Distance correlation vs beta (0.0 to 1.0, step 0.1) -- blue box plot
  2) Impact of M (number of variables) at fixed beta values
  3) Impact of K (number of concepts) at fixed beta values
  4) Impact of overlap regime (low / medium / high) at fixed beta values
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from distance_corr_simulation import run_distance_corr_pipeline


# Fixed dimensions for fair comparison (kept identical to the probability-metric study).
FIXED_N_EXAMPLES = 1000
FIXED_M = 2000
FIXED_K = 300

# Base overlap profile used for beta / M / K sweeps (standard settings).
BASE_OVERLAP_PROFILE = {
    "name": "base",
    "overlap_skew": 30.0,
    "overlap_floor": 0.0,
    "overlap_ceiling": 0.2,
    "overlap_convergence_power": 0.4,
    "overlap_reference_concepts": FIXED_K,
}

# Overlap regimes (fixed M, K; only the overlap generation parameters differ),
# matching the probability-metric overlap study.
OVERLAP_REGIME_CONFIGS: List[Dict[str, object]] = [
    {
        "name": "low_overlap",
        "overlap_skew": 5.0,
        "overlap_floor": 0.0,
        "overlap_ceiling": 0.15,
        "overlap_convergence_power": 0.4,
        "dirichlet_total_assignments_factor": 1.8,
    },
    {
        "name": "medium_overlap",
        "overlap_skew": 2.0,
        "overlap_floor": 0.0,
        "overlap_ceiling": 0.5,
        "overlap_convergence_power": 0.4,
        "dirichlet_total_assignments_factor": 1.8,
    },
    {
        "name": "high_overlap",
        "overlap_skew": 0.5,
        "overlap_floor": 0.0,
        "overlap_ceiling": 1.0,
        "overlap_convergence_power": 0.15,
        "dirichlet_total_assignments_factor": 4.5,
    },
]

RUN_PARAMETERS_FILENAME = "run_parameters.txt"


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def apply_fixed_simulation_dims(args: argparse.Namespace) -> None:
    """Pin N, M, K so CLI flags cannot change the standard comparison settings."""
    args.n_examples = FIXED_N_EXAMPLES
    args.n_variables = FIXED_M
    args.n_concepts = FIXED_K


def resolve_output_dir(output_dir: str = None, run_name: str = None) -> str:
    if output_dir is not None:
        return output_dir
    if run_name is not None and str(run_name).strip():
        return f"plots_{str(run_name).strip()}"
    return DEFAULTS["output_dir"]


def _concept_sizes_for_k(K: int, args: argparse.Namespace) -> tuple:
    """
    Scale concept size so packing density rho = K * mean_size / M stays constant as K
    varies. This keeps coverage and the overlap distribution comparable across K,
    isolating the effect of the number of concepts. Reference point: at K = FIXED_K,
    sizes are (args.size_min, args.size_max).
    """
    base_avg = (args.size_min + args.size_max) / 2.0
    rho = FIXED_K * base_avg / FIXED_M
    target_avg = rho * FIXED_M / K
    smin = max(2, round(target_avg * args.size_min / base_avg))
    smax = max(smin + 1, round(target_avg * args.size_max / base_avg))
    return int(smin), int(smax)


class RunParameterLog:
    """Accumulates per-study parameters and writes run_parameters.txt."""

    def __init__(self, args: argparse.Namespace, out_dir: str) -> None:
        self.args = args
        self.out_dir = out_dir
        self.sections: List[str] = []

    def _global_block(self) -> str:
        a = self.args
        lines = [
            "distance_corr_simulation — run parameters",
            "=" * 60,
            f"Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
            f"Output directory: {self.out_dir}",
            "",
            "Global defaults (fixed unless a study sweeps the dimension)",
            "-" * 40,
            f"n_examples (N): {a.n_examples}",
            f"n_variables (M) when fixed: {a.n_variables}",
            f"n_concepts (K) when fixed: {a.n_concepts}",
            f"size_strategy: {a.size_strategy}",
            f"size_min / size_max: {a.size_min} / {a.size_max}",
            f"max_concept_size: {a.max_concept_size}",
            f"dirichlet_total_assignments_factor: {a.dirichlet_total_assignments_factor}",
            f"variable_mean_strategy: {a.variable_mean_strategy}",
            f"alpha_mode: {a.alpha_mode}",
            f"alpha_exp_scale: {a.alpha_exp_scale}",
            f"seed: {a.seed}",
            "",
            "Base overlap profile (beta / M / K sweeps)",
            "-" * 40,
        ]
        for k, v in BASE_OVERLAP_PROFILE.items():
            lines.append(f"  {k}: {v}")
        lines.append("")
        return "\n".join(lines)

    def add_section(self, title: str, params: Dict[str, object]) -> None:
        lines = [f"Study: {title}", "-" * 40]
        for key, value in params.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        self.sections.append("\n".join(lines))

    def write(self, final: bool = False) -> str:
        path = os.path.join(self.out_dir, RUN_PARAMETERS_FILENAME)
        body = self._global_block() + "\n".join(self.sections)
        if not final:
            body += "\n(Run in progress — file updated after each study.)\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(body)
        if final:
            print(f"Saved: {path}")
        return path


DEFAULTS = {
    "n_examples": FIXED_N_EXAMPLES,
    "n_variables": FIXED_M,
    "n_concepts": FIXED_K,
    "size_strategy": "dirichlet",
    "max_concept_size": 200,
    "size_min": 1,
    "size_max": 200,
    "dirichlet_total_assignments_factor": 1.5,
    "variable_mean_strategy": "mean",
    "alpha_mode": "uniform",
    "alpha_exp_scale": 1.0,
    "seed": 12345,
    "output_dir": "plots_distance_corr_simulation",
    "beta_sweep": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    "fixed_betas": "0.0,0.3,0.6,0.9",
    "M_values": "500,1000,2000,4000",
    "K_values": "50,100,300,500,700",
}

METRIC_COLUMN = "distance_corr"
METRIC_LABEL = "Pearson distance correlation (input vs neuron)"


def _pipeline_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "n_examples": FIXED_N_EXAMPLES,
        "size_strategy": args.size_strategy,
        "max_concept_size": args.max_concept_size,
        "size_min": args.size_min,
        "size_max": args.size_max,
        "dirichlet_total_assignments_factor": args.dirichlet_total_assignments_factor,
        "variable_mean_strategy": args.variable_mean_strategy,
        "alpha_mode": args.alpha_mode,
        "alpha_exp_scale": args.alpha_exp_scale,
        "seed": args.seed,
    }


def _rows_to_dataframe(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "beta" in df.columns:
        df["beta"] = df["beta"].astype(float)
    if METRIC_COLUMN in df.columns:
        df[METRIC_COLUMN] = df[METRIC_COLUMN].astype(float)
    if "n_variables" in df.columns:
        df["n_variables"] = df["n_variables"].astype(int)
    if "n_concepts" in df.columns:
        df["n_concepts"] = df["n_concepts"].astype(int)
    return df


def _require_plot_columns(df: pd.DataFrame, columns: List[str], study_name: str) -> bool:
    if df.empty:
        print(f"[{study_name}] No correlation rows produced; skipping plot.")
        return False
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[{study_name}] Missing columns {missing}; skipping plot.")
        return False
    return True


def _corr_ylim(values: np.ndarray) -> tuple:
    """Keep the top at 1.0 but leave headroom below for any negative correlations."""
    if len(values) == 0:
        return (-0.05, 1.0)
    lo = float(np.nanmin(values))
    return (min(-0.05, lo - 0.05), 1.0)


def study_beta_sweep(
    args: argparse.Namespace,
    out_dir: str,
    param_log: Optional[RunParameterLog] = None,
) -> pd.DataFrame:
    betas = _parse_float_list(args.beta_sweep)
    rows: List[Dict[str, object]] = []
    kw = _pipeline_kwargs(args)

    if param_log is not None:
        param_log.add_section(
            "beta_sweep (01_distance_corr_by_beta.png)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": args.n_variables,
                "K (fixed)": args.n_concepts,
                "beta_values": args.beta_sweep,
                "overlap_profile": BASE_OVERLAP_PROFILE,
                "output_csv": "distance_corr_beta_sweep.csv",
            },
        )

    for beta in betas:
        print(f"[beta sweep] beta={beta:.1f}")
        batch = run_distance_corr_pipeline(
            n_variables=FIXED_M,
            n_concepts=FIXED_K,
            beta=beta,
            overlap_profile=BASE_OVERLAP_PROFILE,
            **kw,
        )
        rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "distance_corr_beta_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["beta", METRIC_COLUMN], "beta sweep"):
        return df

    betas_sorted = sorted(df["beta"].unique())
    data_by_beta = [df.loc[df["beta"] == b, METRIC_COLUMN].values for b in betas_sorted]

    plt.figure(figsize=(12, 6))
    bp = plt.boxplot(
        data_by_beta,
        positions=range(len(betas_sorted)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
    )
    for box in bp["boxes"]:
        box.set(facecolor="steelblue", edgecolor="navy", alpha=0.7)
    for element in ("whiskers", "caps", "medians"):
        for item in bp[element]:
            item.set(color="navy")
    plt.xticks(range(len(betas_sorted)), [f"{b:.1f}" for b in betas_sorted])
    plt.ylim(*_corr_ylim(df[METRIC_COLUMN].values))
    plt.xlabel("beta")
    plt.ylabel(METRIC_LABEL)
    plt.title("Influence of beta on distance-correlation distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "01_distance_corr_by_beta.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def study_M_sweep(
    args: argparse.Namespace,
    out_dir: str,
    param_log: Optional[RunParameterLog] = None,
) -> pd.DataFrame:
    betas = _parse_float_list(args.fixed_betas)
    M_values = _parse_int_list(args.M_values)
    rows: List[Dict[str, object]] = []
    kw = _pipeline_kwargs(args)

    if param_log is not None:
        param_log.add_section(
            "M_sweep (02_distance_corr_by_M.png)",
            {
                "n_examples": args.n_examples,
                "M_values (swept)": args.M_values,
                "K (fixed)": args.n_concepts,
                "beta_values": args.fixed_betas,
                "overlap_profile": BASE_OVERLAP_PROFILE,
                "output_csv": "distance_corr_M_sweep.csv",
            },
        )

    for M in M_values:
        for beta in betas:
            print(f"[M sweep] M={M}, beta={beta:.1f}")
            batch = run_distance_corr_pipeline(
                n_variables=M,
                n_concepts=FIXED_K,
                beta=beta,
                overlap_profile=BASE_OVERLAP_PROFILE,
                **kw,
            )
            for row in batch:
                row["n_variables"] = M
            rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "distance_corr_M_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["n_variables", "beta", METRIC_COLUMN], "M sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="n_variables",
        y=METRIC_COLUMN,
        hue="beta",
        showfliers=False,
    )
    plt.ylim(*_corr_ylim(df[METRIC_COLUMN].values))
    plt.xlabel("M (number of variables)")
    plt.ylabel(METRIC_LABEL)
    plt.title("Impact of M on distance correlation at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "02_distance_corr_by_M.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def study_K_sweep(
    args: argparse.Namespace,
    out_dir: str,
    param_log: Optional[RunParameterLog] = None,
) -> pd.DataFrame:
    betas = _parse_float_list(args.fixed_betas)
    K_values = _parse_int_list(args.K_values)
    rows: List[Dict[str, object]] = []
    kw = _pipeline_kwargs(args)

    size_schedule = {K: _concept_sizes_for_k(K, args) for K in K_values}

    if param_log is not None:
        param_log.add_section(
            "K_sweep (03_distance_corr_by_K.png)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": args.n_variables,
                "K_values (swept)": args.K_values,
                "beta_values": args.fixed_betas,
                "overlap_profile": BASE_OVERLAP_PROFILE,
                "concept_sizes_per_K (density-preserving)": {
                    K: f"{lo}-{hi}" for K, (lo, hi) in size_schedule.items()
                },
                "note": (
                    "Concept size scales as ~M/K so packing density (and overlap) is "
                    "comparable across K; avoids zero-overlap degeneracy at small K."
                ),
                "output_csv": "distance_corr_K_sweep.csv",
            },
        )

    for K in K_values:
        profile = BASE_OVERLAP_PROFILE.copy()
        profile["overlap_reference_concepts"] = K
        size_min, size_max = size_schedule[K]
        K_kw = {**kw, "size_min": size_min, "size_max": size_max, "max_concept_size": size_max}
        for beta in betas:
            print(f"[K sweep] K={K}, sizes=({size_min},{size_max}), beta={beta:.1f}")
            batch = run_distance_corr_pipeline(
                n_variables=FIXED_M,
                n_concepts=K,
                beta=beta,
                overlap_profile=profile,
                **K_kw,
            )
            for row in batch:
                row["n_concepts"] = K
            rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "distance_corr_K_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["n_concepts", "beta", METRIC_COLUMN], "K sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="n_concepts",
        y=METRIC_COLUMN,
        hue="beta",
        showfliers=False,
    )
    plt.ylim(*_corr_ylim(df[METRIC_COLUMN].values))
    plt.xlabel("K (number of concepts)")
    plt.ylabel(METRIC_LABEL)
    plt.title("Impact of K on distance correlation at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "03_distance_corr_by_K.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def study_overlap_sweep(
    args: argparse.Namespace,
    out_dir: str,
    regime_configs: List[Dict[str, object]] = None,
    param_log: Optional[RunParameterLog] = None,
) -> pd.DataFrame:
    """
    Compare low / medium / high overlap regimes at fixed M, K (only the overlap
    generation parameters differ), for several fixed beta values.
    """
    if regime_configs is None:
        regime_configs = OVERLAP_REGIME_CONFIGS

    betas = _parse_float_list(args.fixed_betas)
    rows: List[Dict[str, object]] = []
    kw = _pipeline_kwargs(args)

    if param_log is not None:
        param_log.add_section(
            "overlap_sweep (04_distance_corr_by_overlap.png)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": FIXED_M,
                "K (fixed)": FIXED_K,
                "concept_sizes": f"{args.size_min}-{args.size_max} ({args.size_strategy})",
                "beta_values": args.fixed_betas,
                "overlap_regimes": "; ".join(
                    f"{r['name']}(skew={r['overlap_skew']}, ceiling={r['overlap_ceiling']}, "
                    f"dirichlet={r['dirichlet_total_assignments_factor']})"
                    for r in regime_configs
                ),
                "output_csv": "distance_corr_overlap_sweep.csv",
            },
        )

    for regime in regime_configs:
        profile = {
            "name": str(regime["name"]),
            "overlap_skew": float(regime["overlap_skew"]),
            "overlap_floor": float(regime["overlap_floor"]),
            "overlap_ceiling": float(regime["overlap_ceiling"]),
            "overlap_convergence_power": float(regime["overlap_convergence_power"]),
            "overlap_reference_concepts": FIXED_K,
        }
        # Each regime has its own concept density (dirichlet factor); override the default.
        regime_kw = {
            **kw,
            "dirichlet_total_assignments_factor": float(
                regime["dirichlet_total_assignments_factor"]
            ),
        }
        for beta in betas:
            print(f"[overlap sweep] regime={profile['name']}, beta={beta:.1f}")
            batch = run_distance_corr_pipeline(
                n_variables=FIXED_M,
                n_concepts=FIXED_K,
                beta=beta,
                overlap_profile=profile,
                **regime_kw,
            )
            rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "distance_corr_overlap_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["overlap_profile", "beta", METRIC_COLUMN], "overlap sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="overlap_profile",
        y=METRIC_COLUMN,
        hue="beta",
        order=["low_overlap", "medium_overlap", "high_overlap"],
        showfliers=False,
    )
    plt.ylim(*_corr_ylim(df[METRIC_COLUMN].values))
    plt.xlabel(f"Overlap regime (M={FIXED_M}, K={FIXED_K} fixed)")
    plt.ylabel(METRIC_LABEL)
    plt.title("Impact of overlap regime on distance correlation at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "04_distance_corr_by_overlap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run distance-correlation simulation studies and save plots/CSVs.",
        epilog=(
            "Examples:\n"
            "  python test_distance_corr_simulation.py --run-name dcorr_v1\n"
            "  python test_distance_corr_simulation.py --output-dir results/dcorr_2026\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n-examples", type=int, default=DEFAULTS["n_examples"])
    parser.add_argument("--n-variables", type=int, default=DEFAULTS["n_variables"])
    parser.add_argument("--n-concepts", type=int, default=DEFAULTS["n_concepts"])
    parser.add_argument("--size-strategy", default=DEFAULTS["size_strategy"])
    parser.add_argument("--max-concept-size", type=int, default=DEFAULTS["max_concept_size"])
    parser.add_argument("--size-min", type=int, default=DEFAULTS["size_min"])
    parser.add_argument("--size-max", type=int, default=DEFAULTS["size_max"])
    parser.add_argument(
        "--dirichlet-total-assignments-factor",
        type=float,
        default=DEFAULTS["dirichlet_total_assignments_factor"],
    )
    parser.add_argument("--variable-mean-strategy", default=DEFAULTS["variable_mean_strategy"])
    parser.add_argument("--alpha-mode", default=DEFAULTS["alpha_mode"])
    parser.add_argument("--alpha-exp-scale", type=float, default=DEFAULTS["alpha_exp_scale"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory where all plots and CSVs are saved. "
            f"If omitted, uses DEFAULTS['output_dir'] ({DEFAULTS['output_dir']!r}) "
            "unless --run-name is set."
        ),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Shortcut for output folder: saves to plots_<run-name>/.",
    )
    parser.add_argument("--beta-sweep", type=str, default=DEFAULTS["beta_sweep"])
    parser.add_argument("--fixed-betas", type=str, default=DEFAULTS["fixed_betas"])
    parser.add_argument("--M-values", type=str, default=DEFAULTS["M_values"])
    parser.add_argument("--K-values", type=str, default=DEFAULTS["K_values"])
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer beta/M/K sweep values (N=1000, M=2000, K=300 unchanged).",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default="beta,M,K,overlap",
        help="Comma-separated: beta,M,K,overlap",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = os.path.abspath(resolve_output_dir(args.output_dir, args.run_name))
    apply_fixed_simulation_dims(args)
    if args.quick:
        args.M_values = "500,1000,2000"
        args.K_values = "100,200,300"
        args.fixed_betas = "0.0,0.5,1.0"
        args.beta_sweep = "0.0,0.5,1.0"
        print("[quick] Reduced beta/M/K sweeps; N/M/K defaults unchanged.")

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    print(f"Output directory: {args.output_dir}")
    print(f"Fixed defaults: N={args.n_examples}, M={args.n_variables}, K={args.n_concepts}")

    studies = {s.strip().lower() for s in args.studies.split(",") if s.strip()}
    param_log = RunParameterLog(args, args.output_dir)
    param_log.add_section(
        "studies_requested",
        {"studies": ", ".join(sorted(studies)), "quick_mode": args.quick},
    )

    if "beta" in studies:
        study_beta_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "m" in studies:
        study_M_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "k" in studies:
        study_K_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "overlap" in studies:
        study_overlap_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()

    param_log.write(final=True)
    print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
