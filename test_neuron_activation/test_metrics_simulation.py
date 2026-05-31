"""
End-to-end metrics simulation tests with seaborn figures (one PNG per study).

Fixed simulation dimensions (always applied in main()):
  N (examples) = 1000
  M (variables) when not swept = 2000
  K (concepts) when not swept = 300

Studies:
  1) Probability distributions vs beta (0.0 to 1.0, step 0.1)
  2) Impact of overlap level at fixed beta values
  3) Impact of M (number of variables) at fixed beta values
  4) Impact of K (number of concepts) at fixed beta values
  5) Overlap threshold filter on high-overlap data vs beta (05-07 PNGs)
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_generation import generate_structured_dataset
from metrics_simulation import (
    compute_concept_overlap_matrix,
    compute_reduction_tensor,
    probability_rows_from_reduction,
    run_probability_pipeline,
)


# Fixed dimensions for fair comparison (overlap regimes and baseline studies).
FIXED_N_EXAMPLES = 1000
FIXED_M = 2000
FIXED_K = 300

DEFAULTS = {
    "n_examples": FIXED_N_EXAMPLES,
    "n_variables": FIXED_M,
    "n_concepts": FIXED_K,
    # Dirichlet sizing (matches data_trial_v6): right-skewed sizes (most small, a few
    # up to max_concept_size), density capped by dirichlet_total_assignments_factor.
    "size_strategy": "dirichlet",
    "max_concept_size": 200,
    "size_min": 1,
    "size_max": 200,
    "dirichlet_total_assignments_factor": 1.5,
    "variable_mean_strategy": "mean",
    "alpha_mode": "uniform",
    "alpha_exp_scale": 1.0,
    "seed": 12345,
    # Folder for all PNG/CSV outputs (override via CLI --output-dir or --run-name).
    "output_dir": "plots_metrics_simulation",
    # beta 0, 0.1, ..., 1.0
    "beta_sweep": "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    "fixed_betas": "0.0,0.3,0.6,0.9",
    "M_values": "500,1000,2000,4000",
    "K_values": "50,100,300,500, 700",
    # Used only by threshold study (pairs with overlap < threshold).
    "overlap_thresholds": "0.02,0.05,0.1,0.15,0.25,0.35,0.5,0.65,0.8",
}

# Base overlap profile used for beta / M / K sweeps (standard settings).
# Overlap is directional: O[i, j] = |Ci ∩ Cj| / |Ci| in [0, 1] (normalized by the
# observed concept's size); negative-exponential generation law puts most pairs at low
# overlap (below 0.5) with few high-overlap pairs.
BASE_OVERLAP_PROFILE = {
    "name": "base",
    "overlap_skew": 30.0,
    "overlap_floor": 0.0,
    "overlap_ceiling": 0.2,
    "overlap_convergence_power": 0.4,
    "overlap_reference_concepts": FIXED_K,
}

# Overlap regimes (02): fixed M=2000, K=300; only overlap generation parameters differ.
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

# What each overlap regime means (generation intent + how to read realized overlap).
OVERLAP_REGIME_DEFINITIONS: Dict[str, str] = {
    "low_overlap": (
        "Fixed M=2000, K=300. Directional overlap O[i,j]=|Ci∩Cj|/|Ci|. High overlap_skew and low "
        "overlap_ceiling (little variable sharing). Expected: low pairwise overlap."
    ),
    "medium_overlap": (
        "Fixed M=2000, K=300. Directional overlap O[i,j]=|Ci∩Cj|/|Ci|. Balanced overlap_skew / "
        "overlap_ceiling. Expected: mid-range pairwise overlap."
    ),
    "high_overlap": (
        "Fixed M=2000, K=300. Directional overlap O[i,j]=|Ci∩Cj|/|Ci|. Low overlap_skew and high "
        "overlap_ceiling (heavy sharing). Expected: high pairwise overlap."
    ),
}

# Threshold study (05-07): fixed M, K; wide overlap via generation parameters only.
THRESHOLD_STUDY_CONFIG: Dict[str, object] = {
    "name": "wide_overlap_spectrum",
    "overlap_skew": 1.0,
    "overlap_floor": 0.0,
    "overlap_ceiling": 1.0,
    "overlap_convergence_power": 0.2,
    "dirichlet_total_assignments_factor": 3.8,
}

RUN_PARAMETERS_FILENAME = "run_parameters.txt"


class RunParameterLog:
    """Accumulates per-study parameters and writes run_parameters.txt."""

    def __init__(self, args: argparse.Namespace, out_dir: str) -> None:
        self.args = args
        self.out_dir = out_dir
        self.sections: List[str] = []
        self._header_written = False

    def _global_block(self) -> str:
        a = self.args
        lines = [
            "metrics_simulation — run parameters",
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


def _normalize_overlap_regime(
    regime: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    """Attach fixed M, K, and shared concept-size settings for fair overlap comparison."""
    return {
        **regime,
        "n_variables": FIXED_M,
        "n_concepts": FIXED_K,
        "size_strategy": args.size_strategy,
        "size_min": args.size_min,
        "size_max": args.size_max,
        "max_concept_size": args.max_concept_size,
    }


def _normalize_overlap_regimes(
    regimes: List[Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    return [_normalize_overlap_regime(r, args) for r in regimes]


def _normalize_threshold_config(
    config: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    return _normalize_overlap_regime(config, args)


def _format_regimes_for_log(regimes: List[Dict[str, object]]) -> str:
    parts = []
    for r in regimes:
        parts.append(
            f"{r['name']}(skew={r['overlap_skew']}, ceiling={r['overlap_ceiling']}, "
            f"dirichlet={r['dirichlet_total_assignments_factor']})"
        )
    return "; ".join(parts)


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_int_list(raw: str) -> List[int]:
    return [int(float(x.strip())) for x in raw.split(",") if x.strip()]


def _require_plot_columns(df: pd.DataFrame, columns: List[str], study_name: str) -> bool:
    if df.empty:
        print(f"[{study_name}] No probability rows produced; skipping plot.")
        return False
    missing = [c for c in columns if c not in df.columns]
    if missing:
        print(f"[{study_name}] Missing columns {missing}; skipping plot.")
        return False
    return True


def _rows_to_dataframe(rows: List[Dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if "beta" in df.columns:
        df["beta"] = df["beta"].astype(float)
    if "probability" in df.columns:
        df["probability"] = df["probability"].astype(float)
    if "overlap_threshold" in df.columns:
        df["overlap_threshold"] = df["overlap_threshold"].astype(float)
    if "pair_overlap" in df.columns:
        df["pair_overlap"] = df["pair_overlap"].astype(float)
    if "n_variables" in df.columns:
        df["n_variables"] = df["n_variables"].astype(int)
    if "n_concepts" in df.columns:
        df["n_concepts"] = df["n_concepts"].astype(int)
    return df


def apply_fixed_simulation_dims(args: argparse.Namespace) -> None:
    """Pin N, M, K so CLI flags cannot change the standard comparison settings."""
    args.n_examples = FIXED_N_EXAMPLES
    args.n_variables = FIXED_M
    args.n_concepts = FIXED_K


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


def _regime_to_overlap_profile(regime: Dict[str, object]) -> Dict[str, object]:
    return {
        "name": regime["name"],
        "overlap_skew": float(regime["overlap_skew"]),
        "overlap_floor": float(regime["overlap_floor"]),
        "overlap_ceiling": float(regime["overlap_ceiling"]),
        "overlap_convergence_power": float(regime["overlap_convergence_power"]),
        "overlap_reference_concepts": int(regime["n_concepts"]),
    }


def _ensure_regime_capacity(regime: Dict[str, object]) -> None:
    """Heuristic check: range strategy needs sum of target sizes >= M."""
    M = int(regime["n_variables"])
    K = int(regime["n_concepts"])
    size_min = int(regime["size_min"])
    size_max = int(regime["max_concept_size"])
    expected_sum = K * (size_min + size_max) / 2.0
    if K * size_max < M or expected_sum < M:
        raise ValueError(
            f"Regime '{regime['name']}': likely insufficient capacity for M={M} "
            f"(K={K}, sizes=[{size_min},{size_max}], expected sum≈{expected_sum:.0f}). "
            "Increase n_concepts or size_max."
        )


def _generate_dataset_for_regime(
    regime: Dict[str, object],
    args: argparse.Namespace,
) -> tuple[np.ndarray, Dict[str, List[int]], int, Dict[str, object]]:
    """Generate (X, concept_map) for one overlap regime (M, K fixed; overlap params vary)."""
    _ensure_regime_capacity(regime)
    profile = _regime_to_overlap_profile(regime)
    X, _, concept_map = generate_structured_dataset(
        n_examples=args.n_examples,
        n_variables=int(regime["n_variables"]),
        n_concepts=int(regime["n_concepts"]),
        overlap_skew=float(profile["overlap_skew"]),
        max_concept_size=int(regime["max_concept_size"]),
        size_strategy=str(regime.get("size_strategy", args.size_strategy)),
        size_range=(int(regime["size_min"]), int(regime["size_max"])),
        overlap_floor=float(profile["overlap_floor"]),
        overlap_ceiling=float(profile["overlap_ceiling"]),
        overlap_convergence_power=float(profile["overlap_convergence_power"]),
        overlap_reference_concepts=int(profile["overlap_reference_concepts"]),
        dirichlet_total_assignments_factor=float(regime["dirichlet_total_assignments_factor"]),
        variable_mean_strategy=args.variable_mean_strategy,
        seed=args.seed,
    )
    return X, concept_map, int(regime["n_variables"]), profile


def _pairwise_overlap_values(concept_map: Dict[str, List[int]], n_variables: int) -> np.ndarray:
    """Off-diagonal directional overlap ratios O[i, j] = |Ci ∩ Cj| / |Ci| for all ordered concept pairs."""
    overlap_matrix = compute_concept_overlap_matrix(concept_map, n_variables)
    K = overlap_matrix.shape[0]
    mask = ~np.eye(K, dtype=bool)
    return overlap_matrix[mask]


def _log_pairwise_overlap_stats(concept_map: Dict[str, List[int]], n_variables: int) -> None:
    off_diag = _pairwise_overlap_values(concept_map, n_variables)
    print(
        f"  realized pairwise overlap: min={off_diag.min():.4f}, "
        f"median={np.median(off_diag):.4f}, max={off_diag.max():.4f}, "
        f"mean={off_diag.mean():.4f}"
    )


def _regime_panel_annotation(regime: Dict[str, object], args: argparse.Namespace, off_diag: np.ndarray) -> str:
    """Multi-line context block for a regime subplot."""
    return (
        f"N examples: {args.n_examples}\n"
        f"M variables: {regime['n_variables']}\n"
        f"K concepts: {regime['n_concepts']}\n"
        f"Concept sizes: {regime['size_min']}-{regime['size_max']} ({regime['size_strategy']})\n"
        f"Overlap gen.: skew={regime['overlap_skew']}, "
        f"ceiling={regime['overlap_ceiling']}, "
        f"dirichlet={regime['dirichlet_total_assignments_factor']}\n"
        f"Realized |Ci∩Cj|/|Ci|: min={off_diag.min():.3f}, "
        f"med={np.median(off_diag):.3f}, max={off_diag.max():.3f}"
    )


def _generate_regime_datasets(
    regime_configs: List[Dict[str, object]],
    args: argparse.Namespace,
) -> List[Dict[str, object]]:
    """Generate one dataset per overlap regime (used for distribution plot + probability sweep)."""
    cached: List[Dict[str, object]] = []
    for regime in regime_configs:
        name = str(regime["name"])
        print(f"[overlap regime] generating {name} (M={regime['n_variables']}, K={regime['n_concepts']})")
        X, concept_map, M, profile = _generate_dataset_for_regime(regime, args)
        off_diag = _pairwise_overlap_values(concept_map, M)
        _log_pairwise_overlap_stats(concept_map, M)
        cached.append(
            {
                "regime": regime,
                "name": name,
                "X": X,
                "concept_map": concept_map,
                "M": M,
                "profile": profile,
                "off_diag": off_diag,
            }
        )
    return cached


def plot_overlap_regime_distributions(
    regime_cache: List[Dict[str, object]],
    args: argparse.Namespace,
    out_dir: str,
) -> pd.DataFrame:
    """
    One figure with three panels (low / medium / high): histogram of realized
    pairwise overlap for the dataset generated under each regime's M, K, overlap settings.
    """
    order = ["low_overlap", "medium_overlap", "high_overlap"]
    cache_by_name = {item["name"]: item for item in regime_cache}
    records: List[Dict[str, object]] = []

    n_panels = len(regime_cache)
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=(6.5 * n_panels, 6.5),
        sharey=True,
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    panel_order = [n for n in order if n in cache_by_name] + [
        n for n in cache_by_name if n not in order
    ]

    colors = {
        "low_overlap": "steelblue",
        "medium_overlap": "darkorange",
        "high_overlap": "firebrick",
    }

    for ax, name in zip(axes, panel_order):
        item = cache_by_name[name]
        regime = item["regime"]
        off_diag = item["off_diag"]
        for v in off_diag:
            records.append({"overlap_regime": name, "pair_overlap": float(v)})

        sns.histplot(
            x=off_diag,
            bins=40,
            stat="density",
            kde=True,
            ax=ax,
            color=colors.get(name, "gray"),
        )
        ax.set_xlabel("Directional overlap |Ci ∩ Cj| / |Ci|")
        ax.set_ylabel("Density" if ax is axes[0] else "")
        ax.set_title(name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlim(0.0, 1.0)

        definition = OVERLAP_REGIME_DEFINITIONS.get(name, "")
        ann = _regime_panel_annotation(regime, args, off_diag)
        text = f"{definition}\n\n{ann}" if definition else ann
        ax.text(
            0.98,
            0.98,
            text,
            transform=ax.transAxes,
            fontsize=7.5,
            va="top",
            ha="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        )

    fig.suptitle(
        "Realized pairwise overlap distributions per regime\n"
        f"(fixed M={FIXED_M}, K={FIXED_K}; only overlap generation parameters differ)",
        fontsize=13,
    )
    path = os.path.join(out_dir, "02_overlap_regime_pairwise_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "overlap_regime_pairwise_values.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    defs_path = os.path.join(out_dir, "overlap_regime_definitions.txt")
    with open(defs_path, "w", encoding="utf-8") as f:
        f.write("Overlap regime definitions used in test_metrics_simulation\n")
        f.write("=" * 60 + "\n\n")
        for name in panel_order:
            r = cache_by_name[name]["regime"]
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(OVERLAP_REGIME_DEFINITIONS.get(name, "") + "\n\n")
            f.write(
                f"N examples: {args.n_examples}\n"
                f"M={r['n_variables']}, K={r['n_concepts']}, "
                f"sizes={r['size_min']}-{r['size_max']}, strategy={r['size_strategy']}\n"
                f"overlap_skew={r['overlap_skew']}, overlap_ceiling={r['overlap_ceiling']}, "
                f"dirichlet_factor={r['dirichlet_total_assignments_factor']}\n\n"
            )
    print(f"Saved: {defs_path}")

    return df


def _collect_probs_for_regime(
    X: np.ndarray,
    concept_map: Dict[str, List[int]],
    n_variables: int,
    profile: Dict[str, object],
    betas: List[float],
    args: argparse.Namespace,
    overlap_threshold: float = None,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for beta in betas:
        reduction_tensor, _ = compute_reduction_tensor(
            X=X,
            concept_map=concept_map,
            beta=beta,
            alpha_mode=args.alpha_mode,
            alpha_exp_scale=args.alpha_exp_scale,
        )
        batch = probability_rows_from_reduction(
            reduction_tensor=reduction_tensor,
            concept_map=concept_map,
            n_variables=n_variables,
            beta=beta,
            overlap_profile=str(profile["name"]),
            overlap_threshold=overlap_threshold,
        )
        for row in batch:
            row["n_variables"] = n_variables
            row["n_concepts"] = len(concept_map)
        rows.extend(batch)
    return rows


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
            "beta_sweep (01_probabilities_by_beta.png)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": args.n_variables,
                "K (fixed)": args.n_concepts,
                "beta_values": args.beta_sweep,
                "overlap_profile": BASE_OVERLAP_PROFILE,
                "output_csv": "probabilities_beta_sweep.csv",
            },
        )

    for beta in betas:
        print(f"[beta sweep] beta={beta:.1f}")
        batch = run_probability_pipeline(
            n_variables=FIXED_M,
            n_concepts=FIXED_K,
            beta=beta,
            overlap_profile=BASE_OVERLAP_PROFILE,
            **kw,
        )
        rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "probabilities_beta_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["beta", "probability"], "beta sweep"):
        return df

    betas_sorted = sorted(df["beta"].unique())
    data_by_beta = [df.loc[df["beta"] == b, "probability"].values for b in betas_sorted]

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
    plt.ylim(0.0, 1.0)
    plt.xlabel("beta")
    plt.ylabel("P(self reduction > other reduction)")
    plt.title("Influence of beta on probability distribution")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(out_dir, "01_probabilities_by_beta.png")
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
    Compare low / medium / high overlap at fixed M, K; only overlap parameters differ.
    """
    if regime_configs is None:
        regime_configs = OVERLAP_REGIME_CONFIGS
    regime_configs = _normalize_overlap_regimes(regime_configs, args)

    betas = _parse_float_list(args.fixed_betas)

    if param_log is not None:
        param_log.add_section(
            "overlap_sweep (02, 02b)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": FIXED_M,
                "K (fixed)": FIXED_K,
                "concept_sizes": f"{args.size_min}-{args.size_max} ({args.size_strategy})",
                "beta_values": args.fixed_betas,
                "overlap_regimes": _format_regimes_for_log(regime_configs),
                "output_csv": "probabilities_overlap_sweep.csv",
            },
        )
    rows: List[Dict[str, object]] = []

    regime_cache = _generate_regime_datasets(regime_configs, args)
    plot_overlap_regime_distributions(regime_cache, args, out_dir)

    for item in regime_cache:
        print(f"[overlap sweep] probabilities for {item['name']}")
        rows.extend(
            _collect_probs_for_regime(
                item["X"],
                item["concept_map"],
                item["M"],
                item["profile"],
                betas,
                args,
                overlap_threshold=None,
            )
        )

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "probabilities_overlap_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["overlap_profile", "beta", "probability"], "overlap sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="overlap_profile",
        y="probability",
        hue="beta",
        order=["low_overlap", "medium_overlap", "high_overlap"],
        showfliers=False,
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel(f"Overlap regime (M={FIXED_M}, K={FIXED_K} fixed)")
    plt.ylabel("P(self reduction > other reduction)")
    plt.title("Impact of overlap regime on probabilities at fixed beta values")
    plt.suptitle(
        f"See 02_overlap_regime_pairwise_distributions.png (M={FIXED_M}, K={FIXED_K})",
        fontsize=9,
        y=1.02,
    )
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "02b_probabilities_by_overlap.png")
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
            "M_sweep (03_probabilities_by_M.png)",
            {
                "n_examples": args.n_examples,
                "M_values (swept)": args.M_values,
                "K (fixed)": args.n_concepts,
                "beta_values": args.fixed_betas,
                "overlap_profile": BASE_OVERLAP_PROFILE,
                "output_csv": "probabilities_M_sweep.csv",
            },
        )

    for M in M_values:
        for beta in betas:
            print(f"[M sweep] M={M}, beta={beta:.1f}")
            batch = run_probability_pipeline(
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
    df.to_csv(os.path.join(out_dir, "probabilities_M_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["n_variables", "beta", "probability"], "M sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="n_variables",
        y="probability",
        hue="beta",
        showfliers=False,
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("M (number of variables)")
    plt.ylabel("P(self reduction > other reduction)")
    plt.title("Impact of M on probabilities at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "03_probabilities_by_M.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def study_overlap_threshold_vs_beta(
    args: argparse.Namespace,
    out_dir: str,
    study_config: Dict[str, object] = None,
    param_log: Optional[RunParameterLog] = None,
) -> pd.DataFrame:
    """
    Wide-spectrum overlap at fixed M, K: sweep overlap_threshold.
    Only pairs where the observed concept is not highly contained in the perturbated
    concept, i.e. overlap(observed, perturbated) = |observed ∩ perturbated| / |observed|
    < threshold, enter the probability computation.
    """
    if study_config is None:
        study_config = THRESHOLD_STUDY_CONFIG
    study_config = _normalize_threshold_config(study_config, args)

    betas = _parse_float_list(args.beta_sweep)
    thresholds = _parse_float_list(args.overlap_thresholds)

    if param_log is not None:
        param_log.add_section(
            "overlap_threshold (05-08)",
            {
                "n_examples": args.n_examples,
                "M (fixed)": study_config["n_variables"],
                "K (fixed)": study_config["n_concepts"],
                "concept_sizes": f"{args.size_min}-{args.size_max} ({args.size_strategy})",
                "overlap_skew": study_config["overlap_skew"],
                "overlap_ceiling": study_config["overlap_ceiling"],
                "dirichlet_factor": study_config["dirichlet_total_assignments_factor"],
                "beta_values": args.beta_sweep,
                "overlap_thresholds": args.overlap_thresholds,
                "output_csv": "probabilities_overlap_threshold_sweep.csv",
            },
        )

    print(
        f"[threshold study] generating dataset "
        f"(M={study_config['n_variables']}, K={study_config['n_concepts']})..."
    )
    X, concept_map, M, profile = _generate_dataset_for_regime(study_config, args)
    _log_pairwise_overlap_stats(concept_map, M)

    overlap_matrix = compute_concept_overlap_matrix(concept_map, M)
    K = overlap_matrix.shape[0]
    mask = ~np.eye(K, dtype=bool)
    pair_overlap_df = pd.DataFrame(
        {"pair_overlap": overlap_matrix[mask]}
    )
    plt.figure(figsize=(10, 5))
    sns.histplot(data=pair_overlap_df, x="pair_overlap", bins=40, kde=True)
    plt.xlabel("Directional overlap |g(i) ∩ g(j)| / |g(i)|")
    plt.ylabel("Count")
    plt.title("Realized pairwise overlap distribution (threshold study dataset)")
    plt.tight_layout()
    path_hist = os.path.join(out_dir, "08_pairwise_overlap_distribution.png")
    plt.savefig(path_hist, dpi=150)
    plt.close()
    print(f"Saved: {path_hist}")

    rows: List[Dict[str, object]] = []
    for beta in betas:
        print(f"[threshold study] computing reductions for beta={beta:.1f}")
        reduction_tensor, _ = compute_reduction_tensor(
            X=X,
            concept_map=concept_map,
            beta=beta,
            alpha_mode=args.alpha_mode,
            alpha_exp_scale=args.alpha_exp_scale,
        )
        for thr in thresholds:
            n_included = int(np.sum(overlap_matrix[mask] < thr))
            print(
                f"[threshold study] beta={beta:.1f}, threshold={thr:.4f} "
                f"({n_included} pairs with overlap < threshold)"
            )
            batch = probability_rows_from_reduction(
                reduction_tensor=reduction_tensor,
                concept_map=concept_map,
                n_variables=M,
                beta=beta,
                overlap_profile=str(profile["name"]),
                overlap_threshold=thr,
            )
            rows.extend(batch)

    df = _rows_to_dataframe(rows)
    df.to_csv(os.path.join(out_dir, "probabilities_overlap_threshold_sweep.csv"), index=False)

    if not _require_plot_columns(
        df, ["overlap_threshold", "beta", "probability"], "threshold study"
    ):
        return df

    # Plot 1: mean probability vs overlap threshold, one curve per beta.
    mean_df = (
        df.groupby(["beta", "overlap_threshold"], as_index=False)["probability"]
        .mean()
        .rename(columns={"probability": "mean_probability"})
    )
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=mean_df,
        x="overlap_threshold",
        y="mean_probability",
        hue="beta",
        marker="o",
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("Overlap threshold (include pairs with overlap(observed, perturbated) < threshold)")
    plt.ylabel("Mean P(R_i,i > R_i,j)")
    plt.title("Impact of overlap threshold on specificity (wide overlap spectrum)")
    plt.tight_layout()
    path_mean = os.path.join(out_dir, "05_mean_probability_vs_overlap_threshold.png")
    plt.savefig(path_mean, dpi=150)
    plt.close()
    print(f"Saved: {path_mean}")

    # Plot 2: full probability distributions by threshold and beta.
    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df,
        x="overlap_threshold",
        y="probability",
        hue="beta",
        showfliers=False,
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("Overlap threshold")
    plt.ylabel("P(R_i,i > R_i,j)")
    plt.title("Probability distributions vs overlap threshold and beta")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path_box = os.path.join(out_dir, "06_probability_distributions_threshold_beta.png")
    plt.savefig(path_box, dpi=150)
    plt.close()
    print(f"Saved: {path_box}")

    # Plot 3: number of concept pairs retained per threshold (same for all beta).
    pair_counts = (
        df.groupby("overlap_threshold")["perturbated_concept"]
        .count()
        .reset_index(name="n_pairs")
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=pair_counts, x="overlap_threshold", y="n_pairs", color="steelblue")
    plt.xlabel("Overlap threshold")
    plt.ylabel("Number of (perturbed, observed) pairs included")
    plt.title("Pair count retained (overlap < threshold, wide overlap spectrum)")
    plt.tight_layout()
    path_count = os.path.join(out_dir, "07_pair_count_vs_overlap_threshold.png")
    plt.savefig(path_count, dpi=150)
    plt.close()
    print(f"Saved: {path_count}")

    return df


def _concept_sizes_for_k(K: int, args: argparse.Namespace) -> tuple[int, int]:
    """
    Scale concept size so packing density rho = K * mean_size / M stays constant
    as K varies. This keeps coverage (rho > 1) and the overlap distribution
    comparable across K, isolating the effect of the number of concepts.

    Reference point: at K = FIXED_K, sizes are (args.size_min, args.size_max).
    """
    base_avg = (args.size_min + args.size_max) / 2.0
    rho = FIXED_K * base_avg / FIXED_M
    target_avg = rho * FIXED_M / K
    smin = max(2, round(target_avg * args.size_min / base_avg))
    smax = max(smin + 1, round(target_avg * args.size_max / base_avg))
    return int(smin), int(smax)


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
            "K_sweep (04_probabilities_by_K.png)",
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
                "output_csv": "probabilities_K_sweep.csv",
            },
        )

    for K in K_values:
        profile = BASE_OVERLAP_PROFILE.copy()
        profile["overlap_reference_concepts"] = K
        size_min, size_max = size_schedule[K]
        K_kw = {**kw, "size_min": size_min, "size_max": size_max, "max_concept_size": size_max}
        for beta in betas:
            print(f"[K sweep] K={K}, sizes=({size_min},{size_max}), beta={beta:.1f}")
            batch = run_probability_pipeline(
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
    df.to_csv(os.path.join(out_dir, "probabilities_K_sweep.csv"), index=False)

    if not _require_plot_columns(df, ["n_concepts", "beta", "probability"], "K sweep"):
        return df

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df,
        x="n_concepts",
        y="probability",
        hue="beta",
        showfliers=False,
    )
    plt.ylim(0.0, 1.0)
    plt.xlabel("K (number of concepts); concept size scales as ~M/K to fix overlap density")
    plt.ylabel("P(self reduction > other reduction)")
    plt.title("Impact of K on probabilities at fixed beta values")
    plt.legend(title="beta", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    path = os.path.join(out_dir, "04_probabilities_by_K.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return df


def resolve_output_dir(output_dir: str = None, run_name: str = None) -> str:
    """
    Resolve output folder:
      - --output-dir <path>  -> use exactly this folder name/path
      - --run-name <name>    -> use plots_<name>/
      - otherwise            -> DEFAULTS['output_dir']
    """
    if output_dir is not None:
        return output_dir
    if run_name is not None and str(run_name).strip():
        return f"plots_{str(run_name).strip()}"
    return DEFAULTS["output_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run metrics simulation studies and save seaborn plots.",
        epilog=(
            "Examples:\n"
            "  python test_metrics_simulation.py --run-name trial_v1\n"
            "  python test_metrics_simulation.py --output-dir results/exp_2025_05_28\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=DEFAULTS["n_examples"],
        help=f"Ignored: always {FIXED_N_EXAMPLES}.",
    )
    parser.add_argument(
        "--n-variables",
        type=int,
        default=DEFAULTS["n_variables"],
        help=f"Ignored when M is fixed: always {FIXED_M} (M sweep varies M only).",
    )
    parser.add_argument(
        "--n-concepts",
        type=int,
        default=DEFAULTS["n_concepts"],
        help=f"Ignored when K is fixed: always {FIXED_K} (K sweep varies K only).",
    )
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
        help="Shortcut for output folder: saves to plots_<run-name>/ (e.g. trial_v1 -> plots_trial_v1/).",
    )
    parser.add_argument("--beta-sweep", type=str, default=DEFAULTS["beta_sweep"])
    parser.add_argument("--fixed-betas", type=str, default=DEFAULTS["fixed_betas"])
    parser.add_argument("--M-values", type=str, default=DEFAULTS["M_values"])
    parser.add_argument("--K-values", type=str, default=DEFAULTS["K_values"])
    parser.add_argument(
        "--overlap-thresholds",
        type=str,
        default=DEFAULTS["overlap_thresholds"],
        help="Upper bounds: include pairs with overlap(observed, perturbated) < threshold.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer beta/M/K/threshold sweep values (N=1000, M=2000, K=300 unchanged).",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default="beta,overlap,M,K,threshold",
        help="Comma-separated: beta,overlap,M,K,threshold",
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
        args.overlap_thresholds = "0.05,0.15,0.35,0.6"
        print("[quick] Reduced beta/M/K/threshold sweeps; N/M/K defaults unchanged.")

    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    print(f"Output directory: {args.output_dir}")
    print(
        f"Fixed defaults: N={args.n_examples}, M={args.n_variables}, K={args.n_concepts}"
    )

    studies = {s.strip().lower() for s in args.studies.split(",") if s.strip()}
    param_log = RunParameterLog(args, args.output_dir)
    param_log.add_section(
        "studies_requested",
        {"studies": ", ".join(sorted(studies)), "quick_mode": args.quick},
    )

    if "beta" in studies:
        study_beta_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "overlap" in studies:
        study_overlap_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "m" in studies:
        study_M_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "k" in studies:
        study_K_sweep(args, args.output_dir, param_log=param_log)
        param_log.write()
    if "threshold" in studies:
        study_overlap_threshold_vs_beta(args, args.output_dir, param_log=param_log)
        param_log.write()

    param_log.write(final=True)
    print(f"Done. Outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
