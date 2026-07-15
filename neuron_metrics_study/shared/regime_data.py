"""Generate, cache, and load per-regime structured datasets."""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SIM = os.path.join(_REPO, "neuron_simulation")
if _SIM not in sys.path:
    sys.path.insert(0, _SIM)

from data_generation import generate_structured_dataset
from probability_metrics_simulation import compute_concept_overlap_matrix

from shared.config import (
    FIXED_K,
    FIXED_M,
    FIXED_N_EXAMPLES,
    MAX_CONCEPT_SIZE,
    OVERLAP_REGIME_CONFIGS,
    OVERLAP_REGIME_DEFINITIONS,
    OVERLAP_REGIME_ORDER,
    SEED,
    SIZE_MAX,
    SIZE_MIN,
    SIZE_STRATEGY,
    VARIABLE_MEAN_STRATEGY,
)


from shared.paths import data_dir, regime_dir


def normalize_regime(cfg: Dict[str, object]) -> Dict[str, object]:
    return {
        **cfg,
        "n_variables": FIXED_M,
        "n_concepts": FIXED_K,
        "size_strategy": SIZE_STRATEGY,
        "size_min": int(cfg.get("size_min", SIZE_MIN)),
        "size_max": SIZE_MAX,
        "max_concept_size": MAX_CONCEPT_SIZE,
        "overlap_reference_concepts": FIXED_K,
    }


def pairwise_overlap_values(concept_map: Dict[str, List[int]], n_variables: int) -> np.ndarray:
    overlap_matrix = compute_concept_overlap_matrix(concept_map, n_variables)
    k = overlap_matrix.shape[0]
    mask = ~np.eye(k, dtype=bool)
    return overlap_matrix[mask]


def overlap_summary(off_diag: np.ndarray) -> Dict[str, float]:
    return {
        "mean_overlap": float(np.mean(off_diag)),
        "median_overlap": float(np.median(off_diag)),
        "min_overlap": float(np.min(off_diag)),
        "max_overlap": float(np.max(off_diag)),
    }


def generate_regime_dataset(regime: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[int]], Dict[str, float]]:
    """Return X, concept_activities, concept_map, overlap stats."""
    np.random.seed(SEED)
    X, concept_activities, concept_map = generate_structured_dataset(
        n_examples=FIXED_N_EXAMPLES,
        n_variables=int(regime["n_variables"]),
        n_concepts=int(regime["n_concepts"]),
        overlap_skew=float(regime["overlap_skew"]),
        max_concept_size=int(regime["max_concept_size"]),
        size_strategy=str(regime["size_strategy"]),
        size_range=(int(regime["size_min"]), int(regime["size_max"])),
        overlap_floor=float(regime["overlap_floor"]),
        overlap_ceiling=float(regime["overlap_ceiling"]),
        overlap_convergence_power=float(regime["overlap_convergence_power"]),
        overlap_reference_concepts=int(regime["overlap_reference_concepts"]),
        dirichlet_total_assignments_factor=float(regime["dirichlet_total_assignments_factor"]),
        variable_mean_strategy=VARIABLE_MEAN_STRATEGY,
        seed=SEED,
    )
    off_diag = pairwise_overlap_values(concept_map, int(regime["n_variables"]))
    stats = overlap_summary(off_diag)
    return X, concept_activities, concept_map, stats


def save_regime_cache(
    name: str,
    X: np.ndarray,
    concept_activities: np.ndarray,
    concept_map: Dict[str, List[int]],
    stats: Dict[str, float],
    regime: Dict[str, object],
) -> str:
    out = regime_dir(name)
    os.makedirs(out, exist_ok=True)
    np.save(os.path.join(out, "X.npy"), X)
    np.save(os.path.join(out, "concept_activities.npy"), concept_activities)
    with open(os.path.join(out, "concept_map.json"), "w", encoding="utf-8") as f:
        json.dump(concept_map, f)
    meta = {
        "name": name,
        "regime_params": {k: regime[k] for k in regime if k != "name"},
        "overlap_stats": stats,
        "n_examples": FIXED_N_EXAMPLES,
        "n_variables": FIXED_M,
        "n_concepts": FIXED_K,
        "seed": SEED,
    }
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return out


def load_regime_cache(name: str):
    root = regime_dir(name)
    X = np.load(os.path.join(root, "X.npy"))
    concept_activities = np.load(os.path.join(root, "concept_activities.npy"))
    with open(os.path.join(root, "concept_map.json"), encoding="utf-8") as f:
        concept_map = json.load(f)
    with open(os.path.join(root, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return X, concept_activities, concept_map, meta


def build_all_regime_caches(force: bool = False) -> List[Dict[str, object]]:
    """Generate all overlap-regime datasets; return cache metadata list."""
    os.makedirs(data_dir(), exist_ok=True)
    cached: List[Dict[str, object]] = []

    for cfg in OVERLAP_REGIME_CONFIGS:
        regime = normalize_regime(cfg)
        name = str(regime["name"])
        meta_path = os.path.join(regime_dir(name), "meta.json")
        if not force and os.path.isfile(meta_path):
            X, concept_activities, concept_map, meta = load_regime_cache(name)
            off_diag = pairwise_overlap_values(concept_map, FIXED_M)
            print(
                "[cache] reuse %s (mean overlap=%.4f, median=%.4f)"
                % (name, meta["overlap_stats"]["mean_overlap"], meta["overlap_stats"]["median_overlap"])
            )
        else:
            print("[cache] generating %s ..." % name)
            X, concept_activities, concept_map, stats = generate_regime_dataset(regime)
            save_regime_cache(name, X, concept_activities, concept_map, stats, regime)
            _, _, _, meta = load_regime_cache(name)
            off_diag = pairwise_overlap_values(concept_map, FIXED_M)
            print(
                "  realized overlap: mean=%.4f median=%.4f"
                % (stats["mean_overlap"], stats["median_overlap"])
            )

        cached.append(
            {
                "name": name,
                "regime": regime,
                "meta": meta if not force or os.path.isfile(meta_path) else json.load(open(meta_path)),
                "off_diag": off_diag,
                "overlap_stats": meta["overlap_stats"],
            }
        )

    write_overlap_definitions(cached)
    return cached


def write_overlap_definitions(cached: List[Dict[str, object]]) -> None:
    path = os.path.join(data_dir(), "overlap_regime_definitions.txt")
    lines = ["Overlap regime definitions (neuron_metrics_study)", "=" * 60, ""]
    for item in cached:
        name = item["name"]
        stats = item["overlap_stats"]
        regime = item["regime"]
        lines.extend(
            [
                name,
                "-" * 40,
                OVERLAP_REGIME_DEFINITIONS.get(name, ""),
                "",
                "N=%d, M=%d, K=%d, sizes=%d-%d (%s)"
                % (
                    FIXED_N_EXAMPLES,
                    FIXED_M,
                    FIXED_K,
                    SIZE_MIN,
                    SIZE_MAX,
                    SIZE_STRATEGY,
                ),
                "overlap_skew=%s, overlap_ceiling=%s, dirichlet=%s"
                % (
                    regime["overlap_skew"],
                    regime["overlap_ceiling"],
                    regime["dirichlet_total_assignments_factor"],
                ),
                "Realized mean overlap: %.4f, median: %.4f"
                % (stats["mean_overlap"], stats["median_overlap"]),
                "",
            ]
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Saved: %s" % path)
