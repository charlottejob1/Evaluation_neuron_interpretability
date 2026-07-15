"""Shared simulation defaults for the neuron metrics overlap study."""

from typing import Dict, List

# Fixed dimensions (aligned with test_data_generation / test_*_simulation).
FIXED_N_EXAMPLES = 1000
FIXED_M = 2000
FIXED_K = 300

BETA_SWEEP: List[float] = [round(i * 0.1, 1) for i in range(11)]

SEED = 12345
SIZE_STRATEGY = "dirichlet"
SIZE_MIN = 1
SIZE_MAX = 200
MAX_CONCEPT_SIZE = 200
DIRICHLET_FACTOR = 1.5
VARIABLE_MEAN_STRATEGY = "mean"
ALPHA_MODE = "exponential"
ALPHA_EXP_SCALE = 1.0
OVERLAP_THRESHOLD = None  # no pair filtering (include all off-diagonal pairs)

OVERLAP_REGIME_ORDER = ["low_overlap", "medium_overlap", "high_overlap"]

OVERLAP_REGIME_LABELS = {
    "low_overlap": "Low Overlap",
    "medium_overlap": "Medium Overlap",
    "high_overlap": "High Overlap",
}

# Tuned at M=2000, K=300, seed=12345 (directional overlap |Ci∩Cj|/|Ci|, off-diagonal pairs).
# Target realized means: low ≈ 0, medium ≈ 0.2, high ≈ 0.4.
OVERLAP_REGIME_CONFIGS: List[Dict[str, object]] = [
    {
        "name": "low_overlap",
        "overlap_skew": 50.0,
        "overlap_floor": 0.0,
        "overlap_ceiling": 0.05,
        "overlap_convergence_power": 0.5,
        "dirichlet_total_assignments_factor": 1.5,
        "size_min": 1,
        # realized mean ≈ 0.010, median ≈ 0.000
    },
    {
        "name": "medium_overlap",
        "overlap_skew": 0.2,
        "overlap_floor": 0.02,
        "overlap_ceiling": 1.0,
        "overlap_convergence_power": 0.05,
        "dirichlet_total_assignments_factor": 14.0,
        "size_min": 1,
        # realized mean ≈ 0.204, median ≈ 0.041
    },
    {
        "name": "high_overlap",
        "overlap_skew": 0.001,
        "overlap_floor": 0.35,
        "overlap_ceiling": 1.0,
        "overlap_convergence_power": 0.01,
        "dirichlet_total_assignments_factor": 30.0,
        "size_min": 10,
        # realized mean ≈ 0.402, median ≈ 0.300 (size_min=10 avoids size-1 degeneracy)
    },
]

OVERLAP_REGIME_DEFINITIONS: Dict[str, str] = {
    "low_overlap": (
        "Target mean overlap ≈ 0. High overlap_skew (50), low ceiling (0.05), "
        "dirichlet factor 1.5."
    ),
    "medium_overlap": (
        "Target mean overlap ≈ 0.2. Low skew (0.2), overlap_floor 0.02, "
        "dirichlet factor 14."
    ),
    "high_overlap": (
        "Target mean overlap ≈ 0.4. Very low skew (0.001), overlap_floor 0.35, "
        "dirichlet factor 30, concept size_min 10."
    ),
}

LEGEND_DC = "DC"
LEGEND_KOS = "KOS"
LEGEND_MIG = "MIG"

COMBINED_SHARED_FIGSIZE = (700 / 150, 412 / 150)
COMBINED_SHARED_DPI = 150
# Cropped vertical canvas for 04_combined_metrics_by_overlap_shared_yscale (axes area unchanged).
_OVERLAP_SHARED_AXES_FRAC = 0.83 - 0.24
_OVERLAP_SHARED_CROP_TOP = 0.91
_OVERLAP_SHARED_CROP_BOTTOM = 0.23
OVERLAP_SHARED_FIG_HEIGHT_PX = int(
    round(412 * _OVERLAP_SHARED_AXES_FRAC / (_OVERLAP_SHARED_CROP_TOP - _OVERLAP_SHARED_CROP_BOTTOM))
)
OVERLAP_SHARED_FIGSIZE = (700 / COMBINED_SHARED_DPI, OVERLAP_SHARED_FIG_HEIGHT_PX / COMBINED_SHARED_DPI)
# Per-regime beta shared-yscale plots (02_combined_metrics_*_beta_shared_yscale.png).
BETA_SHARED_FIGSIZE = (1120 / 150, 660 / 150)
COMBINED_AXIS_LABEL_FONTSIZE = 11
COMBINED_TICK_LABEL_FONTSIZE = 10
COMBINED_LEGEND_FONTSIZE = 11
COMBINED_TITLE_FONTSIZE = 12
COMBINED_REGIME_LABEL_FONTSIZE = 10

# Typography and line styling for 04_combined_metrics_by_overlap_shared_yscale.
OVERLAP_SHARED_AXIS_LABEL_FONTSIZE = 6
OVERLAP_SHARED_TICK_LABEL_FONTSIZE = 5
OVERLAP_SHARED_LEGEND_FONTSIZE = 5
OVERLAP_SHARED_TITLE_FONTSIZE = 7
OVERLAP_SHARED_REGIME_LABEL_FONTSIZE = 5
OVERLAP_SHARED_REGIME_LABEL_Y = -0.24
OVERLAP_SHARED_XLABEL_Y = -0.14
OVERLAP_SHARED_XLIM_PAD_LEFT = 0.35
OVERLAP_SHARED_XLIM_PAD_RIGHT = 0.15
OVERLAP_SHARED_FIG04_TITLE = "Neuron Simulations (N=1000, M=2000, K=300)"
OVERLAP_SHARED_LINE_WIDTH = 1.2
OVERLAP_SHARED_MARKER_SIZE = 3
OVERLAP_SHARED_SUBPLOT_MARGINS = dict(left=0.12, right=0.995, top=0.91, bottom=0.23)
OVERLAP_SHARED_TITLE_PAD = 2

# 05_combined_neuron_vega_merged: neuron overlap regimes + VEGA FC section on one axis.
DEFAULT_VEGA_SWEEP_ROOT = "vega_fcn_sweep"
VEGA_SECTION_N_FC = 11
# Pixel output at COMBINED_SHARED_DPI (150): landscape, width > height.
_MERGED_BASE_WIDTH_PX = 700 + 700 / len(BETA_SWEEP) * VEGA_SECTION_N_FC
_MERGED_BASE_HEIGHT_PX = 412
_MERGED_TYPO_REF_HEIGHT_PX = 1800  # typography tuned at this height
MERGED_FIG_HEIGHT_PX = 2700
MERGED_FIG_WIDTH_PX = int(round(_MERGED_BASE_WIDTH_PX * MERGED_FIG_HEIGHT_PX / _MERGED_BASE_HEIGHT_PX))
MERGED_NEURON_VEGA_FIGSIZE = (
    MERGED_FIG_WIDTH_PX / COMBINED_SHARED_DPI,
    MERGED_FIG_HEIGHT_PX / COMBINED_SHARED_DPI,
)
MERGED_SIZE_SCALE = MERGED_FIG_HEIGHT_PX / _MERGED_TYPO_REF_HEIGHT_PX
MERGED_FONT_SCALE = MERGED_FIG_HEIGHT_PX / _MERGED_BASE_HEIGHT_PX
# Typography at the 1800 px reference (scaled proportionally with MERGED_SIZE_SCALE).
_MERGED_TYPO_AT_REF = dict(
    axis=34,
    param=34,
    tick=27,
    title=36,
    regime=30,
    legend_handlelength=1.0,
    legend_symbol_scale=1.0,
)
MERGED_AXIS_LABEL_FONTSIZE = int(round(_MERGED_TYPO_AT_REF["axis"] * MERGED_SIZE_SCALE))
MERGED_PARAM_LABEL_FONTSIZE = int(round(_MERGED_TYPO_AT_REF["param"] * MERGED_SIZE_SCALE))
MERGED_TICK_LABEL_FONTSIZE = int(round(_MERGED_TYPO_AT_REF["tick"] * MERGED_SIZE_SCALE))
MERGED_TITLE_FONTSIZE = int(round(_MERGED_TYPO_AT_REF["title"] * MERGED_SIZE_SCALE))
MERGED_LEGEND_FONTSIZE = MERGED_TITLE_FONTSIZE
MERGED_REGIME_LABEL_FONTSIZE = int(round(_MERGED_TYPO_AT_REF["regime"] * MERGED_SIZE_SCALE))
MERGED_LEGEND_HANDLELENGTH = _MERGED_TYPO_AT_REF["legend_handlelength"] * MERGED_SIZE_SCALE
MERGED_LEGEND_SYMBOL_SCALE = _MERGED_TYPO_AT_REF["legend_symbol_scale"]
MERGED_LINE_WIDTH = OVERLAP_SHARED_LINE_WIDTH * MERGED_FONT_SCALE
MERGED_MARKER_SIZE = OVERLAP_SHARED_MARKER_SIZE * MERGED_FONT_SCALE
MERGED_REF_LINE_WIDTH = 1.5 * MERGED_FONT_SCALE
MERGED_VEGA_SECTION_LABEL = (
    "Percentage of Fully-Connected Neurons\n"
    "(0%: sparse, 100%: fully-connected)"
)
MERGED_PANEL_LETTERS = ["(a)", "(b)", "(c)", "(d)"]
MERGED_VEGA_XLABEL = "gamma"
MERGED_VEGA_SUBTITLE = "VEGA"
MERGED_NEURON_PANEL_TITLES = {
    "low_overlap": "Neuron Simulation (First)",
    "medium_overlap": "Neuron Simulation (Second)",
    "high_overlap": "Neuron Simulation (Third)",
}
MERGED_XLABEL_Y = -0.17
MERGED_SECTION_LABEL_Y = -0.32
MERGED_PANEL_LETTER_Y = -0.54
MERGED_XLIM_PAD_LEFT = 0.50
MERGED_XLIM_PAD_RIGHT = 0.22
MERGED_SUBPLOT_MARGINS = dict(left=0.05, right=0.995, top=0.82, bottom=0.38)
