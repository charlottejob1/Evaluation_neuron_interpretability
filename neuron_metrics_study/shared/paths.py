"""Output paths for a named study run (preserves previous results in other folders)."""

from __future__ import annotations

import os
from typing import Optional

_PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_RUN_NAME: Optional[str] = None


def study_root() -> str:
    return _PACKAGE_ROOT


def get_run_name() -> Optional[str]:
    return _RUN_NAME


def set_run_name(name: Optional[str]) -> str:
    """Set active run folder under runs/<name>/ (None = legacy top-level layout)."""
    global _RUN_NAME
    if name is not None:
        name = str(name).strip()
        if not name:
            name = None
    _RUN_NAME = name
    return run_root()


def run_root() -> str:
    if _RUN_NAME is None:
        return study_root()
    return os.path.join(study_root(), "runs", _RUN_NAME)


def data_dir() -> str:
    return os.path.join(run_root(), "data")


def regime_dir(regime_name: str) -> str:
    return os.path.join(data_dir(), "regimes", regime_name)


def metric_output_dir(metric: str) -> str:
    return os.path.join(run_root(), metric, "outputs")


def aggregate_output_dir() -> str:
    return os.path.join(run_root(), "aggregate", "outputs")


def add_run_name_arg(parser) -> None:
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Save all outputs under runs/<run-name>/ (keeps previous top-level results). "
            "Example: --run-name v2_mean_020_040"
        ),
    )


def apply_run_name_from_args(args) -> str:
    set_run_name(getattr(args, "run_name", None))
    root = run_root()
    os.makedirs(root, exist_ok=True)
    if get_run_name():
        print("Study run: runs/%s/" % get_run_name())
    else:
        print("Study run: (legacy top-level layout)")
    return root
