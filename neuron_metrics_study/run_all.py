#!/usr/bin/env python
"""Run the full neuron metrics overlap study (data + 3 metrics + aggregate)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

STUDY_ROOT = os.path.dirname(os.path.abspath(__file__))


def _run(script: str, extra_args=None):
    cmd = [sys.executable, script] + (extra_args or [])
    print("\n>>> %s\n" % " ".join(cmd))
    subprocess.check_call(cmd, cwd=STUDY_ROOT)


def main():
    p = argparse.ArgumentParser(description="Run neuron_metrics_study end-to-end.")
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Write outputs to runs/<run-name>/ without overwriting top-level results. "
            "Example: --run-name v2_mean_020_040"
        ),
    )
    p.add_argument("--force-data", action="store_true", help="Regenerate cached regime datasets.")
    p.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Only run aggregate (requires existing metric CSVs).",
    )
    p.add_argument(
        "--metrics-only",
        choices=["distance_corr", "probability", "mig", "all"],
        default="all",
    )
    args = p.parse_args()

    run_args = []
    if args.run_name:
        run_args.extend(["--run-name", args.run_name])
    if args.force_data:
        run_args.append("--force-data")

    if not args.skip_metrics:
        if args.metrics_only in ("all", "distance_corr"):
            _run(os.path.join("distance_corr", "run_study.py"), run_args)
        if args.metrics_only in ("all", "probability"):
            _run(os.path.join("probability", "run_study.py"), run_args)
        if args.metrics_only in ("all", "mig"):
            _run(os.path.join("mig", "run_study.py"), run_args)

    agg_args = ["--run-name", args.run_name] if args.run_name else []
    _run(os.path.join("aggregate", "plot_aggregate.py"), agg_args)


if __name__ == "__main__":
    main()
