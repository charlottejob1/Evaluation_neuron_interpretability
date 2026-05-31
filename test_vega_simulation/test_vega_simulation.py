"""
Test runner for VEGA interpretability training on PBMC data.

Uses ``train_vega_pbmc.py`` which:
  - loads PBMC from ``pbmc_data/`` (10x h5 + metadata),
  - preprocesses with top 2000 highly variable genes,
  - splits train / validation / test (stratified),
  - builds the Reactome pathway mask and randomly selects a fraction of latent neurons
    to make fully connected (``fully_connected_neuron_fraction``),
  - trains VEGA2 and evaluates MSE + Pearson on train and test,
  - saves model, metrics.json, plots, and training_info.txt.

Example:
  python test_vega_simulation.py --fully-connected-neuron-fraction 0.5
"""

import argparse
import os

import torch

from train_vega_pbmc import run_pbmc_training


DEFAULTS = {
    "data_dir": "pbmc_data",
    "gmt_path": "vega/vega/data/reactomes.gmt",
    "fully_connected_neuron_fraction": 0,
    "n_top_genes": 2000,
    "train_size": 0.9,
    "batch_size": 128,
    "learning_rate": 0.00062187,
    "n_epochs": 500,
    "train_patience": 25,
    "test_patience": 25,
    "kld_weight": 0.000233864,
    "dropout": 0.0452273 ,
    "seed": 42,
    "output_dir": "results_vega_pbmc",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Train VEGA2 on PBMC with selectable fully-connected decoder neurons."
    )
    p.add_argument("--data-dir", type=str, default=DEFAULTS["data_dir"])
    p.add_argument("--gmt-path", type=str, default=DEFAULTS["gmt_path"])
    p.add_argument(
        "--fully-connected-neuron-fraction",
        type=float,
        default=DEFAULTS["fully_connected_neuron_fraction"],
        help=(
            "Fraction of latent neurons randomly selected to be fully connected "
            "(0=sparse VEGA2, 1=dense decoder)."
        ),
    )
    p.add_argument("--n-top-genes", type=int, default=DEFAULTS["n_top_genes"])
    p.add_argument("--train-size", type=float, default=DEFAULTS["train_size"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--n-epochs", type=int, default=DEFAULTS["n_epochs"])
    p.add_argument("--train-patience", type=int, default=DEFAULTS["train_patience"])
    p.add_argument("--test-patience", type=int, default=DEFAULTS["test_patience"])
    p.add_argument(
        "--kld-weight",
        type=float,
        default=DEFAULTS["kld_weight"],
        help="VEGA2 beta-VAE KL weight (original VEGA2 training parameter).",
    )
    p.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder (default: results_vega_pbmc_fcn<fraction>).",
    )
    p.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Shortcut: saves to results_<run-name>/.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fewer epochs and patience for a fast smoke test.",
    )
    return p.parse_args()


def resolve_output_dir(output_dir, run_name, fully_connected_neuron_fraction):
    if output_dir is not None:
        return os.path.abspath(output_dir)
    if run_name is not None and str(run_name).strip():
        return os.path.abspath("results_%s" % str(run_name).strip())
    fcn_tag = "%.2f" % (
        fully_connected_neuron_fraction / 100.0
        if fully_connected_neuron_fraction > 1.0
        else fully_connected_neuron_fraction
    )
    return os.path.abspath("results_vega_pbmc_fcn%s" % fcn_tag)


def main():
    args = parse_args()
    if args.quick:
        args.n_epochs = 20
        args.train_patience = 5
        args.test_patience = 5
        print("[quick] Reduced epochs and patience for smoke test.")

    out_dir = resolve_output_dir(
        args.output_dir, args.run_name, args.fully_connected_neuron_fraction
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Output directory: %s" % out_dir)
    print("Device: %s" % device)
    print(
        "Fully connected neuron fraction: %s" % args.fully_connected_neuron_fraction
    )
    print("VEGA2 KL weight (beta): %s" % args.kld_weight)

    result = run_pbmc_training(
        data_dir=args.data_dir,
        gmt_path=args.gmt_path,
        fully_connected_neuron_fraction=args.fully_connected_neuron_fraction,
        n_top_genes=args.n_top_genes,
        train_size=args.train_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        n_epochs=args.n_epochs,
        train_patience=args.train_patience,
        test_patience=args.test_patience,
        kld_weight=args.kld_weight,
        dropout=args.dropout,
        seed=args.seed,
        output_dir=out_dir,
        device=device,
    )
    print("\nDone. All outputs saved to: %s" % result["output_dir"])


if __name__ == "__main__":
    main()
