"""
PBMC VEGA2 training pipeline with selectable fully-connected decoder neurons.

Mirrors the workflow in ``train_vega_original.ipynb`` and the preprocessing /
splitting utilities in ``vega_utils_copy.py``, but uses the self-contained
``vega_interpretability_simulation`` module (VEGA2, CustomizedLinear, mask selection).

Pipeline:
  1. Load PBMC data from ``pbmc_data/`` (10x h5 + metadata).
  2. Preprocess with top 2000 highly variable genes.
  3. Split into train / validation / test (stratified by cell type).
  4. Build the Reactome pathway mask and randomly select a fraction of latent neurons
     to make fully connected (``fully_connected_neuron_fraction``).
  5. Train VEGA2 for reconstruction; evaluate MSE + Pearson on train and test.
  6. Save model, metrics.json, plots, and training_info.txt.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import sparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from vega_interpretability_simulation import (
    VEGA2,
    compute_mse,
    compute_pearson,
    make_partially_connected_mask,
    mask_density,
    read_gmt,
    reconstruct,
)


# ---------------------------------------------------------------------------
# Data utilities (ported from vega_utils_copy.py, no external VEGA src imports)
# ---------------------------------------------------------------------------

class UnsupervisedDataset(torch.utils.data.Dataset):
    """PyTorch dataset returning expression vectors only."""

    def __init__(self, data: torch.Tensor, targets=None):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_pathway_mask(
    feature_list: List[str],
    pathway_dict: Dict[str, List[str]],
    add_missing: int = 1,
    fully_connected: bool = True,
) -> np.ndarray:
    """
    Build mask [n_genes, n_pathways (+ add_missing)] where M[g, p]=1 if gene g is in pathway p.
    Extra UNANNOTATED columns are fully connected when ``fully_connected=True``.
    """
    p_mask = np.zeros((len(feature_list), len(pathway_dict)), dtype=float)
    feature_index = {g: i for i, g in enumerate(feature_list)}
    for j, genes in enumerate(pathway_dict.values()):
        for gene in genes:
            i = feature_index.get(gene)
            if i is not None:
                p_mask[i, j] = 1.0
    if add_missing > 0 and fully_connected:
        extra = np.ones((p_mask.shape[0], add_missing), dtype=float)
        p_mask = np.hstack((p_mask, extra))
    return p_mask


def preprocess_adata(adata, n_top_genes: int = 2000):
    """Filter cells/genes and keep top highly variable genes (matches vega_utils_copy)."""
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def extract_x_y_from_adata(adata, column_labels_name: str):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[column_labels_name]
    return X, y


def split_data(X, y, train_size: float, random_seed: int):
    return train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y
    )


def build_adata_from_index(adata, index_df):
    return adata[adata.obs.index.isin(index_df)].copy()


def encode_y(y):
    le = preprocessing.LabelEncoder().fit(y)
    return torch.Tensor(le.transform(y))


def adata_to_array(adata) -> np.ndarray:
    if sparse.issparse(adata.X):
        return adata.X.A.astype(np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def build_vega_dataset(adata, pathway_file: str):
    data = torch.tensor(adata_to_array(adata), dtype=torch.float32)
    dataset = UnsupervisedDataset(data)
    pathway_dict = read_gmt(pathway_file, min_genes=0, max_genes=1000)
    pathway_mask = create_pathway_mask(
        adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True
    )
    list_pathways = list(pathway_dict.keys()) + ["UNANNOTATED_0"]
    return dataset, pathway_dict, pathway_mask, list_pathways


def load_pathway_context(adata, pathway_file: str):
    """Pathway mask and names only (no torch dataset). Used for eval-only workflows."""
    pathway_dict = read_gmt(pathway_file, min_genes=0, max_genes=1000)
    pathway_mask = create_pathway_mask(
        adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True
    )
    list_pathways = list(pathway_dict.keys()) + ["UNANNOTATED_0"]
    return pathway_dict, pathway_mask, list_pathways


def create_vega_test_eval_context(
    adata,
    pathway_file: str,
    column_labels_name: str,
    n_top_genes: int = 2000,
    train_size: float = 0.9,
    random_seed: int = 42,
):
    """
    Preprocess and return only the held-out test AnnData plus pathway context.

    Skips building train/val torch datasets to save time during interpretability eval.
    """
    print("Preprocessing adata (top %d HVGs, test split only for eval)" % n_top_genes)
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)

    X, y = extract_x_y_from_adata(adata, column_labels_name)
    _, X_test, _, _ = split_data(X, y, train_size, random_seed)
    adata_test = build_adata_from_index(adata, X_test.index)

    pathway_dict, pathway_mask, list_pathways = load_pathway_context(
        adata_test, pathway_file
    )
    print(
        "Test split for eval: n_cells=%d | genes=%d | pathways=%d"
        % (len(adata_test), adata.n_vars, pathway_mask.shape[1])
    )
    return {
        "adata_test": adata_test,
        "pathway_dict": pathway_dict,
        "pathway_mask": pathway_mask,
        "list_pathways": list_pathways,
    }


def create_vega_training_splits(
    adata,
    pathway_file: str,
    column_labels_name: str,
    n_top_genes: int = 2000,
    train_size: float = 0.9,
    random_seed: int = 42,
):
    """
    Preprocess once, then split train/val/test like ``train_vega_original.ipynb``:
      - 90% train pool / 10% test (first split)
      - 90% train / 10% val inside the train pool (second split)
    """
    print("Preprocessing adata (top %d HVGs)" % n_top_genes)
    adata = preprocess_adata(adata, n_top_genes=n_top_genes)

    X, y = extract_x_y_from_adata(adata, column_labels_name)
    X_train_pool, X_test, y_train_pool, y_test = split_data(X, y, train_size, random_seed)
    X_train, X_val, y_train, y_val = split_data(
        X_train_pool, y_train_pool, train_size, random_seed
    )

    adata_train = build_adata_from_index(adata, X_train.index)
    adata_val = build_adata_from_index(adata, X_val.index)
    adata_test = build_adata_from_index(adata, X_test.index)

    train_ds, pathway_dict, pathway_mask, list_pathways = build_vega_dataset(
        adata_train, pathway_file
    )
    val_ds, _, _, _ = build_vega_dataset(adata_val, pathway_file)
    test_ds, _, _, _ = build_vega_dataset(adata_test, pathway_file)

    print(
        "Split sizes: train=%d, val=%d, test=%d | genes=%d | pathways=%d"
        % (
            len(adata_train),
            len(adata_val),
            len(adata_test),
            adata.n_vars,
            pathway_mask.shape[1],
        )
    )
    return {
        "adata": adata,
        "adata_train": adata_train,
        "adata_val": adata_val,
        "adata_test": adata_test,
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "pathway_dict": pathway_dict,
        "pathway_mask": pathway_mask,
        "list_pathways": list_pathways,
    }


def load_pbmc_8k(data_dir: str = "pbmc_data"):
    """Load PBMC 8K dataset (10x h5 + cell metadata)."""
    data_dir = os.path.abspath(data_dir)
    metadata_path = os.path.join(data_dir, "PBMC_8K_CellMetainfo_table.tsv")
    expression_path = os.path.join(data_dir, "PBMC_8K_expression.h5")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError("Missing metadata: %s" % metadata_path)
    if not os.path.isfile(expression_path):
        raise FileNotFoundError("Missing expression: %s" % expression_path)

    metadata = pd.read_csv(metadata_path, sep="\t")
    adata = sc.read_10x_h5(expression_path, genome="GRCh38", gex_only=False)
    adata.obs = metadata.set_index("Cell").loc[adata.obs_names]
    column_labels_name = "Celltype (major-lineage)"
    return adata, column_labels_name


def build_dataloaders_from_datasets(
    train_ds,
    val_ds,
    test_ds,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )
    return train_loader, val_loader, test_loader


def apply_fully_connected_neuron_fraction_to_mask(
    pathway_mask: np.ndarray,
    fully_connected_neuron_fraction: float,
    add_nodes: int = 1,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly select latent neurons to make fully connected; UNANNOTATED column stays as-is."""
    return make_partially_connected_mask(
        pathway_mask,
        fully_connected_neuron_fraction=fully_connected_neuron_fraction,
        add_nodes=add_nodes,
        seed=seed,
    )


def describe_model_architecture(model: VEGA2) -> str:
    """Return a detailed multi-line architecture summary."""
    active_dec = int(model.decoder.mask.sum().item())
    dense_dec = model.decoder.mask.numel()
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lines = [
        str(model),
        "",
        "Total parameters: %s | trainable: %s" % (f"{n_params:,}", f"{n_trainable:,}"),
        "Decoder active connections (mask==1): %s / %s (%.1f%% of dense decoder)"
        % (f"{active_dec:,}", f"{dense_dec:,}", 100.0 * active_dec / dense_dec),
    ]
    return "\n".join(lines)


def evaluate_reconstruction(model: VEGA2, adata) -> Dict[str, float]:
    """Compute MSE and Pearson metrics for one AnnData split."""
    X = torch.tensor(adata_to_array(adata), dtype=torch.float32)
    x_true = X.numpy()
    x_pred = reconstruct(model, X)
    pearson = compute_pearson(x_true, x_pred)
    return {
        "mse": compute_mse(x_true, x_pred),
        "pearson_overall": pearson["overall"],
        "pearson_per_cell_mean": pearson["per_cell_mean"],
        "n_cells": int(x_true.shape[0]),
        "n_genes": int(x_true.shape[1]),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_mask(mask, fully_connected_idx, add_nodes, out_path, fully_connected_neuron_fraction):
    n_genes, n_cols = mask.shape
    fig, axes = plt.subplots(
        2, 1, figsize=(min(18, 0.18 * n_cols + 4), 10),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    axes[0].imshow(mask, aspect="auto", cmap="Greys", interpolation="nearest")
    axes[0].set_title(
        "Pathway mask (fully_connected_neuron_fraction=%.2f): %d genes x %d nodes | "
        "%d neurons selected fully connected"
        % (fully_connected_neuron_fraction, n_genes, n_cols, len(fully_connected_idx))
    )
    axes[0].set_xlabel("Latent node index")
    axes[0].set_ylabel("Gene index")
    for idx in fully_connected_idx:
        axes[0].axvline(idx, color="red", alpha=0.35, linewidth=1.0)

    densities = mask.mean(axis=0)
    colors = []
    fc_set = set(fully_connected_idx.tolist())
    for j in range(n_cols):
        if j in fc_set:
            colors.append("firebrick")
        elif j >= n_cols - add_nodes:
            colors.append("darkorange")
        else:
            colors.append("steelblue")
    axes[1].bar(np.arange(n_cols), densities, color=colors)
    axes[1].set_title("Per-node connection density")
    axes[1].set_xlabel("Latent node index")
    axes[1].set_ylabel("Fraction of genes connected")
    axes[1].set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_loss(hist, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(hist["train_loss"], label="train loss")
    if hist["valid_loss"]:
        plt.plot(hist["valid_loss"], label="validation loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("VEGA2 training loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_reconstruction_scatter(x_true, x_pred, split_name, metrics, out_path):
    rng = np.random.RandomState(0)
    n_points = min(20000, x_true.size)
    idx = rng.choice(x_true.size, size=n_points, replace=False)
    t = x_true.ravel()[idx]
    p = x_pred.ravel()[idx]

    plt.figure(figsize=(7, 7))
    plt.scatter(t, p, s=3, alpha=0.2, color="steelblue")
    lims = [min(t.min(), p.min()), max(t.max(), p.max())]
    plt.plot(lims, lims, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Input expression")
    plt.ylabel("Reconstructed expression")
    plt.title(
        "Reconstruction (%s)\nMSE=%.4f | Pearson overall=%.4f | per-cell=%.4f"
        % (
            split_name,
            metrics["mse"],
            metrics["pearson_overall"],
            metrics["pearson_per_cell_mean"],
        )
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def run_pbmc_training(
    *,
    data_dir: str = "pbmc_data",
    gmt_path: str = "vega/vega/data/reactomes.gmt",
    fully_connected_neuron_fraction: float = 0.5,
    n_top_genes: int = 2000,
    train_size: float = 0.9,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    n_epochs: int = 100,
    train_patience: int = 15,
    test_patience: int = 15,
    kld_weight: float = 1e-4,
    dropout: float = 0.1,
    seed: int = 42,
    output_dir: str = "results_vega_pbmc",
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """
    End-to-end PBMC training with randomly selected fully-connected decoder neurons.

    Returns a dict with metrics, paths, and training history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcn_tag = "%.2f" % (
        fully_connected_neuron_fraction / 100.0
        if fully_connected_neuron_fraction > 1.0
        else fully_connected_neuron_fraction
    )
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Data ---
    adata, column_labels_name = load_pbmc_8k(data_dir)
    splits = create_vega_training_splits(
        adata=adata,
        pathway_file=gmt_path,
        column_labels_name=column_labels_name,
        n_top_genes=n_top_genes,
        train_size=train_size,
        random_seed=seed,
    )
    train_loader, val_loader, test_loader = build_dataloaders_from_datasets(
        splits["train_ds"], splits["val_ds"], splits["test_ds"], batch_size
    )

    # --- Mask: select fully connected neurons ---
    base_mask = splits["pathway_mask"]
    add_nodes = 1
    mask, fully_connected_idx = apply_fully_connected_neuron_fraction_to_mask(
        base_mask,
        fully_connected_neuron_fraction=fully_connected_neuron_fraction,
        add_nodes=add_nodes,
        seed=seed,
    )
    n_pathway_cols = base_mask.shape[1] - add_nodes

    fcn_value = (
        fully_connected_neuron_fraction / 100.0
        if fully_connected_neuron_fraction > 1.0
        else fully_connected_neuron_fraction
    )
    plot_mask(
        mask, fully_connected_idx, add_nodes,
        os.path.join(out_dir, "01_fully_connected_neuron_mask.png"),
        fully_connected_neuron_fraction=fcn_value,
    )

    # --- Model ---
    model_path = os.path.join(out_dir, "vega2_pbmc_fcn%s.pt" % fcn_tag)
    model = VEGA2(
        pathway_mask=mask,
        positive_decoder=True,
        device=device,
        beta=kld_weight,
        dropout=dropout,
        save_path=model_path,
    ).to(device)

    arch_summary = describe_model_architecture(model)
    print(arch_summary)

    # --- Train ---
    hist, mse_hist, kld_hist = model.train_model(
        train_loader,
        learning_rate=learning_rate,
        n_epochs=n_epochs,
        train_patience=train_patience,
        test_patience=test_patience,
        test_loader=val_loader,
        save_model=True,
    )
    plot_training_loss(hist, os.path.join(out_dir, "02_training_loss.png"))

    # --- Evaluate train + test ---
    train_metrics = evaluate_reconstruction(model, splits["adata_train"])
    test_metrics = evaluate_reconstruction(model, splits["adata_test"])

    X_train = torch.tensor(adata_to_array(splits["adata_train"]), dtype=torch.float32)
    X_test = torch.tensor(adata_to_array(splits["adata_test"]), dtype=torch.float32)
    plot_reconstruction_scatter(
        X_train.numpy(), reconstruct(model, X_train), "train", train_metrics,
        os.path.join(out_dir, "03_reconstruction_train.png"),
    )
    plot_reconstruction_scatter(
        X_test.numpy(), reconstruct(model, X_test), "test", test_metrics,
        os.path.join(out_dir, "04_reconstruction_test.png"),
    )

    metrics = {
        "dataset": "PBMC_8K",
        "fully_connected_neuron_fraction": fcn_value,
        "n_fully_connected_neurons_selected": int(len(fully_connected_idx)),
        "n_pathway_nodes": int(n_pathway_cols),
        "n_genes": int(mask.shape[0]),
        "n_latent_nodes": int(mask.shape[1]),
        "mask_density": mask_density(mask),
        "n_epochs_trained": len(hist["train_loss"]),
        "train": train_metrics,
        "test": test_metrics,
        "hyperparameters": {
            "n_top_genes": n_top_genes,
            "train_size": train_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "train_patience": train_patience,
            "test_patience": test_patience,
            "kld_weight": kld_weight,
            "dropout": dropout,
            "seed": seed,
            "gmt_path": gmt_path,
        },
    }

    metrics_path = os.path.join(out_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    info_path = os.path.join(out_dir, "training_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("VEGA2 PBMC training run\n")
        f.write("=" * 60 + "\n")
        f.write("Generated (UTC): %s\n" % datetime.now(timezone.utc).isoformat())
        f.write("Output directory: %s\n\n" % out_dir)
        f.write("Data\n")
        f.write("-" * 40 + "\n")
        f.write("data_dir: %s\n" % os.path.abspath(data_dir))
        f.write("gmt_path: %s\n" % gmt_path)
        f.write("n_top_genes: %d\n" % n_top_genes)
        f.write("train cells: %d | val cells: %d | test cells: %d\n\n"
                % (len(splits["adata_train"]), len(splits["adata_val"]), len(splits["adata_test"])))
        f.write("Decoder mask (fully connected neuron selection)\n")
        f.write("-" * 40 + "\n")
        f.write("fully_connected_neuron_fraction: %.4f\n" % fcn_value)
        f.write("fully connected neurons selected: %d / %d\n"
                % (len(fully_connected_idx), n_pathway_cols))
        f.write("original mask density: %.4f\n" % mask_density(base_mask))
        f.write("modified mask density: %.4f\n\n" % mask_density(mask))
        f.write("Hyperparameters\n")
        f.write("-" * 40 + "\n")
        for k, v in metrics["hyperparameters"].items():
            f.write("%s: %s\n" % (k, v))
        f.write("\nModel architecture\n")
        f.write("-" * 40 + "\n")
        f.write(arch_summary + "\n\n")
        f.write("Training\n")
        f.write("-" * 40 + "\n")
        f.write("epochs trained: %d\n" % len(hist["train_loss"]))
        if hist["train_loss"]:
            f.write("final train loss: %.4f\n" % hist["train_loss"][-1])
        if hist["valid_loss"]:
            f.write("final validation loss: %.4f\n" % hist["valid_loss"][-1])
        f.write("saved model: %s\n\n" % model_path)
        f.write("Reconstruction metrics\n")
        f.write("-" * 40 + "\n")
        f.write("TRAIN  MSE=%.6f | Pearson overall=%.6f | per-cell=%.6f\n"
                % (train_metrics["mse"], train_metrics["pearson_overall"],
                   train_metrics["pearson_per_cell_mean"]))
        f.write("TEST   MSE=%.6f | Pearson overall=%.6f | per-cell=%.6f\n"
                % (test_metrics["mse"], test_metrics["pearson_overall"],
                   test_metrics["pearson_per_cell_mean"]))
        f.write("\nOutput files\n")
        f.write("-" * 40 + "\n")
        f.write("01_fully_connected_neuron_mask.png\n")
        f.write("02_training_loss.png\n")
        f.write("03_reconstruction_train.png\n")
        f.write("04_reconstruction_test.png\n")
        f.write("metrics.json\n")
        f.write("training_info.txt\n")
        f.write("vega2_pbmc_fcn%s.pt\n" % fcn_tag)

    print("\nSaved model: %s" % model_path)
    print("Saved metrics: %s" % metrics_path)
    print("Saved training info: %s" % info_path)
    print("TRAIN  MSE=%.4f | Pearson=%.4f" % (train_metrics["mse"], train_metrics["pearson_overall"]))
    print("TEST   MSE=%.4f | Pearson=%.4f" % (test_metrics["mse"], test_metrics["pearson_overall"]))

    return {
        "model": model,
        "metrics": metrics,
        "hist": hist,
        "output_dir": out_dir,
        "model_path": model_path,
        "fully_connected_idx": fully_connected_idx,
        "mask": mask,
    }
