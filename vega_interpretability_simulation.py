"""
VEGA interpretability simulation.

This module builds an "intermediate" VEGA2 variant whose decoder connectivity is
controlled by a single parameter ``fully_connected_neuron_fraction``:

  - The decoder of VEGA2 is a *sparse* linear layer whose connections are given by a
    pathway mask M of shape [n_genes, n_pathways], where M[g, p] = 1 if gene g belongs
    to pathway p (built from a Reactome .gmt file). Each pathway is a latent "node".
  - ``fully_connected_neuron_fraction`` is the fraction (0..1, or a percentage 0..100)
    of latent neurons randomly selected and turned *fully connected*: for each selected
    column p, M[:, p] is set to all 1s, so that neuron connects to every gene.

Special cases:
  - ``fully_connected_neuron_fraction = 0``   -> mask unchanged -> the original sparse VEGA2 decoder.
  - ``fully_connected_neuron_fraction = 1``   -> every pathway column is all 1s -> a fully-connected linear decoder.

Everything needed to build the mask, build/train the model and evaluate reconstruction
is self-contained here (CustomizedLinear, VEGA2, data generation, metrics), so the file
runs in the ``venv_vega`` environment without importing the full ``scvi``-dependent
VEGA package.
"""

import math
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch import optim


# =====================================================================
# 1. Pathway mask construction (from a Reactome .gmt file)
# =====================================================================

def read_gmt(path: str, sep: str = "\t", min_genes: int = 0, max_genes: int = 5000) -> "OrderedDict[str, List[str]]":
    """
    Read a .gmt file into an ordered dict {pathway_name: [genes...]}.
    GMT format per line: name <sep> description/url <sep> gene1 <sep> gene2 ...
    """
    dict_gmv: "OrderedDict[str, List[str]]" = OrderedDict()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            val = line.split(sep)
            genes = val[2:]
            if min_genes <= len(genes) <= max_genes:
                dict_gmv[val[0]] = genes
    return dict_gmv


def make_gmv_mask(feature_list: List[str], dict_gmv: "OrderedDict[str, List[str]]", add_nodes: int = 1) -> np.ndarray:
    """
    Build a mask of shape [n_genes, n_pathways(+add_nodes)] where entry (i, j) = 1 if
    gene i belongs to pathway j, else 0. ``add_nodes`` extra fully-connected (all-ones)
    columns are appended to capture residual variance (as in VEGA).
    """
    p_mask = np.zeros((len(feature_list), len(dict_gmv)), dtype=float)
    feature_index = {g: i for i, g in enumerate(feature_list)}
    for j, k in enumerate(dict_gmv.keys()):
        for gene in dict_gmv[k]:
            i = feature_index.get(gene)
            if i is not None:
                p_mask[i, j] = 1.0
    if add_nodes > 0:
        p_mask = np.hstack((p_mask, np.ones((p_mask.shape[0], add_nodes), dtype=float)))
    return p_mask


def build_pathway_mask(
    gmt_path: str,
    feature_list: Optional[List[str]] = None,
    add_nodes: int = 1,
    min_genes: int = 5,
    max_genes: int = 1000,
    max_pathways: Optional[int] = None,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Construct the pathway mask from a .gmt file.

    Args:
      gmt_path: path to the Reactome .gmt file.
      feature_list: ordered list of genes (mask rows). If None, it is built from the
        union of genes across the selected pathways.
      add_nodes: number of extra fully-connected latent nodes.
      min_genes / max_genes: pathway size filters when reading the gmt.
      max_pathways: optionally keep only the first ``max_pathways`` pathways (useful to
        keep the simulation small/fast).
      seed: reproducibility for pathway subsampling (only used with max_pathways).

    Returns:
      mask: [n_genes, n_pathways + add_nodes] binary matrix.
      gmv_names: list of latent-node names (pathways + UNANNOTATED_k).
      feature_list: ordered list of genes (mask rows).
    """
    dict_gmv = read_gmt(gmt_path, min_genes=min_genes, max_genes=max_genes)
    if len(dict_gmv) == 0:
        raise ValueError("No pathways passed the min_genes/max_genes filter.")

    if max_pathways is not None and max_pathways < len(dict_gmv):
        keys = list(dict_gmv.keys())
        if seed is not None:
            rng = np.random.RandomState(seed)
            keys = list(rng.choice(keys, size=max_pathways, replace=False))
        else:
            keys = keys[:max_pathways]
        dict_gmv = OrderedDict((k, dict_gmv[k]) for k in keys)

    if feature_list is None:
        seen: "OrderedDict[str, None]" = OrderedDict()
        for genes in dict_gmv.values():
            for g in genes:
                seen.setdefault(g, None)
        feature_list = list(seen.keys())

    mask = make_gmv_mask(feature_list, dict_gmv, add_nodes=add_nodes)
    gmv_names = list(dict_gmv.keys()) + [f"UNANNOTATED_{k}" for k in range(add_nodes)]
    return mask, gmv_names, feature_list


# =====================================================================
# 2. Intermediate function: set a random set of pathway nodes fully connected
# =====================================================================

def make_partially_connected_mask(
    mask: np.ndarray,
    fully_connected_neuron_fraction: float,
    add_nodes: int = 0,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Randomly select a fraction of latent neurons and turn them fully connected.

    For each selected pathway column p, the whole column M[:, p] is set to 1, so the
    corresponding latent neuron connects to every gene (a dense connection).

    Args:
      mask: original pathway mask [n_genes, n_columns]. The last ``add_nodes`` columns
        are the already-fully-connected UNANNOTATED nodes and are excluded from the
        selection pool (they are kept as-is).
      fully_connected_neuron_fraction: fraction in [0, 1] (a value > 1 is treated as a
        percentage, e.g. 50 -> 0.5) of selectable latent neurons to make fully connected.
        0 -> mask unchanged (original VEGA2).
        1 -> all pathway columns fully connected (fully-connected linear decoder).
      add_nodes: number of trailing fully-connected nodes to exclude from selection.
      seed: reproducibility for the random choice of neurons.

    Returns:
      new_mask: a copy of the mask with the selected columns set fully connected.
      fully_connected_idx: indices of the latent neurons that were made fully connected.
    """
    if fully_connected_neuron_fraction > 1.0:
        fully_connected_neuron_fraction = fully_connected_neuron_fraction / 100.0
    if not (0.0 <= fully_connected_neuron_fraction <= 1.0):
        raise ValueError(
            "fully_connected_neuron_fraction must be in [0, 1] "
            "(or [0, 100] as a percentage)."
        )

    new_mask = mask.copy().astype(float)
    n_cols = new_mask.shape[1]
    n_pathway_cols = n_cols - add_nodes  # selectable pathway neurons
    if n_pathway_cols <= 0:
        return new_mask, np.array([], dtype=int)

    n_select = int(round(fully_connected_neuron_fraction * n_pathway_cols))
    if n_select <= 0:
        return new_mask, np.array([], dtype=int)

    rng = np.random.RandomState(seed)
    fully_connected_idx = np.sort(
        rng.choice(np.arange(n_pathway_cols), size=n_select, replace=False)
    )
    new_mask[:, fully_connected_idx] = 1.0
    return new_mask, fully_connected_idx


def mask_density(mask: np.ndarray) -> float:
    """Fraction of non-zero entries in the mask (connection density)."""
    return float(np.count_nonzero(mask)) / float(mask.size)


# =====================================================================
# 3. Sparse decoder layer (CustomizedLinear) -- torch-only re-implementation
# =====================================================================

class CustomizedLinearFunction(torch.autograd.Function):
    """Autograd function which masks its weights by ``mask`` (forward and backward)."""

    @staticmethod
    def forward(ctx, input, weight, bias=None, mask=None):
        if mask is not None:
            weight = weight * mask
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        ctx.save_for_backward(input, weight, bias, mask)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias, mask = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = grad_mask = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
            if mask is not None:
                grad_weight = grad_weight * mask
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, grad_mask


class CustomizedLinear(nn.Module):
    """Linear layer whose connections are masked (no learning where mask == 0)."""

    def __init__(self, mask, bias=True):
        super(CustomizedLinear, self).__init__()
        self.input_features = mask.shape[0]
        self.output_features = mask.shape[1]
        if isinstance(mask, torch.Tensor):
            self.mask = mask.type(torch.float).t()
        else:
            self.mask = torch.tensor(mask, dtype=torch.float).t()
        self.mask = nn.Parameter(self.mask, requires_grad=False)
        self.weight = nn.Parameter(torch.Tensor(self.output_features, self.input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.output_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.weight.data = self.weight.data * self.mask

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_params_pos(self):
        """Initialize weights to positive values only."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return CustomizedLinearFunction.apply(input, self.weight, self.bias, self.mask)

    def extra_repr(self):
        return "input_features={}, output_features={}, bias={}".format(
            self.input_features, self.output_features, self.bias is not None
        )


# =====================================================================
# 4. Training utilities (minimal, torch-only)
# =====================================================================

class EarlyStopping:
    """Early stops training if the monitored loss does not improve after `patience`."""

    def __init__(self, patience=7, verbose=False, delta=0, mode="train"):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class WeightClipper(object):
    """Clip weights to positive values (for the positive-decoder constraint)."""

    def __init__(self, frequency=1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "weight"):
            w = module.weight.data
            module.weight.data = w.clamp(0)


# =====================================================================
# 5. VEGA2 model (fully-connected encoder + sparse/partly-sparse decoder)
# =====================================================================

class VEGA2(torch.nn.Module):
    """
    VAE with a fully-connected encoder and a (partially) masked linear decoder built
    from a pathway mask. Mirrors the VEGA2 model in vega_training_copy.py.
    """

    def __init__(self, pathway_mask, positive_decoder=False, **kwargs):
        super(VEGA2, self).__init__()
        self.pathway_mask = pathway_mask
        self.n_genes = self.pathway_mask.shape[0]
        self.n_pathways = self.pathway_mask.shape[1]
        self.dev = kwargs.get("device", torch.device("cpu"))
        # NOTE: `beta` is the VAE KL-divergence weight (original VEGA2 training parameter).
        # Decoder neuron connectivity is controlled separately via
        # ``fully_connected_neuron_fraction`` when building the mask.
        self.beta = kwargs.get("beta", 0.01)
        self.save_path = kwargs.get("save_path", "trained_vega2.pt")
        self.dropout = kwargs.get("dropout", 0.2)
        self.pos_dec = positive_decoder

        self.encoder = nn.Sequential(
            nn.Linear(self.n_genes, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(800, 800),
            nn.BatchNorm1d(800),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        self.mean = nn.Sequential(nn.Linear(800, self.n_pathways), nn.Dropout(self.dropout))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_pathways), nn.Dropout(self.dropout))
        self.decoder = CustomizedLinear(self.pathway_mask.T)
        if self.pos_dec:
            print("Constraining decoder to positive weights", flush=True)
            self.decoder.reset_params_pos()
            self.decoder.weight.data *= self.decoder.mask

    def encode(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def sample_latent(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z

    def forward(self, X):
        z, mu, logvar = self.encode(X)
        X_rec = self.decode(z)
        return X_rec, mu, logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        kld = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp())
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta * kld), mse, kld

    def train_model(self, train_loader, learning_rate, n_epochs, train_patience,
                    test_patience, test_loader=False, save_model=False):
        epoch_hist = {"train_loss": [], "valid_loss": []}
        epoch_mse_hist = {"train_loss": [], "valid_loss": []}
        epoch_kld_hist = {"train_loss": [], "valid_loss": []}
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode="train")
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode="valid")
        clipper = WeightClipper(frequency=1)
        for epoch in range(n_epochs):
            loss_value = mse_loss_value = kld_loss_value = 0
            self.train()
            for x_train in train_loader:
                x_train = x_train.to(self.dev)
                optimizer.zero_grad()
                x_rec, mu, logvar = self.forward(x_train)
                loss, mse, kld = self.vae_loss(x_rec, x_train, mu, logvar)
                loss_value += loss.item()
                mse_loss_value += mse.item()
                kld_loss_value += kld.item()
                loss.backward()
                optimizer.step()
                if self.pos_dec:
                    self.decoder.apply(clipper)
            denom = len(train_loader) * train_loader.batch_size
            epoch_loss = loss_value / denom
            epoch_mse_loss = mse_loss_value / denom
            epoch_kld_loss = kld_loss_value / denom
            epoch_hist["train_loss"].append(epoch_loss)
            epoch_mse_hist["train_loss"].append(epoch_mse_loss)
            epoch_kld_hist["train_loss"].append(epoch_kld_loss)
            train_ES(epoch_loss)
            if test_loader:
                self.eval()
                test_dict, mse_test_dict, kld_test_dict = self.test_model(test_loader)
                test_loss = test_dict["loss"]
                epoch_hist["valid_loss"].append(test_loss)
                epoch_mse_hist["valid_loss"].append(mse_test_dict["loss"])
                epoch_kld_hist["valid_loss"].append(kld_test_dict["loss"])
                valid_ES(test_loss)
                print(
                    "[Epoch %d] | train_loss: %.3f, train_mse: %.3f, train_kld: %.3f | "
                    "test_loss: %.3f, test_mse: %.3f, test_kld: %.3f |"
                    % (epoch + 1, epoch_loss, epoch_mse_loss, epoch_kld_loss,
                       test_loss, mse_test_dict["loss"], kld_test_dict["loss"]),
                    flush=True,
                )
                if valid_ES.early_stop or train_ES.early_stop:
                    print("[Epoch %d] Early stopping" % (epoch + 1), flush=True)
                    break
            else:
                print("[Epoch %d] | loss: %.3f |" % (epoch + 1, epoch_loss), flush=True)
                if train_ES.early_stop:
                    print("[Epoch %d] Early stopping" % (epoch + 1), flush=True)
                    break
        if save_model and self.save_path:
            print("Saving model to ...", self.save_path)
            torch.save(self.state_dict(), self.save_path)
        return epoch_hist, epoch_mse_hist, epoch_kld_hist

    def test_model(self, loader):
        test_dict, mse_test_dict, kld_test_dict = {}, {}, {}
        total_loss = total_mse = total_kld = 0.0
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.dev)
                reconstruct_X, mu, logvar = self.forward(data)
                loss, mse, kld = self.vae_loss(reconstruct_X, data, mu, logvar)
                total_loss += loss.item()
                total_mse += mse.item()
                total_kld += kld.item()
        denom = len(loader) * loader.batch_size
        test_dict["loss"] = total_loss / denom
        mse_test_dict["loss"] = total_mse / denom
        kld_test_dict["loss"] = total_kld / denom
        return test_dict, mse_test_dict, kld_test_dict


# =====================================================================
# 6. Synthetic expression data with pathway structure (learnable reconstruction)
# =====================================================================

def generate_expression_from_mask(
    mask: np.ndarray,
    n_cells: int,
    noise_std: float = 0.1,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate synthetic expression X with structure induced by the pathway mask, so a
    masked decoder can learn to reconstruct it.

      z ~ N(0, 1)              latent pathway activities, shape (n_cells, n_pathways)
      W = U * mask             gene-by-pathway weights (sparse, positive), (n_genes, n_pathways)
      X = z @ W^T + noise      expression, shape (n_cells, n_genes)
    """
    rng = np.random.RandomState(seed)
    n_genes, n_pathways = mask.shape
    W = rng.uniform(0.5, 1.5, size=(n_genes, n_pathways)) * mask
    z = rng.standard_normal((n_cells, n_pathways))
    X = z @ W.T
    X = X + rng.normal(0.0, noise_std, size=X.shape)
    return X.astype(np.float32)


def build_dataloaders(
    X: np.ndarray,
    batch_size: int = 128,
    train_frac: float = 0.85,
    seed: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.Tensor, torch.Tensor]:
    """Split X into train/validation tensors and wrap into DataLoaders."""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32)
    train_loader = torch.utils.data.DataLoader(X_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(X_val, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader, val_loader, X_train, X_val


# =====================================================================
# 7. Reconstruction evaluation: MSE and Pearson correlation
# =====================================================================

@torch.no_grad()
def reconstruct(model: VEGA2, X: torch.Tensor) -> np.ndarray:
    """Return the model reconstruction of X as a numpy array."""
    model.eval()
    X = X.to(model.dev)
    x_rec, _, _ = model.forward(X)
    return x_rec.cpu().numpy()


def compute_mse(x_true: np.ndarray, x_pred: np.ndarray) -> float:
    """Mean squared error over all entries."""
    return float(np.mean((x_true - x_pred) ** 2))


def compute_pearson(x_true: np.ndarray, x_pred: np.ndarray) -> Dict[str, float]:
    """
    Pearson correlation between input and reconstruction.

    Returns:
      overall: correlation of all flattened entries.
      per_cell_mean: mean over cells of the per-cell correlation.
    """
    flat_true = x_true.ravel()
    flat_pred = x_pred.ravel()
    overall, _ = pearsonr(flat_true, flat_pred)

    per_cell = []
    for i in range(x_true.shape[0]):
        if np.std(x_true[i]) > 0 and np.std(x_pred[i]) > 0:
            r, _ = pearsonr(x_true[i], x_pred[i])
            per_cell.append(r)
    per_cell_mean = float(np.mean(per_cell)) if per_cell else float("nan")
    return {"overall": float(overall), "per_cell_mean": per_cell_mean}
