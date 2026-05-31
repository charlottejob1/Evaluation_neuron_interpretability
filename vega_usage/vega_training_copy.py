import sys
import os
from pathlib import Path

#sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/src')
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vega
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
from customized_linear import CustomizedLinear
import torch.nn.functional as F
from torch import nn, optim
from scipy import sparse
#from vanilla_vae import VanillaVAE
from vega_model import VEGA


import torch
import itertools
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import pandas as pd
import scanpy as sc
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import sparse
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import itertools
import argparse
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import anndata as ad
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px  # For color palette
from sklearn.model_selection import train_test_split
import umap
from scipy.sparse import issparse
import anndata as ad
from anndata import AnnData
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import KFold


#sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/clean code')
sys.path.append('/home/BS94_SUR/phD/review/utils/utils_evaluation/')
import vega_utils
from vega_utils import *
import utils_evaluation_models
from utils_evaluation_models import *

#device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
#dev = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


class VEGA2(torch.nn.Module):
    def __init__(self, pathway_mask, positive_decoder=False, **kwargs):
        """ Constructor for class VEGA (VAE Enhanced by Gene Annotations). """
        super(VEGA2, self).__init__()
        self.pathway_mask = pathway_mask
        self.n_pathways = self.pathway_mask.shape[1]
        self.n_genes = self.pathway_mask.shape[0]
        self.dev = kwargs.get('device', torch.device('cpu'))
        self.beta = kwargs.get("beta", 0.01)
        self.save_path = kwargs.get('path_model', "trained_vae.pt")
        self.dropout = kwargs.get('dropout', 0.2)
        self.pos_dec = positive_decoder
        print(self.dropout)
        self.encoder = nn.Sequential(nn.Linear(self.n_genes, 800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(800,800),
                                    nn.BatchNorm1d(800),
                                    nn.ReLU(),
                                    nn.Dropout(self.dropout))
        self.mean = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.logvar = nn.Sequential(nn.Linear(800, self.n_pathways), 
                                    nn.Dropout(self.dropout))
        self.decoder = CustomizedLinear(self.pathway_mask.T)
        # Constraining decoder or not
        if self.pos_dec:
            print('Constraining decoder to positive weights', flush=True)
            self.decoder.reset_params_pos()
            self.decoder.weight.data *= self.decoder.mask        


    def encode(self, X):
        """ Encode data """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z, mu, logvar
    
    def decode(self, z):
        """ Decode data """
        X_rec = self.decoder(z)
        return X_rec
    
    def sample_latent(self, mu, logvar):
        """ Sample latent space with reparametrization trick. First convert to std, sample normal(0,1) and get Z."""
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(self.dev)
        eps = eps.mul_(std).add_(mu)
        return eps

    def to_latent(self, X):
        """ Same as encode, but only returns z (no mu and logvar) """
        y = self.encoder(X)
        mu, logvar = self.mean(y), self.logvar(y)
        z = self.sample_latent(mu, logvar)
        return z
 
    def _average_latent(self, X):
        """ """
        z = self.to_latent(X)
        mean_z = z.mean(0)
        return mean_z

    def bayesian_diff_exp(self, adata1, adata2, n_samples=2000, use_permutations=True, n_permutations=1000, random_seed=False):
        """ Run Bayesian differential expression in latent space.
            Returns Bayes factor of all factors. 
        """
        self.eval()
        # Set seed for reproducibility
        if random_seed:
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)
        epsilon = 1e-12
        # Sample cell from each condition
        idx1 = np.random.choice(np.arange(len(adata1)), n_samples)
        idx2 = np.random.choice(np.arange(len(adata2)), n_samples)
        # To latent
        z1 = self.to_latent(torch.Tensor(adata1[idx1,:].X)).detach().numpy()
        z2 = self.to_latent(torch.Tensor(adata2[idx2,:].X)).detach().numpy()
        # Compare samples by using number of permutations - if 0, just pairwise comparison
        # This estimates the double integral in the posterior of the hypothesis
        if use_permutations:
            z1, z2 = self._scale_sampling(z1, z2, n_perm=n_permutations)
        p_h1 = np.mean(z1 > z2, axis=0)
        p_h2 = 1.0 - p_h1
        mad = np.abs(np.mean(z1 - z2, axis=0))
        bf = np.log(p_h1 + epsilon) - np.log(p_h2 + epsilon) 
        # Wrap results
        res = {'p_h1':p_h1,
                'p_h2':p_h2,
                'bayes_factor': bf,
                'mad':mad}
        return res

    def _scale_sampling(self, arr1, arr2, n_perm=1000):
        """ Use permutation to better estimate double integral (create more pair comparisons)
            Inspired by scVI (Lopez et al., 2018) """
        u, v = (np.random.choice(arr1.shape[0], size=n_perm), np.random.choice(arr2.shape[0], size=n_perm))
        scaled1 = arr1[u]
        scaled2 = arr2[v]
        return scaled1, scaled2

    def forward(self, X):
        """ Forward pass through full network"""
        z, mu, logvar = self.encode(X)
        X_rec = self.decode(z)
        return X_rec, mu, logvar

    def vae_loss(self, y_pred, y_true, mu, logvar):
        """ Custom loss for VAE """
        kld = -0.5 * torch.sum(1. + logvar - mu.pow(2) - logvar.exp(), )
        mse = F.mse_loss(y_pred, y_true, reduction="sum")
        return torch.mean(mse + self.beta*kld), mse, kld

    def train_model(self, train_loader, learning_rate, n_epochs, train_patience, test_patience, test_loader=False, save_model=True):
        """ Train VAE """
        epoch_hist = {}
        epoch_hist['train_loss'] = []
        epoch_hist['valid_loss'] = []
        epoch_mse_hist = {}
        epoch_mse_hist['train_loss'] = []
        epoch_mse_hist['valid_loss'] = []
        epoch_kld_hist = {}
        epoch_kld_hist['train_loss'] = []
        epoch_kld_hist['valid_loss'] = []
        optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=5e-4)
        train_ES = EarlyStopping(patience=train_patience, verbose=True, mode='train')
        if test_loader:
            valid_ES = EarlyStopping(patience=test_patience, verbose=True, mode='valid')
        clipper = WeightClipper(frequency=1)
        # Train
        for epoch in range(n_epochs):
            loss_value = 0
            mse_loss_value = 0
            kld_loss_value = 0
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
            # Get epoch loss
            epoch_loss = loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_mse_loss = mse_loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_kld_loss = kld_loss_value / (len(train_loader) * train_loader.batch_size)
            epoch_hist['train_loss'].append(epoch_loss)
            epoch_mse_hist['train_loss'].append(epoch_mse_loss)
            epoch_kld_hist['train_loss'].append(epoch_kld_loss)
            train_ES(epoch_loss)
            # Eval
            if test_loader:
                self.eval()
                test_dict, mse_test_dict, kld_test_dict = self.test_model(test_loader)
                test_loss, mse_test_loss, kld_test_loss = test_dict['loss'], mse_test_dict['loss'], kld_test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                epoch_mse_hist['valid_loss'].append(mse_test_loss)
                epoch_kld_hist['valid_loss'].append(kld_test_loss)
                valid_ES(test_loss)
                print('[Epoch %d] | train_loss: %.3f, train_mse: %.3f, train_kld: %.3f | test_loss: %.3f, test_mse: %.3f, test_kld: %.3f |'%(epoch+1, epoch_loss, epoch_mse_loss, epoch_kld_loss, test_loss, mse_test_loss, kld_test_loss), flush=True)
                if valid_ES.early_stop or train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
            else:
                print('[Epoch %d] | loss: %.3f |' % (epoch + 1, epoch_loss), flush=True)
                if train_ES.early_stop:
                    print('[Epoch %d] Early stopping' % (epoch+1), flush=True)
                    break
        # Save model
        if save_model:
            print('Saving model to ...', self.save_path)
            torch.save(self.state_dict(), self.save_path)

        return epoch_hist, epoch_mse_hist, epoch_kld_hist

    def test_model(self, loader):
        """Test model on input loader."""
        test_dict = {}
        mse_test_dict = {}
        kld_test_dict = {}
        total_loss = 0.0
        total_mse = 0.0
        total_kld = 0.0
        self.eval()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.dev)
                reconstruct_X, mu, logvar = self.forward(data)
                loss, mse, kld = self.vae_loss(reconstruct_X, data, mu, logvar)
                total_loss += loss.item()
                total_mse += mse.item()
                total_kld += kld.item()
        test_dict['loss'] = total_loss/(len(loader)*loader.batch_size)
        mse_test_dict['loss'] = total_mse/(len(loader)*loader.batch_size)
        kld_test_dict['loss'] = total_kld/(len(loader)*loader.batch_size)
        return test_dict, mse_test_dict, kld_test_dict
    


def random_search_hyperparameters(
    train_data,
    test_data,
    pathway_mask,
    device,
    save_path='random_hyperparameter_search_results.csv',
    n_iterations=50,
    n_epochs=2000,
    train_p=25,
    test_p=25,
    random_seed=42
):
    """
    Perform random search for hyperparameter optimization.
    
    Parameters:
    -----------
    train_data : torch.utils.data.Dataset
        Training dataset
    test_data : torch.utils.data.Dataset
        Test dataset
    pathway_mask : torch.Tensor
        Pathway mask for VEGA model
    device : torch.device
        Device to run training on
    save_path : str
        Path where to save the results CSV file (results saved after each iteration)
    n_iterations : int
        Number of random search iterations
    n_epochs : int
        Number of epochs per training
    train_p : int
        Training patience for early stopping
    test_p : int
        Test patience for early stopping
    random_seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame : DataFrame with hyperparameters and corresponding losses
    """
    
    # Set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    # Define hyperparameter search spaces
    param_distributions = {
        'beta': (1e-9, 1e-3, 'log'),  # (min, max, scale)
        'learning_rate': (1e-5, 1e-2, 'log'),
        'batch_size': ([32, 64, 128, 256, 512], 'choice'),
        'dropout': (0.0, 1, 'uniform')
    }
    
    # Initialize results storage
    results = []
    
    # Check if file exists and load previous results
    try:
        existing_df = pd.read_csv(save_path)
        results = existing_df.to_dict('records')
        print(f"Loaded {len(results)} existing results from {save_path}")
        start_iteration = len(results)
    except FileNotFoundError:
        print(f"No existing results found. Starting fresh.")
        start_iteration = 0
    
    print(f"Starting random search with {n_iterations} iterations...")
    print("="*70)
    
    for iteration in range(start_iteration, start_iteration + n_iterations):
        print(f"\nIteration {iteration + 1}/{start_iteration + n_iterations}")
        print("-"*70)
        
        # Sample hyperparameters
        if param_distributions['beta'][2] == 'log':
            beta = 10 ** np.random.uniform(
                np.log10(param_distributions['beta'][0]),
                np.log10(param_distributions['beta'][1])
            )
        
        if param_distributions['learning_rate'][2] == 'log':
            lr = 10 ** np.random.uniform(
                np.log10(param_distributions['learning_rate'][0]),
                np.log10(param_distributions['learning_rate'][1])
            )
        
        if param_distributions['batch_size'][1] == 'choice':
            batch_size = int(np.random.choice(param_distributions['batch_size'][0]))
        
        if param_distributions['dropout'][2] == 'uniform':
            dropout = np.random.uniform(
                param_distributions['dropout'][0],
                param_distributions['dropout'][1]
            )
        
        # Print current hyperparameters
        print(f"beta: {beta:.2e}")
        print(f"learning_rate: {lr:.2e}")
        print(f"batch_size: {batch_size}")
        print(f"dropout: {dropout:.3f}")
        
        # Create data loaders with current batch size
        train_loader = torch.utils.data.DataLoader(
            train_data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_data, 
            batch_size=batch_size, 
            shuffle=True, 
            drop_last=True
        )
        
        # Create model with current hyperparameters
        dict_params = {
            "pathway_mask": pathway_mask,
            "n_pathways": pathway_mask.shape[1],
            "n_genes": pathway_mask.shape[0],
            "device": device,
            "beta": beta,
            "save_path": False,
            "dropout": dropout,
            "pos_dec": True,
        }
        
        try:
            # Initialize model
            model = VEGA2(**dict_params).to(device)
            
            # Train model
            hist, mse_hist, kld_hist = model.train_model(
                train_loader,
                lr,
                n_epochs,
                train_p,
                test_p,
                test_loader,
                save_model=False
            )
            
            # Get final validation loss
            final_loss = hist["valid_loss"][-1]
            final_train_loss = hist["train_loss"][-1]
            final_mse_loss = mse_hist["valid_loss"][-1]
            final_mse_train_loss = mse_hist["train_loss"][-1]
            final_kld_loss = kld_hist["valid_loss"][-1]
            final_kld_train_loss = kld_hist["train_loss"][-1]
            n_epochs_trained = len(hist["train_loss"])
            
            print(f"Final validation loss: {final_loss:.4f}")
            print(f"Final training loss: {final_train_loss:.4f}")
            print(f"Final validation mse loss: {final_mse_loss:.4f}")
            print(f"Final training mse loss: {final_mse_train_loss:.4f}")
            print(f"Final validation kld loss: {final_kld_loss:.4f}")
            print(f"Final training kld loss: {final_kld_train_loss:.4f}")
            print(f"Epochs trained: {n_epochs_trained}")
            
            # Store results
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'final_valid_loss': final_loss,
                'final_train_loss': final_train_loss,
                'final_mse_valid_loss': final_mse_loss,
                'final_mse_train_loss': final_mse_train_loss,
                'final_kld_valid_loss': final_kld_loss,
                'final_kld_train_loss': final_kld_train_loss,
                'n_epochs_trained': n_epochs_trained,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Save results immediately after each iteration
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            results.append({
                'iteration': iteration + 1,
                'beta': beta,
                'learning_rate': lr,
                'batch_size': batch_size,
                'dropout': dropout,
                'final_valid_loss': np.nan,
                'final_train_loss': np.nan,
                'n_epochs_trained': 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': str(e)
            })
            
            # Save results even on error
            results_df = pd.DataFrame(results)
            results_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by validation loss
    results_df = results_df.sort_values('final_valid_loss', ascending=True)
    
    print("\n" + "="*70)
    print("Random search completed!")
    print(f"\nBest hyperparameters (lowest validation loss):")
    print("-"*70)
    best_idx = results_df['final_valid_loss'].idxmin()
    print(results_df.loc[best_idx, ['beta', 'learning_rate',
                                     'batch_size', 'dropout', 'final_valid_loss']])
    
    return results_df


class Vega_train_multiple_times:
    def __init__(self,
                 adata: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n_training: int, 
                 n_epochs: int,
                 lr:int,
                 pathway_file,
                 batch_size:int,
                 beta: int,
                 dropout: int,
                 train_p,
                 test_p,
                 dev: str,
                 save_path: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                        ) -> None:

        self.adata = adata
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_training = n_training
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_file = pathway_file
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.train_p = train_p
        self.test_p = test_p
        self.dev = dev
        self.save_path = save_path
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, val_data, test_data, batch_size, name_model):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader, test_loader

    def train_VEGA_n_times(self, adata, n_top_genes, val_resolution, train_size, random_seed_list, column_labels_name, path_to_save_embeddings, path_to_save_reconstructed, save_path_results, path_save_fig):
        results = []
        for n in range(self.n_training):
            random_seed = random_seed_list[n]
            print(f'Training {n} - seed {random_seed}')
            adata, adata_train, adata_test, train_data, test_data, pathway_dict, pathway_mask = create_vega_training_data(self.name_model, False, n_top_genes, random_seed, train_size, column_labels_name, adata, self.pathway_file)
            adata, adata_train, adata_val, train_data, val_data, pathway_dict, pathway_mask = create_vega_training_data(self.name_model, False, n_top_genes, random_seed, train_size, column_labels_name, adata_train, self.pathway_file)

            train_loader, val_loader, test_loader = self.build_data_loader(train_data, val_data, test_data, self.batch_size, self.name_model)    
            dict_params = {'pathway_mask': pathway_mask, 'n_pathways':pathway_mask.shape[1], 'n_genes':pathway_mask.shape[0], 'device':self.dev, 'beta':self.beta, 'save_path':self.save_path,  'dropout':self.dropout, 'pos_dec':True}
            pathway_list = list(pathway_dict.keys())+['UNANNOTATED_'+str(k) for k in range(1)]  

            model = VEGA2(**dict_params).to(dev)
            hist, mse_hist, kld_hist = model.train_model(train_loader, self.lr, self.n_epochs, self.train_p, self.test_p, val_loader, save_model=True)

            final_loss = hist["valid_loss"][-1]
            final_train_loss = hist["train_loss"][-1]
            final_mse_loss = mse_hist["valid_loss"][-1]
            final_mse_train_loss = mse_hist["train_loss"][-1]
            final_kld_loss = kld_hist["valid_loss"][-1]
            final_kld_train_loss = kld_hist["train_loss"][-1]
            n_epochs_trained = len(hist["train_loss"])

            embedding_train = model.to_latent(torch.Tensor(adata_train.X.A if sparse.issparse(adata_train.X) else adata_train.X).to(dev))
            embedding_train_array = embedding_train.cpu().detach().numpy()
            pd.DataFrame(embedding_train_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_train_{n}_trial_{random_seed}.txt')
            X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(dev))
            X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_train_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_train_{n}_trial_{random_seed}.txt')

            embedding_val = model.to_latent(torch.Tensor(adata_val.X.A if sparse.issparse(adata_val.X) else adata_val.X).to(dev))
            embedding_val_array = embedding_val.cpu().detach().numpy()
            pd.DataFrame(embedding_val_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_val_{n}_trial_{random_seed}.txt')
            X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(dev))
            X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_val_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_val_{n}_trial_{random_seed}.txt')

            embedding_test = model.to_latent(torch.Tensor(adata_test.X.A if sparse.issparse(adata_test.X) else adata_test.X).to(dev))
            embedding_test_array = embedding_test.cpu().detach().numpy()
            pd.DataFrame(embedding_test_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_test_{n}_trial_{random_seed}.txt')
            X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(dev))
            X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_test_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_test_{n}_trial_{random_seed}.txt')

            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)

            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)

            adata_latent_train = build_adata_latent(embedding_train_array, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_array, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_array, adata_test, column_labels_name)

            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', val_resolution)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', val_resolution)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', val_resolution)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)          
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test) 
            
            if path_save_fig:
                plot_umap_orig_and_clusters(embedding_train_array, true_labels_train, self.name_dataset, self.name_model 
                , clusters_train, 
                    ari_train, nmi_train, 'Train', n, 'Leiden', path_save_fig)
                plot_umap_orig_and_clusters(embedding_val_array, true_labels_val, self.name_dataset, self.name_model, clusters_val, 
                    ari_val, nmi_val, 'Val', n, 'Leiden', path_save_fig)
                plot_umap_orig_and_clusters(embedding_test_array, true_labels_test, self.name_dataset, self.name_model, clusters_test, 
                    ari_test, nmi_test, 'Test', n, 'Leiden', path_save_fig)
                
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)


            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'id_training': n,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'final_train_loss': final_train_loss,
                'final_mse_train_loss': final_mse_train_loss,
                'final_kld_train_loss': final_kld_train_loss,
                'final_valid_loss': final_loss,
                'final_mse_valid_loss': final_mse_loss,
                'final_kld_valid_loss': final_kld_loss,
                'n_epochs_trained': n_epochs_trained,
                'mse_score_train': mse_score_train,
                'mse_score_val': mse_score_val,
                'mse_score_test': mse_score_test,
                'corr_train': corr_train,
                'corr_val': corr_val,
                'corr_test': corr_test,
                'ari_train': ari_train,
                'nmi_train': nmi_train,
                'ari_val': ari_val,
                'nmi_val': nmi_val,
                'ari_test': ari_test,
                'nmi_test': nmi_test,
                'accuracy_train_rf': accuracy_train_rf,
                'precision_train_rf': precision_train_rf,
                'recall_train_rf': recall_train_rf,
                'f1_train_rf': f1_train_rf,
                'roc_auc_train_rf': roc_auc_train_rf,
                'accuracy_val_rf': accuracy_val_rf,
                'precision_val_rf': precision_val_rf,
                'recall_val_rf': recall_val_rf,
                'f1_val_rf': f1_val_rf,
                'roc_auc_val_rf': roc_auc_val_rf,
                'accuracy_test_rf': accuracy_test_rf,
                'precision_test_rf': precision_test_rf,
                'recall_test_rf': recall_test_rf,
                'f1_test_rf': f1_test_rf,
                'roc_auc_test_rf': roc_auc_test_rf,
                'accuracy_train_xg': accuracy_train_xg,
                'precision_train_xg': precision_train_xg,
                'recall_train_xg': recall_train_xg,
                'f1_train_xg': f1_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'accuracy_val_xg': accuracy_val_xg,
                'precision_val_xg': precision_val_xg,
                'recall_val_xg': recall_val_xg,
                'f1_val_xg': f1_val_xg,
                'roc_auc_val_xg': roc_auc_val_xg,
                'accuracy_test_xg': accuracy_test_xg,
                'precision_test_xg': precision_test_xg,
                'recall_test_xg': recall_test_xg,
                'f1_test_xg': f1_test_xg,
                'roc_auc_test_xg': roc_auc_test_xg
            })

        # Create DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path_results + f'{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv', index=False)
        print(f"Results saved to {save_path_results}{self.name_model}_{self.name_dataset}_{self.n_training}_training_results.csv")
        
        return results_df

        
        
class Vega_train_different_datasets_sizes:
    def __init__(self,
                 adata: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n_training: int, 
                 n_epochs: int,
                 lr:int,
                 pathway_file,
                 batch_size:int,
                 beta: int,
                 dropout: int,
                 train_p,
                 test_p,
                 dev: str,
                 save_path: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                        ) -> None:

        self.adata = adata
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_training = n_training
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_file = pathway_file
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.train_p = train_p
        self.test_p = test_p
        self.dev = dev
        self.save_path = save_path
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, val_data, test_data, batch_size, name_model):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader, test_loader

    def train_VEGA_n_times_reduced_dataset(self, adata, size_data_selected, n_top_genes, val_resolution, train_size, random_seed_list, column_labels_name, path_to_save_embeddings, path_to_save_reconstructed, save_path_results, path_save_fig):
        results = []
        for n in range(self.n_training):
            random_seed = random_seed_list[n]
            n_percent = size_data_selected * 100
            print(f'Training {n} - seed {random_seed} - {n_percent} % of dataset selected')
            
            np.random.seed(random_seed) 
            n_cells = adata.n_obs
            n_select = int(size_data_selected * n_cells)
            selected_idx = np.random.choice(
                adata.obs_names,
                size=n_select,
                replace=False
            )
            adata_selected = adata[selected_idx].copy()
            print(adata_selected.shape)
            
            adata, adata_train, adata_test, train_data, test_data, pathway_dict, pathway_mask = create_vega_training_data(self.name_model, False, n_top_genes, random_seed, train_size, column_labels_name, adata_selected, self.pathway_file)
            adata, adata_train, adata_val, train_data, val_data, pathway_dict, pathway_mask = create_vega_training_data(self.name_model, False, n_top_genes, random_seed, train_size, column_labels_name, adata_train, self.pathway_file)

            train_loader, val_loader, test_loader = self.build_data_loader(train_data, val_data, test_data, self.batch_size, self.name_model)    
            dict_params = {'pathway_mask': pathway_mask, 'n_pathways':pathway_mask.shape[1], 'n_genes':pathway_mask.shape[0], 'device':self.dev, 'beta':self.beta, 'save_path':self.save_path,  'dropout':self.dropout, 'pos_dec':True}
            pathway_list = list(pathway_dict.keys())+['UNANNOTATED_'+str(k) for k in range(1)]  

            model = VEGA2(**dict_params).to(dev)
            hist, mse_hist, kld_hist = model.train_model(train_loader, self.lr, self.n_epochs, self.train_p, self.test_p, val_loader, save_model=True)

            final_loss = hist["valid_loss"][-1]
            final_train_loss = hist["train_loss"][-1]
            final_mse_loss = mse_hist["valid_loss"][-1]
            final_mse_train_loss = mse_hist["train_loss"][-1]
            final_kld_loss = kld_hist["valid_loss"][-1]
            final_kld_train_loss = kld_hist["train_loss"][-1]
            n_epochs_trained = len(hist["train_loss"])

            embedding_train = model.to_latent(torch.Tensor(adata_train.X.A if sparse.issparse(adata_train.X) else adata_train.X).to(dev))
            embedding_train_array = embedding_train.cpu().detach().numpy()
            pd.DataFrame(embedding_train_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_{n_percent}%_embeddings_train_{n}_trial_{random_seed}.txt')
            X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(dev))
            X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_train_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_{n_percent}%_reconstruction_train_{n}_trial_{random_seed}.txt')

            embedding_val = model.to_latent(torch.Tensor(adata_val.X.A if sparse.issparse(adata_val.X) else adata_val.X).to(dev))
            embedding_val_array = embedding_val.cpu().detach().numpy()
            pd.DataFrame(embedding_val_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_{n_percent}%_embeddings_val_{n}_trial_{random_seed}.txt')
            X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(dev))
            X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_val_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_{n_percent}%_reconstruction_val_{n}_trial_{random_seed}.txt')

            embedding_test = model.to_latent(torch.Tensor(adata_test.X.A if sparse.issparse(adata_test.X) else adata_test.X).to(dev))
            embedding_test_array = embedding_test.cpu().detach().numpy()
            pd.DataFrame(embedding_test_array).to_csv(path_to_save_embeddings + f'/vega_{self.name_dataset}_{n_percent}%_embeddings_test_{n}_trial_{random_seed}.txt')
            X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(dev))
            X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_test_array).to_csv(path_to_save_reconstructed + f'/vega_{self.name_dataset}_{n_percent}%_reconstruction_test_{n}_trial_{random_seed}.txt')

            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)

            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)

            adata_latent_train = build_adata_latent(embedding_train_array, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_array, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_array, adata_test, column_labels_name)

            clusters_train, true_labels_train = apply_clustering_algo(adata_latent_train, column_labels_name, 'Leiden', val_resolution)
            clusters_val, true_labels_val = apply_clustering_algo(adata_latent_val, column_labels_name, 'Leiden', val_resolution)
            clusters_test, true_labels_test = apply_clustering_algo(adata_latent_test, column_labels_name, 'Leiden', val_resolution)
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)          
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test) 
            
            if path_save_fig:
                plot_umap_orig_and_clusters(embedding_train_array, true_labels_train, self.name_dataset, self.name_model 
                , clusters_train, 
                    ari_train, nmi_train, 'Train', n, 'Leiden', path_save_fig)
                plot_umap_orig_and_clusters(embedding_val_array, true_labels_val, self.name_dataset, self.name_model, clusters_val, 
                    ari_val, nmi_val, 'Val', n, 'Leiden', path_save_fig)
                plot_umap_orig_and_clusters(embedding_test_array, true_labels_test, self.name_dataset, self.name_model, clusters_test, 
                    ari_test, nmi_test, 'Test', n, 'Leiden', path_save_fig)
                
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)

            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'id_training': n,
                'size_data_selected': size_data_selected,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'final_train_loss': final_train_loss,
                'final_mse_train_loss': final_mse_train_loss,
                'final_kld_train_loss': final_kld_train_loss,
                'final_valid_loss': final_loss,
                'final_mse_valid_loss': final_mse_loss,
                'final_kld_valid_loss': final_kld_loss,
                'n_epochs_trained': n_epochs_trained,
                'mse_score_train': mse_score_train,
                'mse_score_val': mse_score_val,
                'mse_score_test': mse_score_test,
                'corr_train': corr_train,
                'corr_val': corr_val,
                'corr_test': corr_test,
                'ari_train': ari_train,
                'nmi_train': nmi_train,
                'ari_val': ari_val,
                'nmi_val': nmi_val,
                'ari_test': ari_test,
                'nmi_test': nmi_test,
                'accuracy_train_rf': accuracy_train_rf,
                'precision_train_rf': precision_train_rf,
                'recall_train_rf': recall_train_rf,
                'f1_train_rf': f1_train_rf,
                'roc_auc_train_rf': roc_auc_train_rf,
                'accuracy_val_rf': accuracy_val_rf,
                'precision_val_rf': precision_val_rf,
                'recall_val_rf': recall_val_rf,
                'f1_val_rf': f1_val_rf,
                'roc_auc_val_rf': roc_auc_val_rf,
                'accuracy_test_rf': accuracy_test_rf,
                'precision_test_rf': precision_test_rf,
                'recall_test_rf': recall_test_rf,
                'f1_test_rf': f1_test_rf,
                'roc_auc_test_rf': roc_auc_test_rf,
                'accuracy_train_xg': accuracy_train_xg,
                'precision_train_xg': precision_train_xg,
                'recall_train_xg': recall_train_xg,
                'f1_train_xg': f1_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'accuracy_val_xg': accuracy_val_xg,
                'precision_val_xg': precision_val_xg,
                'recall_val_xg': recall_val_xg,
                'f1_val_xg': f1_val_xg,
                'roc_auc_val_xg': roc_auc_val_xg,
                'accuracy_test_xg': accuracy_test_xg,
                'precision_test_xg': precision_test_xg,
                'recall_test_xg': recall_test_xg,
                'f1_test_xg': f1_test_xg,
                'roc_auc_test_xg': roc_auc_test_xg
            })

        # Create DataFrame
        results_df = pd.DataFrame(results)
        results_df.to_csv(save_path_results + f'{self.name_model}_{n_percent}%_selected_{self.name_dataset}_{self.n_training}_training_results.csv', index=False)
        print(f"Results saved to {save_path_results}{self.name_model}_{n_percent}%_selected_{self.name_dataset}_{self.n_training}_training_results.csv")
        
        return results_df
    
    

class Vega_nfold_cross_validation:
    def __init__(self,
                 adata: AnnData,
                 name_model: str,
                 name_dataset: str,
                 n_folds: int,  # Changed from n_training to n_folds
                 n_epochs: int,
                 lr: int,
                 pathway_file,
                 batch_size: int,
                 beta: int,
                 dropout: int,
                 train_p,
                 test_p,
                 dev: str,
                 save_path: bool,
                 path_to_save_embeddings: str,
                 path_to_save_reconstructed: str
                 ) -> None:

        self.adata = adata
        self.name_model = name_model
        self.name_dataset = name_dataset
        self.n_folds = n_folds
        self.n_epochs = n_epochs
        self.lr = lr
        self.pathway_file = pathway_file
        self.batch_size = batch_size
        self.beta = beta
        self.dropout = dropout
        self.train_p = train_p
        self.test_p = test_p
        self.dev = dev
        self.save_path = save_path
        self.path_to_save_embeddings = path_to_save_embeddings
        self.path_to_save_reconstructed = path_to_save_reconstructed

    def build_data_loader(self, train_data, val_data, test_data, batch_size, name_model):
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, drop_last=True)

        return train_loader, val_loader, test_loader

    def cross_validate_VEGA(self, adata, n_top_genes, val_resolution, random_seed, column_labels_name, 
                           path_to_save_embeddings, path_to_save_reconstructed, save_path_results, path_save_fig):
        """
        Perform n-fold cross-validation on VEGA model
        
        Args:
            random_seed: seed for reproducibility of fold splitting
        """
        results = []
        
        labels = adata.obs[column_labels_name].values
        
        # Initialize StratifiedKFold cross-validator
        skfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=random_seed)
        
        # Get indices for splitting
        indices = np.arange(adata.n_obs)
        
        # Perform k-fold cross-validation
        for fold, (train_val_idx, test_idx) in enumerate(skfold.split(indices, labels)):
            print(f'Fold {fold + 1}/{self.n_folds}')
            
            # Split data into train+val and test sets
            adata_train_val = adata[train_val_idx].copy()
            adata_test = adata[test_idx].copy()
            
            # Get labels for train_val set for further stratification
            train_val_labels = adata_train_val.obs[column_labels_name].values
            
            # Further stratified split of train_val into train and validation
            train_idx_relative, val_idx_relative = train_test_split(
                np.arange(len(train_val_idx)),
                test_size=0.1,
                random_state=random_seed,
                stratify=train_val_labels
            )
            
            adata_train = adata_train_val[train_idx_relative].copy()
            adata_val = adata_train_val[val_idx_relative].copy()
            
            _, _, _, train_data, _, pathway_dict, pathway_mask = create_vega_training_data(
                self.name_model, False, n_top_genes, random_seed, None, 
                column_labels_name, adata_train, self.pathway_file
            )
            _, _, _, val_data, _, _, _ = create_vega_training_data(
                self.name_model, False, n_top_genes, random_seed, None, 
                column_labels_name, adata_val, self.pathway_file
            )
            _, _, _, test_data, _, _, _ = create_vega_training_data(
                self.name_model, False, n_top_genes, random_seed, None, 
                column_labels_name, adata_test, self.pathway_file
            )

            train_loader, val_loader, test_loader = self.build_data_loader(
                train_data, val_data, test_data, self.batch_size, self.name_model
            )
            
            dict_params = {
                'pathway_mask': pathway_mask, 
                'n_pathways': pathway_mask.shape[1], 
                'n_genes': pathway_mask.shape[0], 
                'device': self.dev, 
                'beta': self.beta, 
                'save_path': self.save_path,  
                'dropout': self.dropout, 
                'pos_dec': True
            }
            pathway_list = list(pathway_dict.keys()) + ['UNANNOTATED_' + str(k) for k in range(1)]  

            model = VEGA2(**dict_params).to(self.dev)
            hist, mse_hist, kld_hist = model.train_model(
                train_loader, self.lr, self.n_epochs, self.train_p, 
                self.test_p, val_loader, save_model=True
            )

            final_loss = hist["valid_loss"][-1]
            final_train_loss = hist["train_loss"][-1]
            final_mse_loss = mse_hist["valid_loss"][-1]
            final_mse_train_loss = mse_hist["train_loss"][-1]
            final_kld_loss = kld_hist["valid_loss"][-1]
            final_kld_train_loss = kld_hist["train_loss"][-1]
            n_epochs_trained = len(hist["train_loss"])

            # Get embeddings and reconstructions
            embedding_train = model.to_latent(torch.Tensor(
                adata_train.X.A if sparse.issparse(adata_train.X) else adata_train.X
            ).to(self.dev))
            embedding_train_array = embedding_train.cpu().detach().numpy()
            pd.DataFrame(embedding_train_array).to_csv(
                path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_train_fold_{fold}.txt'
            )
            X_reconstructed_train = model.decode(torch.Tensor(embedding_train).to(self.dev))
            X_reconstructed_train_array = X_reconstructed_train.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_train_array).to_csv(
                path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_train_fold_{fold}.txt'
            )

            embedding_val = model.to_latent(torch.Tensor(
                adata_val.X.A if sparse.issparse(adata_val.X) else adata_val.X
            ).to(self.dev))
            embedding_val_array = embedding_val.cpu().detach().numpy()
            pd.DataFrame(embedding_val_array).to_csv(
                path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_val_fold_{fold}.txt'
            )
            X_reconstructed_val = model.decode(torch.Tensor(embedding_val).to(self.dev))
            X_reconstructed_val_array = X_reconstructed_val.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_val_array).to_csv(
                path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_val_fold_{fold}.txt'
            )

            embedding_test = model.to_latent(torch.Tensor(
                adata_test.X.A if sparse.issparse(adata_test.X) else adata_test.X
            ).to(self.dev))
            embedding_test_array = embedding_test.cpu().detach().numpy()
            pd.DataFrame(embedding_test_array).to_csv(
                path_to_save_embeddings + f'/vega_{self.name_dataset}_embeddings_test_fold_{fold}.txt'
            )
            X_reconstructed_test = model.decode(torch.Tensor(embedding_test).to(self.dev))
            X_reconstructed_test_array = X_reconstructed_test.cpu().detach().numpy()
            pd.DataFrame(X_reconstructed_test_array).to_csv(
                path_to_save_reconstructed + f'/vega_{self.name_dataset}_reconstruction_test_fold_{fold}.txt'
            )

            # Compute metrics
            mse_score_train = compute_mse_score(adata_train, X_reconstructed_train_array)
            mse_score_val = compute_mse_score(adata_val, X_reconstructed_val_array)
            mse_score_test = compute_mse_score(adata_test, X_reconstructed_test_array)

            corr_train = compute_pearson_score(adata_train, X_reconstructed_train_array)
            corr_val = compute_pearson_score(adata_val, X_reconstructed_val_array)
            corr_test = compute_pearson_score(adata_test, X_reconstructed_test_array)

            adata_latent_train = build_adata_latent(embedding_train_array, adata_train, column_labels_name)
            adata_latent_val = build_adata_latent(embedding_val_array, adata_val, column_labels_name)
            adata_latent_test = build_adata_latent(embedding_test_array, adata_test, column_labels_name)

            clusters_train, true_labels_train = apply_clustering_algo(
                adata_latent_train, column_labels_name, 'Leiden', val_resolution
            )
            clusters_val, true_labels_val = apply_clustering_algo(
                adata_latent_val, column_labels_name, 'Leiden', val_resolution
            )
            clusters_test, true_labels_test = apply_clustering_algo(
                adata_latent_test, column_labels_name, 'Leiden', val_resolution
            )
            
            ari_train, nmi_train = apply_clustering_metrics(true_labels_train, clusters_train)
            ari_val, nmi_val = apply_clustering_metrics(true_labels_val, clusters_val)          
            ari_test, nmi_test = apply_clustering_metrics(true_labels_test, clusters_test) 
            
            if path_save_fig:
                plot_umap_orig_and_clusters(
                    embedding_train_array, true_labels_train, self.name_dataset, 
                    self.name_model, clusters_train, ari_train, nmi_train, 
                    'Train', fold, 'Leiden', path_save_fig
                )
                plot_umap_orig_and_clusters(
                    embedding_val_array, true_labels_val, self.name_dataset, 
                    self.name_model, clusters_val, ari_val, nmi_val, 
                    'Val', fold, 'Leiden', path_save_fig
                )
                plot_umap_orig_and_clusters(
                    embedding_test_array, true_labels_test, self.name_dataset, 
                    self.name_model, clusters_test, ari_test, nmi_test, 
                    'Test', fold, 'Leiden', path_save_fig
                )
                
            accuracy_train_rf, precision_train_rf, recall_train_rf, f1_train_rf, roc_auc_train_rf, accuracy_val_rf, precision_val_rf, recall_val_rf, f1_val_rf, roc_auc_val_rf, accuracy_test_rf, precision_test_rf, recall_test_rf, f1_test_rf, roc_auc_test_rf = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'random_forest', test_size=0.1)
            accuracy_train_xg, precision_train_xg, recall_train_xg, f1_train_xg, roc_auc_train_xg, accuracy_val_xg, precision_val_xg, recall_val_xg, f1_val_xg, roc_auc_val_xg, accuracy_test_xg, precision_test_xg, recall_test_xg, f1_test_xg, roc_auc_test_xg = apply_classification_algo2(adata_latent_train, adata_latent_val, adata_latent_test, column_labels_name, 'xgboost', test_size=0.1)

            results.append({
                'model_name': self.name_model,
                'dataset_name': self.name_dataset,
                'fold': fold + 1,
                'random_seed': random_seed,
                'beta': self.beta,
                'learning_rate': self.lr,
                'batch_size': self.batch_size,
                'dropout': self.dropout,
                'final_train_loss': final_train_loss,
                'final_mse_train_loss': final_mse_train_loss,
                'final_kld_train_loss': final_kld_train_loss,
                'final_valid_loss': final_loss,
                'final_mse_valid_loss': final_mse_loss,
                'final_kld_valid_loss': final_kld_loss,
                'n_epochs_trained': n_epochs_trained,
                'mse_score_train': mse_score_train,
                'mse_score_val': mse_score_val,
                'mse_score_test': mse_score_test,
                'corr_train': corr_train,
                'corr_val': corr_val,
                'corr_test': corr_test,
                'ari_train': ari_train,
                'nmi_train': nmi_train,
                'ari_val': ari_val,
                'nmi_val': nmi_val,
                'ari_test': ari_test,
                'nmi_test': nmi_test,
                'accuracy_train_rf': accuracy_train_rf,
                'precision_train_rf': precision_train_rf,
                'recall_train_rf': recall_train_rf,
                'f1_train_rf': f1_train_rf,
                'roc_auc_train_rf': roc_auc_train_rf,
                'accuracy_val_rf': accuracy_val_rf,
                'precision_val_rf': precision_val_rf,
                'recall_val_rf': recall_val_rf,
                'f1_val_rf': f1_val_rf,
                'roc_auc_val_rf': roc_auc_val_rf,
                'accuracy_test_rf': accuracy_test_rf,
                'precision_test_rf': precision_test_rf,
                'recall_test_rf': recall_test_rf,
                'f1_test_rf': f1_test_rf,
                'roc_auc_test_rf': roc_auc_test_rf,
                'accuracy_train_xg': accuracy_train_xg,
                'precision_train_xg': precision_train_xg,
                'recall_train_xg': recall_train_xg,
                'f1_train_xg': f1_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'roc_auc_train_xg': roc_auc_train_xg,
                'accuracy_val_xg': accuracy_val_xg,
                'precision_val_xg': precision_val_xg,
                'recall_val_xg': recall_val_xg,
                'f1_val_xg': f1_val_xg,
                'roc_auc_val_xg': roc_auc_val_xg,
                'accuracy_test_xg': accuracy_test_xg,
                'precision_test_xg': precision_test_xg,
                'recall_test_xg': recall_test_xg,
                'f1_test_xg': f1_test_xg,
                'roc_auc_test_xg': roc_auc_test_xg
            })

        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Add summary statistics
        print("\n=== Cross-Validation Summary ===")
        metrics_to_summarize = ['ari_test', 'nmi_test', 'mse_score_test', 'corr_test', 
                               'accuracy_test_rf', 'f1_test_rf', 'accuracy_test_xg', 'f1_test_xg']
        for metric in metrics_to_summarize:
            mean_val = results_df[metric].mean()
            std_val = results_df[metric].std()
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        results_df.to_csv(
            save_path_results + f'{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv', 
            index=False
        )
        print(f"\nResults saved to {save_path_results}{self.name_model}_{self.name_dataset}_{self.n_folds}fold_cv_results.csv")
        
        return results_df

        
            
            