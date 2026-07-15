import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import scanpy as sc
import seaborn as sns
import matplotlib.ticker as ticker
import ast
from scipy.stats import pearsonr
from sklearn.metrics import pairwise_distances
import sys

import utils_compute_scores2
from utils_compute_scores2 import *

def load_embedding_original(path_to_save_embeddings_original, type_file, name_model, name_dataset, split, list_pathways):
    if type_file == 'txt':
        path_embeddings_original = path_to_save_embeddings_original + f'{name_model}_{name_dataset}_embeddings_{split}_original.txt'
        embeddings_original = load_embeddings(path_embeddings_original, list_pathways)
    elif type_file == 'layers':
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_layers_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
    elif type_file == 'all_dim':
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_embeddings_all_dim_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
    else:
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
        
    return embeddings_original
        
        

def compute_distance_corr_one_pathway_one_dim(pathway_selected, adata, embeddings_original, pathway_dict, path_save_heatmap):
   
    list_genes_pathway = pathway_dict[pathway_selected]
    if len(list_genes_pathway) == 0:
        return None
    
    df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    df_pathway = df[[gene for gene in list_genes_pathway if gene in adata.var_names]]
    if df_pathway.empty:
        return None
    
    dist_matrix_pathway = pairwise_distances(df_pathway.values, metric='euclidean')
    dist_matrix_neuron = pairwise_distances(embeddings_original[pathway_selected].values.reshape(-1, 1) , metric='euclidean')
    
    triu_idx = np.triu_indices_from(dist_matrix_pathway, k=1)  # k=1 excludes diagonal
    vec1 = dist_matrix_pathway[triu_idx]
    vec2 = dist_matrix_neuron[triu_idx]

    # Compute Pearson correlation
    corr, p_value = pearsonr(vec1, vec2)
    print(f"Pearson correlation: {corr:.4f}, p-value: {p_value:.4e}")
    
    if path_save_heatmap:
        plot_dist_heatmap_input(dist_matrix_pathway, pathway_selected, list_genes_pathway, path_save_heatmap)
        plot_dist_heatmap_neuron(dist_matrix_neuron, pathway_selected, path_save_heatmap)
    
    return corr

def compute_distance_corr_one_pathway_multiple_dim(pathway_selected, adata, embeddings_original, pathway_dict, path_save_heatmap):
   
    list_genes_pathway = pathway_dict[pathway_selected]
    if len(list_genes_pathway) == 0:
        return None
    
    df = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
    df_pathway = df[[gene for gene in list_genes_pathway if gene in adata.var_names]]
    if df_pathway.empty:
        return None
    
    #df_pathway_neuron = embeddings_original.filter(regex=f"^{pathway_selected}-")
    df_pathway_neuron = embeddings_original.filter(
        regex=f"^{re.escape(pathway_selected)}-"
    )

    dist_matrix_pathway = pairwise_distances(df_pathway.values, metric='euclidean')
    dist_matrix_neuron = pairwise_distances(df_pathway_neuron.values , metric='euclidean')
    
    triu_idx = np.triu_indices_from(dist_matrix_pathway, k=1)  # k=1 excludes diagonal
    vec1 = dist_matrix_pathway[triu_idx]
    vec2 = dist_matrix_neuron[triu_idx]

    # Compute Pearson correlation
    corr, p_value = pearsonr(vec1, vec2)
    print(f"Pearson correlation: {corr:.4f}, p-value: {p_value:.4e}")
    
    if path_save_heatmap:
        plot_dist_heatmap_input(dist_matrix_pathway, pathway_selected, list_genes_pathway, path_save_heatmap)
        plot_dist_heatmap_neuron(dist_matrix_neuron, pathway_selected, path_save_heatmap)
    
    return corr


def plot_dist_heatmap_input(dist_matrix_pathway, pathway_selected, list_genes_pathway, path_save_heatmap):
    
    plt.figure(figsize=(12, 10))  # make it larger for clarity
    ax = sns.heatmap(dist_matrix_pathway, cmap='viridis')  # or 'magma', 'coolwarm', etc.

    cbar = ax.collections[0].colorbar
    cbar.set_label("Euclidean Distance", fontsize=14)

    plt.title(f"{pathway_selected}", fontsize=16)
    plt.xlabel(f"Cells In Input Data ({len(list_genes_pathway)} genes)")
    plt.ylabel(f"Cells In Input Data ({len(list_genes_pathway)} genes)")
    plt.tight_layout()
    if path_save_heatmap:
        plt.savefig(path_save_heatmap + f"heatmap_input_{pathway_selected}.png", dpi=300, bbox_inches="tight")
    plt.show()
    

def plot_dist_heatmap_neuron(dist_matrix_neuron, pathway_selected, path_save_heatmap):
    plt.figure(figsize=(12, 10))  # make it larger for clarity
    ax = sns.heatmap(dist_matrix_neuron, cmap='viridis')  # or 'magma', 'coolwarm', etc.

    cbar = ax.collections[0].colorbar
    cbar.set_label("Euclidean Distance", fontsize=14)

    plt.title(f"{pathway_selected}", fontsize=16)
    plt.xlabel(f"Cells In Neuron Space (1 Dim)")
    plt.ylabel(f"Cells In Neuron Space (1 Dim)")
    plt.tight_layout()
    if path_save_heatmap:
        plt.savefig(path_save_heatmap + f"heatmap_latent_{pathway_selected}.png", dpi=300, bbox_inches="tight")
    plt.show()
    
def compute_corr_one_pathways(pathway_selected, adata, path_to_save_embeddings_original, type_file, name_model, name_dataset, split, list_pathways, pathway_dict, path_save_heatmap):
    embeddings_original = load_embedding_original(path_to_save_embeddings_original, type_file, name_model, name_dataset, split, list_pathways)
    if name_model == 'pmVAE' or name_model == 'OntoVAE':
        corr = compute_distance_corr_one_pathway_multiple_dim(pathway_selected, adata, embeddings_original, pathway_dict, path_save_heatmap)
    else:
        corr = compute_distance_corr_one_pathway_one_dim(pathway_selected, adata, embeddings_original, pathway_dict, path_save_heatmap)
    df = pd.DataFrame({
    'Model': [name_model],
    'Dataset': [name_dataset],
    'Split': [split],
    'Pathway Name': [pathway_selected],
    'Corr Distance Score': [corr]
    })
    return df