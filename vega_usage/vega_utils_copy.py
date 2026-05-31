import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import sparse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch
import scanpy as sc
from anndata import AnnData

#sys.path.append('/home/user/Review-Interpretable-VAEs/Review-Interpretable-VAEs/cloned_github_models/vega/vega-reproducibility/src')
sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vega
import utils
from utils import *
from learning_utils import *
from vega_model import VEGA
import torch
import itertools

def load_pathways_vega(load, data_path, pathway_file):
    if load == True:
        adata = sc.read(data_path)
    else:
        adata = data_path
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = list(pathway_dict.keys()) + ['UNANNOTATED_'+str(k) for k in range(1)]
    return list_pathways

def create_path_embeddings(name_model, name_dataset, split, random_seed, perturbation, source, pathway_selected, path_saved_embeddings):
    if source == 'original':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_original_seed_{random_seed}_trial.txt'
    if source == 'perturbated':
        path_embeddings = path_saved_embeddings + f'{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected}_{perturbation}_seed_{random_seed}.txt'
    return path_embeddings


def load_embeddings(path_embeddings_pathway, list_pathways):
    embeddings_pathway = np.loadtxt(path_embeddings_pathway)
    if list_pathways:
        embeddings_pathway= pd.DataFrame(embeddings_pathway, columns=list_pathways)
    return embeddings_pathway


def build_overlap_matrix_Vega(pathway_mask, adata, list_pathways):
    df = pd.DataFrame(pathway_mask, columns=list_pathways, index=adata.var_names)
    pathway_dict_dataset = {pathway_name: df.index[df[pathway_name] == 1].tolist() for pathway_name in df.columns}
    all_results = []  # accumulate results across all pathways

    for pathway_selected in list_pathways:
        # On cherche les gènes (index) là où la colonne du pathway sélectionné vaut 1
        list_genes_to_perturbate = df.index[df[pathway_selected] == 1].tolist()
        genes1 = [gene for gene in list_genes_to_perturbate if gene in adata.var_names]

        for pathway_compared, genes in pathway_dict_dataset.items():
            genes2 = [gene for gene in genes if gene in adata.var_names]
            intersection = set(genes1).intersection(set(genes2))

            # if intersection:  # only keep if non-empty
            all_results.append({
                "Pathway Selected": pathway_selected,   # which pathway we started with
                "Nb Genes Pathway Selected": len(genes1),
                "Compared Pathway": pathway_compared,   # the pathway we are comparing against
                "Nb Genes Compared Pathway": len(genes2),
                "Genes Overlap": len(intersection),
                "Commun Genes": list(intersection),
                "Overlap Proportion": len(intersection)/len(genes1) if len(genes1) > 0 else 0
            })

    # Convert to a single dataframe
    overlap_matrix = pd.DataFrame(all_results)
    overlap_matrix  = overlap_matrix .sort_values(
        by=["Pathway Selected", "Genes Overlap"],
        ascending=[True, False]
    ).reset_index(drop=True)
    
    return overlap_matrix

def build_df_genespathways_vega(pathway_dict, adata, list_pathways):
    rows = []
    for pathway_selected in list_pathways[:-1]:
        pathway_position = [list_pathways.index(word) for word in list_pathways if word == pathway_selected][0]
        list_genes_to_perturbate = pathway_dict[pathway_selected]
        genes_in_adata = [gene for gene in list_genes_to_perturbate if gene in adata.var_names]
        #print(genes_in_adata)
        other_genes = list(set(adata.var_names).symmetric_difference(genes_in_adata))

        rows.append(
            pd.DataFrame({
            'pathway': pathway_selected,
            'pathway_position': pathway_position,
            'list_genes_in_adata': [genes_in_adata],
            'number of genes': len(genes_in_adata)
            })
        )
    df_genespathways = pd.concat(rows, axis=0, ignore_index=True).sort_values(by='number of genes', ascending=True)
    
    return df_genespathways


def access_data_vega(data_path, adata, pathway_file):
    if data_path is not None:
        adata = sc.read(data_path)
    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)
    list_pathways = load_pathways_vega(False, adata, pathway_file)
    
    overlap_matrix = build_overlap_matrix_Vega(pathway_mask, adata, list_pathways)
    print("Overlap matrix available")
    
    df_genespathways = build_df_genespathways_vega(pathway_dict, adata, list_pathways)
    
    return adata, pathway_dict, pathway_mask, list_pathways, df_genespathways, overlap_matrix


def extract_x_y_from_adata(adata, column_labels_name: pd.Series):
    X = pd.DataFrame(adata.X, index=adata.obs.index)
    y = adata.obs[column_labels_name]
    return X, y

def split_data(X, y, train_size, random_seed):
    X_train,  X_test, labels_train,  labels_test = train_test_split(
        X, y, train_size=train_size, random_state=random_seed, stratify=y)
    return X_train,  X_test, labels_train,  labels_test

def extract_index(X):
    index_df = X.index
    return index_df
    
def build_adata_from_X(adata, index_df):
    adata = adata[adata.obs.index.isin(index_df)]
    return adata, index_df

def encode_y(y):
    le = preprocessing.LabelEncoder().fit(y)
    y_encoded = torch.Tensor(le.transform(y))
    return y_encoded

def build_vega_dataset(adata, y_encoded, pathway_file):
    if sparse.issparse(adata.X):
        data = adata.X.A
    else:
        data = adata.X

    data = torch.Tensor(data)
    data = UnsupervisedDataset(data, targets=y_encoded)

    pathway_dict = read_gmt(pathway_file, min_g=0, max_g=1000)
    pathway_mask = create_pathway_mask(adata.var.index.tolist(), pathway_dict, add_missing=1, fully_connected=True)

    return data, pathway_dict, pathway_mask

def preprocess_adata(adata, n_top_genes=5000):
    """ Simple (default) sc preprocessing function before autoencoders """
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    #sc.pp.normalize_total(adata, target_sum=1e4)
    #sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    adata.raw = adata
    adata = adata[:, adata.var.highly_variable]
    return adata


def create_vega_training_data(name_model, preprocess, select_hvg, n_top_genes, random_seed, train_size, column_labels_name, adata, pathway_file):
    if preprocess == True:
        print('Preprocessing adata')
        adata = preprocess_adata(adata, n_top_genes=n_top_genes)
    else:
        adata = adata.copy()
    X, y = extract_x_y_from_adata(adata, column_labels_name)
    X_train,  X_test, labels_train,  labels_test = split_data(X, y, train_size, random_seed)
    y_train = encode_y(labels_train)
    y_test = encode_y(labels_test)
    index_train = extract_index(X_train)
    index_test = extract_index(X_test)
    adata_train, index_train = build_adata_from_X(adata, index_train)
    adata_test, index_test = build_adata_from_X(adata, index_test)

    train_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_train, y_train, pathway_file)
    test_ds, pathway_dict, pathway_mask = build_vega_dataset(adata_test, y_test, pathway_file)
    return adata, adata_train, adata_test, train_ds, test_ds, pathway_dict, pathway_mask
