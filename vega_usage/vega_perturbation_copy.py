import sys
import vega
import os
from pathlib import Path
import random

sys.path.append('/home/BS94_SUR/phD/review/models reproductibility/VEGA/vega-reproducibility/src')
import vanilla_vae
import train_vanilla_vae_suppFig1
import utils
from utils import *
from learning_utils import *
from vanilla_vae import VanillaVAE
from vega_model import VEGA


def get_genes_in_pathway(adata, pathway_selected, list_pathways, pathway_dict):
    pathway_position = [list_pathways.index(word) for word in list_pathways if word == pathway_selected][0]
    list_genes_to_perturbate = pathway_dict[pathway_selected]
    genes_in_adata = [gene for gene in list_genes_to_perturbate if gene in adata.var_names]

    return genes_in_adata


def vega_perturbation(model, name_dataset, adata, pathway_selected, list_pathways, pathway_dict, perturbation, mode, split, path_to_save_embeddings, path_to_save_reconstructed):
    #select list of genes to perturb involved in pathway_name
    #print(pathway_selected)
    genes_in_adata = get_genes_in_pathway(adata, pathway_selected, list_pathways, pathway_dict)
    adata_perturbated = adata.copy()

    if genes_in_adata != []:
        #select type of perturbation to apply
        if perturbation == 'inhibition' and mode == 'one_vs_all':
            adata_perturbated[:, genes_in_adata].X = 0

    embeddings = model.to_latent(torch.Tensor(adata_perturbated.X.toarray()))
    X_reconstructed = model.decode(torch.Tensor(embeddings))
    
    n = f'{split}_{pathway_selected}_{perturbation}'
    
    if path_to_save_embeddings: 
        np.savetxt(Path(path_to_save_embeddings + f'/vega_{name_dataset}_embeddings_{n}.txt'), embeddings.cpu().detach().numpy())
    if path_to_save_reconstructed:
        np.savetxt(Path(path_to_save_reconstructed + f'/vega_{name_dataset}_reconstruction_{n}.txt'), X_reconstructed.cpu().detach().numpy())
    
    return embeddings, X_reconstructed
        
        
        
def vega_perturbation_random(model, name_dataset, adata, pathway_selected, list_pathways, pathway_dict, perturbation, mode, split, path_to_save_embeddings, path_to_save_reconstructed):
    #select list of genes to perturb involved in pathway_name
    #print(pathway_selected)
    genes_in_adata = get_genes_in_pathway(adata, pathway_selected, list_pathways, pathway_dict)
    adata_perturbated = adata.copy()
    
    list_genes_available = list(set(adata.var_names) - set(genes_in_adata))
    
    n_genes_to_select = len(genes_in_adata)
    
    if len(list_genes_available) < n_genes_to_select:
        print(f"Warning: Not enough genes available ({len(available_genes)} < {n_genes_to_select})")
        n_genes_to_select = len(list_genes_available)
        
    random_genes = random.sample(list_genes_available , n_genes_to_select)

    if random_genes != []:
        #select type of perturbation to apply
        if perturbation == 'random' and mode == 'one_vs_all':
            adata_perturbated[:, random_genes].X = 0

    embeddings = model.to_latent(torch.Tensor(adata_perturbated.X.toarray()))
    X_reconstructed = model.decode(torch.Tensor(embeddings))
    
    n = f'{split}_{pathway_selected}_{perturbation}'
    
    if path_to_save_embeddings: 
        np.savetxt(Path(path_to_save_embeddings + f'/vega_{name_dataset}_embeddings_{n}.txt'), embeddings.cpu().detach().numpy())
    if path_to_save_reconstructed:
        np.savetxt(Path(path_to_save_reconstructed + f'/vega_{name_dataset}_reconstruction_{n}.txt'), X_reconstructed.cpu().detach().numpy())
    
    return embeddings, X_reconstructed