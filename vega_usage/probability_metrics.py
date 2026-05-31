import pandas as pd
import numpy as np
import argparse
import os
import sys
import gc
import re
from threading import Lock


#compute reduction scores

def compute_scores(pathway_selected, split, embeddings_original, embeddings_perturbated, list_pathways, name_model, path_to_save):
    #print(pathway_selected)
    rows = []
    for pathway in list_pathways:
        rows.append(pd.DataFrame({
        'model': [name_model] * len(embeddings_original), 
        'cell_id': [f'cell_{i}' for i in range(len(embeddings_original))],
        'perturbation': [0] * len(embeddings_original),
        'pathway perturbated': [pathway_selected] * len(embeddings_original),
        'pathway observed': [pathway] * len(embeddings_original), 
        'original neuron activation': embeddings_original[pathway],
        'perturbated neuron activation': embeddings_perturbated[pathway],
        'reduction_score': abs(embeddings_original[pathway] - embeddings_perturbated[pathway]),
        #'reduction_score': abs(embeddings_original[pathway]) - abs(embeddings_perturbated[pathway]),
        'reduction_rate': ((abs(embeddings_original[pathway] - embeddings_perturbated[pathway]))/abs(embeddings_original[pathway])),
        'reduction_z-score': ((abs(embeddings_original[pathway] - embeddings_perturbated[pathway]))/embeddings_original[pathway].std()),
        
        }))
                    
    df = pd.concat(rows, axis=0, ignore_index=True)
    
    if path_to_save:
        os.makedirs(path_to_save, exist_ok=True)   
        file_path = os.path.join(path_to_save, f'{name_model}_reduction_scores_{pathway_selected}_{split}_perturbation.parquet')
        df.to_parquet(file_path, engine='pyarrow')
    return df


def compute_one_df_scores_perturbation(list_pathways, name_dataset, split, pathway_selected, perturbation, name_model, type_file, path_to_save_embeddings_original, path_to_save_embeddings_perturbated, path_to_save_reduction_scores):
    if type_file == 'txt':
        path_embeddings_original = path_to_save_embeddings_original + f'{name_model}_{name_dataset}_embeddings_{split}_original.txt'
        path_embeddings_perturbated = path_to_save_embeddings_perturbated + f'/{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected}_{perturbation}.txt'
        embeddings_original = load_embeddings(path_embeddings_original, list_pathways)
        embeddings_perturbated = load_embeddings(path_embeddings_perturbated, list_pathways)
    elif type_file == 'layers':
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_layers_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
        path_embeddings_perturbated = os.path.join(path_to_save_embeddings_perturbated, f'{name_model}_{name_dataset}_layers_embeddings_{split}_{pathway_selected}_{perturbation}.parquet')
        embeddings_perturbated = pd.read_parquet(path_embeddings_perturbated, engine="pyarrow")
    else:
        path_embeddings_original = os.path.join(path_to_save_embeddings_original, f'{name_model}_{name_dataset}_embeddings_{split}_original.parquet')
        embeddings_original = pd.read_parquet(path_embeddings_original, engine="pyarrow")
        path_embeddings_perturbated = os.path.join(path_to_save_embeddings_perturbated, f'{name_model}_{name_dataset}_embeddings_{split}_{pathway_selected}_{perturbation}.parquet')
        embeddings_perturbated = pd.read_parquet(path_embeddings_perturbated, engine="pyarrow")

    df = compute_scores(pathway_selected, split, embeddings_original, embeddings_perturbated, list_pathways, name_model, path_to_save_reduction_scores)
    return df


#compute probas scores

def proba_impact_pathway_perturbation(df_scores, overlap_matrix, pathway_perturbated, pathway_compared, name_reduction_metric, name_activation_column, name_overlap_compared_pathway, name_overlap_column, activ_threshold, overlap_threshold):
    if df_scores.empty or overlap_matrix.empty:
        return None
    overlap_score = overlap_matrix[(overlap_matrix['Pathway Selected'] == pathway_perturbated) & (overlap_matrix[name_overlap_compared_pathway] == pathway_compared)][name_overlap_column].values[0]
    if overlap_score >= overlap_threshold:
        #print(overlap_score)
        return None
    
    df_selected = df_scores[
        (df_scores['pathway perturbated'] == pathway_perturbated) & 
        (df_scores['pathway observed'] == pathway_compared)
    ]
    
    
    df_selected = df_selected[abs(df_selected[name_activation_column]) > activ_threshold]
    
    if df_selected.empty:
        return None 
    
    observed_filter = (df_scores['pathway observed'] == pathway_perturbated) & (df_scores['pathway perturbated'] == pathway_perturbated) & (df_scores['cell_id'].isin(df_selected['cell_id']))
    
    if observed_filter.sum() == 0:
        return None
    
    observed_values  = df_scores[observed_filter][name_reduction_metric].values
    #print(len(observed_values))
    
    count_pathway_p_over_pathway_q = (observed_values > df_selected[name_reduction_metric].values).sum()
    count_threshold_requirements = len(df_selected)
    
    del df_selected
    gc.collect()
    
    proba = count_pathway_p_over_pathway_q / count_threshold_requirements
    
    return proba


def proba_impact_pathway_perturbation2(df_scores, overlap_matrix, pathway_perturbated, pathway_compared, name_reduction_metric, name_activation_column, name_overlap_compared_pathway, name_overlap_column, overlap_threshold):
    if df_scores.empty or overlap_matrix.empty:
        return None
    overlap_score = overlap_matrix[(overlap_matrix['Pathway Selected'] == pathway_compared) & (overlap_matrix[name_overlap_compared_pathway] == pathway_perturbated)][name_overlap_column].values[0]
    if overlap_score >= overlap_threshold:
        #print(overlap_score)
        return None
    
    df_selected = df_scores[
        (df_scores['pathway perturbated'] == pathway_perturbated) & 
        (df_scores['pathway observed'] == pathway_compared)
    ]
    
    
    if df_selected.empty:
        return None 
    
    observed_filter = (df_scores['pathway observed'] == pathway_perturbated) & (df_scores['pathway perturbated'] == pathway_perturbated)
    
    if observed_filter.sum() == 0:
        return None
    
    observed_values  = df_scores[observed_filter][name_reduction_metric].values
    #print(len(observed_values))
    
    count_pathway_p_over_pathway_q = (observed_values > df_selected[name_reduction_metric].values).sum()
    count_threshold_requirements = len(df_selected)
    
    del df_selected
    gc.collect()
    
    proba = count_pathway_p_over_pathway_q / count_threshold_requirements
    
    return proba




def overall_proba_one_pathway_perturbated(
    pathway_perturbated,
    list_pathways,
    df_scores,
    overlap_matrix,
    name_reduction_metric,
    name_activation_column,
    name_overlap_compared_pathway,
    name_overlap_column,
    #activ_threshold,
    overlap_threshold
):
    """
    Compute probabilities for all comparisons of one pathway perturbated.
    """
    records = []
    
    for pathway_compared in list_pathways:
        proba = proba_impact_pathway_perturbation2(
            df_scores=df_scores,
            overlap_matrix=overlap_matrix,
            pathway_perturbated=pathway_perturbated,
            pathway_compared=pathway_compared,
            name_reduction_metric=name_reduction_metric,
            name_activation_column=name_activation_column,
            name_overlap_compared_pathway=name_overlap_compared_pathway,
            name_overlap_column=name_overlap_column,
            #activ_threshold=activ_threshold,
            overlap_threshold=overlap_threshold
        )

        records.append({
                'Pathway Perturbated': pathway_perturbated,
                'Pathway Compared': pathway_compared,
                f'{name_reduction_metric} Probability': proba
            })
    df_results = pd.DataFrame(records)
    overall_proba = (df_results[f"{name_reduction_metric} Probability"].sum())/ (df_results[f"{name_reduction_metric} Probability"].notna().sum())
    
    return  df_results, overall_proba


def compute_metrics_for_pathway(
    name_model,
    name_dataset,
    pathway_perturbated,
    list_pathways,
    path_to_save_reduction_scores,
    split,
    overlap_matrix,
    name_activation_column,
    name_overlap_compared_pathway,
    name_overlap_column,
    #activ_threshold,
    overlap_threshold,
    path_to_save_results
):
    """
    Compute all three reduction metrics for one pathway_perturbated.
    """
    if name_model == 'OntoVAE':
        pathway_perturbated_2 = safe_filename(pathway_perturbated)
        file_path = os.path.join(path_to_save_reduction_scores, f'{name_model}_reduction_scores_{pathway_perturbated_2}_{split}_perturbation.parquet')
        df_scores = pd.read_parquet(file_path, engine="pyarrow")
    else: 
        file_path = os.path.join(path_to_save_reduction_scores, f'{name_model}_reduction_scores_{pathway_perturbated}_{split}_perturbation.parquet')
        df_scores = pd.read_parquet(file_path, engine="pyarrow")
    
    results = {}
    final_df_results = pd.DataFrame()

    # Compute the three metrics
    for name_reduction_metric in ['reduction_score', 'reduction_rate', 'reduction_z-score']:
        df_results, overall_proba = overall_proba_one_pathway_perturbated(
            pathway_perturbated=pathway_perturbated,
            list_pathways=list_pathways,
            df_scores=df_scores,
            overlap_matrix=overlap_matrix,
            name_reduction_metric=name_reduction_metric,
            name_activation_column=name_activation_column,
            name_overlap_compared_pathway=name_overlap_compared_pathway,
            name_overlap_column=name_overlap_column,
            #activ_threshold=activ_threshold,
            overlap_threshold=overlap_threshold
        )
        
        results['pathway_perturbated'] = pathway_perturbated
        #results['activation_threshold'] = activ_threshold
        results['overlap_threshold'] = overlap_threshold
        
        
        # Save metric results
        new_col_name = ''
        if name_reduction_metric == 'reduction_score':
            new_col_name = 'Reduction Score Probability'
            results[new_col_name] = overall_proba
        elif name_reduction_metric == 'reduction_rate':
            new_col_name = 'Reduction Rate Probability'
            results[new_col_name] = overall_proba
        elif name_reduction_metric == 'reduction_z-score':
            new_col_name = 'Reduction Z-Score Probability'
            results[new_col_name] = overall_proba
        

        df_results = df_results.rename(
            columns={f"{name_reduction_metric} Probability": new_col_name}
        )
        
        if final_df_results.empty:
            final_df_results = df_results
        else:
            final_df_results = final_df_results.merge(df_results, on=['Pathway Perturbated', 'Pathway Compared'])

        del df_results
        gc.collect()
        
    if path_to_save_results:
        os.makedirs(path_to_save_results, exist_ok=True)   
        if name_model == 'OntoVAE':
            file_path = os.path.join(path_to_save_results, f'{name_model}_{name_dataset}_overall_proba_{pathway_perturbated_2}_{split}_perturbation.parquet')
            final_df_results.to_parquet(file_path, engine='pyarrow')
        else:
            file_path = os.path.join(path_to_save_results, f'{name_model}_{name_dataset}_overall_proba_{pathway_perturbated}_{split}_perturbation.parquet')
            final_df_results.to_parquet(file_path, engine='pyarrow')

    return pd.DataFrame([results]), final_df_results


