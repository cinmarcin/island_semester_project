from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm

def create_index_rdm(n_trials, repetitions, relative_distance=False, save_fig=False):
    """
    Create an index RDM where dissimilarity increases with the distance between trial indices.
    
    Parameters
    ----------
    n_trials : int
        Number of unique trials.
    repetitions : int
        Number of repetitions per trial.

    save_folder : str, optional
        Folder to save the RDM plot. If None, the plot is not saved.
    Returns
    -------
    np.ndarray
        An (n_trials * repetitions) x (n_trials * repetitions) RDM matrix.
    """
    total_trials = n_trials * repetitions
    rdm = np.zeros((total_trials, total_trials))
    
    if relative_distance:
        dist = 64
    else:
        dist = 0
    for i in range(total_trials):
        for j in range(total_trials):
            rdm[i, j] = min(abs((i % (n_trials)) - (j % (n_trials))), abs((i % (n_trials) + dist) - (j % (n_trials))), abs((i % (n_trials) - dist) - (j % (n_trials))))

    # Inverse the RDM to represent dissimilarity
    rdm = rdm.max() - rdm
    # plot
    if save_fig is not None:
        plt.imshow(rdm, cmap='viridis')
        plt.colorbar(label='Correlation')
        plt.title(f'Index RDM for {n_trials} trials with {repetitions} repetitions each')
        plt.xlabel('Trial Index')
        plt.ylabel('Trial Index')
        rdm_plot_path = os.path.join('figures/', f"index_{'relative' if relative_distance else 'absolute'}_rdm.png")
        plt.savefig(rdm_plot_path)
        plt.close()
        print(f"Saved index RDM plot to {rdm_plot_path}")
        
    return rdm

def correlation_accross_repetitions(correlation_matrix, trial_per_repetition=64, repetitions=6):
    avg_correlation_across_repetition = []
    for trial in range(trial_per_repetition):
        trial_indices = [trial + r * trial_per_repetition for r in range(repetitions)]
        cross_rep_correlations = []
        for i in range(len(trial_indices)):
            for j in range(i + 1, len(trial_indices)):
                cross_rep_correlations.append(correlation_matrix[trial_indices[i], trial_indices[j]])
        avg_cross_rep_correlation = np.mean(cross_rep_correlations)
        avg_correlation_across_repetition.append(avg_cross_rep_correlation)
        print(f"Average correlation for trial {trial + 1} across repetitions: {avg_cross_rep_correlation:.4f}")
    overall_avg_cross_rep_correlation = np.mean(avg_correlation_across_repetition)
    print(f"Overall average correlation across all repetitions: {overall_avg_cross_rep_correlation:.4f}")
    return overall_avg_cross_rep_correlation

def correlation_within_repetitions(correlation_matrix, trial_per_repetition=64, repetitions=6):
    avg_correlation_within_repetition = []
    for rep in range(repetitions):
        start_idx = rep * trial_per_repetition
        end_idx = (rep + 1) * trial_per_repetition
        avg_correlation = correlation_matrix[start_idx:end_idx, start_idx:end_idx].mean()
        avg_correlation_within_repetition.append(avg_correlation)
        print(f"Average correlation within repetition {rep + 1}: {avg_correlation:.4f}")
    overall_avg_correlation = np.mean(avg_correlation_within_repetition)
    print(f"Overall average correlation within all repetitions: {overall_avg_correlation:.4f}")
    return overall_avg_correlation

def extract_beta_maps(fmri_glm, idxs=None, run_label=None):
    """
    Extract beta maps for specified trial indices from a fitted GLM model.
    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM model from nilearn.
    idxs : list of int, optional
        List of trial indices to extract. If None, all trials are extracted.
    run_label : str, optional
        Label for the run (e.g., 'pre' or 'post') to filter trials. If None, no filtering is applied.
    Returns
    -------
    np.ndarray
        Array of shape (n_trials, n_voxels) containing the extracted beta maps.
    """
    print("Extracting beta maps...")
    exclude = ['derivative', 'dispersion']
    # Get column names
    columns = fmri_glm.design_matrices_[0].columns
    # Filter columns for trials
    if run_label is not None and idxs is not None:
        trial_columns = [
            i
            for i, col in enumerate(columns)
            if any(f"{run_label}_trial_{idx+1}" == col for idx in idxs)
            and all(e not in col for e in exclude)
        ]
    else:
        trial_columns = [
        i for i, col in enumerate(columns)
        if 'trial_' in col and all(e not in col for e in exclude)  # all trials
        ]
    beta_maps_list = []
    for i in tqdm(trial_columns):
        contrast_vector = np.zeros(len(columns))
        contrast_vector[i] = 1
        beta_map = fmri_glm.compute_contrast(contrast_vector, output_type='effect_size')
        beta_maps_list.append(beta_map)

    beta_maps = np.stack([bm.get_fdata() for bm in beta_maps_list], axis=-1)
    print(f"Extracted {beta_maps.shape[-1]} beta maps with shape {beta_maps.shape[:-1]} flattened for similarity computation.")

    # flatten beta maps for similarity computation
    n_voxels = np.prod(beta_maps.shape[:-1])
    n_trials = beta_maps.shape[-1]
    trial_data = beta_maps.reshape((n_voxels, n_trials)).T  #
    return trial_data

def get_model_rdm(sub: str, path_to_raw_data, spatial=True, min_confidence=0):
    """
    Compute model matrices for a given subject on its judgment distances.
    Parameters
    ----------
    sub : str
        Subject identifier.
    path_to_raw_data : str
        Path to the raw data directory.
    spatial : bool
        Whether to compute spatial or temporal RDMs.
    min_confidence : float
        Minimum confidence level for including trials. (on a scale 0-100)
    Returns
    -------
    np.ndarray
        The computed RDM matrix.
    selected_trials : np.ndarray
        The trials that were included in the RDM computation.
    """
    from scipy.spatial.distance import pdist, squareform
    df = pd.read_csv(path_to_raw_data)
    df = df[df['Participant'] == sub]

    if spatial:
        df = df[df['Placed Confidence'] >= min_confidence]
        selected_trials = df['V4'].unique()
        df = df[['X', 'Y', 'Z', 'Placed Confidence']]
        judged_positions = df[['X', 'Y', 'Z']].to_numpy()

    else:
        df = df[df['Timed (days)'] != 100]
        df = df[df['Timed Confidence'] >= min_confidence]

        if len(df) < 2:
            raise ValueError(f'Participant {sub} has less than 2 valid temporal judgments & should probably be discarded.')
        
        selected_trials = df['V4'].unique()
        df = df[['Timed (days)', 'Timed Confidence']]
        judged_positions = df[['Timed (days)']].to_numpy()

    dist_matrix = squareform(pdist(judged_positions))

    return dist_matrix, selected_trials

def compute_rsa(beta_maps, beta_maps_idxs):
    """
    Compute Representational Similarity Analysis (RSA) by spearman correlating beta maps. 
    Only correlate the beta map of a trial with every other trial in every other repetition.
    Parameters
    ----------
    beta_maps : np.ndarray
        Array of shape (n_repetitions * n_trials, n_voxels) containing the beta maps.
    beta_maps_idxs : np.ndarray
        2D array of shape (n_repetitions, n_trials) containing the indices of beta maps.
    Returns 
    -------
    np.ndarray
        The computed correlation matrix in the shape (n_trials, n_trials).
    """
    from scipy.stats import spearmanr
    n_trials = beta_maps_idxs.shape[1]
    correlation_matrix = spearmanr(beta_maps, axis=1).correlation
    return correlation_matrix