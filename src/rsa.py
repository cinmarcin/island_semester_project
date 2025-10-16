import numpy as np
import matplotlib.pyplot as plt
import os
from utils.judgment_distances import load_memory_data
import re

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

def extract_beta_maps(fmri_glm, filter_labels=None, run_label=None, rep=None):
    """
    Extract beta maps for specified trial indices from a fitted GLM model.
    
    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM model from nilearn.
    filter_labels : list of str, optional
        List of trial labels to extract. If None, all trials are extracted.
    run_label : str, optional
        Label for the run (e.g., 'pre' or 'post') to filter trials. If None, no filtering is applied.
    rep : str, optional
        Specific repetition to filter trials (e.g., 'rep1'). If None, no filtering is applied.
    
    Returns
    -------
    dict
        Dictionary mapping trial names to flattened beta maps (1D arrays).
    """
    print("Extracting beta maps...")
    if rep == None:
        rep = 'rep'
    exclude = ['derivative', 'dispersion']
    # Get column names
    columns = fmri_glm.design_matrices_[0].columns
    # check that all filter_labels are in columns
    if filter_labels is not None:
        missing_labels = [label for label in filter_labels if not any(label in col for col in columns)]
        if missing_labels:
            raise ValueError(f"The following filter_labels are not found in the design matrix columns: {missing_labels}")
    # Only keep columns if contains any of the filter_labels
    if filter_labels is not None:
        idxs = [
            i for i, col in enumerate(columns)
            if run_label in col
            and rep in col
            and any(
                label == re.sub(f"{run_label}_|{rep}\\d+_", "", col)
                for label in filter_labels
            )
            and all(e not in col for e in exclude)
        ]
    else:
        idxs = [i for i, col in enumerate(columns) 
                if run_label in col and rep in col 
                and all(e not in col for e in exclude)]

    beta_maps_dict = {}

    for i in idxs:
        contrast_vector = np.zeros(len(columns))
        contrast_vector[i] = 1
        beta_map = fmri_glm.compute_contrast(contrast_vector, output_type='effect_size')
        beta_maps_dict[columns[i]] = beta_map

    # flatten beta maps for each trial
    trial_data_dict = {name: bm.get_fdata().ravel() for name, bm in beta_maps_dict.items()}
    print(f"Extracted {len(trial_data_dict)} beta maps.")

    return trial_data_dict


def get_model_rdm(sub: str, spatial=True, min_confidence=0, save_path=None):
    """
    Compute model matrices for a given subject on its judgment distances.
    Parameters
    ----------
    sub : str
        Subject identifier.
    path_to_preprocessed_data : str
        Path to the preprocessed data directory.
    spatial : bool
        Whether to compute spatial or temporal RDMs.
    min_confidence : float
        Minimum confidence level for including trials. (on a scale 0-100)
    Returns
    -------
    np.ndarray
        The computed RDM matrix.
    filtered_trials : np.ndarray
        The trials that were included after filtering.
    """
    from scipy.spatial.distance import pdist, squareform

    # Load preprocessed data
    raw_df = load_memory_data()

    # Careful here sub is given as 'sub-PXX' but in the dataframe it's just 'PXX'
    sub = sub.replace('sub-', '')
    # Filter for the specific subject
    sub_df = raw_df[raw_df['Participant'] == sub].copy()

    # Filter for only correctly classified as old trials
    df = sub_df[(sub_df['Classified as old'] == 1) & (sub_df['Old'] == 1)].copy()

    filtered_trials = df['V4'].copy()
    print(f"Subject {sub} has {len(filtered_trials)} trials after filtering for correctly classified old items.")

    if spatial:
        df = df[df['Placed Confidence'] >= min_confidence]
        df = df[['X', 'Y', 'Z', 'Placed Confidence']]
        judged_positions = df[['X', 'Y', 'Z']].to_numpy()
    else:
        df = df[df['Timed Confidence'] >= min_confidence]
        if len(df) < 2:
            raise ValueError(f'Participant {sub} has less than 2 valid temporal judgments & should probably be discarded.')
        
        df = df[['Timed', 'Timed Confidence']]
        judged_positions = df[['Timed']].to_numpy()

    dist_matrix = squareform(pdist(judged_positions, metric='euclidean'))
    np.save(save_path, dist_matrix)

    if save_path is not None:
        rdm_type = 'spatial' if spatial else 'temporal'
        plt.imshow(dist_matrix, cmap='viridis')
        plt.colorbar(label='Dissimilarity')
        plt.title(f'{rdm_type.capitalize()} RDM for {sub}{" with min confidence " + str(min_confidence) if min_confidence > 0 else ""}')
        plt.xlabel('Trial Index')
        plt.ylabel('Trial Index')
        plot_path = str(save_path).replace('.npy', '.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {rdm_type} RDM plot to {plot_path}")

    return dist_matrix, filtered_trials

def compute_rsa(beta_maps_dict, save_path=None, single_repetition=False):
    """
    Compute Representational Similarity Analysis (RSA) by spearman correlating beta maps. 
    Only correlate the beta map of a trial with every other trial in every other repetition.
    Parameters
    ----------
    beta_maps_dict : dict
        Dictionary mapping trial names to flattened beta maps (1D arrays).
    save_path : str, optional
        Path to save the computed RSA matrix. If None, the matrix is not saved.
    Returns 
    -------
    np.ndarray
        The computed correlation matrix in the shape (n_trials, n_trials).
    """
    from scipy.stats import spearmanr
    beta_maps = get_ordered_array_from_beta_dict(beta_maps_dict)
    correlation_matrix = spearmanr(beta_maps, axis=1).correlation
    print(f"Computed correlation matrix with shape {correlation_matrix.shape}")
    n_repetitions, n_trials = 6, int(len(beta_maps_dict) / 6)
    if single_repetition:
        n_repetitions = 1
        n_trials = len(beta_maps_dict)
    beta_maps_idxs = np.array([[i + r * n_trials for i in range(n_trials)] for r in range(n_repetitions)])
    # We only want to keep the correlations between different repetitions
    rsa_matrix = np.zeros((n_trials, n_trials))
    for i in range(n_trials):
        for j in range(i, n_trials):
            correlations = []
            for r1 in range(n_repetitions):
                for r2 in range(n_repetitions):
                    if single_repetition or r1 != r2: # only correlate different repetitions if not single_repetition
                        correlations.append(correlation_matrix[beta_maps_idxs[r1, i], beta_maps_idxs[r2, j]])
            rsa_matrix[i, j] = np.mean(correlations)
            rsa_matrix[j, i] = rsa_matrix[i, j]  # Symmetric matrix
    print(f"Computed RSA matrix with shape {rsa_matrix.shape}")
    if save_path is not None:
        np.save(save_path, rsa_matrix)
        plot_path = str(save_path).replace('.npy', '.png')
        create_rsa_figure(rsa_matrix, title='RSA Matrix', save_path=plot_path)
        print(f"Saved RSA matrix to {save_path}")
    return rsa_matrix


def create_rsa_figure(rsa_matrix, title, save_path):
    """
    Create and optionally save a figure of the RSA matrix.
    Parameters
    ----------
    rsa_matrix : np.ndarray
        The RSA matrix to visualize.
    title : str
        Title for the plot.
    save_path : str
        Path to save the figure. If None, the figure is not saved.
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(rsa_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='RSA Correlation')
    plt.title(title)
    plt.xlabel('Trial Index')
    plt.ylabel('Trial Index')
    plt.savefig(save_path)
    print(f"Saved RSA figure to {save_path}")
    plt.close()


def compute_rsa_from_glm(fmri_glm, sub, filtered_trials, pre_save_path, n_repetitions=6, downsample=False):
    """
    Compute RSA from a fitted GLM model by extracting beta maps for old trials and correlating them.
    Parameters
    ----------
    fmri_glm : FirstLevelModel
        Fitted GLM model from nilearn.
    sub : str
        Subject identifier.
    filtered_trials : list of str
        List of trials to include based on previous filtering.
    n_trials : int
        Number of unique trials.
    n_repetitions : int
        Number of repetitions per trial.
    pre_save_path : str
        Path to save the computed RSA matrix for the pre run. If None, the matrix is not saved.
    downsample : bool
        Whether to downsample the trials for testing purposes.
    Returns
    -------
    pre_rsa : np.ndarray
        The computed RSA matrix for the pre run.
    post_rsa : np.ndarray
        The computed RSA matrix for the post run.
    """
    # Transform filtered idxs to include all repetitions
    if downsample:
        filtered_trials = filtered_trials[0] # only keep the first trial for test purposes

    n_tasks = len(filtered_trials)
    print(f"Computing RSA Matrix for sub : {sub}, retrieved {n_tasks} tasks and {n_repetitions} repetitions")
    

    # Extract beta maps for old trials only
    pre_beta_maps = extract_beta_maps(fmri_glm, filter_labels=filtered_trials, run_label='pre')
    post_beta_maps = extract_beta_maps(fmri_glm, filter_labels=filtered_trials, run_label='post')

    # Correlate every beta maps with every other beta maps of every other repetition
    pre_rsa = compute_rsa(pre_beta_maps, save_path=pre_save_path)
    print(f"Pre RSA shape: {pre_rsa.shape}")
    post_save_path = str(pre_save_path).replace('pre', 'post')
    post_rsa = compute_rsa(post_beta_maps, save_path=post_save_path)
    print(f"Post RSA shape: {post_rsa.shape}")
    return pre_rsa, post_rsa

def get_ordered_array_from_beta_dict(beta_maps_dict):
    # Parse trial names and repetitions
    reps = []
    trials = []
    run_label = 'pre' if 'pre' in next(iter(beta_maps_dict.keys())) else 'post'
    for key in beta_maps_dict.keys():
        # take the second part of the key split by '_'
        keys_split = key.split('_')
        rep_idx = keys_split[1] # first is run label
        reps.append(rep_idx)
        # trial name = everything after the repetition index
        trial_name = '_'.join(keys_split[2:])
        trials.append(trial_name)
    
    unique_reps = sorted(list(set(reps)))
    unique_trials = sorted(list(set(trials)))
    print(f"Identified {len(unique_trials)} unique trials and {len(unique_reps)} unique repetitions.")
    
    # Build beta array in order rep x trial
    beta_array = np.zeros((len(unique_reps) * len(unique_trials), len(next(iter(beta_maps_dict.values())))))
    for i, rep in enumerate(unique_reps):
        for j, trial in enumerate(unique_trials):
            key = f"{run_label}_{rep}_{trial}"
            if key not in beta_maps_dict:
                raise ValueError(f"Missing beta map for {key}")
            beta_array[i * len(unique_trials) + j, :] = beta_maps_dict[key]
    print(f"Constructed beta array with shape {beta_array.shape}")
    return beta_array