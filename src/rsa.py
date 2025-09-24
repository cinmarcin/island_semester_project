from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import os

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
    # plot
    if save_fig is not None:
        plt.imshow(rdm, cmap='viridis')
        plt.colorbar(label='Dissimilarity')
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

def extract_beta_maps(fmri_glm):
    print("Extracting beta maps...")
    # Get column names
    columns = fmri_glm.design_matrices_[0].columns
    trial_columns = [i for i, col in enumerate(columns) if "trial_" in col]
    beta_maps_list = []
    for i in trial_columns:
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