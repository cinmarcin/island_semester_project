import numpy as np
import matplotlib.pyplot as plt
import os
import re
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform

from src.data.judgment_distances import load_memory_data
from src.data.data import get_model_rdm, get_rsa_matrix
from src.utils.plots import plot_bar, plot_split
from src.utils.utils import make_dir

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
    if rep == 'all_reps':
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
                label == re.sub(f"{run_label}_|{rep}\\d*_", "", col)
                for label in filter_labels
            )
            and all(e not in col for e in exclude)
        ]
        print(f"Columns : {[columns[i] for i in idxs]} selected.")
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


def compute_model_rdm(sub: str, spatial=True, min_confidence=0, save_path=None):
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

    # Load preprocessed data
    raw_df = load_memory_data()

    # Careful here sub is given as 'sub-PXX' but in the dataframe it's just 'PXX'
    sub = sub.replace('sub-', '')
    # Filter for the specific subject
    sub_df = raw_df[raw_df['Participant'] == sub].copy()

    # Filter for only correctly classified as old trials
    df = sub_df[(sub_df['Classified as old'] == 1) & (sub_df['Old'] == 1)].copy()

    # Define min confidence as median confidence
    # median_spatial_confidence = df['Placed Confidence'].median()
    # print(f"Subject {sub} median spatial confidence: {median_spatial_confidence}")
    # df = df[df['Placed Confidence'] >= median_spatial_confidence]

    filtered_trials = df['V4'].copy()
    print(f"Subject {sub} has {len(filtered_trials)} trials after filtering for correctly classified old items.")

    if spatial:
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

def compute_rsa(beta_maps_dict, save_path=None, rep='all_reps', euclidean=False):
    """
    Compute Representational Similarity Analysis (RSA) by spearman correlating beta maps. 
    Only correlate the beta map of a trial with every other trial in every other repetition.
    Parameters
    ----------
    beta_maps_dict : dict
        Dictionary mapping trial names to flattened beta maps (1D arrays).
    save_path : str, optional
        Path to save the computed RSA matrix. If None, the matrix is not saved.
    single_repetition : bool
        Whether to compute RSA using a single repetition (default is False).
    euclidean : bool
        Whether to use Euclidean distance instead of Spearman correlation.
    Returns
    -------
    np.ndarray
        The computed correlation matrix in the shape (n_trials, n_trials).
    """
    single_repetition = (rep != 'all_reps')
    beta_maps = get_ordered_array_from_beta_dict(beta_maps_dict)
    if euclidean:
        correlation_matrix = squareform(pdist(beta_maps, metric='euclidean'))
        save_path = str(save_path).replace('.npy', '_euclidean.npy') if save_path is not None else None
    else:
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


def compute_rsa_from_glm(fmri_glm, sub, filtered_trials, pre_save_path, n_repetitions=6, downsample=False, rep='all_reps', euclidean=False):
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

    pre_save_path = str(pre_save_path).replace('.npy', f'_{rep}.npy')

    n_tasks = len(filtered_trials)
    print(f"---- Computing RSA Matrix for sub : {sub}, retrieved {n_tasks} tasks and {n_repetitions} repetitions for rep {rep}.")
    

    # Extract beta maps for old trials only
    pre_beta_maps = extract_beta_maps(fmri_glm, filter_labels=filtered_trials, run_label='pre', rep=rep)
    post_beta_maps = extract_beta_maps(fmri_glm, filter_labels=filtered_trials, run_label='post', rep=rep)

    # Correlate every beta maps with every other beta maps of every other repetition
    pre_rsa = compute_rsa(pre_beta_maps, save_path=pre_save_path, rep=rep, euclidean=euclidean)
    print(f"Pre RSA shape: {pre_rsa.shape}")
    post_save_path = str(pre_save_path).replace('pre', 'post')
    post_rsa = compute_rsa(post_beta_maps, save_path=post_save_path, rep=rep, euclidean=euclidean)
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


def run_subject_level_rsa(sub, config, config_file, rep='all_reps', only_post_session=False):
    """
    Run RSA analysis for a single subject, correlating behavioral RDMs with neural RSA matrix.
    Parameters
    ----------
    sub : str
        Subject identifier.
    config : dict
        Configuration dictionary.
    config_file : str
        Path to the configuration file.
    Returns
    -------
    tuple
        Tuple containing correlation results.
    """
    # Load behavioral RDMs
    temporal_rdm = get_model_rdm(sub, spatial=False, config=config, config_file=config_file)
    spatial_rdm = get_model_rdm(sub, spatial=True, config=config, config_file=config_file)

    print(f"\n{'-'*70}")
    print(f"Running RSA for {sub}")
    print(f"Config file: {config_file}")
    print(f"RDM dimensions: {temporal_rdm.shape[0]} x {temporal_rdm.shape[1]}")
    print(f"{'-'*70}")

    # Sanity checks
    assert temporal_rdm.shape == spatial_rdm.shape, "Temporal and spatial RDMs must have the same shape"
    assert temporal_rdm.shape[0] == temporal_rdm.shape[1], "RDMs must be square matrices"

    # Extract upper triangle (excluding diagonal)
    iu = np.triu_indices(len(temporal_rdm), k=1)
    temporal_vals = temporal_rdm[iu]
    spatial_vals = spatial_rdm[iu]

    # Compute median
    spat_median = np.median(spatial_vals)
    print(f"Median spatial distance:  {spat_median:.4f}")

    # Correlation between temporal and spatial behavioral RDMs
    tempo_spatial_corr = spearmanr(temporal_vals, spatial_vals)
    print(f"Temporal vs. spatial RDM: ρ = {tempo_spatial_corr.correlation:.4f}, p = {tempo_spatial_corr.pvalue:.4f}")

    # Load neural RSA matrix (PS′)
    pre_post_rsa = get_rsa_matrix(sub, config=config, config_file=config_file, run_label='pre-post' if not only_post_session else 'post', rep=rep, euclidean=config.get('euclidean_rsa', False))
    rsa_vals = pre_post_rsa[iu]

    # Compute Spearman correlations between model and neural RDMs (full)
    corr_temp = spearmanr(temporal_vals, rsa_vals)
    corr_spat = spearmanr(spatial_vals, rsa_vals)


    # Compute Z effect
    real_corr_temp, z_temp, _ = permutation_test_spearman(temporal_vals, rsa_vals)
    real_corr_spat, z_spat, _ = permutation_test_spearman(spatial_vals, rsa_vals)
    print(f"Temporal RDM ↔ RSA : real ρ = {real_corr_temp:.4f}, Z = {z_temp:.4f}, (p-value from spearman: {corr_temp.pvalue:.4f})")
    print(f"Spatial RDM  ↔ RSA : real ρ = {real_corr_spat:.4f}, Z = {z_spat:.4f}, (p-value from spearman: {corr_spat.pvalue:.4f})")

    # Median split indices for high / low spatial distances
    spat_low_mask = spatial_vals <= spat_median
    spat_high_mask = spatial_vals > spat_median

    # Safely compute spearman on subsets (handle degenerate cases)
    def safe_spearman(x, y):
        # if constant array -> return nan p=1
        if np.all(x == x[0]) or np.all(y == y[0]):
            return (np.nan, np.nan)
        return spearmanr(x, y).correlation, spearmanr(x, y).pvalue

    # Spatial low/high correlations
    corr_spat_low = np.mean(rsa_vals[spat_low_mask])
    corr_spat_high = np.mean(rsa_vals[spat_high_mask])

    print(f"Spatial  RDM ↔ RSA (low dist):   ρ = {np.nan_to_num(corr_spat_low):.4f}")
    print(f"Spatial  RDM ↔ RSA (high dist):  ρ = {np.nan_to_num(corr_spat_high):.4f}")

    return (corr_temp, z_temp, corr_spat, z_spat, tempo_spatial_corr,
            corr_spat_low, corr_spat_high)

def run_subject_level_rsa_v2(sub, config, config_file, rep='all_reps', only_post_session=False):
    """
    Calculate RSA separately for positive and negative correlations.
    """
    temporal_rdm = get_model_rdm(sub, spatial=False, config=config, config_file=config_file)
    spatial_rdm = get_model_rdm(sub, spatial=True, config=config, config_file=config_file)

    print(f"\n{'-'*70}")
    print(f"Running RSA v2 for {sub}")
    print(f"Config file: {config_file}")
    print(f"RDM dimensions: {temporal_rdm.shape[0]} x {temporal_rdm.shape[1]}")
    print(f"{'-'*70}")  

    # Sanity checks
    assert temporal_rdm.shape == spatial_rdm.shape, "Temporal and spatial RDMs must have the same shape"
    assert temporal_rdm.shape[0] == temporal_rdm.shape[1], "RDMs must be square matrices"

    # Extract upper triangle (excluding diagonal)
    iu = np.triu_indices(len(temporal_rdm), k=1)
    temporal_vals = temporal_rdm[iu]
    spatial_vals = spatial_rdm[iu]

    # Load neural RSA matrix (PS′)
    run_label = 'post' if only_post_session else 'pre-post'
    pre_rsa = get_rsa_matrix(sub, config=config, config_file=config_file, run_label=run_label, rep=rep, euclidean=config.get('euclidean_rsa', False))
    positive_effect_rsa = np.where(pre_rsa > 0, pre_rsa, 0)
    negative_effect_rsa = np.where(pre_rsa < 0, pre_rsa, 0)
    positive_rsa_vals = positive_effect_rsa[iu]
    negative_rsa_vals = negative_effect_rsa[iu]

    # Compute Spearman correlations between model and neural RDMs (full)
    corr_temp = spearmanr(temporal_vals, positive_rsa_vals)
    corr_spat = spearmanr(spatial_vals, positive_rsa_vals)
    corr_temp_neg = spearmanr(temporal_vals, negative_rsa_vals)
    corr_spat_neg = spearmanr(spatial_vals, negative_rsa_vals)
    print(f"Temporal RDM ↔ RSA (positive): ρ = {corr_temp[0]:.4f}, p = {corr_temp[1]:.4f}")
    print(f"Spatial RDM  ↔ RSA (positive): ρ = {corr_spat[0]:.4f}, p = {corr_spat[1]:.4f}")
    print(f"Temporal RDM ↔ RSA (negative): ρ = {corr_temp_neg[0]:.4f}, p = {corr_temp_neg[1]:.4f}")
    print(f"Spatial RDM  ↔ RSA (negative): ρ = {corr_spat_neg[0]:.4f}, p = {corr_spat_neg[1]:.4f}")

    return (corr_temp, corr_spat,
            corr_temp_neg, corr_spat_neg)

# ------------------------------
# Summary helpers
# ------------------------------
def mean_sem(array_of_vals):
    arr = np.array([v for v in array_of_vals if not np.isnan(v)])
    if arr.size == 0:
        return np.nan, np.nan
    mean = np.mean(arr)
    sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, sem

def compute_group_mean_sem(results_dicts, subs):
    """Compute mean and SEM for multiple conditions."""
    vals = [mean_sem([res[s][0] for s in subs]) for res in results_dicts]
    means, sems = zip(*vals)
    return means, sems

def summarize_results(name, results):
    corrs = np.array([results[sub][0] for sub in results])
    ps = np.array([results[sub][1] for sub in results])
    mean_corr, sem_corr = mean_sem(corrs)
    mean_p = np.nanmean(ps)
    print(f"\n{name} (N={len(results)}):")
    print(f"  Mean ρ = {mean_corr:.4f} ± {sem_corr:.4f} (SEM)")
    print(f"  Mean p = {mean_p:.4f}")

# summarize high/low splits
def summarize_split(name, results_low, results_high):
    from scipy.stats import sem
    mean_low = np.mean([results_low[s] for s in results_low])
    sem_low = sem([results_low[s] for s in results_low])

    mean_high = np.mean([results_high[s] for s in results_high])
    sem_high = sem([results_high[s] for s in results_high])
    print(f"\n{name} (median split):")
    print(f"  Low-dist mean ρ = {mean_low:.4f} ± {sem_low:.4f} (SEM)")
    print(f"  High-dist mean ρ = {mean_high:.4f} ± {sem_high:.4f} (SEM)")

def permutation_test_spearman(x, y, n_perm=10000, random_state=None):
    """
    Generate surrogate distribution by shuffling y and computing Spearman correlations.
    Returns the real correlation, z-score, and surrogate distribution.
    """
    rng = np.random.default_rng(random_state)
    real_corr, _ = spearmanr(x, y)
    null_corrs = np.empty(n_perm)
    for i in range(n_perm):
        y_shuffled = rng.permutation(y)
        null_corrs[i], _ = spearmanr(x, y_shuffled)
    mean_null = np.mean(null_corrs)
    std_null = np.std(null_corrs)
    z = (real_corr - mean_null) / std_null if std_null > 0 else np.nan
    return real_corr, z, null_corrs

def second_level_sign_flip(z_values, n_perm=10000, random_state=None):
    """
    Perform group-level sign-flip permutation test on subject-level z-values.
    
    Parameters
    ----------
    z_values : array-like
        First-level z-values for all participants (e.g. one per subject).
    n_perm : int
        Number of sign-flip permutations (default: 10000).
    random_state : int or None
        Random seed for reproducibility.
    
    Returns
    -------
    dict
        Contains observed mean z, surrogate mean z's, z-statistic, and p-value.
    """
    z_values = np.asarray(z_values)
    N = len(z_values)
    rng = np.random.default_rng(random_state)
    
    # Real group mean
    mean_real = np.mean(z_values)
    
    # Generate surrogate means via random sign flips
    null_means = np.empty(n_perm)
    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=N)
        null_means[i] = np.mean(z_values * signs)
    
    # Compute z-statistic for the observed mean
    mean_null = np.mean(null_means)
    std_null = np.std(null_means)
    z_group = (mean_real - mean_null) / std_null if std_null > 0 else np.nan
    
    # Empirical two-tailed p-value
    p_empirical = (np.sum(np.abs(null_means) >= abs(mean_real)) + 1) / (n_perm + 1)
    
    return {
        "mean_real": mean_real,
        "z_group": z_group,
        "p_value": p_empirical,
        "null_means": null_means
    }

def plot_and_save_rsa_results(results, figure_dir, subs, second_level_results_spatial, second_level_results_temporal, euclidean=False):
    
        # --- Print summary
    print("\n" + "="*70)
    print("Summary of RSA results across subjects")
    print("="*70)
    summarize_results("Temporal RDM ↔ RSA (full)", results["temp"])
    print(f"Group-level permutation test (spatial): p = {second_level_results_spatial['p_value']:.4f}, observed mean Z = {second_level_results_spatial['mean_real']:.4f}")
    print(f"Group-level permutation test (temporal): p = {second_level_results_temporal['p_value']:.4f}, observed mean Z = {second_level_results_temporal['mean_real']:.4f}")
    summarize_results("Spatial RDM  ↔ RSA (full)", results["spat"])
    summarize_split("Spatial  RDM ↔ RSA", results["spat_low"], results["spat_high"])
    summarize_results("Temporal RDM ↔ RSA (positive)", results["pos_temp"])
    summarize_results("Spatial RDM  ↔ RSA (positive)", results["pos_spat"])
    summarize_results("Temporal RDM ↔ RSA (negative)", results["neg_temp"])
    summarize_results("Spatial RDM  ↔ RSA (negative)", results["neg_spat"])
    print("="*70)

    # --- Per-subject plots
    for sub in subs:
        vals = [results["temp"][sub][0], results["spat"][sub][0]]
        sub_folder = f"{figure_dir}/{sub}/"
        make_dir(sub_folder)
        plot_bar(
            labels=['Temporal', 'Spatial'],
            values=vals,
            title=f"RSA correlations - {sub}",
            path=f"{sub_folder}/{sub}_rsa_barplot.png",
            colors=['skyblue', 'salmon', 'lightgreen']
        )
        pos_neg_vals = [results["pos_temp"][sub][0], results["pos_spat"][sub][0], results["neg_temp"][sub][0], results["neg_spat"][sub][0]]
        plot_bar(
            labels=['Pos Temporal', 'Pos Spatial', 'Neg Temporal', 'Neg Spatial'],
            values=pos_neg_vals,
            title=f"Positive/Negative RSA correlations - {sub}",
            path=f"{sub_folder}/{sub}_rsa_pos_neg_barplot.png",
            colors=['lightcoral', 'salmon', 'lightblue', 'skyblue'],
            ylim=(-1, 1)
        )
        plot_split(
            sub=sub,
            spat_low=results["spat_low"][sub],
            spat_high=results["spat_high"][sub],
            path=sub_folder
        )
    # --- Group-level bar plot
    mean_vals, sem_vals = compute_group_mean_sem(
        [results["temp"], results["spat"]],
        subs
    )
    plot_bar(
        labels=['Temporal ↔ RSA', 'Spatial ↔ RSA'],
        values=mean_vals,
        errors=sem_vals,
        title="Group-level RSA correlations",
        ylabel="Spearman ρ (mean ± SEM)",
        path=f"{figure_dir}/group_rsa_barplot.png",
        colors=['skyblue', 'salmon', 'lightgreen']
    )

    # --- Group-level positive/negative bar plot
    mean_vals_pos_neg, sem_vals_pos_neg = compute_group_mean_sem(
        [results["pos_temp"], results["pos_spat"], results["neg_temp"], results["neg_spat"]],
        subs
    )
    plot_bar(
        labels=['Pos Temporal', 'Pos Spatial', 'Neg Temporal', 'Neg Spatial'],
        values=mean_vals_pos_neg,
        errors=sem_vals_pos_neg,
        title="Group-level Positive/Negative RSA correlations",
        ylabel="Spearman ρ (mean ± SEM)",
        path=f"{figure_dir}/group_rsa_pos_neg_barplot.png",
        colors=['lightcoral', 'salmon', 'lightblue', 'skyblue'],
        ylim=(-1, 1)
    )
    # --- Group-level median split
    def group_mean_sem(low_dict, high_dict):
        from scipy.stats import sem
        mean_low = np.mean([low_dict[s] for s in subs])
        sem_low = sem([low_dict[s] for s in subs])

        mean_high = np.mean([high_dict[s] for s in subs])
        sem_high = sem([high_dict[s] for s in subs])
        return [mean_low, mean_high], [sem_low, sem_high]

    spat_vals, spat_err = group_mean_sem(results["spat_low"], results["spat_high"])

    scale = 1e1


    plt.figure(figsize=(6, 4))  # Adjusted size since we have only 1 plot

    if euclidean:
        metric = "Euclidean"
        scale = 1e-4
        ylim = None
    else:
        metric = "Spearman ρ (×10⁻¹)"
        scale = 1e1
        ylim = (-1, 1)

    vals, errs, title, colors = spat_vals, spat_err, "Spatial RDM ↔ RSA", ['salmon', 'red']

    plt.bar(['Low', 'High'],
            np.nan_to_num(vals) * scale,
            yerr=np.nan_to_num(errs) * scale,
            capsize=5,
            color=colors)

    plt.ylim(ylim)
    plt.ylabel(metric)
    plt.title(title)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{figure_dir}/group_rsa_median_split.png")
    plt.close()

    # --- Group-level histograms for permutation ---
        # plot histograms of z-values
    z_spat_values = [results['z_spat'][sub] for sub in subs]
    z_temp_values = [results['z_temp'][sub] for sub in subs]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(z_spat_values, bins=10, color='salmon', alpha=0.7)
    plt.title('Histogram of Spatial RSA Z-values')
    plt.xlabel('Z-value')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(z_temp_values, bins=10, color='skyblue', alpha=0.7)
    plt.title('Histogram of Temporal RSA Z-values')
    plt.xlabel('Z-value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{figure_dir}/group_rsa_zvalue_histograms.png")

