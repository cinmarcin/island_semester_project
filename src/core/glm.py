import pandas as pd
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.plotting import plot_design_matrix
from nilearn.glm.first_level import FirstLevelModel
from nilearn.glm.second_level import SecondLevelModel
from nilearn.reporting import make_glm_report
import numpy as np
import nibabel as nib
import inspect
import matplotlib.pyplot as plt
from nilearn import datasets, image

from src.core.contrast_types import ContrastType



BLUE = "\033[34m"
RESET = "\033[0m"

# Keep reference to original print
original_print = print

def print(*args, **kwargs):
    # Get the caller function name
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    # Format message
    msg = " ".join(str(a) for a in args)
    original_print(f"{BLUE}[{func_name}] {msg}{RESET}", **kwargs)

def create_gm_mask(func_img):
    """
    Create a gray matter mask resampled to the functional image.
    Args:
        func_img (nibabel.Nifti1Image): Functional image to which the mask will be resampled.
    Returns:
        gm_mask_img (nibabel.Nifti1Image): Resampled gray matter mask image.
    """
    gm_mask = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)
    gm_mask_resampled = image.resample_to_img(gm_mask, func_img, interpolation='nearest', force_resample=True, copy_header=True)
    gm_mask_img = nib.Nifti1Image((gm_mask_resampled.get_fdata() > 0).astype(int), gm_mask_resampled.affine, gm_mask_resampled.header)
    return gm_mask_img

def create_hippocampal_gm_mask(func_img):
    """
    Create a hippocampal gray matter mask resampled to the functional image.
    Args:
        func_img (nibabel.Nifti1Image): Functional image in MNI space.
    Returns:
        hippocampal_gm_mask_img (nibabel.Nifti1Image): Resampled hippocampal GM mask image.
    """
    print("Creating hippocampal gray matter mask...")
    # Load Harvard-Oxford cortical atlas
    atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
    atlas_img = atlas['maps']
    atlas_data = atlas_img.get_fdata()

    # Select left and right hippocampus (indices 17 and 53)
    hippocampus_mask_data = np.isin(atlas_data, [17, 53])
    hippocampus_mask_img = nib.Nifti1Image(hippocampus_mask_data.astype(np.uint8), atlas_img.affine)

    # Load MNI gray matter mask
    gm_mask_img = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)

    # Resample hippocampal mask to GM mask space
    hippocampus_resampled = image.resample_to_img(hippocampus_mask_img, gm_mask_img, interpolation='nearest', force_resample=True, copy_header=True)

    # Intersect with GM mask (binary AND)
    intersection_data = (hippocampus_resampled.get_fdata() > 0) & (gm_mask_img.get_fdata() > 0)
    hippocampal_gm_mask_img = nib.Nifti1Image(intersection_data.astype(np.uint8), gm_mask_img.affine, gm_mask_img.header)

    # Resample to functional image space
    hippocampal_gm_mask_resampled = image.resample_to_img(hippocampal_gm_mask_img, func_img, interpolation='nearest', force_resample=True, copy_header=True)
    hippocampal_gm_mask_resampled = nib.Nifti1Image((hippocampal_gm_mask_resampled.get_fdata() > 0).astype(np.uint8), hippocampal_gm_mask_resampled.affine, 
                                                    hippocampal_gm_mask_resampled.header)

    return hippocampal_gm_mask_resampled


def create_group_mask(subject_masks, threshold=0.8):
    """
    Combine subject masks into a group mask.
    
    Args:
        subject_masks (list of Nifti1Image): list of individual masks in same space
        threshold (float): fraction of subjects that must have voxel present (0-1)
        
    Returns:
        Nifti1Image: group mask
    """
    data_stack = np.stack([mask.get_fdata() > 0 for mask in subject_masks], axis=-1)
    coverage = data_stack.mean(axis=-1)
    group_data = (coverage >= threshold).astype(np.uint8)
    
    # Use affine/header from first mask
    group_mask_img = nib.Nifti1Image(group_data, subject_masks[0].affine, subject_masks[0].header)
    return group_mask_img

def events_to_stimuli(events):
    """
    Convert events DataFrame to stimuli format.
    Args:
        events (pandas.DataFrame): DataFrame containing event information with at least three columns: onset, duration, and trial_type.
    Returns:
        stimuli (pandas.DataFrame): DataFrame formatted for stimuli with columns: onset, duration, trial_type.
    """
    events = events.iloc[:,0:3]
    events.columns = ['onset', 'duration', 'trial_type']
    events = events.sort_values(by='onset').reset_index(drop=True)
    return events

def create_design_matrix(n_scans, events, metadata, confounds, drift_model="cosine", high_pass=0.01):
    """
    Create a design matrix for fMRI analysis.
    Args:
        n_scans (int): Number of scans (time points) in the fMRI data.
        events (pandas.DataFrame): DataFrame containing event information with at least three columns: onset, duration, and trial_type.
        metadata (dict): Metadata dictionary containing at least the "RepetitionTime" key.
        confounds (pandas.DataFrame): DataFrame containing confound time series.
        drift_model (str): Drift model to use (default is "cosine").
        high_pass (float): High-pass filter cutoff frequency (default is 0.01).
    Returns:
        design_matrix (pandas.DataFrame): Design matrix for fMRI analysis.
    """
    tr = metadata["RepetitionTime"]
    frame_times = (
        np.arange(n_scans) * tr
    )
    if "trans_x" in confounds.columns:
        add_regressors = confounds[["trans_x","trans_y","trans_z","rot_x","rot_y","rot_z","global_signal","csf","white_matter"]].copy()
    else:
        add_regressors = confounds.copy()

    design_matrix = make_first_level_design_matrix(
        frame_times,
        events,
        hrf_model='spm + derivative + dispersion',
        drift_model=drift_model,
        high_pass=high_pass,
        add_regs=add_regressors,
        add_reg_names=add_regressors.columns.tolist(),
    )
    print(f"Created design matrix with shape: {design_matrix.shape}")  
    return design_matrix

def save_design_matrix(design_matrix, filepath):
    """
    Save the design matrix to a CSV + PNG file.
    Args:
        design_matrix (pandas.DataFrame): Design matrix to be saved.
        filepath (str): Path to the output CSV file.
    """
    design_matrix.to_csv(filepath, index=False)
    plot_design_matrix(design_matrix)
    plot_path = filepath.replace('data/processed', 'figures').replace('.csv', '.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved design matrix to {filepath} and plot to {plot_path}")


def combined_first_level_analysis(fmri_data_list, events_list, metadata_list, confounds_list, contrast_type: ContrastType, smoothing_fwhm=None, hippocampus_only=False):
    """
    Perform first-level GLM analysis on concatenated fMRI data.
    Args:
        fmri_data_list (list of nibabel.Nifti1Image): List of 4D fMRI data images to concatenate.
        events_list (list of pandas.DataFrame): List of DataFrames containing event information for each fMRI data.
        metadata_list (list of dict): List of metadata dictionaries for each fMRI data.
        confounds_list (list of pandas.DataFrame): List of DataFrames containing confound time series for each fMRI data.
        contrast_type (ContrastType): Type of contrast to compute.
        smoothing_fwhm (float): Smoothing FWHM in mm.
    Returns:
        fmri_glm (FirstLevelModel): Fitted FirstLevelModel object.
        pre_contrast (nibabel.Nifti1Image): Contrast image for the pre session.
        post_contrast (nibabel.Nifti1Image): Contrast image for the post session.
        pre_post_contrast (nibabel.Nifti1Image): Contrast image for the pre vs post comparison.
        gm_mask_img (nibabel.Nifti1Image): Gray matter mask used in the analysis.
    """
    print("Starting combined first-level analysis...")
    
    # Concatenate fMRI data
    fmri_data = image.concat_imgs(fmri_data_list)
    tr = metadata_list[0]["RepetitionTime"]
    n_volumes_pre = fmri_data_list[0].shape[3]
    n_volumes_post = fmri_data_list[1].shape[3]
    n_scans = fmri_data.shape[3]
    # Use metadata from the first session (assuming same TR)
    metadata = metadata_list[0]

    # Preprocess pre 
    pre_events = events_list[0]
    pre_events = events_to_stimuli(pre_events)
    pre_events = contrast_type.preprocess_stimuli(pre_events, run_label='pre')
    pre_duration = tr * n_volumes_pre

    # Preprocess post
    post_events = events_list[1]
    post_events = events_to_stimuli(post_events, )
    post_events = contrast_type.preprocess_stimuli(post_events, run_label='post')


    # Generate confounds regressor for per and post separately
    pre_conf_matrix = create_design_matrix(n_volumes_pre, None, metadata, confounds_list[0])
    post_conf_matrix = create_design_matrix(n_volumes_post, None, metadata, confounds_list[1])

    # Concatenate events
    post_events['onset'] += pre_duration  # shift onsets forward
    events = pd.concat([pre_events, post_events], ignore_index=True)

    # Rename confound columns to avoid duplicates
    pre_conf_matrix = pre_conf_matrix.add_prefix('pre_')
    post_conf_matrix = post_conf_matrix.add_prefix('post_')

    # Concatenate confounds + fill nans with 0
    confounds = pd.concat([pre_conf_matrix, post_conf_matrix], ignore_index=True)
    confounds = confounds.fillna(0)

    # Generate event design matrix
    design_matrix = create_design_matrix(n_scans, events, metadata, confounds, drift_model=None, high_pass=None)

    # drop intercept column to avoid collinearity
    if 'constant' in design_matrix.columns:
        design_matrix = design_matrix.drop(columns=['constant'])

    # Create mask
    if hippocampus_only:
        gm_mask_img = create_hippocampal_gm_mask(fmri_data)
    else:
        gm_mask_img = create_gm_mask(fmri_data)
    

    # Fit GLM
    fmri_glm = FirstLevelModel(mask_img=gm_mask_img, smoothing_fwhm=smoothing_fwhm, n_jobs=-1)
    print("Fitting FirstLevelModel...")
    fmri_glm = fmri_glm.fit(fmri_data, design_matrices=design_matrix)
    print("Fitted FirstLevelModel.")

    contrast_vec = contrast_type.get_vector(design_matrix, combined=True)

    if contrast_type not in {ContrastType.PER_TRIAL, ContrastType.ORDER}:
        pre_contrast = fmri_glm.compute_contrast(contrast_vec['Pre_' + contrast_type.value], output_type='z_score')
        post_contrast = fmri_glm.compute_contrast(contrast_vec['Post_' + contrast_type.value], output_type='z_score')
        pre_post_contrast = fmri_glm.compute_contrast(contrast_vec['Pre_vs_Post_' + contrast_type.value], output_type='z_score')
    else:
        # We do not compute contrasts for per_trial and order here we will use this for RSA analysis
        pre_contrast = None
        post_contrast = None
        pre_post_contrast = None

    return fmri_glm, pre_contrast, post_contrast, pre_post_contrast, gm_mask_img

def first_level_analysis(fmri_data, events, metadata, confounds, contrast_type: ContrastType, smoothing_fwhm=None, hippocampus_only=False):
    """
    Perform first-level GLM analysis on single fMRI data.
    Args:
        fmri_data (nibabel.Nifti1Image): 4D fMRI data image.
        events (pandas.DataFrame): DataFrame containing event information.
        metadata (dict): Metadata dictionary containing at least the "RepetitionTime" key.
        confounds (pandas.DataFrame): DataFrame containing confound time series.
        contrast_type (ContrastType): Type of contrast to compute.
        smoothing_fwhm (float): Smoothing FWHM in mm.
    Returns:
        fmri_glm (FirstLevelModel): Fitted FirstLevelModel object.
        contrast (nibabel.Nifti1Image): Contrast image for the specified contrast type.
    """
    print("Starting first-level analysis...")
    n_scans = fmri_data.shape[3]

    # Preprocess events
    events = events_to_stimuli(events)
    events = contrast_type.preprocess_stimuli(events)

    # Create design matrix
    design_matrix = create_design_matrix(n_scans, events, metadata, confounds)

    # Create mask
    if hippocampus_only:
        gm_mask_img = create_hippocampal_gm_mask(fmri_data)
    else:
        gm_mask_img = create_gm_mask(fmri_data)

    # Fit GLM
    fmri_glm = FirstLevelModel(mask_img=gm_mask_img, smoothing_fwhm=smoothing_fwhm, n_jobs=-1)
    print("Fitting FirstLevelModel...")
    fmri_glm = fmri_glm.fit(fmri_data, design_matrices=design_matrix)
    print("Fitted FirstLevelModel.")

    # Define contrast
    contrast_vec = contrast_type.get_vector(design_matrix)
    contrast = fmri_glm.compute_contrast(contrast_vec, output_type='z_score')
    print(f"Computed contrast for {contrast_type}.")

    return fmri_glm, contrast

def group_level_analysis(contrast_maps, height_control, alpha, save_path):
    """
    Perform second-level GLM analysis.
    Args:
        contrast_maps (list of nibabel.Nifti1Image): List of contrast images from first-level
        height_control (str): Method for height control ('fdr' or 'bonferroni').
        alpha (float): Significance level for thresholding.
        smoothing_fwhm (float): Smoothing FWHM in mm.
        contrast_type (ContrastType): Type of contrast to compute.
        save_path (str): Path to save the report.
    Returns:
        second_level_model (SecondLevelModel): Fitted SecondLevelModel object.
        second_level_contrast (nibabel.Nifti1Image): Contrast image for the second-level analysis.
    """
    design_matrix = pd.DataFrame({"intercept": [1] * len(contrast_maps)})
    second_level_model = SecondLevelModel().fit(contrast_maps, design_matrix=design_matrix)
    second_level_contrast = second_level_model.compute_contrast(second_level_contrast='intercept', output_type='z_score')    
    report = make_glm_report(second_level_model, 'intercept', height_control=height_control, alpha=alpha)
    report.save_as_html(save_path)
    print(f"Saved second-level GLM report to {save_path}")
    return second_level_model, second_level_contrast

def compute_pre_post_contrast(pre_contrast, post_contrast):
    """
    Compute the difference between post and pre contrast images.
    Args:
        pre_contrast (nibabel.Nifti1Image): Pre contrast image.
        post_contrast (nibabel.Nifti1Image): Post contrast image.
    Returns:
        diff_contrast (nibabel.Nifti1Image): Difference image (post - pre).
    """
    post_resampled = image.resample_to_img(post_contrast, pre_contrast, interpolation='nearest', force_resample=True, copy_header=True)
    diff_contrast = image.math_img("post - pre", post=post_resampled, pre=pre_contrast)
    return diff_contrast
