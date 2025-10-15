import os
from utils.data import load_subject_data, remove_subject_data
from glm import combined_first_level_analysis
from rsa import compute_rsa_from_glm, get_model_rdm
from nilearn.reporting import make_glm_report
from nilearn.plotting import plot_design_matrix
from joblib import dump, load
from contrast_types import ContrastType
import argparse
import yaml
from paths import get_processed_data_file_path, get_processed_data_subject_folder, is_glm_computed, PRE_SES, POST_SES, get_model_rdm_path

## --------- GET ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=str, required=True, help="Subject ID, e.g., sub-P10")
parser.add_argument("--config", type=str, nargs='+', required=True, help="Path(s) to YAML config file(s)")
args = parser.parse_args()
SUB = args.sub
config_files = args.config

# --------- RUN GLM FOR EACH CONFIG
for config_file in config_files:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Extract config parameters
    CONTRAST_TYPE = ContrastType(config["contrast_type"])
    SMOOTHING_FWHM = config["smoothing_fwhm"]
    HEIGHT_CONTROL = config["height_control"]
    P_VALUE = config["p_value"]
    HIPPOCAMPUS_ONLY = config.get("hippocampus_only", False)
    SAVE_RSA = config.get("save_rsa", False)

    print(f"Running GLM for {SUB} with config {config_file}")
    if SAVE_RSA:
        print("RSA saving enabled, GLM models will not be saved to save space.")

    ## Data Loading
    if is_glm_computed(config_file=config_file, sub=SUB, config=config):
        print(f"Loading existing models from {get_processed_data_subject_folder(config_file=config_file, analysis='glm', sub=SUB)}")
        fmri_glm = load(get_processed_data_file_path(config_file=config_file, analysis='glm', sub=SUB, config=config))
    else:
        # Load raw data
        pre_fmri_data, pre_confounds_df, pre_events_df, pre_metadata = load_subject_data(SUB, PRE_SES)
        post_fmri_data, post_confounds_df, post_events_df, post_metadata = load_subject_data(SUB, POST_SES)

        # Concatenate pre and post 
        fmri_concat = [pre_fmri_data, post_fmri_data]
        confounds_concat = [pre_confounds_df, post_confounds_df]
        events_concat = [pre_events_df.copy(), post_events_df.copy()]
        metadata_concat = [pre_metadata, post_metadata]

        # Run GLM
        fmri_glm, pre_contrast, post_contrast, pre_post_contrast, mask_img = combined_first_level_analysis(fmri_concat, events_concat, metadata_concat, confounds_concat, CONTRAST_TYPE, SMOOTHING_FWHM, HIPPOCAMPUS_ONLY)
        
        # Save results
        if SAVE_RSA:
            temporal_rdm, filtered_trials = get_model_rdm(SUB, spatial=False, min_confidence=0, save_path=get_model_rdm_path(sub=SUB, spatial=False, config_file=config_file, config=config))
            spatial_rdm, _ = get_model_rdm(SUB, spatial=True, min_confidence=2, save_path=get_model_rdm_path(sub=SUB, spatial=True, config_file=config_file, config=config))
            pre_rsa, post_rsa = compute_rsa_from_glm(fmri_glm, SUB, filtered_trials, pre_save_path=get_processed_data_file_path(config_file=config_file, analysis='rsa', sub=SUB, config=config, session='pre'), n_repetitions=6)
        else:
            dump(fmri_glm, get_processed_data_file_path(config_file=config_file, analysis='glm', sub=SUB, config=config))

        # Save contrasts only if they exist (not the case for some contrast types e.g. ORDER)
        if pre_post_contrast is not None:
            pre_contrast.to_filename(get_processed_data_file_path(config_file=config_file, analysis='contrasts', sub=SUB, config=config, session='pre'))
            post_contrast.to_filename(get_processed_data_file_path(config_file=config_file, analysis='contrasts', sub=SUB, config=config, session='post'))
            pre_post_contrast.to_filename(get_processed_data_file_path(config_file=config_file, analysis='contrasts', sub=SUB, config=config, session='pre-post'))

        # Save mask for later group analysis
        mask_img.to_filename(get_processed_data_file_path(config_file=config_file, analysis='masks', sub=SUB, config=config, session=None))
        print(f"Saved combined model to {get_processed_data_file_path(config_file=config_file, analysis='glm', sub=SUB, config=config)} and contrasts.")

    ## Generate and save reports
    if CONTRAST_TYPE in {ContrastType.PER_TRIAL, ContrastType.ORDER}:
        # only save design matrix plot for per_trial and order
        design_matrix = fmri_glm.design_matrices_[0]
        plot_design_matrix(design_matrix, output_file=get_processed_data_file_path(config_file=config_file, analysis='design_matrix', sub=SUB, config=config))
        print(f"Saved design matrix figure to {get_processed_data_subject_folder(config_file=config_file, analysis='design_matrix', sub=SUB)}")
    else:
        contrast_vec = CONTRAST_TYPE.get_vector(fmri_glm.design_matrices_[0], combined=True)

        report = make_glm_report(fmri_glm, contrast_vec, height_control=HEIGHT_CONTROL, alpha=P_VALUE)
        report.save_as_html(get_processed_data_file_path(config_file=config_file, analysis='reports', sub=SUB, config=config, height_control=HEIGHT_CONTROL))
        print(f"Saved GLM report to {get_processed_data_subject_folder(config_file=config_file, analysis='reports', sub=SUB)}")


## Clean up data to save space
remove_subject_data(SUB)