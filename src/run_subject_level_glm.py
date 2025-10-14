import os
from utils.data import load_subject_data, remove_subject_data
from glm import combined_first_level_analysis
from nilearn.reporting import make_glm_report
from nilearn.plotting import plot_design_matrix
from joblib import dump, load
from contrast_types import ContrastType
import argparse
import yaml

## --------- GET ARGUMENTS
parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=str, required=True, help="Subject ID, e.g., sub-P10")
parser.add_argument("--config", type=str, nargs='+', required=True, help="Path(s) to YAML config file(s)")
args = parser.parse_args()

SUB = args.sub
config_files = args.config

# Load YAML
for config_file in config_files:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    CONTRAST_TYPE = ContrastType(config["contrast_type"])
    SMOOTHING_FWHM = config["smoothing_fwhm"]
    HEIGHT_CONTROL = config["height_control"]
    P_VALUE = config["p_value"]
    HIPPOCAMPUS_ONLY = config.get("hippocampus_only", False)

    print(f"Running GLM for {SUB} with config {config_file}")


    pre_ses = 'ses-01'
    post_ses = 'ses-05'
    FIGURE_OUTPUT_PATH = 'figures'
    DATA_OUTPUT_PATH = f"data/processed/glm/{config_file.split('/')[-1].replace('.yaml','')}"
    REPORT_PATH = f"reports/glm/{CONTRAST_TYPE.value}/{config_file.split('/')[-1].replace('.yaml','')}"
    MASK_PATH = f"data/processed/masks/{SUB}/{SUB}_{'hippo_' if HIPPOCAMPUS_ONLY else ''}gm_mask.nii.gz"
    if not os.path.exists(FIGURE_OUTPUT_PATH):
        os.makedirs(FIGURE_OUTPUT_PATH)
    if not os.path.exists(DATA_OUTPUT_PATH):
        os.makedirs(DATA_OUTPUT_PATH)
    if not os.path.exists(REPORT_PATH):
        os.makedirs(REPORT_PATH)
    if not os.path.exists(os.path.dirname(MASK_PATH)):
        os.makedirs(os.path.dirname(MASK_PATH))
    SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, SUB)
    MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{SUB}_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    if not os.path.exists(SUBJECT_FOLDER):
        os.makedirs(SUBJECT_FOLDER)


    ## Data Loading
    if os.path.exists(MODEL_PATH):
        print(f"Loading existing models from {SUBJECT_FOLDER}")
        fmri_glm = load(MODEL_PATH)
    else:
        # Load raw data
        pre_fmri_data, pre_confounds_df, pre_events_df, pre_metadata = load_subject_data(SUB, pre_ses)
        post_fmri_data, post_confounds_df, post_events_df, post_metadata = load_subject_data(SUB, post_ses)

        # Concatenate pre and post 
        fmri_concat = [pre_fmri_data, post_fmri_data]
        confounds_concat = [pre_confounds_df, post_confounds_df]
        events_concat = [pre_events_df.copy(), post_events_df.copy()]
        metadata_concat = [pre_metadata, post_metadata]

        # Run GLM
        fmri_glm, pre_contrast, post_contrast, pre_post_contrast, mask_img = combined_first_level_analysis(fmri_concat, events_concat, metadata_concat, confounds_concat, CONTRAST_TYPE, SMOOTHING_FWHM, HIPPOCAMPUS_ONLY)
        dump(fmri_glm, MODEL_PATH)

        if pre_post_contrast is not None:
            pre_contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'pre_contrast.nii.gz'))
            post_contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'post_contrast.nii.gz'))
            pre_post_contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'pre_post_contrast.nii.gz'))
        mask_img.to_filename(MASK_PATH)
        print(f"Saved combined model to {MODEL_PATH} and contrasts.")


    ## Produce Reports
    SUBJECT_REPORT_FOLDER = os.path.join(REPORT_PATH, SUB)
    if not os.path.exists(SUBJECT_REPORT_FOLDER):
        os.makedirs(SUBJECT_REPORT_FOLDER)

    if CONTRAST_TYPE in {ContrastType.PER_TRIAL, ContrastType.ORDER}:
        # only save design matrix plot for per_trial and order
        design_matrix = fmri_glm.design_matrices_[0]
        fig_path = os.path.join(SUBJECT_REPORT_FOLDER, f'{SUB}_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}design_matrix.png')
        plot_design_matrix(design_matrix, output_file=fig_path)
        print(f"Saved design matrix figure to {SUBJECT_REPORT_FOLDER}")
    else:
        contrast_vec = CONTRAST_TYPE.get_vector(fmri_glm.design_matrices_[0], combined=True)

        report = make_glm_report(fmri_glm, contrast_vec, height_control=HEIGHT_CONTROL, alpha=P_VALUE)
        report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f'{SUB}_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}glm_report.html'))
        print(f"Saved GLM report to {SUBJECT_REPORT_FOLDER}")


## Clean up data to save space
remove_subject_data(SUB)