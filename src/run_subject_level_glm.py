import os
from utils.data import load_subject_data
from glm import combined_first_level_analysis, first_level_analysis, pre_vs_post_second_level_analysis
from nilearn.reporting import make_glm_report
from joblib import dump, load
from contrast_types import ContrastType
import argparse
import yaml

## Controls
parser = argparse.ArgumentParser()
parser.add_argument("--sub", type=str, required=True, help="Subject ID, e.g., sub-P10")
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

SUB = args.sub
config_file = args.config

# Load YAML
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

CONTRAST_TYPE = ContrastType(config["contrast_type"])
SMOOTHING_FWHM = config["smoothing_fwhm"]
HEIGHT_CONTROL = config["height_control"]
P_VALUE = config["p_value"]

print(f"Running GLM for {SUB} with config {config_file}")


pre_ses = 'ses-01'
post_ses = 'ses-05'
FIGURE_OUTPUT_PATH = 'figures'
DATA_OUTPUT_PATH = f'data/processed/glm/{CONTRAST_TYPE.value}'
REPORT_PATH = f"reports/glm/{CONTRAST_TYPE.value}"
if not os.path.exists(FIGURE_OUTPUT_PATH):
    os.makedirs(FIGURE_OUTPUT_PATH)
if not os.path.exists(DATA_OUTPUT_PATH):
    os.makedirs(DATA_OUTPUT_PATH)
SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, SUB)
MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{SUB}_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_fitted_glm.pkl')
PRE_MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{SUB}_pre_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_fitted_glm.pkl')
POST_MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{SUB}_post_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_fitted_glm.pkl')
SECOND_LEVEL_MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{SUB}_second_level_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_fitted_glm.pkl')
if not os.path.exists(SUBJECT_FOLDER):
    os.makedirs(SUBJECT_FOLDER)
LOAD_MODELS = os.path.exists(MODEL_PATH) and os.path.exists(PRE_MODEL_PATH) and os.path.exists(POST_MODEL_PATH) and os.path.exists(SECOND_LEVEL_MODEL_PATH)

## Data Loading
if LOAD_MODELS:
    print(f"Loading existing models from {SUBJECT_FOLDER}")
    fmri_glm = load(MODEL_PATH)
    pre_glm = load(PRE_MODEL_PATH)
    post_glm = load(POST_MODEL_PATH)
    second_level_model = load(SECOND_LEVEL_MODEL_PATH)
else:
    # Load raw data
    pre_fmri_data, pre_confounds_df, pre_events_df, pre_metadata = load_subject_data(SUB, pre_ses)
    post_fmri_data, post_confounds_df, post_events_df, post_metadata = load_subject_data(SUB, post_ses)
    
    # Concatenate pre and post 
    fmri_concat = [pre_fmri_data, post_fmri_data]
    confounds_concat =[pre_confounds_df, post_confounds_df]
    events_concat = [pre_events_df, post_events_df]
    metadata_concat = [pre_metadata, post_metadata]
    
    ## COMBINED Analysis
    fmri_glm, contrast, csf_contrast, wm_contrast = combined_first_level_analysis(fmri_concat, events_concat, metadata_concat, confounds_concat, CONTRAST_TYPE, SMOOTHING_FWHM)
    dump(fmri_glm, MODEL_PATH)
    contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'contrast.nii.gz'))
    csf_contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'csf_contrast.nii.gz'))
    wm_contrast.to_filename(MODEL_PATH.replace('fitted_glm.pkl', 'wm_contrast.nii.gz'))
    print(f"Saved combined model to {MODEL_PATH}")

    ## SEPARATE Analysis
    pre_glm, pre_contrast, pre_csf_contrast, pre_wm_contrast = first_level_analysis(pre_fmri_data, pre_events_df, pre_metadata, pre_confounds_df, CONTRAST_TYPE, SMOOTHING_FWHM, title=f"Pre Session")
    post_glm, post_contrast, post_csf_contrast, post_wm_contrast = first_level_analysis(post_fmri_data, post_events_df, post_metadata, post_confounds_df, CONTRAST_TYPE, SMOOTHING_FWHM, title=f"Post Session")

    dump(pre_glm, PRE_MODEL_PATH)
    pre_contrast.to_filename(PRE_MODEL_PATH.replace('fitted_glm.pkl', 'contrast.nii.gz'))
    pre_csf_contrast.to_filename(PRE_MODEL_PATH.replace('fitted_glm.pkl', 'csf_contrast.nii.gz'))
    pre_wm_contrast.to_filename(PRE_MODEL_PATH.replace('fitted_glm.pkl', 'wm_contrast.nii.gz'))
    dump(post_glm, POST_MODEL_PATH)
    post_contrast.to_filename(POST_MODEL_PATH.replace('fitted_glm.pkl', 'contrast.nii.gz'))
    post_csf_contrast.to_filename(POST_MODEL_PATH.replace('fitted_glm.pkl', 'csf_contrast.nii.gz'))
    post_wm_contrast.to_filename(POST_MODEL_PATH.replace('fitted_glm.pkl', 'wm_contrast.nii.gz'))
    print(f"Saved separate models to {PRE_MODEL_PATH} and {POST_MODEL_PATH}")

    ## SECOND LEVEL - PRE vs POST
    second_level_model, second_level_contrast = pre_vs_post_second_level_analysis(pre_contrast, post_contrast)
    dump(second_level_model, SECOND_LEVEL_MODEL_PATH)
    second_level_contrast.to_filename(SECOND_LEVEL_MODEL_PATH.replace('fitted_glm.pkl', 'contrast.nii.gz'))
    print(f"Saved second-level model to {SECOND_LEVEL_MODEL_PATH}")


## Produce Reports

# COMBINED
contrast_vec = CONTRAST_TYPE.get_vector(fmri_glm.design_matrices_[0], combined=True)
SUBJECT_REPORT_FOLDER = os.path.join(REPORT_PATH, SUB)
if not os.path.exists(SUBJECT_REPORT_FOLDER):
    os.makedirs(SUBJECT_REPORT_FOLDER)
report = make_glm_report(fmri_glm, contrast_vec, height_control=HEIGHT_CONTROL, alpha=P_VALUE)
report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f"{SUB}_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))
print(f"Saved GLM report to {SUBJECT_REPORT_FOLDER}")


# SEPARATE
pre_report = make_glm_report(pre_glm, CONTRAST_TYPE.get_vector(pre_glm.design_matrices_[0]), title="Pre Session", height_control=HEIGHT_CONTROL, alpha=P_VALUE)
pre_report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f"{SUB}_pre_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))

post_report = make_glm_report(post_glm, CONTRAST_TYPE.get_vector(post_glm.design_matrices_[0]), title="Post Session", height_control=HEIGHT_CONTROL, alpha=P_VALUE)
post_report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f"{SUB}_post_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))

# SECOND LEVEL
pre_vs_post_report = make_glm_report(second_level_model, 'session', height_control=HEIGHT_CONTROL, alpha=P_VALUE)
pre_vs_post_report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f"{SUB}_second_level_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))
print(f"Saved second-level pre vs post GLM report to {SUBJECT_REPORT_FOLDER}")