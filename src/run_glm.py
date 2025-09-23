import os
from utils.data import load_subject_data
from glm import second_level_analysis, combined_first_level_analysis
from nilearn.reporting import make_glm_report
from nilearn.glm.second_level import SecondLevelModel
from joblib import dump, load
from contrast_types import ContrastType
from tqdm import tqdm
import pandas as pd 

## Controls
CONTRAST_TYPE = ContrastType.ORDER
SMOOTHING_FWHM = None # in mm
HEIGHT_CONTROL = "bonferroni"  # "fdr" or "bonferroni"
P_VALUE = 0.05

## Paths
list_of_subs = [
                'sub-P10',
        #         'sub-P12',
        #         'sub-P13',
        #         'sub-P14',
        #         'sub-P15',
        #         'sub-P16',
        #         'sub-P17',
        #         'sub-P18',
        #         'sub-P19',
        #         'sub-P20',
        #         'sub-P21',
        #         'sub-P22',
        #         'sub-P23',
        #         'sub-P25',
        # #        'sub-P26', missing event file
        #         'sub-P27',
        #         'sub-P30',
        #         'sub-P31',
        #         'sub-P34',
        #         'sub-P35',
        #         'sub-P36',
        #         'sub-P37',
        #         'sub-P41',
        #         'sub-P42',
        #         'sub-P43',
        #         'sub-P44',
        #         'sub-P45',
        #         'sub-P47',
        #         'sub-P48',
        #         'sub-P49',
        #         'sub-P50',
        #         'sub-P51',
                ]
pre_ses = 'ses-01'
post_ses = 'ses-05'
FIGURE_OUTPUT_PATH = '../figures'
DATA_OUTPUT_PATH = f'../data/processed/glm/{CONTRAST_TYPE.value}'
REPORT_PATH = f"../reports/glm/{CONTRAST_TYPE.value}"
SECOND_MODEL_FOLDER = os.path.join(DATA_OUTPUT_PATH, 'second_level')
if not os.path.exists(SECOND_MODEL_FOLDER):
    os.makedirs(SECOND_MODEL_FOLDER)
SECOND_MODEL_PATH = os.path.join(SECOND_MODEL_FOLDER, f'second_level_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_fitted_glm.pkl')
if not os.path.exists(FIGURE_OUTPUT_PATH):
    os.makedirs(FIGURE_OUTPUT_PATH)
if not os.path.exists(DATA_OUTPUT_PATH):
    os.makedirs(DATA_OUTPUT_PATH)

contrast_maps = []

print(f"Starting analysis for {CONTRAST_TYPE} with smoothing FWHM: {SMOOTHING_FWHM} mm, height control: {HEIGHT_CONTROL} for {len(list_of_subs)} subjects.")
for sub in tqdm(list_of_subs):
    print(f"--- Processing subject: {sub}")
    SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, sub)
    MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{sub}_smooth_{SMOOTHING_FWHM}_fitted_glm.pkl')
    if not os.path.exists(SUBJECT_FOLDER):
        os.makedirs(SUBJECT_FOLDER)
    # Check if models already exist
    LOAD_GLM = os.path.exists(MODEL_PATH)

    ## Data Loading
    if LOAD_GLM:
        # Load pre-fitted models
        fmri_glm = load(MODEL_PATH)
        print(f"Loaded pre-fitted GLM model from {MODEL_PATH}")
    else:
        # Load raw data
        pre_fmri_data, pre_confounds_df, pre_events_df, pre_metadata = load_subject_data(sub, pre_ses)
        post_fmri_data, post_confounds_df, post_events_df, post_metadata = load_subject_data(sub, post_ses)
        # Concatenate pre and post 
        fmri_concat = [pre_fmri_data, post_fmri_data]
        confounds_concat =[pre_confounds_df, post_confounds_df]
        events_concat = [pre_events_df, post_events_df]
        metadata_concat = [pre_metadata, post_metadata]

        fmri_glm = combined_first_level_analysis(fmri_concat, events_concat, metadata_concat, confounds_concat, CONTRAST_TYPE, SMOOTHING_FWHM)
        dump(fmri_glm, MODEL_PATH)

        print(f"Saved fitted GLM model to {MODEL_PATH}")


    # Define contrasts
    contrast_vec = CONTRAST_TYPE.get_vector(fmri_glm.design_matrices_[0])
    # Generate and save report

    SUBJECT_REPORT_FOLDER = os.path.join(REPORT_PATH, sub)
    if not os.path.exists(SUBJECT_REPORT_FOLDER):
        os.makedirs(SUBJECT_REPORT_FOLDER)
    report = make_glm_report(fmri_glm, contrast_vec, height_control=HEIGHT_CONTROL, alpha=P_VALUE)
    report.save_as_html(os.path.join(SUBJECT_REPORT_FOLDER, f"{sub}_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))
    print(f"Saved GLM report to {SUBJECT_REPORT_FOLDER}")

## Second Level Analysis
print("--- Starting second-level analysis...")

second_level_model, second_level_contrast = second_level_analysis(contrast_maps, HEIGHT_CONTROL, P_VALUE,
                                                                          os.path.join(REPORT_PATH, f"second_level_pre_session_smooth_{SMOOTHING_FWHM}_{HEIGHT_CONTROL}_{CONTRAST_TYPE}_glm_report.html"))
dump(second_level_model, SECOND_MODEL_PATH)

print(f"Saved second-level GLM model to {SECOND_MODEL_PATH}")