
from contrast_types import ContrastType
import os
from rsa import extract_beta_maps
from tqdm import tqdm
from joblib import load
from glm import get_old_trials_idx

## Controls
CONTRAST_TYPE = ContrastType.ORDER
SMOOTHING_FWHM = None # in mm
HIPPOCAMPUS_ONLY = True  # If True, restrict analysis to hippocampus only

list_of_subs = [
                'sub-P10',
                'sub-P12',
                'sub-P13',
                'sub-P14',
                'sub-P15',
                'sub-P16',
                'sub-P17',
                'sub-P18',
                'sub-P19',
            #    'sub-P20', Unattentive
                'sub-P21',
                'sub-P22',
                'sub-P23',
                'sub-P25',
        #        'sub-P26', missing event file
                'sub-P27',
        #        'sub-P30', incomplete data
                'sub-P31',
                'sub-P34',
                'sub-P35',
                'sub-P36',
                'sub-P37',
                'sub-P41',
                'sub-P42',
                'sub-P43',
                'sub-P44',
                'sub-P45',
                'sub-P47',
                'sub-P48',
        #        'sub-P49',   Unattentive
                'sub-P50',
                'sub-P51',
                ]

DATA_INPUT_PATH = f'data/processed/glm/{CONTRAST_TYPE.value}'
DATA_OUTPUT_PATH = f'data/processed/rsa/'
FIGURE_OUTPUT_PATH = 'figures/rsa'
if not os.path.exists(FIGURE_OUTPUT_PATH):
    os.makedirs(FIGURE_OUTPUT_PATH)

for sub in tqdm(list_of_subs):
    print(f"--- Processing subject: {sub}")
    SUBJECT_FOLDER_IN = os.path.join(DATA_INPUT_PATH, sub)
    SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, sub)
    PRE_GLM_MODEL_PATH = os.path.join(SUBJECT_FOLDER_IN, f'{sub}_pre_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    POST_GLM_MODEL_PATH = os.path.join(SUBJECT_FOLDER_IN, f'{sub}_post_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    if not os.path.exists(SUBJECT_FOLDER):
        os.makedirs(SUBJECT_FOLDER)

    pre_fmri_glm = load(PRE_GLM_MODEL_PATH)
    post_fmri_glm = load(POST_GLM_MODEL_PATH)
    print(f"Loaded GLM models from {PRE_GLM_MODEL_PATH} and {POST_GLM_MODEL_PATH}")

    # Get only old indexs
    pre_old_trials_idx = get_old_trials_idx(sub, 'ses-01')
    post_old_trials_idx = get_old_trials_idx(sub, 'ses-05')

    pre_trial_data = extract_beta_maps(pre_fmri_glm)
    post_trial_data = extract_beta_maps(post_fmri_glm)

    # Reshape to (repetitions, tasks, n_voxels)
    pre_reshaped = pre_trial_data.reshape(6, 48, -1)
    post_reshaped = post_trial_data.reshape(6, 48, -1)

    # Average across repetitions TODO: we need to improve this processing
    pre_avg = pre_reshaped.mean(axis=0)
    post_avg = post_reshaped.mean(axis=0)