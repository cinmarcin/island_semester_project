
from contrast_types import ContrastType
import os
from rsa import extract_beta_maps, compute_rsa
from tqdm import tqdm
from joblib import load
from glm import get_old_trials_idx
import numpy as np

## Controls
CONTRAST_TYPE = ContrastType.ORDER
SMOOTHING_FWHM = None # in mm
HIPPOCAMPUS_ONLY = True  # If True, restrict analysis to hippocampus only
NB_REPETITIONS = 6  # Number of repetitions during picture-viewing task

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
        #     #    'sub-P20', Unattentive
        #         'sub-P21',
        #         'sub-P22',
        #         'sub-P23',
        #         'sub-P25',
        # #        'sub-P26', missing event file
        #         'sub-P27',
        # #        'sub-P30', incomplete data
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
        # #        'sub-P49',   Unattentive
        #         'sub-P50',
        #         'sub-P51',
                ]

DATA_INPUT_PATH = f'data/processed/glm/{"order_hippocampus" if HIPPOCAMPUS_ONLY else "order"}'
DATA_OUTPUT_PATH = f'data/processed/rsa/'
FIGURE_OUTPUT_PATH = 'figures/rsa'
if not os.path.exists(FIGURE_OUTPUT_PATH):
    os.makedirs(FIGURE_OUTPUT_PATH)

for sub in tqdm(list_of_subs):
    print(f"--- Processing subject: {sub}")
    SUBJECT_FOLDER_IN = os.path.join(DATA_INPUT_PATH, sub)
    SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, sub)
    MODEL_PATH = os.path.join(SUBJECT_FOLDER_IN, f'{sub}_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    if not os.path.exists(SUBJECT_FOLDER):
        os.makedirs(SUBJECT_FOLDER)

    fmri_glm = load(MODEL_PATH)
    print(f"Loaded GLM model from {MODEL_PATH}")

    # Get only old indexs only pre is necessary as they are the same for pre and post
    old_trials_idx = get_old_trials_idx(sub)
    
    # sub-sample for test purposes
    old_trials_idx = old_trials_idx[::48]  # using only the 6 repetitions of the first trial
    print(f"Using {old_trials_idx} ")
    # Extract beta maps for old trials only
    pre_beta_maps = extract_beta_maps(fmri_glm, idxs=old_trials_idx, run_label='pre')
    # post_beta_maps = extract_beta_maps(fmri_glm, idxs=old_trials_idx, run_label='post')

    # Create 2D version of beta maps indexes (n_rep, n_trials)
    beta_maps_idxs = np.arange(len(old_trials_idx)).reshape((NB_REPETITIONS, -1))

    # Correlate every beta maps with every other beta maps of every other repetition
    pre_rsa = compute_rsa(pre_beta_maps, beta_maps_idxs)
    print(f"Pre RSA shape: {pre_rsa.shape}")