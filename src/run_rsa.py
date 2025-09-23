import os
from pkgutil import get_data
from joblib import dump, load
from contrast_types import ContrastType
from tqdm import tqdm
from rsa import create_index_rdm
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform


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

DATA_INPUT_PATH = f'../data/processed/glm/{CONTRAST_TYPE.value}'

for sub in tqdm(list_of_subs):
    print(f"--- Processing subject: {sub}")
    SUBJECT_FOLDER = os.path.join(DATA_INPUT_PATH, sub)
    GLM_MODEL_PATH = os.path.join(SUBJECT_FOLDER, f'{sub}_smooth_{SMOOTHING_FWHM}_fitted_glm.pkl')

    fmri_glm = load(GLM_MODEL_PATH)
    print(f"Loaded GLM model from {GLM_MODEL_PATH}")

    # Extract beta maps
    print("Extracting beta maps...")
    beta_maps = fmri_glm.compute_contrast(
    np.eye(len(fmri_glm.design_matrices_[0].columns)),  
    output_type="effect_size"
    )
    print(f"Extracted {len(beta_maps)} beta maps.")
    # Flatten beta maps into vectors (voxels × trials)
    trial_data = []
    for beta_map in tqdm(beta_maps):
        trial_data.append(get_data(beta_map).ravel())
    trial_data = np.array(trial_data)   # shape = (384, n_voxels)
    print(f"Extracted and flattened beta maps. Shape of trial data: {trial_data.shape}")

    # Compute similarity matrix
    print("Computing similarity matrix...")
    similarity_matrix = 1 - squareform(pdist(trial_data, metric='correlation'))
    # Save similarity matrix
    similarity_matrix_path = os.path.join(SUBJECT_FOLDER, f'{sub}_similarity_matrix.npy')
    dump(similarity_matrix, similarity_matrix_path)
    print(f"Saved similarity matrix to {similarity_matrix_path}")

    # Create RDM
    n_trials = 64 
    repetitions = 6
    index_rdm = create_index_rdm(n_trials, repetitions, relative_distance=False, save_fig=True)


