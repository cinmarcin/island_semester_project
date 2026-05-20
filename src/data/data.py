import os
import nibabel as nib
import pandas as pd
import inspect
import json
import numpy as np

from src.data.paths import get_processed_data_file_path, get_model_rdm_path
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

## Paths
FMRI_PREP_OUTPUT_PATH = 'data/fmriprep_output'
EVENTS_PATH = 'data/bids_ackbar'

def load_subject_data(sub, ses):
    """
    Load fMRI data and confounds for a given subject and session. The first and last sessions are viewing tasks while the 3 middle sones are encoding sessions.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
        ses (str): Session identifier (e.g., 'ses-01').
    Returns:
        fmri_data (numpy.ndarray): 4D fMRI data array.
        confounds_df (pandas.DataFrame): DataFrame containing confound time series.
        events_df (pandas.DataFrame): DataFrame containing event information.
        metadata (dict): Metadata from the JSON sidecar file.
    """
    # Define file paths
    # check the number of the session selected
    if ses not in ['ses-01', 'ses-05']:
        task = 'task-encoding'
        raise NotImplementedError("Session not recognized. Only 'ses-01' and 'ses-05' are implemented.")
    else:
        task = 'task-viewing'
        run = 'run-01'
    


    func_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    confounds_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_desc-confounds_timeseries.tsv')
    events_file = os.path.join(EVENTS_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_events.tsv')
    meta_data_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.json')

    required_files = [func_file, confounds_file, events_file, meta_data_file]
    missing = [f for f in required_files if not os.path.exists(f)]

    if missing:
        raise FileNotFoundError(
            f"Missing files for {sub}, {ses}:\n" + "\n".join(missing)
        )

    print (f"Loading data for {sub}, {ses}, {task}, {run}...")

    # Load files
    fmri_data = nib.load(func_file)
    confounds_df = pd.read_csv(confounds_file, sep='\t')
    events_df = pd.read_csv(events_file, sep='\t')

    with open(meta_data_file, 'r') as f:
        metadata = json.load(f)


    print(f"Loaded fMRI data shape: {fmri_data.shape}, confounds shape: {confounds_df.shape}, events shape: {events_df.shape}, metadata keys: {list(metadata.keys())}")

    return fmri_data, confounds_df, events_df, metadata

def download_subject_data(sub):

    raise RuntimeError(

        "Automatic download is disabled. Data are accessed directly via "

        "data/bids_ackbar and data/fmriprep_output symlinks."

    )

def remove_subject_data(sub):
    """
    Remove local fMRI data and necessary files for a given subject.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
    """
    local_base_path = os.path.join(FMRI_PREP_OUTPUT_PATH, sub)
    if os.path.exists(local_base_path):
        print(f"Removing local data for {sub} at {local_base_path}...")
        os.system(f"rm -rf {local_base_path}")
        print(f"Removed local data for {sub}.")
    else:
        print(f"No local data found for {sub} at {local_base_path}.")

def download_event_file(sub, session):
    raise RuntimeError(
        "Automatic event download is disabled. Events are accessed directly via "
        "data/bids_ackbar."
    )


def get_model_rdm(sub, config, config_file, spatial=False):
    """
    Load the model RDM (Representational Dissimilarity Matrix) for a given subject.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
        spatial (bool): Whether to compute the spatial RDM. If False, computes temporal RDM.
        config (dict): Configuration dictionary.
        config_file (str): Path to the configuration file.
    Returns:
        rdm (numpy.ndarray): The computed or loaded RDM.
    """
    return np.load(get_model_rdm_path(sub=sub, spatial=spatial, config_file=config_file, config=config))

def get_rsa_matrix(sub, config, config_file, run_label, rep, euclidean=False):
    """
    Load the RSA matrix for a given subject.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
        config (dict): Configuration dictionary.
        config_file (str): Path to the configuration file.
        run_label (str): Label for the run/session (e.g., 'pre', 'post').
        rep (str): Repetition label (e.g., 'all_reps', 'rep1', etc.).
        euclidean (bool): Whether to use Euclidean distance instead of correlation.
    Returns:
        rsa_matrix (numpy.ndarray): The RSA matrix.
    """
    path = str(get_processed_data_file_path(config_file=config_file, analysis='rsa', sub=sub, config=config, session=run_label)).replace('.npy', f'_{rep}.npy')
    if euclidean:
        path = path.replace('.npy', '_euclidean.npy')
    return np.load(path)
