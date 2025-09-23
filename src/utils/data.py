import os
import nibabel as nib
import pandas as pd
import builtins
import inspect
import json

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
FMRI_PREP_OUTPUT_PATH = '../../../../media/RCPNAS/Data3/Alison_island/new_data/fmriprep_output'
EVENTS_PATH = '../../../../media/RCPNAS/Data3/Alison_island/bids_ackbar'

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

    print (f"Loading data for {sub}, {ses}, {task}, {run}...")

    func_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    confounds_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_desc-confounds_timeseries.tsv')
    events_file = os.path.join(EVENTS_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_events.tsv')
    meta_data_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.json')

    # Load files
    fmri_data = nib.load(func_file)
    confounds_df = pd.read_csv(confounds_file, sep='\t')
    events_df = pd.read_csv(events_file, sep='\t')

    with open(meta_data_file, 'r') as f:
        metadata = json.load(f)


    print(f"Loaded fMRI data shape: {fmri_data.shape}, confounds shape: {confounds_df.shape}, events shape: {events_df.shape}, metadata keys: {list(metadata.keys())}")

    return fmri_data, confounds_df, events_df, metadata