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
FMRI_PREP_OUTPUT_PATH = 'data/preprocessed/'
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
    


    func_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz')
    confounds_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_desc-confounds_timeseries.tsv')
    events_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_events.tsv')
    meta_data_file = os.path.join(FMRI_PREP_OUTPUT_PATH, sub, ses, 'func', f'{sub}_{ses}_{task}_{run}_space-MNI152NLin2009cAsym_desc-preproc_bold.json')

    if not os.path.exists(func_file) or not os.path.exists(confounds_file) or not os.path.exists(events_file) or not os.path.exists(meta_data_file):
        print(f"Data files for {sub}, {ses} not found locally. Downloading...")
        download_subject_data(sub)

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
    """
    Download fMRI data and necessary files using ssh for a given subject,
    separating them in two sessions so that they are copied in the correct folder.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
    """
    remote_base_path = f"boesch@miplabsrv3:/media/RCPNAS/Data3/Alison_island/"
    remote_fmri_path = os.path.join(remote_base_path, f"new_data/fmriprep_output/{sub}/")
    remote_events_path = os.path.join(remote_base_path, f"bids_ackbar/{sub}/")

    local_base_path = os.path.join(FMRI_PREP_OUTPUT_PATH, sub)
    os.makedirs(local_base_path, exist_ok=True)

    sessions = ['ses-01', 'ses-05']
    fmri_suffixes = [
        "space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "desc-confounds_timeseries.tsv",
        "space-MNI152NLin2009cAsym_desc-preproc_bold.json"
    ]

    for ses in sessions:
        # Create local folder for this session
        local_ses_func = os.path.join(local_base_path, ses, "func")
        os.makedirs(local_ses_func, exist_ok=True)

        # Remote fMRI files
        remote_files = [
            os.path.join(remote_fmri_path, ses, "func", f"{sub}_{ses}_task-viewing_run-01_{suffix}")
            for suffix in fmri_suffixes
        ]

        # Remote events file
        remote_events_file = os.path.join(remote_events_path, ses, "func", f"{sub}_{ses}_task-viewing_run-01_events.tsv")
        remote_files.append(remote_events_file)

        # Build single SCP command for this session
        scp_command = f"scp {' '.join(remote_files)} {local_ses_func}/"
        print(f"Executing command: {scp_command}")
        os.system(scp_command)

    print(f"Downloaded all files for {sub} to {local_base_path}.")

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
    """
    Download event file using ssh for a given subject and session.
    Args:
        sub (str): Subject identifier (e.g., 'sub-P10').
        session (str): Session identifier (e.g., 'ses-01').
    """
    remote_base_path = f"boesch@miplabsrv3:/media/RCPNAS/Data3/Alison_island/bids_ackbar/"
    remote_events_file = os.path.join(remote_base_path, f"{sub}/{session}/func/{sub}_{session}_task-memory_events.tsv")

    local_base_path = os.path.join('data/raw', sub, 'func')
    os.makedirs(local_base_path, exist_ok=True)

    scp_command = f"scp {remote_events_file} {local_base_path}/"
    print(f"Executing command: {scp_command}")
    os.system(scp_command)
    print(f"Downloaded event file for {sub}, {session} to {local_base_path}.")
    return os.path.join(local_base_path, f"{sub}_{session}_task-memory_events.tsv")
