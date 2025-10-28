from pathlib import Path
import os

# Level 0

# we are in island_semester_project/src/data/paths.py we want to go three levels up
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Level 1
DATA_DIR = BASE_DIR / 'data'
FIGURES_DIR = BASE_DIR / 'figures'
REPORTS_DIR = BASE_DIR / 'reports'
CONFIGS_DIR = BASE_DIR / 'configs'

# Level 2
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
RAW_DATA_DIR = DATA_DIR / 'raw'
PREPROCESSED_DATA_DIR = DATA_DIR / 'preprocessed'

# Level 3
GLM_DATA_DIR = PROCESSED_DATA_DIR / 'glm'
RSA_DATA_DIR = PROCESSED_DATA_DIR / 'rsa'
MASKS_DATA_DIR = PROCESSED_DATA_DIR / 'masks'


## Others
PRE_SES = 'ses-01'
POST_SES = 'ses-05'

def get_processed_data_folder(config_file: str, analysis: str) -> Path:
    if '/' in config_file:
        config_file = config_file.split('/')[-1]
    if analysis not in ['glm', 'rsa', 'masks', 'contrasts', 'reports', 'design_matrix']:
        raise ValueError("Analysis must be either 'glm', 'rsa', 'masks', 'contrasts' or 'reports'")
    if analysis == 'glm' or analysis == 'contrasts':
        folder_path = GLM_DATA_DIR / config_file.replace('.yaml', '')
    elif analysis == 'rsa':
        folder_path = RSA_DATA_DIR / config_file.replace('.yaml', '')
    elif analysis == 'reports' or analysis == 'design_matrix':
        folder_path = REPORTS_DIR / config_file.replace('.yaml', '')
    elif analysis == 'masks':
        folder_path = MASKS_DATA_DIR / config_file.replace('.yaml', '')

    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_processed_data_subject_folder(config_file: str, analysis: str, sub: str) -> Path:
    folder_path = get_processed_data_folder(config_file, analysis) / sub
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_processed_data_file_path(config_file: str, analysis: str, sub: str, config, session: str = '', height_control=None) -> Path:
    subject_folder = get_processed_data_subject_folder(config_file, analysis, sub)

    if analysis == 'glm':
        filename = f"{get_file_basename_from_config(sub, session, config)}_fitted_glm.pkl"
    elif analysis == 'rsa':
        filename = f"{get_file_basename_from_config(sub, session, config)}_rsa.npy"
    elif analysis == 'contrasts':
        filename = f"{get_file_basename_from_config(sub, session, config)}_contrast.nii.gz"
    elif analysis == 'masks':
        if session is not None:
            raise ValueError("Session must be None when analysis is 'masks' since masks are not session-specific.")
        filename = f"{sub}_{'hippo_' if config.get('hippocampus_only', False) else ''}gm_mask.nii.gz"
    elif analysis == 'reports':
        filename = f"{get_file_basename_from_config(sub, session, config, height_control)}_glm_report.html"
    elif analysis == 'design_matrix':
        filename = f"{get_file_basename_from_config(sub, session, config)}_design_matrix.png"
    else:
        raise ValueError(f"Unknown analysis type: {analysis}")

    return subject_folder / filename


def get_file_basename_from_config(sub: str, session: str, config, height_control=None) -> Path:
    if height_control is not None:
        height_control_str = f"_{height_control}"
    else:
        height_control_str = ""
    return Path(f"{sub}_smooth_{config['smoothing_fwhm']}{height_control_str}_{config['contrast_type']}_{'hippo_' if config.get('hippocampus_only', False) else ''}{session}")


def is_glm_computed(config_file: str, sub: str, config) -> bool:
    glm_file = get_processed_data_file_path(config_file=config_file, analysis='glm', sub=sub, config=config)
    return glm_file.exists()


def get_raw_memory_data_file_path() -> Path:
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    return RAW_DATA_DIR / 'final_merged_data.csv'

def get_preprocessed_memory_data_file_path() -> Path:
    os.makedirs(PREPROCESSED_DATA_DIR, exist_ok=True)
    return PREPROCESSED_DATA_DIR / 'memory_preprocessed.csv'

def get_config_folder() -> Path:
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    return CONFIGS_DIR


def get_model_rdm_path(sub: str, spatial: bool, config_file: str, config) -> Path:
    sub_rsa_path = get_processed_data_file_path(config_file, 'rsa', sub, config)
    rdm_type = 'spatial' if spatial else 'temporal'

    # Convert to string before replacing
    sub_rsa_str = str(sub_rsa_path)
    model_rdm_path = sub_rsa_str.replace('_rsa.npy', f'_{rdm_type}_model_rdm.npy')

    return Path(model_rdm_path)

def get_neuronal_rdm_path(sub: str, config_file: str, config) -> Path:
    sub_rsa_path = get_processed_data_file_path(config_file, 'rsa', sub, config, session='pre-post')

    return Path(sub_rsa_path)

import re
from pathlib import Path

def get_next_run_id(base_folder, base_name="group_level_rsa_results_run", ext=".json"):
    base_folder = Path(base_folder)
    base_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all existing runs
    pattern = re.compile(rf"{re.escape(base_name)}(\d+){re.escape(ext)}$")
    existing_runs = []
    for f in base_folder.glob(f"{base_name}*{ext}"):
        match = pattern.match(f.name)
        if match:
            existing_runs.append(int(match.group(1)))
    
    next_run = max(existing_runs, default=-1) + 1
    return base_folder / f"{base_name}{next_run}{ext}"
