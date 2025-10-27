import os
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.reporting import make_glm_report
from nilearn import image
import argparse
import yaml

from src.core.glm import create_group_mask
from src.core.contrast_types import ContrastType



## Controls
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
args = parser.parse_args()

config_file = args.config

# Load YAML
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

CONTRAST_TYPE = ContrastType(config["contrast_type"])
SMOOTHING_FWHM = config["smoothing_fwhm"]
HIPPOCAMPUS_ONLY = config.get("hippocampus_only", False)

# Redefine height control and p-value for group level
HEIGHT_CONTROLS_AND_P_VALUES = {"fdr": 0.05,
                               "bonferroni": 0.05,
                               "fpr": 0.001}

CONTRAST_PATH_ENDINGS= [f'_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}pre_contrast.nii.gz',
                        f'_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}post_contrast.nii.gz',
                        f'_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}pre_post_contrast.nii.gz']
DATA_PATH = f"data/processed/glm/{config_file.split('/')[-1].replace('.yaml','')}"
REPORT_PATH = f"reports/glm/{CONTRAST_TYPE.value}/{config_file.split('/')[-1].replace('.yaml','')}/group_level"
if not os.path.exists(REPORT_PATH):
    os.makedirs(REPORT_PATH)

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
            #    'sub-P31',
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

masks = []
group_mask = None
for CONTRAST_PATH_ENDING in CONTRAST_PATH_ENDINGS:
    print(f"Processing group-level GLM for: {CONTRAST_PATH_ENDING}")

    contrasts = []
    for sub in list_of_subs:
        MASK_PATH = f"data/processed/masks/{sub}/{sub}_{'hippo_' if HIPPOCAMPUS_ONLY else ''}gm_mask.nii.gz"
        SUBJECT_FOLDER = os.path.join(DATA_PATH, sub)
        CONTRAST_PATH = os.path.join(SUBJECT_FOLDER, f'{sub}{CONTRAST_PATH_ENDING}')
        if not os.path.exists(CONTRAST_PATH):
            raise FileNotFoundError(f"Contrast file not found for {sub} at {CONTRAST_PATH}")
        print(f"Loading contrast for {sub} from {CONTRAST_PATH}")
        contrast = image.load_img(CONTRAST_PATH)
        contrasts.append(contrast)
        if group_mask is None:
            masks.append(image.load_img(MASK_PATH))
        
    if group_mask is None:
        group_mask = create_group_mask(masks, threshold=0.8)
        print(f"Created group mask from {len(masks)} individual masks.")

    ## Second Level Model
    # The design matrix needs an index for subjects
    design_matrix = make_second_level_design_matrix(subjects_label=list_of_subs)
    second_level_model = SecondLevelModel(smoothing_fwhm=None, mask_img=group_mask)
    second_level_model.fit(contrasts, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast('intercept', output_type="z_score")

    # Generate report
    for HEIGHT_CONTROL, P_VALUE in HEIGHT_CONTROLS_AND_P_VALUES.items():
        print(f"Generating report with height control: {HEIGHT_CONTROL} and p-value: {P_VALUE}")
        report = make_glm_report(second_level_model, contrasts="intercept", height_control=HEIGHT_CONTROL, alpha=P_VALUE)

        report.save_as_html(os.path.join(REPORT_PATH, f"group_level_{HEIGHT_CONTROL}_{CONTRAST_PATH_ENDING.replace('.nii.gz', '')}_glm_report.html"))