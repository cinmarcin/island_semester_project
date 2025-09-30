from contrast_types import ContrastType
import os
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.reporting import make_glm_report
from nilearn import image


## Controls
CONTRAST_TYPE = ContrastType.OLD_VS_NEW
SMOOTHING_FWHM = 5 # in mm
HEIGHT_CONTROL = "fdr"  # "fdr" or "bonferroni"
HIPPOCAMPUS_ONLY = False
P_VALUE = 0.05

CONTRAST_PATH_ENDINGS= [f'_post_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}contrast.nii.gz',
                        f'_pre_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}contrast.nii.gz',
                        f'_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}contrast.nii.gz', 
                        f'_second_level_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}contrast.nii.gz']
DATA_PATH = f'data/processed/glm/{CONTRAST_TYPE.value}'
REPORT_PATH = f"reports/glm/{CONTRAST_TYPE.value}/group_level"
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
for CONTRAST_PATH_ENDING in CONTRAST_PATH_ENDINGS:
    print(f"Processing group-level GLM for: {CONTRAST_PATH_ENDING}")

    contrasts = []
    for sub in list_of_subs:
        SUBJECT_FOLDER = os.path.join(DATA_PATH, sub)
        CONTRAST_PATH = os.path.join(SUBJECT_FOLDER, f'{sub}{CONTRAST_PATH_ENDING}')
        if not os.path.exists(CONTRAST_PATH):
            raise FileNotFoundError(f"Contrast file not found for {sub} at {CONTRAST_PATH}")
        print(f"Loading contrast for {sub} from {CONTRAST_PATH}")
        contrast = image.load_img(CONTRAST_PATH)
        contrasts.append(contrast)

    ## Second Level Model
    # The design matrix needs an index for subjects
    design_matrix = make_second_level_design_matrix(subjects_label=list_of_subs)
    second_level_model = SecondLevelModel(smoothing_fwhm=None)
    second_level_model.fit(contrasts, design_matrix=design_matrix)
    z_map = second_level_model.compute_contrast('intercept', output_type="z_score")

    # Generate report
    report = make_glm_report(second_level_model, contrasts="intercept", height_control=HEIGHT_CONTROL, alpha=P_VALUE)

    report.save_as_html(os.path.join(REPORT_PATH, f"group_level_{HEIGHT_CONTROL}_{CONTRAST_PATH_ENDING.replace('.nii.gz', '')}_glm_report.html"))