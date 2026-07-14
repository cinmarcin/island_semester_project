# Island semester project — Kroczak fMRI GLM analyses

This repository contains my semester-project analysis code for the Island VR episodic-memory fMRI study. The project investigates how hippocampal and whole-brain activity during image viewing changes after participants encoded images in a structured virtual island environment.

The main contribution in this repository is the cleaned GLM analysis workflow in `notebooks_kroczak/` and the reusable helper code in `src_kroczak/island_glm/`. Some exploratory work has been done under `notebooks_old_kroczak/`.

## Project structure

```text
island_semester_project/
├── README_kroczak.md
├── requirements.txt
├── requirements_island.txt
├── configs/
├── data/
│   ├── bids_ackbar -> shared BIDS/event data
│   ├── fmriprep_output -> shared fMRIPrep derivatives
│   ├── processed/       # generated analysis outputs
│   └── rois/            # hippocampus/amygdala ROI and SVC masks
├── notebooks_kroczak/
│   ├── 00_prepare_common_inputs.ipynb
│   ├── 01_faces_objects_pre_post_validation.ipynb
│   ├── 02_03_04_old_new_pre_post_viewing_glm.ipynb
│   ├── 05_post_relevance_old_new_viewing_glm.ipynb
│   ├── 05b_pre_post_relevance_old_new_viewing_glm.ipynb
│   ├── 06_post_episode_e1_e2_e3_viewing_glm.ipynb
│   ├── 07_post_relevance_episode_interaction_viewing_glm.ipynb
│   ├── 08_post_encoding_time_early_middle_late_viewing_glm.ipynb
│   ├── 09_post_episode_x_encoding_time_interaction_viewing_glm.ipynb
│   ├── 10_pre_post_episode_e1_e2_e3_control_glm copy 2.ipynb
│   ├── 11_pre_post_encoding_time_early_middle_late_control_glm.ipynb
│   ├── 12_pre_post_time_x_encoding_space_interaction_glm cop.ipynb
│   └── 12b_pre_post_temporal_minus_spatial_glm.ipynb
├── notebooks_old_kroczak/       # earlier exploratory notebooks
├── reports/
├── src_kroczak/
│   └── island_glm/
└── src_Oskar/
```

The cleaned analysis workflow is mainly in:

```text
notebooks/
src_kroczak/island_glm/
data/processed/
data/rois/
```

`notebooks_old_kroczak/` is kept for provenance but is not the main reproducible workflow.

## Data requirements

The notebooks expect these folders to exist:

```text
data/bids_ackbar/
data/fmriprep_output/
```

In the MIP Lab NAS environment, these are symbolic links to shared data locations:

```text
data/bids_ackbar -> /media/RCPNAS/Data3/Alison_island/bids_ackbar
data/fmriprep_output -> /media/RCPNAS/Data3/Alison_island/new_data/fmriprep_output
```

Most notebooks use:

```python
PROJECT_ROOT = Path("/media/RCPNAS/Data3/Alison_island/kroczak/island_semester_project")
```

If the repository is copied elsewhere, update `PROJECT_ROOT` in the notebooks or recreate the same path structure.

Large raw data, BIDS data, and fMRIPrep derivatives are not intended to be stored directly in Git.

## Environment

The analyses were run in a Python environment named `island`.

Example setup:

```bash
cd /path/to/island_semester_project

conda create -n island python=3.10 -y
conda activate island

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

python -m ipykernel install --user --name island --display-name "Python (island)"
jupyter lab
```

Then select the Jupyter kernel:

```text
Python (island)
```

## Helper package

Reusable GLM functions are stored in:

```text
src_kroczak/island_glm/
```

Important modules include:

```text
paths.py          # project paths and analysis directories
common.py         # shared input loading and subject selection
events.py         # event-table construction
contrasts.py      # contrast definitions and contrast QC
first_level.py    # first-level GLM fitting
second_level.py   # group-level GLM, thresholding, cluster tables
reporting.py      # table formatting and reporting utilities
plotting.py       # common plotting helpers
```

## Recommended notebook order

Run or inspect the cleaned notebooks in this order:

```text
00_prepare_common_inputs.ipynb
01_faces_objects_pre_post_validation.ipynb
02_03_04_old_new_pre_post_viewing_glm.ipynb
05_post_relevance_old_new_viewing_glm.ipynb
05b_pre_post_relevance_old_new_viewing_glm.ipynb
06_post_episode_e1_e2_e3_viewing_glm.ipynb
07_post_relevance_episode_interaction_viewing_glm.ipynb
08_post_encoding_time_early_middle_late_viewing_glm.ipynb
09_post_episode_x_encoding_time_interaction_viewing_glm.ipynb
10_pre_post_episode_e1_e2_e3_control_glm copy 2.ipynb
11_pre_post_encoding_time_early_middle_late_control_glm.ipynb
12_pre_post_time_x_encoding_space_interaction_glm cop.ipynb
12b_pre_post_temporal_minus_spatial_glm.ipynb
```

Notebook 00 should be run first because it prepares shared inputs used by later notebooks.

## Notebook overview

### `00_prepare_common_inputs.ipynb`

Prepares shared project inputs:

- subject inclusion and exclusion tables;
- PRE/POST subject availability;
- image identity and event-label checks;
- item-level labels for old/new, relevance, episode, checkpoint, and route-position thirds;
- common mean T1w anatomical background for plotting;
- shared files used by later GLM notebooks.

### `01_faces_objects_pre_post_validation.ipynb`

Validation GLM for the viewing-task pipeline. It checks whether expected visual-category responses can be recovered, especially faces versus objects.

### `02_03_04_old_new_pre_post_viewing_glm.ipynb`

Main old/new PRE-to-POST viewing GLM.

Regressors:

```text
pre_old
pre_new
post_old
post_new
```

Central contrast:

```text
(post_old - post_new) - (pre_old - pre_new)
```

This tests whether old/new image responses changed from PRE to POST after encoding.

### `05_post_relevance_old_new_viewing_glm.ipynb`

POST-only relevance GLM.

Regressors:

```text
post_oldRel
post_oldIrrel
post_new
```

Main contrasts include:

```text
post_oldRel > post_new
post_oldIrrel > post_new
post_oldRel > post_oldIrrel
post_oldIrrel > post_oldRel
```

### `05b_pre_post_relevance_old_new_viewing_glm.ipynb`

PRE-to-POST relevance GLM.

Central contrast family:

```text
(post_oldRel - post_oldIrrel) - (pre_oldRel - pre_oldIrrel)
```

This controls for baseline differences between relevant and irrelevant image sets.

### `06_post_episode_e1_e2_e3_viewing_glm.ipynb`

POST-only episode GLM.

Regressors:

```text
post_oldE1
post_oldE2
post_oldE3
post_new
```

Main contrasts include ordered episode comparisons such as:

```text
E1 > E2
E1 > E3
E2 > E3
E2 > E1
E3 > E1
E3 > E2
```

### `07_post_relevance_episode_interaction_viewing_glm.ipynb`

POST-only relevance × episode GLM.

Regressors:

```text
post_oldRelE1     post_oldIrrelE1
post_oldRelE2     post_oldIrrelE2
post_oldRelE3     post_oldIrrelE3
post_new
```

This tests whether the relevance effect differs across episodes.

### `08_post_encoding_time_early_middle_late_viewing_glm.ipynb`

POST-only route-position GLM.

Regressors:

```text
post_oldEarly
post_oldMiddle
post_oldLate
post_new
```

Checkpoint thirds:

```text
Early  = checkpoints 1–5
Middle = checkpoints 6–10
Late   = checkpoints 11–16
```

### `09_post_episode_x_encoding_time_interaction_viewing_glm.ipynb`

POST-only episode × route-position GLM.

Regressors:

```text
post_oldE1Early    post_oldE1Middle    post_oldE1Late
post_oldE2Early    post_oldE2Middle    post_oldE2Late
post_oldE3Early    post_oldE3Middle    post_oldE3Late
post_new
```

This tests whether route-position effects differ across episodes.

### `10_pre_post_episode_e1_e2_e3_control_glm copy 2.ipynb`

PRE-to-POST episode-control GLM.

Example contrast:

```text
(post_E1 - post_E2) - (pre_E1 - pre_E2)
```

This tests whether episode-specific effects emerge after encoding rather than being present at baseline.

### `11_pre_post_encoding_time_early_middle_late_control_glm.ipynb`

PRE-to-POST route-position-control GLM.

Example contrast:

```text
(post_Early - post_Late) - (pre_Early - pre_Late)
```

This tests whether route-position effects emerge after encoding while controlling for baseline image-set differences.

### `12_pre_post_time_x_encoding_space_interaction_glm cop.ipynb`

PRE-to-POST Temporal × Spatial interaction GLM.

Example contrast:

```text
[(E3_Early - E3_Late) - (E1_Early - E1_Late)]POST
-
[(E3_Early - E3_Late) - (E1_Early - E1_Late)]PRE
```

This can be written as:

```text
E3 > E1 × Early > Late, POST > PRE
```

Equivalent same-sign forms include:

```text
Early > Late × E3 > E1
Late > Early × E1 > E3
E1 > E3 × Late > Early
```

This is a true factorial interaction. It asks whether the effect of route position depends on episode time, or equivalently whether the effect of episode time depends on route position.

### `12b_pre_post_temporal_minus_spatial_glm.ipynb`

PRE-to-POST Temporal − Spatial marginal comparison GLM.

This notebook tests a complementary contrast family. Instead of a factorial interaction, it compares a marginal temporal episode contrast against a marginal route-position contrast.

Example contrast:

```text
[(E1 - E2) - (Middle - Early)]POST
-
[(E1 - E2) - (Middle - Early)]PRE
```

Written as:

```text
(E1 > E2) - (Middle > Early), POST > PRE
```

This is not a Temporal × Spatial interaction. It asks whether an averaged temporal contrast is stronger than an averaged route-position contrast after encoding.

Every Temporal − Spatial contrast has an equivalent Spatial − Temporal form. For example:

```text
(E1 > E2) - (Middle > Early)
=
(Early > Middle) - (E2 > E1)
```

These are algebraically equivalent same-sign contrasts and should not be counted as independent results. The Spatial − Temporal form is included only to make interpretation clearer.

## Run flags

Most GLM notebooks use the same control flags:

```python
RUN_FIRST_LEVEL = False
OVERWRITE_FIRST_LEVEL = False
RUN_SECOND_LEVEL = True
OVERWRITE_SECOND_LEVEL = True
USE_MEAN_T1W_BACKGROUND = True
```

For a fresh run:

```python
RUN_FIRST_LEVEL = True
OVERWRITE_FIRST_LEVEL = False
RUN_SECOND_LEVEL = True
OVERWRITE_SECOND_LEVEL = True
```

After first-level maps have been generated:

```python
RUN_FIRST_LEVEL = False
OVERWRITE_FIRST_LEVEL = False
```

To intentionally recompute all subject-level maps:

```python
RUN_FIRST_LEVEL = True
OVERWRITE_FIRST_LEVEL = True
```

## Outputs

Generated outputs are written mainly under:

```text
data/processed/glm/<analysis_name>/
```

Typical subfolders include:

```text
first_level_maps/
first_level_designs/
first_level_figures/
second_level_maps/
figures/
tables/
interactive_views/
```

Useful files for inspection:

```text
tables/*cluster*.csv
tables/*second_level_table*.csv
tables/*first_level*_qc*.csv
tables/*SVC*.csv
figures/*.png
figures/*.pdf
interactive_views/*.html
```

## Plotting convention

The final figures use the common mean anatomical T1w background image created in Notebook 00.

Typical plotting settings:

```python
threshold = 2.33
vmax = 5.0
cmap = "cold_hot"
symmetric_cbar = True
black_bg = False
draw_cross = False
USE_MEAN_T1W_BACKGROUND = True
```

The high-resolution T1w background is passed directly to Nilearn as `bg_img`. It should not be manually resampled down to the statistical map grid because that makes the anatomical background look blocky.

## Small-volume correction

Several notebooks include hippocampus/amygdala ROI analyses and literature-based small-volume correction.

General procedure:

1. define a literature coordinate;
2. create a 6 mm sphere around that coordinate;
3. extract the peak z-value inside the sphere;
4. compute within-sphere FDR and Bonferroni-corrected p-values;
5. save SVC tables and figures.

SVC results are exploratory and literature-guided. They should be interpreted separately from whole-brain corrected inference.

## Main exploratory result schema

The cleaned analyses suggest the following hippocampal motifs:

```text
Episode PRE-to-POST analysis:
Left posterior hippocampus for E1-related episode contrasts.

Route-position PRE-to-POST analysis:
Anterior hippocampus for Early/Middle/Late route-position structure.

Temporal × Spatial interaction:
Right posterior hippocampus for large temporal × large spatial interaction.
Left anterior hippocampus for local E2/E3 and Middle/Late interaction structure.

Temporal − Spatial marginal comparison:
Left posterior hippocampus for E1 / beginning-of-experience contrasts.
Left anterior hippocampus for local E2/E3 and Middle/Late contrasts.
```

These patterns are exploratory. The safest interpretation is that hippocampal subregions show sensitivity to structured temporal and spatial/event information after encoding, with possible anterior–posterior and hemispheric organization.

## Restart-safe workflow

Recommended workflow:

1. Run Notebook 00 first.
2. For a new GLM notebook, run with first-level enabled.
3. Once first-level maps exist, set `RUN_FIRST_LEVEL = False`.
4. Rerun second-level, SVC, figure, and table cells as needed.
5. Inspect output tables and figures in `data/processed/glm/<analysis_name>/`.

If first-level outputs are incomplete:

```python
RUN_FIRST_LEVEL = True
OVERWRITE_FIRST_LEVEL = False
```

If a full recomputation is needed:

```python
RUN_FIRST_LEVEL = True
OVERWRITE_FIRST_LEVEL = True
```

## Notes for review and submission

This repository is intended for semester-project review and reproducibility within the MIP Lab NAS environment.

Important notes:

- Large raw, BIDS, and fMRIPrep data are accessed through NAS symlinks and are not stored in Git.
- Generated outputs in `data/processed/` may be large and should be handled carefully before committing.
- Whole-brain results are exploratory unless otherwise specified.
- SVC results are literature-guided and should not be interpreted as whole-brain corrected findings.
- Atlas labels and peak labels should be visually checked, especially near anatomical boundaries.
- The final presentation figures were generated from the cleaned notebooks in `notebooks/`.

## Minimal Git submission suggestion

Check files before committing:

```bash
git status
```

Recommended files to include:

```text
README.md
requirements.txt
requirements_island.txt
src/island_glm/
notebooks/*.ipynb
configs/
```

Be careful with large generated folders such as:

```text
data/
reports/
figures/
logs/
```

Typical submission workflow:

```bash
git status
git add README.md requirements.txt requirements_island.txt src/island_glm notebooks configs
git status
git commit -m "Add Kroczak Island semester project GLM analyses"
git push
```
