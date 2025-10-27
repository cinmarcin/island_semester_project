
# Island Experiment — MIP Lab Semester Project

This repository contains the  analysis code for the "Island" experiment, developed as part of a semester project at the MIP lab (EPFL). The project investigates the neural encoding of spatial and temporal aspects of episodic memory, and how goal relevance modulates memory, using an immersive VR-like island and fMRI recordings.

---

## Quick summary

- Final sample: 26 participants (after exclusions). Age range 19–36 (mean = 24.35, SD = 4.15). 15 females, 11 males. Participants were recruited at the University of Geneva and gave written informed consent. All reported no neurological or psychiatric conditions and showed average to good spatial skills on the Santa Barbara Sense of Direction Scale.
- The experiment uses an immersive VR island (Unity) with checkpoints where images are presented. Some images are goal-relevant depending on the mission. The protocol includes behavioral familiarization, a sprint task, three weekly encoding missions (two fMRI runs per session plus resting-state), pre- and post-viewing tasks, and a surprise memory test.
- Analysis in this repository focuses on first-level GLM analyses of the fMRI data and Representational Similarity Analysis (RSA) derived from GLM betas, plus correlation / group-level statistics.

---

## Repository layout

Top-level folders (important ones):

- `configs/` — YAML configuration files describing analysis options (contrast types, smoothing, p-value thresholds, whether to restrict to hippocampus, etc.).
- `data/`
	- `raw/` — original files collected from the experiment (events, dicom/NIfTI, behavioral logs).
	- `preprocessed/` — intermediate preprocessing outputs (subject-level extracted/cleaned data used for GLM).
	- `processed/` — analysis outputs (GLM models, contrasts, RSA results, masks). Example subfolders: `glm/`, `rsa/`, `masks/`, `contrasts/`.
- `figures/` — generated figures for correlation, RSA and other visualizations.
- `notebooks/` — Jupyter notebooks used for exploratory analyses and plotting (e.g., `neural_correlation.ipynb`).
- `reports/` — group or condition-specific reports and saved outputs.
- `src/` — project source code. Important modules:
	- `src/core/` — core analysis modules (GLM, RSA, contrast definitions).
	- `src/data/` — data-loading and path utilities.
	- `src/utils/` — helper utilities (plotting, debugging, IO helpers).
	- `src/run_subject_level_glm.py` — entrypoint to run combined first-level GLM for a subject.
	- `src/run_rsa.py` — run subject- and group-level RSA and save figures.
	- `src/run_correlation.py` — run correlation analyses across subjects and repetitions.
    - `src/run_group_level_glm.py` — run the analysis at the group level from a given config file

- `requirements` - all necessaey packages
- `submit_all_subjects_glm.sh` - srun to be used on a cluster. It runs 'run_subject_level_glm.py' on all subjects in parallel for efficiency purposes for all the given configs.
- `submit_main.sh` - srun to be used on a cluster. It runs all necessary analysis RSA + GLM of all configs present in the folder 'configs'
- `submit_run_correlation.sh` - srun to be used on a cluster. Run the script run_correlation.py
- `submit_run_rsa.sh` - srun to be used on a cluster. Run the script run_rsa.py

---

## Experiment overview (short)

Participants explored a VR island and completed missions across three weeks. At checkpoints they viewed images (humans: pirates, vikings, mayas; objects: food, drink, wealth). In each mission some images were designated as goal-relevant. The experiment includes:

- Familiarization (build spatial representation)
- Sprint task (verify navigation)
- Pre-viewing task (view stimuli in scanner)
- Three encoding missions (two runs each; mark goal relevance)
- Post-viewing and memory tasks (old/new, spatial placement, temporal memory)

---

## Analysis pipeline

Primary analyses implemented in this repo:

1. Preprocessing & confounds extraction (preprocessed data under `data/preprocessed/`).
2. First-level GLM (combined pre/post):
	 - The code builds per-run confound-only design matrices, prefixes per-run columns, concatenates events (post onsets shifted by pre duration), and creates a final design matrix for the concatenated fMRI runs. See `src/core/glm.py` for the exact transformation.
	 - Contrasts are defined in `src/core/contrast_types.py` and support combined (pre/post) contrasts.
3. RSA: compute representational similarity matrices from trial-level beta maps and compare to model RDMs (`src/core/rsa.py`).
4. Group-level statistics and permutation testing for RSA results (see `src/run_rsa.py`).
5. Correlation analyses: within- and across-repetition correlations and group-level statistics (see `src/run_correlation.py`).

---

## Running analyses

All example commands below assume you run them from the repository root. Very important for now the raw data is directly downloaded from the MIPlab server for disk quota reasons. This process is done and linked to my izar account and the logic needs thus to be adapted to your account or you need to pre download the raw data and place it in data/raw.

1) Run the combined first-level GLM for a single subject

```bash
python -m src.run_subject_level_glm.py --sub sub-P10 --config configs/faces_vs_objects.yaml
```

`--config` accepts one or more YAML files; the script will iterate through provided configs. Check `configs/` to select the appropriate contrast and analysis options.

2) Run RSA (group-level runner)

```bash
python -m src.run_rsa.py --config configs/order_hippocampus.yaml
```

This will compute subject-level RSA and group-level sign-flip permutation tests, saving figures to `figures/rsa/<config-name>/`. You need to have run the GLM with the same config prior to running this script.

3) Run correlation analysis (group-level plotting and stats)

```bash
python -m src.run_correlation.py
```

---

## Where to find results

- GLM fitted models and contrasts: `data/processed/glm/<contrast>/<sub>/`
- RSA outputs: `data/processed/rsa/` and `figures/rsa/` for plots
- Design matrix figures and reports: `reports/` and `figures/` depending on configuration

---

## Notes, caveats and troubleshooting

- Several participants were excluded (missing events, incomplete scans or unresponsiveness). The scripts assume valid event files exist for a subject; missing events will raise errors. See `src/data` utilities for the loader and error handling.
- The GLM code uses Nilearn's `FirstLevelModel` and assumes images are correctly preprocessed and in the same space.

---

## Contact

This code was done to be as reproducible as possible, but a semester project is quite short so there might be lengthy code in some place. However, I do think this code creates a strong baseline of analysis on the island experiement so please contact me if you have any questions regarding the repo.

Oskar.Boesch@epfl.ch

---
# island_semester_project

This project is strongly inspired by the work of Deuker et al. (2016), An event map of memory space in the hippocampus (https://elifesciences.org/articles/16534).
