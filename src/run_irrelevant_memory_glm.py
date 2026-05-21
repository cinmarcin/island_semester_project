import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn import image, datasets
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix


REPO_ROOT = Path(".")
FMRIPREP_DIR = REPO_ROOT / "data" / "fmriprep_output"
RELABELLED_EVENTS_DIR = REPO_ROOT / "data" / "preprocessed" / "relabeled_events"
LABEL_FILE = REPO_ROOT / "data" / "preprocessed" / "irrelevant_seen_old_memory_labels_clean_fmri.csv"

OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "glm" / "irrelevant_seen_true_vs_schema"
REPORT_DIR = REPO_ROOT / "reports" / "irrelevant_seen_true_vs_schema"


CONFOUND_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x", "rot_y", "rot_z",
    "global_signal", "csf", "white_matter",
]


def load_session(sub: str, ses: str):
    """Load fMRI image, relabelled events, confounds, and metadata for one session."""
    func_dir = FMRIPREP_DIR / sub / ses / "func"

    bold_file = func_dir / f"{sub}_{ses}_task-viewing_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
    confounds_file = func_dir / f"{sub}_{ses}_task-viewing_run-01_desc-confounds_timeseries.tsv"
    metadata_file = func_dir / f"{sub}_{ses}_task-viewing_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.json"
    events_file = RELABELLED_EVENTS_DIR / f"{sub}_{ses}_task-viewing_run-01_events_memory_labels.tsv"

    required = [bold_file, confounds_file, metadata_file, events_file]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {sub} {ses}:\n" + "\n".join(missing))

    bold_img = nib.load(str(bold_file))
    confounds = pd.read_csv(confounds_file, sep="\t")
    events = pd.read_csv(events_file, sep="\t")

    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    return bold_img, events, confounds, metadata


def prepare_events(events: pd.DataFrame, run_label: str):
    """Convert relabelled events to Nilearn format and prefix by session."""
    out = events[["onset", "duration", "trial_type_memory"]].copy()
    out = out.rename(columns={"trial_type_memory": "trial_type"})
    out["trial_type"] = run_label + "_" + out["trial_type"].astype(str)
    return out


def prepare_confounds(pre_confounds: pd.DataFrame, post_confounds: pd.DataFrame):
    """Select confounds, prefix pre/post columns, concatenate."""
    available_cols = [
        c for c in CONFOUND_COLS
        if c in pre_confounds.columns and c in post_confounds.columns
    ]

    pre_conf = pre_confounds[available_cols].fillna(0).add_prefix("pre_")
    post_conf = post_confounds[available_cols].fillna(0).add_prefix("post_")

    combined = pd.concat([pre_conf, post_conf], ignore_index=True).fillna(0)
    return combined


def build_design_matrix(combined_img, combined_events, combined_confounds, tr):
    """Build first-level design matrix for concatenated pre/post viewing data."""
    n_scans = combined_img.shape[-1]
    frame_times = np.arange(n_scans) * tr

    design_matrix = make_first_level_design_matrix(
        frame_times,
        combined_events,
        hrf_model="spm + derivative + dispersion",
        drift_model="cosine",
        high_pass=0.01,
        add_regs=combined_confounds,
        add_reg_names=combined_confounds.columns.tolist(),
    )

    return design_matrix


def make_contrast_vector(design_matrix, weights):
    """Create contrast vector from a dictionary {column_name: weight}."""
    columns = design_matrix.columns.tolist()
    vec = np.zeros(len(columns))

    missing = []
    for name, weight in weights.items():
        if name in columns:
            vec[columns.index(name)] = weight
        else:
            missing.append(name)

    if missing:
        raise ValueError(f"Missing contrast columns: {missing}")

    return vec


def run_subject(sub: str, smoothing_fwhm: float = 5.0):
    print("=" * 80)
    print(f"Running irrelevant memory GLM for {sub}")
    print("=" * 80)

    pre_img, pre_events, pre_confounds, pre_meta = load_session(sub, "ses-01")
    post_img, post_events, post_confounds, post_meta = load_session(sub, "ses-05")

    print(f"Pre image shape:  {pre_img.shape}")
    print(f"Post image shape: {post_img.shape}")
    print("Pre event labels:")
    print(pre_events["trial_type_memory"].value_counts())
    print("Post event labels:")
    print(post_events["trial_type_memory"].value_counts())

    tr = pre_meta["RepetitionTime"]
    pre_duration = pre_img.shape[-1] * tr

    pre_events_glm = prepare_events(pre_events, "pre")
    post_events_glm = prepare_events(post_events, "post")
    post_events_glm["onset"] += pre_duration

    combined_events = pd.concat([pre_events_glm, post_events_glm], ignore_index=True)

    combined_img = image.concat_imgs([pre_img, post_img])
    combined_confounds = prepare_confounds(pre_confounds, post_confounds)

    design_matrix = build_design_matrix(
        combined_img=combined_img,
        combined_events=combined_events,
        combined_confounds=combined_confounds,
        tr=tr,
    )

    memory_cols = [c for c in design_matrix.columns if "irrelevant_seen" in c]
    print("Memory-related columns:")
    for c in memory_cols:
        print("  ", c)

    gm_mask = datasets.load_mni152_gm_mask(resolution=2, threshold=0.2, n_iter=2)
    gm_mask_resampled = image.resample_to_img(
        gm_mask,
        combined_img,
        interpolation="nearest",
        force_resample=True,
        copy_header=True,
    )

    fmri_glm = FirstLevelModel(
        mask_img=gm_mask_resampled,
        smoothing_fwhm=smoothing_fwhm,
        n_jobs=-1,
    )

    print("Fitting GLM...")
    fmri_glm = fmri_glm.fit(
        combined_img,
        design_matrices=design_matrix,
    )
    print("GLM fitted.")

    # Contrasts
    pre_true_vs_schema = make_contrast_vector(
        design_matrix,
        {
            "pre_irrelevant_seen_true_week": 1,
            "pre_irrelevant_seen_schema_week": -1,
        },
    )

    post_true_vs_schema = make_contrast_vector(
        design_matrix,
        {
            "post_irrelevant_seen_true_week": 1,
            "post_irrelevant_seen_schema_week": -1,
        },
    )

    post_minus_pre_true_vs_schema = post_true_vs_schema - pre_true_vs_schema

    z_pre = fmri_glm.compute_contrast(pre_true_vs_schema, output_type="z_score")
    z_post = fmri_glm.compute_contrast(post_true_vs_schema, output_type="z_score")
    z_post_pre = fmri_glm.compute_contrast(post_minus_pre_true_vs_schema, output_type="z_score")

    # Save outputs
    sub_out = OUTPUT_DIR / sub
    sub_out.mkdir(parents=True, exist_ok=True)

    z_pre.to_filename(sub_out / f"{sub}_smooth_{smoothing_fwhm}_pre_true_vs_schema.nii.gz")
    z_post.to_filename(sub_out / f"{sub}_smooth_{smoothing_fwhm}_post_true_vs_schema.nii.gz")
    z_post_pre.to_filename(sub_out / f"{sub}_smooth_{smoothing_fwhm}_post-pre_true_vs_schema.nii.gz")

    design_matrix.to_csv(sub_out / f"{sub}_smooth_{smoothing_fwhm}_design_matrix.csv", index=False)

    print(f"Saved outputs to: {sub_out}")

    return {
        "sub": sub,
        "output_dir": str(sub_out),
        "n_pre_true": int((pre_events["trial_type_memory"] == "irrelevant_seen_true_week").sum()),
        "n_pre_schema": int((pre_events["trial_type_memory"] == "irrelevant_seen_schema_week").sum()),
        "n_post_true": int((post_events["trial_type_memory"] == "irrelevant_seen_true_week").sum()),
        "n_post_schema": int((post_events["trial_type_memory"] == "irrelevant_seen_schema_week").sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=str, default=None, help="Subject ID, e.g. sub-P10. If omitted, runs all clean subjects.")
    parser.add_argument("--smoothing", type=float, default=5.0, help="Smoothing FWHM in mm.")
    args = parser.parse_args()

    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Label file not found: {LABEL_FILE}")

    labels = pd.read_csv(LABEL_FILE)
    clean_subjects = sorted(labels["sub"].unique())

    if args.sub is not None:
        subjects = [args.sub]
    else:
        subjects = clean_subjects

    print("Subjects to run:")
    print(subjects)

    summary = []

    for sub in subjects:
        try:
            res = run_subject(sub, smoothing_fwhm=args.smoothing)
            summary.append(res)
        except Exception as e:
            print(f"ERROR for {sub}: {e}")

    if summary:
        summary_df = pd.DataFrame(summary)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        summary_file = OUTPUT_DIR / "run_summary.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Saved summary to: {summary_file}")
        print(summary_df)


if __name__ == "__main__":
    main()
