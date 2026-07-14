from __future__ import annotations

from pathlib import Path
import gc
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.fmriprep import load_confounds
from nilearn.plotting import plot_design_matrix
import matplotlib.pyplot as plt

from .contrasts import build_contrast_vector, preview_contrast_vectors
from .paths import ProjectPaths, get_bold_file
from .events import make_faces_objects_events


def infer_tr_from_bold(bold_path: str | Path) -> float:
    img = nib.load(str(bold_path))
    return float(img.header.get_zooms()[3])


def load_basic_confounds_for_bold(bold_path: str | Path) -> tuple[pd.DataFrame, np.ndarray | None]:
    confounds, sample_mask = load_confounds(
        str(bold_path),
        strategy=["motion", "wm_csf", "high_pass"],
        motion="full",
        wm_csf="basic",
    )
    confounds = confounds.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return confounds, sample_mask


def prepare_sample_masks_for_nilearn(sample_masks, bold_paths):
    """Return None when all sample masks are None; otherwise return one mask per run.

    Nilearn can error if a multi-run list mixes None and arrays. If any run has a
    real sample mask, runs without censoring receive a full-scan mask.
    """
    if all(mask is None for mask in sample_masks):
        return None

    prepared = []
    for mask, bold_path in zip(sample_masks, bold_paths):
        if mask is None:
            n_scans = nib.load(str(bold_path)).shape[-1]
            prepared.append(np.arange(n_scans, dtype=int))
        else:
            prepared.append(np.asarray(mask, dtype=int))
    return prepared


def fit_two_run_first_level(
    subject: str,
    paths: ProjectPaths,
    events_by_phase: dict[str, pd.DataFrame],
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    smoothing_fwhm: float = 5.0,
    smoothing_label: str = "smoothed5mm",
    hrf_model: str = "spm",
    task: str = "viewing",
    run: str = "01",
    space: str = "MNI152NLin2009cAsym",
    save_outputs: bool = True,
    make_design_plot: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pre_bold = get_bold_file(paths, subject, sessions["pre"], task=task, run=run, space=space)
    post_bold = get_bold_file(paths, subject, sessions["post"], task=task, run=run, space=space)
    bold_paths = [pre_bold, post_bold]

    for bold_path in bold_paths:
        if not bold_path.exists():
            raise FileNotFoundError(f"Missing BOLD file: {bold_path}")

    pre_confounds, pre_sample_mask = load_basic_confounds_for_bold(pre_bold)
    post_confounds, post_sample_mask = load_basic_confounds_for_bold(post_bold)
    sample_masks = prepare_sample_masks_for_nilearn([pre_sample_mask, post_sample_mask], bold_paths)

    tr = infer_tr_from_bold(pre_bold)

    model = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        drift_model=None,
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        smoothing_fwhm=smoothing_fwhm,
        minimize_memory=False,
    )

    fit_kwargs = dict(
        run_imgs=[str(pre_bold), str(post_bold)],
        events=[events_by_phase["pre"], events_by_phase["post"]],
        confounds=[pre_confounds, post_confounds],
    )
    if sample_masks is not None:
        fit_kwargs["sample_masks"] = sample_masks

    model = model.fit(**fit_kwargs)
    design_matrices = model.design_matrices_

    subject_qc = pd.DataFrame([{
        "subject": subject,
        "tr": tr,
        "n_pre_scans": nib.load(str(pre_bold)).shape[-1],
        "n_post_scans": nib.load(str(post_bold)).shape[-1],
        "n_pre_events": len(events_by_phase["pre"]),
        "n_post_events": len(events_by_phase["post"]),
        "pre_event_types": ";".join(sorted(events_by_phase["pre"]["trial_type"].unique())),
        "post_event_types": ";".join(sorted(events_by_phase["post"]["trial_type"].unique())),
        "n_pre_confounds": pre_confounds.shape[1],
        "n_post_confounds": post_confounds.shape[1],
        "pre_sample_mask_originally_none": pre_sample_mask is None,
        "post_sample_mask_originally_none": post_sample_mask is None,
        "sample_masks_passed_to_nilearn": sample_masks is not None,
        "n_design_columns_run_1": len(design_matrices[0].columns),
        "n_design_columns_run_2": len(design_matrices[1].columns),
        "design_columns_run_1": ";".join(design_matrices[0].columns),
        "design_columns_run_2": ";".join(design_matrices[1].columns),
    }])

    if save_outputs:
        for run_idx, design_matrix in enumerate(design_matrices, start=1):
            design_path = output_dirs["first_level_designs"] / f"{subject}_run-{run_idx}_design_matrix_{smoothing_label}.csv"
            design_matrix.to_csv(design_path, index=False)
            if make_design_plot:
                fig = plt.figure(figsize=(14, 6))
                plot_design_matrix(design_matrix)
                plt.title(f"{subject} run {run_idx} design matrix")
                plt.tight_layout()
                fig_path = output_dirs["first_level_figures"] / f"{subject}_run-{run_idx}_design_matrix_{smoothing_label}.png"
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.show()

    contrast_qc = preview_contrast_vectors(design_matrices, contrast_specs)
    contrast_qc.insert(0, "subject", subject)

    for contrast_name, spec in contrast_specs.items():
        run_vectors = []
        for design_matrix in design_matrices:
            vec, _ = build_contrast_vector(design_matrix.columns, spec["weights"])
            run_vectors.append(vec)

        # Skip only if every run vector is null. Single-run descriptive contrasts
        # can legitimately have a null vector for the other run.
        if all(np.all(vec == 0) for vec in run_vectors):
            continue

        z_map = model.compute_contrast(run_vectors, output_type="z_score")
        effect_map = model.compute_contrast(run_vectors, output_type="effect_size")

        if save_outputs:
            z_path = output_dirs["first_level_maps"] / f"{subject}_{contrast_name}_{smoothing_label}_zmap.nii.gz"
            effect_path = output_dirs["first_level_maps"] / f"{subject}_{contrast_name}_{smoothing_label}_effect.nii.gz"
            z_map.to_filename(str(z_path))
            effect_map.to_filename(str(effect_path))

    del model
    gc.collect()
    return subject_qc, contrast_qc


def run_faces_objects_first_level_subject(
    subject: str,
    paths: ProjectPaths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events_by_phase = {
        "pre": make_faces_objects_events(viewing_item_labels, subject, "pre"),
        "post": make_faces_objects_events(viewing_item_labels, subject, "post"),
    }
    return fit_two_run_first_level(
        subject=subject,
        paths=paths,
        events_by_phase=events_by_phase,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        sessions=sessions,
        **kwargs,
    )



def run_old_new_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the two-run PRE/POST old-new first-level GLM for one subject."""
    from .events import make_old_new_events

    events_by_phase = {
        "pre": make_old_new_events(viewing_item_labels, subject, "pre"),
        "post": make_old_new_events(viewing_item_labels, subject, "post"),
    }
    return fit_two_run_first_level(
        subject=subject,
        paths=paths,
        events_by_phase=events_by_phase,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        sessions=sessions,
        **kwargs,
    )




def fit_single_run_first_level(
    subject: str,
    paths,
    events: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    session: str,
    phase_label: str = "post",
    smoothing_fwhm: float = 5.0,
    smoothing_label: str = "smoothed5mm",
    hrf_model: str = "spm",
    task: str = "viewing",
    run: str = "01",
    space: str = "MNI152NLin2009cAsym",
    save_outputs: bool = True,
    make_design_plot: bool = False,
    return_model: bool = False,
):
    """Fit a single-run first-level GLM and save contrast maps immediately.

    This is useful for POST-only or PRE-only analyses.

    If return_model=True, the fitted FirstLevelModel is returned as well.
    This is mainly useful for displaying the design matrix in notebooks.
    """
    from .contrasts import build_contrast_vector, preview_contrast_vectors
    from .paths import get_bold_file

    bold_path = get_bold_file(
        paths,
        subject,
        session,
        task=task,
        run=run,
        space=space,
    )

    if not bold_path.exists():
        raise FileNotFoundError(f"Missing BOLD file: {bold_path}")

    confounds, sample_mask = load_basic_confounds_for_bold(bold_path)

    sample_masks = prepare_sample_masks_for_nilearn(
        [sample_mask],
        [bold_path],
    )

    tr = infer_tr_from_bold(bold_path)

    model = FirstLevelModel(
        t_r=tr,
        hrf_model=hrf_model,
        drift_model=None,
        noise_model="ar1",
        standardize=False,
        signal_scaling=0,
        smoothing_fwhm=smoothing_fwhm,
        minimize_memory=False,
    )

    fit_kwargs = dict(
        run_imgs=[str(bold_path)],
        events=[events],
        confounds=[confounds],
    )

    if sample_masks is not None:
        fit_kwargs["sample_masks"] = sample_masks

    model = model.fit(**fit_kwargs)

    design_matrices = model.design_matrices_

    if isinstance(design_matrices, pd.DataFrame):
        design_matrices = [design_matrices]

    design_matrix = design_matrices[0]

    subject_qc = pd.DataFrame(
        [
            {
                "subject": subject,
                "phase": phase_label,
                "session": session,
                "tr": tr,
                "n_scans": nib.load(str(bold_path)).shape[-1],
                "n_events": len(events),
                "event_types": ";".join(sorted(events["trial_type"].unique())),
                "n_confounds": confounds.shape[1],
                "sample_mask_originally_none": sample_mask is None,
                "sample_masks_passed_to_nilearn": sample_masks is not None,
                "n_design_columns": len(design_matrix.columns),
                "design_columns": ";".join(design_matrix.columns),
            }
        ]
    )

    # ------------------------------------------------------------
    # Save design matrix / optional design plot
    # ------------------------------------------------------------

    if save_outputs:
        design_path = (
            output_dirs["first_level_designs"]
            / f"{subject}_{phase_label}_design_matrix_{smoothing_label}.csv"
        )

        design_matrix.to_csv(design_path, index=False)

        if make_design_plot:
            fig = plt.figure(figsize=(14, 6))
            plot_design_matrix(design_matrix)
            plt.title(f"{subject} {phase_label} design matrix")
            plt.tight_layout()

            fig_path = (
                output_dirs["first_level_figures"]
                / f"{subject}_{phase_label}_design_matrix_{smoothing_label}.png"
            )

            fig.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.show()

    # ------------------------------------------------------------
    # Contrast-vector QC
    # ------------------------------------------------------------

    contrast_qc = preview_contrast_vectors(
        design_matrices,
        contrast_specs,
    )

    contrast_qc.insert(0, "subject", subject)

    # ------------------------------------------------------------
    # Compute and save maps immediately
    # ------------------------------------------------------------

    for contrast_name, spec in contrast_specs.items():
        vec, _ = build_contrast_vector(
            design_matrix.columns,
            spec["weights"],
        )

        if np.all(vec == 0):
            continue

        z_map = model.compute_contrast(
            vec,
            output_type="z_score",
        )

        effect_map = model.compute_contrast(
            vec,
            output_type="effect_size",
        )

        if save_outputs:
            z_path = (
                output_dirs["first_level_maps"]
                / f"{subject}_{contrast_name}_{smoothing_label}_zmap.nii.gz"
            )

            effect_path = (
                output_dirs["first_level_maps"]
                / f"{subject}_{contrast_name}_{smoothing_label}_effect.nii.gz"
            )

            z_map.to_filename(str(z_path))
            effect_map.to_filename(str(effect_path))

    if return_model:
        return {
            "subject_qc": subject_qc,
            "contrast_qc": contrast_qc,
            "model": model,
            "design_matrices": design_matrices,
        }

    del model
    gc.collect()

    return subject_qc, contrast_qc



def run_post_relevance_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    return_model: bool = False,
    **kwargs,
):
    """Run a POST-only relevance/old-new first-level GLM for one subject."""
    from .events import make_post_relevance_events

    events = make_post_relevance_events(
        viewing_item_labels,
        subject,
        phase="post",
    )

    return fit_single_run_first_level(
        subject=subject,
        paths=paths,
        events=events,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        session=sessions["post"],
        phase_label="post",
        return_model=return_model,
        **kwargs,
    )


def run_post_episode_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    return_model: bool = False,
    **kwargs,
):
    """Run a POST-only episode e1/e2/e3 first-level GLM for one subject.

    This assumes fit_single_run_first_level supports return_model=True, as patched
    for Notebook 05.
    """
    from .events import make_post_episode_events

    events = make_post_episode_events(
        viewing_item_labels=viewing_item_labels,
        subject=subject,
        phase="post",
    )

    return fit_single_run_first_level(
        subject=subject,
        paths=paths,
        events=events,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        session=sessions["post"],
        phase_label="post",
        return_model=return_model,
        **kwargs,
    )



def run_post_relevance_episode_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    return_model: bool = False,
    **kwargs,
):
    """Run a POST-only relevance x episode first-level GLM for one subject.

    This assumes fit_single_run_first_level supports return_model=True, as patched
    for Notebook 05.
    """
    from .events import make_post_relevance_episode_events

    events = make_post_relevance_episode_events(
        viewing_item_labels=viewing_item_labels,
        subject=subject,
        phase="post",
    )

    return fit_single_run_first_level(
        subject=subject,
        paths=paths,
        events=events,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        session=sessions["post"],
        phase_label="post",
        return_model=return_model,
        **kwargs,
    )



def run_post_encoding_time_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    return_model: bool = False,
    **kwargs,
):
    """Run a POST-only encoding-time first-level GLM for one subject."""
    from .events import make_post_encoding_time_events

    events = make_post_encoding_time_events(
        viewing_item_labels,
        subject,
        phase="post",
    )

    return fit_single_run_first_level(
        subject=subject,
        paths=paths,
        events=events,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        session=sessions["post"],
        phase_label="post",
        return_model=return_model,
        **kwargs,
    )



def run_post_episode_encoding_time_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    return_model: bool = False,
    **kwargs,
):
    """Run a POST-only episode x encoding-time first-level GLM for one subject.

    This assumes fit_single_run_first_level supports return_model=True, as patched
    for Notebook 05.
    """
    from .events import make_post_episode_encoding_time_events

    events = make_post_episode_encoding_time_events(
        viewing_item_labels=viewing_item_labels,
        subject=subject,
        phase="post",
    )

    return fit_single_run_first_level(
        subject=subject,
        paths=paths,
        events=events,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        session=sessions["post"],
        phase_label="post",
        return_model=return_model,
        **kwargs,
    )


def run_pre_post_episode_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    **kwargs,
):
    """Run a two-run PRE/POST episode-control first-level GLM for one subject.

    The PRE run acts as an image-level baseline. POST-PRE interaction contrasts
    therefore test episode structure emerging after Island navigation rather
    than pre-existing image salience or visual differences.
    """
    from .events import make_pre_post_episode_events

    events_by_phase = {
        "pre": make_pre_post_episode_events(viewing_item_labels, subject, phase="pre"),
        "post": make_pre_post_episode_events(viewing_item_labels, subject, phase="post"),
    }

    return fit_two_run_first_level(
        subject=subject,
        paths=paths,
        events_by_phase=events_by_phase,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        sessions=sessions,
        **kwargs,
    )



def run_pre_post_encoding_space_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    **kwargs,
):
    """Run a two-run PRE/POST encoding-space-control first-level GLM.

    The PRE run acts as an image-level baseline. POST-PRE interaction contrasts
    therefore test route-position structure emerging after Island navigation
    rather than pre-existing image salience or visual differences.
    """
    from .events import make_pre_post_encoding_space_events

    events_by_phase = {
        "pre": make_pre_post_encoding_space_events(viewing_item_labels, subject, phase="pre"),
        "post": make_pre_post_encoding_space_events(viewing_item_labels, subject, phase="post"),
    }

    return fit_two_run_first_level(
        subject=subject,
        paths=paths,
        events_by_phase=events_by_phase,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        sessions=sessions,
        **kwargs,
    )



def run_pre_post_time_space_first_level_subject(
    subject: str,
    paths,
    viewing_item_labels: pd.DataFrame,
    contrast_specs: dict[str, dict],
    output_dirs: dict[str, Path],
    sessions: dict[str, str],
    **kwargs,
):
    """Run a two-run PRE/POST TIME × SPACE first-level GLM.

    PRE acts as an image-level baseline. The resulting interaction contrasts test
    whether episode-dependent space effects emerge after Island navigation.
    """
    from .events import make_pre_post_time_space_events

    events_by_phase = {
        "pre": make_pre_post_time_space_events(viewing_item_labels, subject, phase="pre"),
        "post": make_pre_post_time_space_events(viewing_item_labels, subject, phase="post"),
    }

    return fit_two_run_first_level(
        subject=subject,
        paths=paths,
        events_by_phase=events_by_phase,
        contrast_specs=contrast_specs,
        output_dirs=output_dirs,
        sessions=sessions,
        **kwargs,
    )




