from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    data_dir: Path
    bids_dir: Path
    fmriprep_dir: Path
    src_dir: Path
    common_manifest_path: Path

    @classmethod
    def from_root(cls, project_root: str | Path) -> "ProjectPaths":
        project_root = Path(project_root)
        data_dir = project_root / "data"
        return cls(
            project_root=project_root,
            data_dir=data_dir,
            bids_dir=data_dir / "bids_ackbar",
            fmriprep_dir=data_dir / "fmriprep_output",
            src_dir=project_root / "src",
            common_manifest_path=(
                data_dir
                / "processed"
                / "common_inputs"
                / "restart"
                / "common_inputs_manifest.json"
            ),
        )


def make_analysis_dirs(paths: ProjectPaths, analysis_name: str, smoothing_label: str = "smoothed5mm") -> dict[str, Path]:
    out_dir = paths.data_dir / "processed" / "glm" / analysis_name
    dirs = {
        "out": out_dir,
        "tables": out_dir / "tables",
        "figures": out_dir / "figures",
        "events": paths.data_dir / "preprocessed" / f"relabeled_events_{analysis_name}",
        "first_level": out_dir / f"first_level_{smoothing_label}",
        "first_level_maps": out_dir / f"first_level_{smoothing_label}" / "maps",
        "first_level_designs": out_dir / f"first_level_{smoothing_label}" / "design_matrices",
        "first_level_figures": out_dir / f"first_level_{smoothing_label}" / "figures",
        "second_level": out_dir / f"second_level_from_{smoothing_label}",
        "second_level_maps": out_dir / f"second_level_from_{smoothing_label}" / "maps",
        "thresholded_maps": out_dir / f"second_level_from_{smoothing_label}" / "thresholded_maps",
        "cluster_tables": out_dir / f"second_level_from_{smoothing_label}" / "cluster_tables",
        "restart": out_dir / "restart",
        "interactive": out_dir / "interactive_views",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def get_bold_file(
    paths: ProjectPaths,
    subject: str,
    session: str,
    task: str = "viewing",
    run: str = "01",
    space: str = "MNI152NLin2009cAsym",
) -> Path:
    return (
        paths.fmriprep_dir
        / subject
        / session
        / "func"
        / f"{subject}_{session}_task-{task}_run-{run}_space-{space}_desc-preproc_bold.nii.gz"
    )
