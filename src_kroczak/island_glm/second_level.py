from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from nilearn.glm.second_level import SecondLevelModel
from nilearn.glm.thresholding import threshold_stats_img
from nilearn.reporting import get_clusters_table


def first_level_map_path(output_dirs: dict[str, Path], subject: str, contrast_name: str, smoothing_label: str, map_type: str) -> Path:
    suffix = "zmap" if map_type == "z" else "effect"
    return output_dirs["first_level_maps"] / f"{subject}_{contrast_name}_{smoothing_label}_{suffix}.nii.gz"


def verify_first_level_maps(subjects: list[str], contrast_specs: dict[str, dict], output_dirs: dict[str, Path], smoothing_label: str) -> pd.DataFrame:
    rows = []
    for contrast_name, spec in contrast_specs.items():
        z_paths = [first_level_map_path(output_dirs, s, contrast_name, smoothing_label, "z") for s in subjects]
        effect_paths = [first_level_map_path(output_dirs, s, contrast_name, smoothing_label, "effect") for s in subjects]
        rows.append({
            "contrast": contrast_name,
            "priority": spec.get("priority", ""),
            "n_subjects": len(subjects),
            "n_z_maps": sum(p.exists() for p in z_paths),
            "n_effect_maps": sum(p.exists() for p in effect_paths),
        })
    return pd.DataFrame(rows)


def run_second_level_for_contrast(
    subjects: list[str],
    contrast_name: str,
    output_dirs: dict[str, Path],
    smoothing_label: str,
) -> tuple[Path, Path]:
    effect_paths = [first_level_map_path(output_dirs, s, contrast_name, smoothing_label, "effect") for s in subjects]
    missing = [str(p) for p in effect_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing first-level effect maps for {contrast_name}: {missing[:5]}")

    design_matrix = pd.DataFrame(np.ones((len(effect_paths), 1)), columns=["intercept"])
    model = SecondLevelModel(smoothing_fwhm=None, n_jobs=-1)
    model = model.fit([str(p) for p in effect_paths], design_matrix=design_matrix)

    z_map = model.compute_contrast("intercept", output_type="z_score")
    effect_map = model.compute_contrast("intercept", output_type="effect_size")

    z_path = output_dirs["second_level_maps"] / f"group_{contrast_name}_{smoothing_label}_zmap.nii.gz"
    effect_path = output_dirs["second_level_maps"] / f"group_{contrast_name}_{smoothing_label}_effect.nii.gz"
    z_map.to_filename(str(z_path))
    effect_map.to_filename(str(effect_path))
    return z_path, effect_path


def run_second_level_all_contrasts(subjects: list[str], contrast_specs: dict[str, dict], output_dirs: dict[str, Path], smoothing_label: str) -> pd.DataFrame:
    rows = []
    for contrast_name, spec in contrast_specs.items():
        z_path, effect_path = run_second_level_for_contrast(subjects, contrast_name, output_dirs, smoothing_label)
        rows.append({
            "contrast": contrast_name,
            "priority": spec.get("priority", ""),
            "formula": spec.get("formula", ""),
            "n_subjects": len(subjects),
            "z_map_path": str(z_path),
            "effect_map_path": str(effect_path),
            "z_map_exists": z_path.exists(),
            "effect_map_exists": effect_path.exists(),
        })
    return pd.DataFrame(rows)


def threshold_group_maps(second_level_table: pd.DataFrame, output_dirs: dict[str, Path]) -> pd.DataFrame:
    threshold_specs = [
        ("uncorrected_p001_one_sided", None, 0.001),
        ("fdr_q05_one_sided", "fdr", 0.05),
        ("fdr_q10_one_sided", "fdr", 0.10),
        ("bonferroni_p05_one_sided", "bonferroni", 0.05),
    ]
    rows = []
    for _, row in second_level_table.iterrows():
        z_img = row["z_map_path"]
        for label, height_control, alpha in threshold_specs:
            thresholded_img, z_threshold = threshold_stats_img(
                z_img,
                alpha=alpha,
                height_control=height_control,
                cluster_threshold=0,
                two_sided=False,
            )
            data = thresholded_img.get_fdata()
            n_voxels = int(np.sum(data != 0))
            max_z = float(np.nanmax(data)) if n_voxels > 0 else np.nan
            out_path = output_dirs["thresholded_maps"] / f"group_{row['contrast']}_{label}_thresholded.nii.gz"
            thresholded_img.to_filename(str(out_path))
            rows.append({
                "contrast": row["contrast"],
                "priority": row.get("priority", ""),
                "formula": row.get("formula", ""),
                "threshold_label": label,
                "height_control": "uncorrected" if height_control is None else height_control,
                "alpha": alpha,
                "two_sided": False,
                "z_threshold": float(z_threshold),
                "n_nonzero_voxels": n_voxels,
                "max_z": max_z,
                "thresholded_map_path": str(out_path),
            })
    return pd.DataFrame(rows)


def make_cluster_tables(
    second_level_table: pd.DataFrame,
    output_dirs: dict[str, Path],
    thresholds: tuple[float, ...] = (2.33, 3.0),
    cluster_threshold_voxels: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create cluster tables for each group-level z-map.

    This wrapper is intentionally robust to mixed dtypes returned by
    nilearn.reporting.get_clusters_table(), especially for subpeaks where
    Cluster Size may be stored as strings or missing values.
    """

    all_tables = []
    summary_rows = []

    for _, row in second_level_table.iterrows():
        for stat_threshold in thresholds:
            threshold_label = f"z_gt_{stat_threshold:.2f}"

            table = get_clusters_table(
                row["z_map_path"],
                stat_threshold=stat_threshold,
                cluster_threshold=cluster_threshold_voxels,
                two_sided=False,
            )

            if len(table) > 0:
                table = table.copy()

                # ------------------------------------------------------------
                # Robust numeric conversion
                # ------------------------------------------------------------
                for col in ["X", "Y", "Z", "Peak Stat", "Cluster Size (mm3)"]:
                    if col in table.columns:
                        table[col] = pd.to_numeric(table[col], errors="coerce")

                table["contrast"] = row["contrast"]
                table["priority"] = row.get("priority", "")
                table["formula"] = row.get("formula", "")
                table["threshold_label"] = threshold_label
                table["stat_threshold"] = stat_threshold
                table["cluster_threshold_voxels"] = cluster_threshold_voxels

                peak_stat_numeric = pd.to_numeric(
                    table["Peak Stat"],
                    errors="coerce",
                )

                cluster_size_numeric = pd.to_numeric(
                    table["Cluster Size (mm3)"],
                    errors="coerce",
                )

                max_peak_stat = (
                    float(peak_stat_numeric.max())
                    if peak_stat_numeric.notna().any()
                    else np.nan
                )

                largest_cluster_mm3 = (
                    float(cluster_size_numeric.max())
                    if cluster_size_numeric.notna().any()
                    else np.nan
                )

                n_clusters = int(
                    table["Cluster ID"]
                    .astype(str)
                    .str.match(r"^\d+$")
                    .sum()
                )

            else:
                table = pd.DataFrame(
                    columns=[
                        "Cluster ID",
                        "X",
                        "Y",
                        "Z",
                        "Peak Stat",
                        "Cluster Size (mm3)",
                        "contrast",
                        "priority",
                        "formula",
                        "threshold_label",
                        "stat_threshold",
                        "cluster_threshold_voxels",
                    ]
                )

                max_peak_stat = np.nan
                largest_cluster_mm3 = np.nan
                n_clusters = 0

            table_path = (
                output_dirs["cluster_tables"]
                / f"clusters_{row['contrast']}_{threshold_label}_k{cluster_threshold_voxels}.csv"
            )

            table.to_csv(table_path, index=False)
            all_tables.append(table)

            summary_rows.append(
                {
                    "contrast": row["contrast"],
                    "priority": row.get("priority", ""),
                    "threshold_label": threshold_label,
                    "stat_threshold": stat_threshold,
                    "cluster_threshold_voxels": cluster_threshold_voxels,
                    "n_clusters": n_clusters,
                    "max_peak_stat": max_peak_stat,
                    "largest_cluster_mm3": largest_cluster_mm3,
                    "cluster_table_path": str(table_path),
                }
            )

    details = (
        pd.concat(all_tables, ignore_index=True)
        if all_tables
        else pd.DataFrame()
    )

    summary = pd.DataFrame(summary_rows)

    return summary, details