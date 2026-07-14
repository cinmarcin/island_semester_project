from __future__ import annotations

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import datasets, image


def _ensure_img(maps):
    if isinstance(maps, nib.spatialimages.SpatialImage):
        return maps
    return nib.load(str(maps))


def load_harvard_oxford_atlases():
    cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
    sub = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    return {
        "cortical_img": _ensure_img(cort.maps),
        "cortical_labels": cort.labels,
        "subcortical_img": _ensure_img(sub.maps),
        "subcortical_labels": sub.labels,
    }


def label_coordinate_harvard_oxford(x: float, y: float, z: float, reference_img=None, atlases=None) -> dict:
    if atlases is None:
        atlases = load_harvard_oxford_atlases()

    coord = np.array([[float(x), float(y), float(z)]])
    out = {}
    for prefix, img_key, labels_key in [
        ("harvard_oxford_cortical", "cortical_img", "cortical_labels"),
        ("harvard_oxford_subcortical", "subcortical_img", "subcortical_labels"),
    ]:
        atlas_img = atlases[img_key]
        if reference_img is not None:
            atlas_img = image.resample_to_img(
                atlas_img,
                reference_img,
                interpolation="nearest",
                force_resample=True,
                copy_header=True,
            )
        data = atlas_img.get_fdata()
        ijk = nib.affines.apply_affine(np.linalg.inv(atlas_img.affine), coord)[0]
        ijk = np.round(ijk).astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(data.shape)):
            value = 0
        else:
            value = int(data[tuple(ijk)])
        labels = atlases[labels_key]
        label = labels[value] if 0 <= value < len(labels) else "Unknown"
        out[f"{prefix}_label"] = label
        out[f"{prefix}_value"] = value
    return out


def label_cluster_table(cluster_table: pd.DataFrame, reference_img=None) -> pd.DataFrame:
    if cluster_table.empty:
        return cluster_table.copy()
    atlases = load_harvard_oxford_atlases()
    rows = []
    for _, row in cluster_table.iterrows():
        labels = label_coordinate_harvard_oxford(row["X"], row["Y"], row["Z"], reference_img=reference_img, atlases=atlases)
        rows.append({**row.to_dict(), **labels})
    return pd.DataFrame(rows)
