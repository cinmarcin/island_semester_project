from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from .paths import ProjectPaths


def load_common_manifest(paths: ProjectPaths) -> dict:
    manifest_path = paths.common_manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing common manifest: {manifest_path}. Run notebook 00 first."
        )
    with open(manifest_path, "r") as f:
        return json.load(f)


def load_common_inputs(paths: ProjectPaths) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    manifest = load_common_manifest(paths)
    subject_table = pd.read_csv(manifest["created_outputs"]["subject_table"])
    viewing_item_labels = pd.read_csv(manifest["created_outputs"]["viewing_item_labels"])
    return manifest, subject_table, viewing_item_labels


def get_pre_post_subjects(manifest: dict, no_high_motion: bool = False) -> list[str]:
    key = "pre_post_no_high_motion_N25" if no_high_motion else "pre_post_N27"
    return list(manifest["subjects"][key])
