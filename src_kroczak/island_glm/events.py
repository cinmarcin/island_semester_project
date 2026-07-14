from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def label_face_object_trial(row: pd.Series, phase: str) -> str:
    """Return phase-specific trial type for the faces-vs-objects validation GLM.

    Old human images are labelled as faces; old object images are labelled as objects.
    New images are modelled as phase-specific 'other' so they do not contaminate
    the face-object comparison.
    """
    old_new = row.get("old_new")
    category = row.get("object_category_from_filename")

    if old_new == "old" and category == "human":
        return f"{phase}_face"
    if old_new == "old" and category == "object":
        return f"{phase}_object"
    return f"{phase}_other"


def make_faces_objects_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str,
) -> pd.DataFrame:
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "object_category_from_filename"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(lambda row: label_face_object_trial(row, phase), axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_faces_objects_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phases: tuple[str, str] = ("pre", "post"),
) -> dict[str, Path]:
    events_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in phases:
        events = make_faces_objects_events(viewing_item_labels, subject, phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_faces_objects_events.tsv"
        events.to_csv(path, sep="\t", index=False)
        paths[phase] = path
    return paths


def build_faces_objects_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_events = []
    for subject in subjects:
        for phase in ["pre", "post"]:
            events = make_faces_objects_events(viewing_item_labels, subject, phase)
            if events_dir is not None:
                path = events_dir / f"{subject}_{phase}_task-viewing_faces_objects_events.tsv"
            else:
                path = ""
            counts = events["trial_type"].value_counts().to_dict()
            rows.append({
                "subject": subject,
                "phase": phase,
                "n_events": len(events),
                "n_face": counts.get(f"{phase}_face", 0),
                "n_object": counts.get(f"{phase}_object", 0),
                "n_other": counts.get(f"{phase}_other", 0),
                "events_path": str(path),
            })
            tmp = events.copy()
            tmp["subject"] = subject
            tmp["phase"] = phase
            all_events.append(tmp)
    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)





def label_old_new_trial(row: pd.Series, phase: str) -> str:
    """Return phase-specific old/new trial type for the pre/post old-new GLM."""
    old_new = row.get("old_new")
    if old_new == "old":
        return f"{phase}_old"
    if old_new == "new":
        return f"{phase}_new"
    return f"{phase}_other"


def make_old_new_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str,
) -> pd.DataFrame:
    """Build BIDS-compatible events for one subject and phase.

    Expected output columns: onset, duration, trial_type, modulation.
    Old and new images are modelled explicitly. Any unexpected rows are kept as
    phase-specific 'other' regressors to avoid dropping data silently.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(lambda row: label_old_new_trial(row, phase), axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_old_new_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phases: tuple[str, str] = ("pre", "post"),
) -> dict[str, Path]:
    events_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in phases:
        events = make_old_new_events(viewing_item_labels, subject, phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_old_new_events.tsv"
        events.to_csv(path, sep="\t", index=False)
        paths[phase] = path
    return paths


def build_old_new_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_events = []
    for subject in subjects:
        for phase in ["pre", "post"]:
            events = make_old_new_events(viewing_item_labels, subject, phase)
            path = events_dir / f"{subject}_{phase}_task-viewing_old_new_events.tsv" if events_dir is not None else ""
            counts = events["trial_type"].value_counts().to_dict()
            rows.append({
                "subject": subject,
                "phase": phase,
                "n_events": len(events),
                "n_old": counts.get(f"{phase}_old", 0),
                "n_new": counts.get(f"{phase}_new", 0),
                "n_other": counts.get(f"{phase}_other", 0),
                "events_path": str(path),
            })
            tmp = events.copy()
            tmp["subject"] = subject
            tmp["phase"] = phase
            all_events.append(tmp)
    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)


def _normalise_relevance_code(value):
    """Return 'relevant', 'irrelevant', or None from common relevance encodings."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"r", "re", "rel", "relevant"}:
        return "relevant"
    if value in {"nr", "irrel", "irrelevant", "not_relevant", "nonrelevant", "non-relevant"}:
        return "irrelevant"
    return None


def label_post_relevance_trial(row: pd.Series) -> str:
    """Return POST trial type for relevant/irrelevant old images and new controls."""
    old_new = row.get("old_new")
    relevance = _normalise_relevance_code(row.get("relevance"))

    if old_new == "new":
        return "post_new"
    if old_new == "old" and relevance == "relevant":
        return "post_oldRel"
    if old_new == "old" and relevance == "irrelevant":
        return "post_oldIrrel"
    return "post_other"


def make_post_relevance_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str = "post",
) -> pd.DataFrame:
    """Build BIDS-compatible POST relevance events for one subject.

    Expected output columns: onset, duration, trial_type, modulation.
    Trial types are post_oldRel, post_oldIrrel, post_new, and post_other.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "relevance"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(label_post_relevance_trial, axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_post_relevance_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phase: str = "post",
) -> Path:
    events_dir.mkdir(parents=True, exist_ok=True)
    events = make_post_relevance_events(viewing_item_labels, subject, phase=phase)
    path = events_dir / f"{subject}_{phase}_task-viewing_post_relevance_events.tsv"
    events.to_csv(path, sep="\t", index=False)
    return path


def build_post_relevance_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phase: str = "post",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    all_events = []
    for subject in subjects:
        events = make_post_relevance_events(viewing_item_labels, subject, phase=phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_post_relevance_events.tsv" if events_dir is not None else ""
        counts = events["trial_type"].value_counts().to_dict()
        rows.append({
            "subject": subject,
            "phase": phase,
            "n_events": len(events),
            "n_post_oldRel": counts.get("post_oldRel", 0),
            "n_post_oldIrrel": counts.get("post_oldIrrel", 0),
            "n_post_new": counts.get("post_new", 0),
            "n_post_other": counts.get("post_other", 0),
            "events_path": str(path),
        })
        tmp = events.copy()
        tmp["subject"] = subject
        tmp["phase"] = phase
        all_events.append(tmp)
    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)


def _normalise_episode_code(value):
    """Return e1/e2/e3 or None from an episode value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"1", "e1", "episode1", "episode_1"}:
        return "e1"
    if value in {"2", "e2", "episode2", "episode_2"}:
        return "e2"
    if value in {"3", "e3", "episode3", "episode_3"}:
        return "e3"
    return None


def label_post_episode_trial(row: pd.Series) -> str:
    """Return POST trial type for old episode images and new controls."""
    old_new = str(row.get("old_new", "")).strip().lower()
    episode = _normalise_episode_code(row.get("episode"))

    if old_new == "new":
        return "post_new"
    if old_new == "old" and episode in {"e1", "e2", "e3"}:
        return f"post_{episode}"
    return "post_other"


def make_post_episode_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str = "post",
) -> pd.DataFrame:
    """Build BIDS-compatible POST episode events for one subject.

    Expected output columns: onset, duration, trial_type, modulation.
    Trial types are post_e1, post_e2, post_e3, post_new, and post_other.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "episode"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(label_post_episode_trial, axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_post_episode_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phase: str = "post",
) -> Path:
    """Save one subject's POST episode events."""
    events_dir.mkdir(parents=True, exist_ok=True)
    events = make_post_episode_events(viewing_item_labels, subject, phase=phase)
    path = events_dir / f"{subject}_{phase}_task-viewing_post_episode_events.tsv"
    events.to_csv(path, sep="\t", index=False)
    return path


def build_post_episode_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phase: str = "post",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for POST episode model."""
    rows = []
    all_events = []

    for subject in subjects:
        events = make_post_episode_events(viewing_item_labels, subject, phase=phase)
        path = (
            events_dir / f"{subject}_{phase}_task-viewing_post_episode_events.tsv"
            if events_dir is not None
            else ""
        )
        counts = events["trial_type"].value_counts().to_dict()

        rows.append({
            "subject": subject,
            "phase": phase,
            "n_events": len(events),
            "n_post_e1": counts.get("post_e1", 0),
            "n_post_e2": counts.get("post_e2", 0),
            "n_post_e3": counts.get("post_e3", 0),
            "n_post_new": counts.get("post_new", 0),
            "n_post_other": counts.get("post_other", 0),
            "events_path": str(path),
        })

        tmp = events.copy()
        tmp["subject"] = subject
        tmp["phase"] = phase
        all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)


def _normalise_episode_code(value):
    """Return e1/e2/e3 or None from an episode value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"1", "e1", "episode1", "episode_1"}:
        return "e1"
    if value in {"2", "e2", "episode2", "episode_2"}:
        return "e2"
    if value in {"3", "e3", "episode3", "episode_3"}:
        return "e3"
    return None


def _normalise_relevance_code(value):
    """Return relevant/irrelevant or None from a relevance value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"r", "re", "rel", "relevant", "oldrel"}:
        return "relevant"
    if value in {"nr", "irrel", "irrelevant", "oldirrel", "not_relevant"}:
        return "irrelevant"
    return None


def label_post_relevance_episode_trial(row: pd.Series) -> str:
    """Return POST trial type for relevance x episode old images and new controls.

    Regressors:
    post_oldRelE1, post_oldIrrelE1,
    post_oldRelE2, post_oldIrrelE2,
    post_oldRelE3, post_oldIrrelE3,
    post_new.
    """
    old_new = str(row.get("old_new", "")).strip().lower()
    episode = _normalise_episode_code(row.get("episode"))
    relevance = _normalise_relevance_code(row.get("relevance"))

    if old_new == "new":
        return "post_new"

    if old_new == "old" and episode in {"e1", "e2", "e3"}:
        ep_suffix = episode.upper()
        if relevance == "relevant":
            return f"post_oldRel{ep_suffix}"
        if relevance == "irrelevant":
            return f"post_oldIrrel{ep_suffix}"

    return "post_other"


def make_post_relevance_episode_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str = "post",
) -> pd.DataFrame:
    """Build BIDS-compatible POST relevance x episode events for one subject.

    Expected output columns: onset, duration, trial_type, modulation.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "episode", "relevance"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(label_post_relevance_episode_trial, axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_post_relevance_episode_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phase: str = "post",
) -> Path:
    """Save one subject's POST relevance x episode events."""
    events_dir.mkdir(parents=True, exist_ok=True)
    events = make_post_relevance_episode_events(viewing_item_labels, subject, phase=phase)
    path = events_dir / f"{subject}_{phase}_task-viewing_post_relevance_episode_events.tsv"
    events.to_csv(path, sep="\t", index=False)
    return path


def build_post_relevance_episode_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phase: str = "post",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for POST relevance x episode model."""
    rows = []
    all_events = []
    trial_types = [
        "post_oldRelE1", "post_oldIrrelE1",
        "post_oldRelE2", "post_oldIrrelE2",
        "post_oldRelE3", "post_oldIrrelE3",
        "post_new", "post_other",
    ]

    for subject in subjects:
        events = make_post_relevance_episode_events(viewing_item_labels, subject, phase=phase)
        path = (
            events_dir / f"{subject}_{phase}_task-viewing_post_relevance_episode_events.tsv"
            if events_dir is not None
            else ""
        )
        counts = events["trial_type"].value_counts().to_dict()

        row = {
            "subject": subject,
            "phase": phase,
            "n_events": len(events),
            "events_path": str(path),
        }
        for trial_type in trial_types:
            row[f"n_{trial_type}"] = counts.get(trial_type, 0)
        rows.append(row)

        tmp = events.copy()
        tmp["subject"] = subject
        tmp["phase"] = phase
        all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)



def _normalise_encoding_time_code(value):
    """Return early/middle/late or None from a checkpoint-third value."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"early", "e", "first", "first_third", "1", "third1", "third_1"}:
        return "early"
    if value in {"middle", "mid", "m", "second", "second_third", "2", "third2", "third_2"}:
        return "middle"
    if value in {"late", "l", "last", "third", "third_third", "3", "third3", "third_3"}:
        return "late"

    return None


def _encoding_time_from_checkpoint(value):
    """Return early/middle/late from checkpoint index 1--16."""
    if pd.isna(value):
        return None
    try:
        checkpoint = int(value)
    except Exception:
        return None

    if 1 <= checkpoint <= 5:
        return "early"
    if 6 <= checkpoint <= 10:
        return "middle"
    if 11 <= checkpoint <= 16:
        return "late"
    return None


def get_encoding_time_label(row: pd.Series) -> str | None:
    """Get encoding-time third from the common item-label table.

    Notebook 00 may provide either checkpoint_third or encoding_phase_third.
    If neither is present, this falls back to checkpoint_index.
    """
    for col in ["checkpoint_third", "encoding_phase_third", "encoding_time", "encoding_third"]:
        if col in row.index:
            label = _normalise_encoding_time_code(row.get(col))
            if label is not None:
                return label

    for col in ["checkpoint_index", "checkpoint", "checkpoint_number"]:
        if col in row.index:
            label = _encoding_time_from_checkpoint(row.get(col))
            if label is not None:
                return label

    return None


def label_post_encoding_time_trial(row: pd.Series) -> str:
    """Return POST trial type for old images split by encoding-time third.

    Regressors:
    post_oldEarly, post_oldMiddle, post_oldLate, post_new.
    """
    old_new = str(row.get("old_new", "")).strip().lower()

    if old_new == "new":
        return "post_new"

    if old_new == "old":
        encoding_time = get_encoding_time_label(row)
        if encoding_time == "early":
            return "post_oldEarly"
        if encoding_time == "middle":
            return "post_oldMiddle"
        if encoding_time == "late":
            return "post_oldLate"

    return "post_other"


def make_post_encoding_time_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str = "post",
) -> pd.DataFrame:
    """Build BIDS-compatible POST events for the encoding-time GLM.

    Expected output columns: onset, duration, trial_type, modulation.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    has_time_col = any(
        col in viewing_item_labels.columns
        for col in [
            "checkpoint_third",
            "encoding_phase_third",
            "encoding_time",
            "encoding_third",
            "checkpoint_index",
            "checkpoint",
            "checkpoint_number",
        ]
    )
    if not has_time_col:
        raise ValueError(
            "No encoding-time/checkpoint column found. Expected one of: "
            "checkpoint_third, encoding_phase_third, encoding_time, encoding_third, "
            "checkpoint_index, checkpoint, checkpoint_number."
        )

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(label_post_encoding_time_trial, axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_post_encoding_time_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phase: str = "post",
) -> Path:
    """Save one subject's POST encoding-time event table."""
    events_dir.mkdir(parents=True, exist_ok=True)
    events = make_post_encoding_time_events(viewing_item_labels, subject, phase=phase)
    path = events_dir / f"{subject}_{phase}_task-viewing_post_encoding_time_events.tsv"
    events.to_csv(path, sep="\t", index=False)
    return path


def build_post_encoding_time_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phase: str = "post",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for the POST encoding-time model."""
    rows = []
    all_events = []
    trial_types = [
        "post_oldEarly",
        "post_oldMiddle",
        "post_oldLate",
        "post_new",
        "post_other",
    ]

    for subject in subjects:
        events = make_post_encoding_time_events(viewing_item_labels, subject, phase=phase)
        path = (
            events_dir / f"{subject}_{phase}_task-viewing_post_encoding_time_events.tsv"
            if events_dir is not None
            else ""
        )
        counts = events["trial_type"].value_counts().to_dict()

        row = {
            "subject": subject,
            "phase": phase,
            "n_events": len(events),
            "events_path": str(path),
        }
        for trial_type in trial_types:
            row[f"n_{trial_type}"] = counts.get(trial_type, 0)
        rows.append(row)

        tmp = events.copy()
        tmp["subject"] = subject
        tmp["phase"] = phase
        all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)



def _normalise_episode_code(value):
    """Return e1/e2/e3 or None from an episode value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"1", "e1", "episode1", "episode_1"}:
        return "e1"
    if value in {"2", "e2", "episode2", "episode_2"}:
        return "e2"
    if value in {"3", "e3", "episode3", "episode_3"}:
        return "e3"
    return None


def _normalise_encoding_time_code(value):
    """Return early/middle/late or None from a checkpoint-third value."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"early", "e", "first", "first_third", "1", "third1", "third_1"}:
        return "early"
    if value in {"middle", "mid", "m", "second", "second_third", "2", "third2", "third_2"}:
        return "middle"
    if value in {"late", "l", "last", "third", "third_third", "3", "third3", "third_3"}:
        return "late"

    return None


def _encoding_time_from_checkpoint(value):
    """Return early/middle/late from checkpoint index 1--16."""
    if pd.isna(value):
        return None
    try:
        checkpoint = int(value)
    except Exception:
        return None

    if 1 <= checkpoint <= 5:
        return "early"
    if 6 <= checkpoint <= 10:
        return "middle"
    if 11 <= checkpoint <= 16:
        return "late"
    return None


def get_encoding_time_label(row: pd.Series) -> str | None:
    """Get encoding-time third from the common item-label table.

    Notebook 00 may provide either checkpoint_third or encoding_phase_third.
    If neither is present, this falls back to checkpoint_index.
    """
    for col in ["checkpoint_third", "encoding_phase_third", "encoding_time", "encoding_third"]:
        if col in row.index:
            label = _normalise_encoding_time_code(row.get(col))
            if label is not None:
                return label

    for col in ["checkpoint_index", "checkpoint", "checkpoint_number"]:
        if col in row.index:
            label = _encoding_time_from_checkpoint(row.get(col))
            if label is not None:
                return label

    return None


def label_post_episode_encoding_time_trial(row: pd.Series) -> str:
    """Return POST trial type for episode x encoding-time old images and new controls.

    Regressors:
    post_oldE1Early, post_oldE1Middle, post_oldE1Late,
    post_oldE2Early, post_oldE2Middle, post_oldE2Late,
    post_oldE3Early, post_oldE3Middle, post_oldE3Late,
    post_new.
    """
    old_new = str(row.get("old_new", "")).strip().lower()

    if old_new == "new":
        return "post_new"

    if old_new == "old":
        episode = _normalise_episode_code(row.get("episode"))
        encoding_time = get_encoding_time_label(row)

        if episode in {"e1", "e2", "e3"} and encoding_time in {"early", "middle", "late"}:
            ep_suffix = episode.upper()
            time_suffix = {
                "early": "Early",
                "middle": "Middle",
                "late": "Late",
            }[encoding_time]
            return f"post_old{ep_suffix}{time_suffix}"

    return "post_other"


def make_post_episode_encoding_time_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str = "post",
) -> pd.DataFrame:
    """Build BIDS-compatible POST episode x encoding-time events for one subject.

    Expected output columns: onset, duration, trial_type, modulation.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "episode"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    has_time_col = any(
        col in viewing_item_labels.columns
        for col in [
            "checkpoint_third",
            "encoding_phase_third",
            "encoding_time",
            "encoding_third",
            "checkpoint_index",
            "checkpoint",
            "checkpoint_number",
        ]
    )
    if not has_time_col:
        raise ValueError(
            "No encoding-time/checkpoint column found. Expected one of: "
            "checkpoint_third, encoding_phase_third, encoding_time, encoding_third, "
            "checkpoint_index, checkpoint, checkpoint_number."
        )

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(label_post_episode_encoding_time_trial, axis=1)
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_post_episode_encoding_time_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phase: str = "post",
) -> Path:
    """Save one subject's POST episode x encoding-time event table."""
    events_dir.mkdir(parents=True, exist_ok=True)
    events = make_post_episode_encoding_time_events(viewing_item_labels, subject, phase=phase)
    path = events_dir / f"{subject}_{phase}_task-viewing_post_episode_encoding_time_events.tsv"
    events.to_csv(path, sep="\t", index=False)
    return path


def build_post_episode_encoding_time_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phase: str = "post",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for POST episode x encoding-time model."""
    rows = []
    all_events = []
    trial_types = [
        "post_oldE1Early", "post_oldE1Middle", "post_oldE1Late",
        "post_oldE2Early", "post_oldE2Middle", "post_oldE2Late",
        "post_oldE3Early", "post_oldE3Middle", "post_oldE3Late",
        "post_new", "post_other",
    ]

    for subject in subjects:
        events = make_post_episode_encoding_time_events(viewing_item_labels, subject, phase=phase)
        path = (
            events_dir / f"{subject}_{phase}_task-viewing_post_episode_encoding_time_events.tsv"
            if events_dir is not None
            else ""
        )
        counts = events["trial_type"].value_counts().to_dict()

        row = {
            "subject": subject,
            "phase": phase,
            "n_events": len(events),
            "events_path": str(path),
        }
        for trial_type in trial_types:
            row[f"n_{trial_type}"] = counts.get(trial_type, 0)
        rows.append(row)

        tmp = events.copy()
        tmp["subject"] = subject
        tmp["phase"] = phase
        all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)



def _normalise_episode_code(value):
    """Return e1/e2/e3 or None from an episode value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"1", "e1", "episode1", "episode_1"}:
        return "e1"
    if value in {"2", "e2", "episode2", "episode_2"}:
        return "e2"
    if value in {"3", "e3", "episode3", "episode_3"}:
        return "e3"
    return None


def label_pre_post_episode_trial(row: pd.Series, phase: str) -> str:
    """Return phase-specific trial type for PRE/POST episode controls.

    Old images are labelled by their future/past Island episode (e1/e2/e3).
    New images are modelled as phase-specific new controls.
    """
    old_new = str(row.get("old_new", "")).strip().lower()
    episode = _normalise_episode_code(row.get("episode"))

    if old_new == "new":
        return f"{phase}_new"
    if old_new == "old" and episode in {"e1", "e2", "e3"}:
        return f"{phase}_{episode}"
    return f"{phase}_other"


def make_pre_post_episode_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str,
) -> pd.DataFrame:
    """Build BIDS-compatible PRE or POST episode events for one subject.

    Expected output columns: onset, duration, trial_type, modulation.
    Trial types are pre_e1/pre_e2/pre_e3/pre_new for PRE and
    post_e1/post_e2/post_e3/post_new for POST.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "episode"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(
        lambda row: label_pre_post_episode_trial(row, phase=phase),
        axis=1,
    )
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_pre_post_episode_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phases: tuple[str, str] = ("pre", "post"),
) -> dict[str, Path]:
    """Save PRE and POST episode-control events for one subject."""
    events_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in phases:
        events = make_pre_post_episode_events(viewing_item_labels, subject, phase=phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_pre_post_episode_events.tsv"
        events.to_csv(path, sep="\t", index=False)
        paths[phase] = path
    return paths


def build_pre_post_episode_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phases: tuple[str, str] = ("pre", "post"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for the PRE/POST episode-control model."""
    rows = []
    all_events = []

    for subject in subjects:
        for phase in phases:
            events = make_pre_post_episode_events(viewing_item_labels, subject, phase=phase)
            path = (
                events_dir / f"{subject}_{phase}_task-viewing_pre_post_episode_events.tsv"
                if events_dir is not None
                else ""
            )
            counts = events["trial_type"].value_counts().to_dict()

            rows.append({
                "subject": subject,
                "phase": phase,
                "n_events": len(events),
                f"n_{phase}_e1": counts.get(f"{phase}_e1", 0),
                f"n_{phase}_e2": counts.get(f"{phase}_e2", 0),
                f"n_{phase}_e3": counts.get(f"{phase}_e3", 0),
                f"n_{phase}_new": counts.get(f"{phase}_new", 0),
                f"n_{phase}_other": counts.get(f"{phase}_other", 0),
                "events_path": str(path),
            })

            tmp = events.copy()
            tmp["subject"] = subject
            tmp["phase"] = phase
            all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)





def _normalise_encoding_time_code(value):
    """Return early/middle/late or None from a checkpoint-third value."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"early", "e", "first", "first_third", "1", "third1", "third_1"}:
        return "early"
    if value in {"middle", "mid", "m", "second", "second_third", "2", "third2", "third_2"}:
        return "middle"
    if value in {"late", "l", "last", "third", "third_third", "3", "third3", "third_3"}:
        return "late"

    return None


def _encoding_time_from_checkpoint(value):
    """Return early/middle/late from checkpoint index 1--16."""
    if pd.isna(value):
        return None
    try:
        checkpoint = int(value)
    except Exception:
        return None

    if 1 <= checkpoint <= 5:
        return "early"
    if 6 <= checkpoint <= 10:
        return "middle"
    if 11 <= checkpoint <= 16:
        return "late"
    return None


def get_encoding_time_label(row: pd.Series) -> str | None:
    """Get encoding-time third from the common item-label table.

    Notebook 00 may provide either checkpoint_third or encoding_phase_third.
    If neither is present, this falls back to checkpoint_index.
    """
    for col in ["checkpoint_third", "encoding_phase_third", "encoding_time", "encoding_third"]:
        if col in row.index:
            label = _normalise_encoding_time_code(row.get(col))
            if label is not None:
                return label

    for col in ["checkpoint_index", "checkpoint", "checkpoint_number"]:
        if col in row.index:
            label = _encoding_time_from_checkpoint(row.get(col))
            if label is not None:
                return label

    return None


def label_pre_post_encoding_space_trial(row: pd.Series, phase: str) -> str:
    """Return phase-specific trial type for PRE/POST encoding-time controls.

    Old images are labelled by their future/past route position during Island
    encoding: early, middle, late. New images are modelled as phase-specific
    controls.
    """
    old_new = str(row.get("old_new", "")).strip().lower()

    if old_new == "new":
        return f"{phase}_new"

    if old_new == "old":
        encoding_time = get_encoding_time_label(row)
        if encoding_time == "early":
            return f"{phase}_oldEarly"
        if encoding_time == "middle":
            return f"{phase}_oldMiddle"
        if encoding_time == "late":
            return f"{phase}_oldLate"

    return f"{phase}_other"


def make_pre_post_encoding_space_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str,
) -> pd.DataFrame:
    """Build BIDS-compatible PRE or POST encoding-time events.

    Expected output columns: onset, duration, trial_type, modulation.
    Trial types are pre_oldEarly/pre_oldMiddle/pre_oldLate/pre_new and
    post_oldEarly/post_oldMiddle/post_oldLate/post_new.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    has_time_col = any(
        col in viewing_item_labels.columns
        for col in [
            "checkpoint_third",
            "encoding_phase_third",
            "encoding_time",
            "encoding_third",
            "checkpoint_index",
            "checkpoint",
            "checkpoint_number",
        ]
    )
    if not has_time_col:
        raise ValueError(
            "No encoding-time/checkpoint column found. Expected one of: "
            "checkpoint_third, encoding_phase_third, encoding_time, encoding_third, "
            "checkpoint_index, checkpoint, checkpoint_number."
        )

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(
        lambda row: label_pre_post_encoding_space_trial(row, phase=phase),
        axis=1,
    )
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_pre_post_encoding_space_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phases: tuple[str, str] = ("pre", "post"),
) -> dict[str, Path]:
    """Save PRE and POST encoding-time-control events for one subject."""
    events_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in phases:
        events = make_pre_post_encoding_space_events(viewing_item_labels, subject, phase=phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_pre_post_encoding_time_events.tsv"
        events.to_csv(path, sep="\t", index=False)
        paths[phase] = path
    return paths


def build_pre_post_encoding_space_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phases: tuple[str, str] = ("pre", "post"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for the PRE/POST encoding-time-control model."""
    rows = []
    all_events = []
    suffixes = ["oldEarly", "oldMiddle", "oldLate", "new", "other"]

    for subject in subjects:
        for phase in phases:
            events = make_pre_post_encoding_space_events(viewing_item_labels, subject, phase=phase)
            path = (
                events_dir / f"{subject}_{phase}_task-viewing_pre_post_encoding_space_events.tsv"
                if events_dir is not None
                else ""
            )
            counts = events["trial_type"].value_counts().to_dict()

            row = {
                "subject": subject,
                "phase": phase,
                "n_events": len(events),
                "events_path": str(path),
            }
            for suffix in suffixes:
                trial_type = f"{phase}_{suffix}"
                row[f"n_{trial_type}"] = counts.get(trial_type, 0)
            rows.append(row)

            tmp = events.copy()
            tmp["subject"] = subject
            tmp["phase"] = phase
            all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)




def _normalise_episode_code(value):
    """Return e1/e2/e3 or None from an episode value."""
    if pd.isna(value):
        return None
    value = str(value).strip().lower()
    if value in {"1", "e1", "episode1", "episode_1"}:
        return "e1"
    if value in {"2", "e2", "episode2", "episode_2"}:
        return "e2"
    if value in {"3", "e3", "episode3", "episode_3"}:
        return "e3"
    return None


def _normalise_encoding_space_code(value):
    """Return early/middle/late or None from a checkpoint-third value."""
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"early", "e", "first", "first_third", "1", "third1", "third_1"}:
        return "early"
    if value in {"middle", "mid", "m", "second", "second_third", "2", "third2", "third_2"}:
        return "middle"
    if value in {"late", "l", "last", "third", "third_third", "3", "third3", "third_3"}:
        return "late"

    return None


def _encoding_space_from_checkpoint(value):
    """Return early/middle/late from checkpoint index 1--16."""
    if pd.isna(value):
        return None
    try:
        checkpoint = int(value)
    except Exception:
        return None

    if 1 <= checkpoint <= 5:
        return "early"
    if 6 <= checkpoint <= 10:
        return "middle"
    if 11 <= checkpoint <= 16:
        return "late"
    return None


def get_encoding_space_label(row: pd.Series) -> str | None:
    """Get encoding-space third from the common item-label table.

    Notebook 00 may provide checkpoint_third or encoding_phase_third.
    If neither is present, this falls back to checkpoint_index.
    """
    for col in ["checkpoint_third", "encoding_phase_third", "encoding_space", "encoding_time", "encoding_third"]:
        if col in row.index:
            label = _normalise_encoding_space_code(row.get(col))
            if label is not None:
                return label

    for col in ["checkpoint_index", "checkpoint", "checkpoint_number"]:
        if col in row.index:
            label = _encoding_space_from_checkpoint(row.get(col))
            if label is not None:
                return label

    return None


def label_pre_post_time_space_trial(row: pd.Series, phase: str) -> str:
    """Return phase-specific trial type for TIME × SPACE controls.

    Old images are labelled by both temporal episode (E1/E2/E3) and route-position
    / encoding-space bin (Early/Middle/Late). New images are phase-specific controls.
    """
    old_new = str(row.get("old_new", "")).strip().lower()

    if old_new == "new":
        return f"{phase}_new"

    if old_new == "old":
        episode = _normalise_episode_code(row.get("episode"))
        encoding_space = get_encoding_space_label(row)

        if episode in {"e1", "e2", "e3"} and encoding_space in {"early", "middle", "late"}:
            ep_suffix = episode.upper()
            space_suffix = {
                "early": "Early",
                "middle": "Middle",
                "late": "Late",
            }[encoding_space]
            return f"{phase}_old{ep_suffix}{space_suffix}"

    return f"{phase}_other"


def make_pre_post_time_space_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    phase: str,
) -> pd.DataFrame:
    """Build BIDS-compatible PRE or POST TIME × SPACE events.

    Expected output columns: onset, duration, trial_type, modulation.

    Trial types are:
    pre_oldE1Early ... pre_oldE3Late, pre_new
    post_oldE1Early ... post_oldE3Late, post_new.
    """
    required_cols = ["subject", "phase", "onset", "duration", "old_new", "episode"]
    missing = [col for col in required_cols if col not in viewing_item_labels.columns]
    if missing:
        raise ValueError(f"Missing columns in viewing_item_labels: {missing}")

    has_space_col = any(
        col in viewing_item_labels.columns
        for col in [
            "checkpoint_third",
            "encoding_phase_third",
            "encoding_space",
            "encoding_time",
            "encoding_third",
            "checkpoint_index",
            "checkpoint",
            "checkpoint_number",
        ]
    )
    if not has_space_col:
        raise ValueError(
            "No encoding-space/checkpoint column found. Expected one of: "
            "checkpoint_third, encoding_phase_third, encoding_space, encoding_time, "
            "encoding_third, checkpoint_index, checkpoint, checkpoint_number."
        )

    events = viewing_item_labels.query("subject == @subject and phase == @phase").copy()
    if events.empty:
        raise ValueError(f"No events found for {subject}, phase={phase}")

    events["trial_type"] = events.apply(
        lambda row: label_pre_post_time_space_trial(row, phase=phase),
        axis=1,
    )
    events["modulation"] = 1.0

    out = events[["onset", "duration", "trial_type", "modulation"]].copy()
    out = out.sort_values("onset").reset_index(drop=True)
    return out


def save_pre_post_time_space_subject_events(
    viewing_item_labels: pd.DataFrame,
    subject: str,
    events_dir: Path,
    phases: tuple[str, str] = ("pre", "post"),
) -> dict[str, Path]:
    """Save PRE and POST TIME × SPACE event files for one subject."""
    events_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for phase in phases:
        events = make_pre_post_time_space_events(viewing_item_labels, subject, phase=phase)
        path = events_dir / f"{subject}_{phase}_task-viewing_pre_post_time_space_events.tsv"
        events.to_csv(path, sep="\t", index=False)
        paths[phase] = path
    return paths


def build_pre_post_time_space_event_qc(
    viewing_item_labels: pd.DataFrame,
    subjects: list[str],
    events_dir: Path | None = None,
    phases: tuple[str, str] = ("pre", "post"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create event-count QC for the PRE/POST TIME × SPACE model."""
    rows = []
    all_events = []

    cell_suffixes = [
        "oldE1Early", "oldE1Middle", "oldE1Late",
        "oldE2Early", "oldE2Middle", "oldE2Late",
        "oldE3Early", "oldE3Middle", "oldE3Late",
        "new", "other",
    ]

    for subject in subjects:
        for phase in phases:
            events = make_pre_post_time_space_events(viewing_item_labels, subject, phase=phase)
            path = (
                events_dir / f"{subject}_{phase}_task-viewing_pre_post_time_space_events.tsv"
                if events_dir is not None
                else ""
            )
            counts = events["trial_type"].value_counts().to_dict()

            row = {
                "subject": subject,
                "phase": phase,
                "n_events": len(events),
                "events_path": str(path),
            }
            for suffix in cell_suffixes:
                trial_type = f"{phase}_{suffix}"
                row[f"n_{trial_type}"] = counts.get(trial_type, 0)
            rows.append(row)

            tmp = events.copy()
            tmp["subject"] = subject
            tmp["phase"] = phase
            all_events.append(tmp)

    return pd.DataFrame(rows), pd.concat(all_events, ignore_index=True)





