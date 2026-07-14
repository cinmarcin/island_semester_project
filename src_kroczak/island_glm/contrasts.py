from __future__ import annotations

import numpy as np
import pandas as pd


def make_faces_objects_contrasts() -> dict[str, dict]:
    """Contrasts for the streamlined pre/post faces-vs-objects validation GLM."""
    return {
        "pre_faces_gt_objects": {
            "priority": "pre_validation",
            "formula": r"$$\beta_{\mathrm{pre,face}}-\beta_{\mathrm{pre,object}}$$",
            "interpretation": "Pre-viewing face > object validation contrast.",
            "weights": {"pre_face": 1.0, "pre_object": -1.0},
        },
        "post_faces_gt_objects": {
            "priority": "post_validation",
            "formula": r"$$\beta_{\mathrm{post,face}}-\beta_{\mathrm{post,object}}$$",
            "interpretation": "Post-viewing face > object validation contrast.",
            "weights": {"post_face": 1.0, "post_object": -1.0},
        },
        "both_faces_gt_objects": {
            "priority": "combined_validation",
            "formula": r"$$\frac{1}{2}(\beta_{\mathrm{pre,face}}+\beta_{\mathrm{post,face}})-\frac{1}{2}(\beta_{\mathrm{pre,object}}+\beta_{\mathrm{post,object}})$$",
            "interpretation": "Average face > object effect across PRE and POST viewing.",
            "weights": {
                "pre_face": 0.5,
                "post_face": 0.5,
                "pre_object": -0.5,
                "post_object": -0.5,
            },
        },
        "faces_gt_objects_post_gt_pre": {
            "priority": "post_pre_interaction",
            "formula": r"$$(\beta_{\mathrm{post,face}}-\beta_{\mathrm{post,object}})-(\beta_{\mathrm{pre,face}}-\beta_{\mathrm{pre,object}})$$",
            "interpretation": "Tests whether the face > object effect is stronger at POST than PRE.",
            "weights": {
                "post_face": 1.0,
                "post_object": -1.0,
                "pre_face": -1.0,
                "pre_object": 1.0,
            },
        },
        "objects_gt_faces_both": {
            "priority": "reverse_control",
            "formula": r"$$\frac{1}{2}(\beta_{\mathrm{pre,object}}+\beta_{\mathrm{post,object}})-\frac{1}{2}(\beta_{\mathrm{pre,face}}+\beta_{\mathrm{post,face}})$$",
            "interpretation": "Reverse validation control: average object > face effect across PRE and POST.",
            "weights": {
                "pre_object": 0.5,
                "post_object": 0.5,
                "pre_face": -0.5,
                "post_face": -0.5,
            },
        },
    }


def contrast_specs_to_table(contrast_specs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, spec in contrast_specs.items():
        row = {
            "contrast": name,
            "priority": spec.get("priority", ""),
            "formula": spec.get("formula", ""),
            "interpretation": spec.get("interpretation", ""),
        }
        for reg, weight in spec["weights"].items():
            row[reg] = weight
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def build_contrast_vector(design_columns, weights: dict[str, float]) -> tuple[np.ndarray, list[str]]:
    vec = np.zeros(len(design_columns), dtype=float)
    missing = []
    for regressor, weight in weights.items():
        if regressor in design_columns:
            vec[list(design_columns).index(regressor)] = float(weight)
        else:
            missing.append(regressor)
    return vec, missing


def preview_contrast_vectors(design_matrices: list[pd.DataFrame], contrast_specs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for contrast_name, spec in contrast_specs.items():
        for run_idx, design_matrix in enumerate(design_matrices, start=1):
            vec, missing = build_contrast_vector(design_matrix.columns, spec["weights"])
            nz = np.where(vec != 0)[0]
            rows.append({
                "contrast": contrast_name,
                "priority": spec.get("priority", ""),
                "run": run_idx,
                "n_design_columns": len(design_matrix.columns),
                "n_nonzero_weights": len(nz),
                "nonzero_regressors": ";".join([str(design_matrix.columns[i]) for i in nz]),
                "nonzero_weights": ";".join([str(vec[i]) for i in nz]),
                "missing_regressors_expected": ";".join(missing),
                "is_null_vector": bool(np.all(vec == 0)),
            })
    return pd.DataFrame(rows)




def make_old_new_pre_post_contrasts() -> dict[str, dict]:
    """Contrasts for PRE old/new, POST old/new, and POST-PRE old-new interaction."""
    return {
        "pre_old_gt_new": {
            "priority": "02_pre_old_vs_new",
            "formula": r"$$\beta_{\mathrm{pre,old}}-\beta_{\mathrm{pre,new}}$$",
            "interpretation": "Baseline old > new difference during PRE viewing.",
            "weights": {"pre_old": 1.0, "pre_new": -1.0},
        },
        "post_old_gt_new": {
            "priority": "03_post_old_vs_new",
            "formula": r"$$\beta_{\mathrm{post,old}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "Old > new difference during POST viewing after island exposure.",
            "weights": {"post_old": 1.0, "post_new": -1.0},
        },
        "old_gt_new_post_gt_pre": {
            "priority": "04_pre_post_old_new_interaction",
            "formula": r"$$(\beta_{\mathrm{post,old}}-\beta_{\mathrm{post,new}})-(\beta_{\mathrm{pre,old}}-\beta_{\mathrm{pre,new}})$$",
            "interpretation": "Tests whether the old-new difference is stronger at POST than PRE.",
            "weights": {
                "post_old": 1.0,
                "post_new": -1.0,
                "pre_old": -1.0,
                "pre_new": 1.0,
            },
        },
        "old_post_gt_pre": {
            "priority": "descriptive_old_post_pre",
            "formula": r"$$\beta_{\mathrm{post,old}}-\beta_{\mathrm{pre,old}}$$",
            "interpretation": "Descriptive POST > PRE change for old images only.",
            "weights": {"post_old": 1.0, "pre_old": -1.0},
        },
        "new_post_gt_pre": {
            "priority": "descriptive_new_post_pre",
            "formula": r"$$\beta_{\mathrm{post,new}}-\beta_{\mathrm{pre,new}}$$",
            "interpretation": "Descriptive POST > PRE change for new images only.",
            "weights": {"post_new": 1.0, "pre_new": -1.0},
        },
        "all_post_gt_pre": {
            "priority": "descriptive_all_post_pre",
            "formula": r"$$\frac{1}{2}(\beta_{\mathrm{post,old}}+\beta_{\mathrm{post,new}})-\frac{1}{2}(\beta_{\mathrm{pre,old}}+\beta_{\mathrm{pre,new}})$$",
            "interpretation": "Descriptive average POST > PRE viewing-session effect.",
            "weights": {
                "post_old": 0.5,
                "post_new": 0.5,
                "pre_old": -0.5,
                "pre_new": -0.5,
            },
        },
    }


def make_post_relevance_old_new_contrasts() -> dict[str, dict]:
    """POST-only relevance/old-new contrasts.

    Regressors expected in the design matrix:
    post_oldRel, post_oldIrrel, post_new.
    """
    return {
        "post_oldRel_gt_new": {
            "priority": "05_post_relevant_old_vs_new",
            "formula": r"$$\beta_{\mathrm{post,oldRel}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST relevant old images > POST new images.",
            "weights": {"post_oldRel": 1.0, "post_new": -1.0},
        },
        "post_oldIrrel_gt_new": {
            "priority": "05_post_irrelevant_old_vs_new",
            "formula": r"$$\beta_{\mathrm{post,oldIrrel}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST irrelevant old images > POST new images.",
            "weights": {"post_oldIrrel": 1.0, "post_new": -1.0},
        },
        "post_rel_gt_irrel": {
            "priority": "05_post_relevance_effect",
            "formula": r"$$\beta_{\mathrm{post,oldRel}}-\beta_{\mathrm{post,oldIrrel}}$$",
            "interpretation": "POST relevant old images > POST irrelevant old images.",
            "weights": {"post_oldRel": 1.0, "post_oldIrrel": -1.0},
        },
        "post_irrel_gt_rel": {
            "priority": "05_post_reverse_relevance_control",
            "formula": r"$$\beta_{\mathrm{post,oldIrrel}}-\beta_{\mathrm{post,oldRel}}$$",
            "interpretation": "Reverse control: POST irrelevant old images > POST relevant old images.",
            "weights": {"post_oldIrrel": 1.0, "post_oldRel": -1.0},
        },
        "post_oldRel_gt_oldIrrelNew_avg": {
            "priority": "05_post_relevant_vs_other_controls",
            "formula": r"$$\beta_{\mathrm{post,oldRel}}-\frac{1}{2}(\beta_{\mathrm{post,oldIrrel}}+\beta_{\mathrm{post,new}})$$",
            "interpretation": "POST relevant old images > average of irrelevant old and new controls.",
            "weights": {"post_oldRel": 1.0, "post_oldIrrel": -0.5, "post_new": -0.5},
        },
        "post_old_gt_new_avg": {
            "priority": "05_post_all_old_vs_new_average",
            "formula": r"$$\frac{1}{2}(\beta_{\mathrm{post,oldRel}}+\beta_{\mathrm{post,oldIrrel}})-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST old images, averaging relevant and irrelevant, > POST new images.",
            "weights": {"post_oldRel": 0.5, "post_oldIrrel": 0.5, "post_new": -1.0},
        },
    }



def make_post_episode_contrasts() -> dict[str, dict]:
    """POST-only episode contrasts for e1/e2/e3 old images and new controls.

    Regressors expected in the design matrix:
    post_e1, post_e2, post_e3, post_new.
    """
    contrasts = {
        "post_e1_gt_new": {
            "priority": "06_episode_vs_new",
            "formula": r"$$\beta_{\mathrm{post,e1}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST old episode-1 images > POST new images.",
            "weights": {"post_e1": 1.0, "post_new": -1.0},
        },
        "post_e2_gt_new": {
            "priority": "06_episode_vs_new",
            "formula": r"$$\beta_{\mathrm{post,e2}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST old episode-2 images > POST new images.",
            "weights": {"post_e2": 1.0, "post_new": -1.0},
        },
        "post_e3_gt_new": {
            "priority": "06_episode_vs_new",
            "formula": r"$$\beta_{\mathrm{post,e3}}-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST old episode-3 images > POST new images.",
            "weights": {"post_e3": 1.0, "post_new": -1.0},
        },
        "post_episode_old_gt_new_avg": {
            "priority": "06_episode_average_old_vs_new",
            "formula": r"$$\frac{1}{3}(\beta_{\mathrm{post,e1}}+\beta_{\mathrm{post,e2}}+\beta_{\mathrm{post,e3}})-\beta_{\mathrm{post,new}}$$",
            "interpretation": "POST old images averaged across episodes > POST new images.",
            "weights": {"post_e1": 1/3, "post_e2": 1/3, "post_e3": 1/3, "post_new": -1.0},
        },
    }

    # All ordered pairwise episode contrasts.
    pairwise_specs = [
        ("e1", "e2"),
        ("e2", "e1"),
        ("e1", "e3"),
        ("e3", "e1"),
        ("e2", "e3"),
        ("e3", "e2"),
    ]

    for a, b in pairwise_specs:
        name = f"post_{a}_gt_{b}"
        contrasts[name] = {
            "priority": "06_episode_pairwise",
            "formula": rf"$$\beta_{{\mathrm{{post,{a}}}}}-\beta_{{\mathrm{{post,{b}}}}}$$",
            "interpretation": f"POST old episode {a.upper()} images > POST old episode {b.upper()} images.",
            "weights": {f"post_{a}": 1.0, f"post_{b}": -1.0},
        }

    # Optional descriptive adjacent average contrast.
    contrasts["post_late_episodes_gt_e1_avg"] = {
        "priority": "06_episode_descriptive_late_vs_early",
        "formula": r"$$\frac{1}{2}(\beta_{\mathrm{post,e2}}+\beta_{\mathrm{post,e3}})-\beta_{\mathrm{post,e1}}$$",
        "interpretation": "POST later episodes averaged across E2/E3 > POST E1.",
        "weights": {"post_e2": 0.5, "post_e3": 0.5, "post_e1": -1.0},
    }

    return contrasts


def make_post_relevance_episode_contrasts() -> dict[str, dict]:
    """POST-only relevance x episode contrasts.

    Expected regressors:
    post_oldRelE1, post_oldIrrelE1,
    post_oldRelE2, post_oldIrrelE2,
    post_oldRelE3, post_oldIrrelE3,
    post_new.
    """
    contrasts = {}

    # Episode-specific relevance effects.
    for ep in ["E1", "E2", "E3"]:
        ep_lower = ep.lower()
        contrasts[f"post_rel_gt_irrel_{ep_lower}"] = {
            "priority": "07_episode_specific_relevance_effect",
            "formula": rf"$$\beta_{{\mathrm{{post,oldRel{ep}}}}}-\beta_{{\mathrm{{post,oldIrrel{ep}}}}}$$",
            "interpretation": f"POST relevant old images > irrelevant old images within {ep}.",
            "weights": {f"post_oldRel{ep}": 1.0, f"post_oldIrrel{ep}": -1.0},
        }
        contrasts[f"post_irrel_gt_rel_{ep_lower}"] = {
            "priority": "07_episode_specific_reverse_control",
            "formula": rf"$$\beta_{{\mathrm{{post,oldIrrel{ep}}}}}-\beta_{{\mathrm{{post,oldRel{ep}}}}}$$",
            "interpretation": f"POST irrelevant old images > relevant old images within {ep}.",
            "weights": {f"post_oldIrrel{ep}": 1.0, f"post_oldRel{ep}": -1.0},
        }

    contrasts["post_rel_gt_irrel_avg"] = {
        "priority": "07_average_relevance_effect",
        "formula": r"$$\frac{1}{3}(\beta_{\mathrm{post,oldRelE1}}+\beta_{\mathrm{post,oldRelE2}}+\beta_{\mathrm{post,oldRelE3}})-\frac{1}{3}(\beta_{\mathrm{post,oldIrrelE1}}+\beta_{\mathrm{post,oldIrrelE2}}+\beta_{\mathrm{post,oldIrrelE3}})$$",
        "interpretation": "POST relevant old images > irrelevant old images, averaged across episodes.",
        "weights": {
            "post_oldRelE1": 1/3, "post_oldRelE2": 1/3, "post_oldRelE3": 1/3,
            "post_oldIrrelE1": -1/3, "post_oldIrrelE2": -1/3, "post_oldIrrelE3": -1/3,
        },
    }

    contrasts["post_irrel_gt_rel_avg"] = {
        "priority": "07_average_reverse_control",
        "formula": r"$$\frac{1}{3}(\beta_{\mathrm{post,oldIrrelE1}}+\beta_{\mathrm{post,oldIrrelE2}}+\beta_{\mathrm{post,oldIrrelE3}})-\frac{1}{3}(\beta_{\mathrm{post,oldRelE1}}+\beta_{\mathrm{post,oldRelE2}}+\beta_{\mathrm{post,oldRelE3}})$$",
        "interpretation": "POST irrelevant old images > relevant old images, averaged across episodes.",
        "weights": {
            "post_oldIrrelE1": 1/3, "post_oldIrrelE2": 1/3, "post_oldIrrelE3": 1/3,
            "post_oldRelE1": -1/3, "post_oldRelE2": -1/3, "post_oldRelE3": -1/3,
        },
    }

    # All ordered relevance x episode interactions.
    pairwise = [
        ("E1", "E2"), ("E2", "E1"),
        ("E1", "E3"), ("E3", "E1"),
        ("E2", "E3"), ("E3", "E2"),
    ]
    for a, b in pairwise:
        name = f"post_relIrrel_{a.lower()}_gt_{b.lower()}"
        contrasts[name] = {
            "priority": "07_relevance_by_episode_interaction",
            "formula": rf"$$(\beta_{{\mathrm{{post,oldRel{a}}}}}-\beta_{{\mathrm{{post,oldIrrel{a}}}}})-(\beta_{{\mathrm{{post,oldRel{b}}}}}-\beta_{{\mathrm{{post,oldIrrel{b}}}}})$$",
            "interpretation": f"Relevance effect in {a} > relevance effect in {b}.",
            "weights": {
                f"post_oldRel{a}": 1.0,
                f"post_oldIrrel{a}": -1.0,
                f"post_oldRel{b}": -1.0,
                f"post_oldIrrel{b}": 1.0,
            },
        }

    # Descriptive controls: relevant or irrelevant episode old images versus new controls.
    for ep in ["E1", "E2", "E3"]:
        ep_lower = ep.lower()
        contrasts[f"post_oldRel{ep}_gt_new"] = {
            "priority": "07_relevant_episode_vs_new_control",
            "formula": rf"$$\beta_{{\mathrm{{post,oldRel{ep}}}}}-\beta_{{\mathrm{{post,new}}}}$$",
            "interpretation": f"POST relevant old {ep} images > new images.",
            "weights": {f"post_oldRel{ep}": 1.0, "post_new": -1.0},
        }
        contrasts[f"post_oldIrrel{ep}_gt_new"] = {
            "priority": "07_irrelevant_episode_vs_new_control",
            "formula": rf"$$\beta_{{\mathrm{{post,oldIrrel{ep}}}}}-\beta_{{\mathrm{{post,new}}}}$$",
            "interpretation": f"POST irrelevant old {ep} images > new images.",
            "weights": {f"post_oldIrrel{ep}": 1.0, "post_new": -1.0},
        }

    return contrasts


def make_post_encoding_time_contrasts() -> dict[str, dict]:
    """POST-only old-image encoding-time contrasts.

    Expected regressors:
    post_oldEarly, post_oldMiddle, post_oldLate, post_new.
    """
    contrasts = {}

    timepoints = [
        ("early", "Early"),
        ("middle", "Middle"),
        ("late", "Late"),
    ]

    # Old images from each encoding-time bin versus new controls.
    for label, suffix in timepoints:
        contrasts[f"post_old{suffix}_gt_new"] = {
            "priority": "08_encoding_time_vs_new_control",
            "formula": rf"$$\beta_{{\mathrm{{post,old{suffix}}}}}-\beta_{{\mathrm{{post,new}}}}$$",
            "interpretation": f"POST response to old images encoded in the {label} part of the route > new images.",
            "weights": {f"post_old{suffix}": 1.0, "post_new": -1.0},
        }

    contrasts["post_oldEncoding_avg_gt_new"] = {
        "priority": "08_all_old_encoding_time_vs_new_average",
        "formula": r"$$\frac{1}{3}(\beta_{\mathrm{post,oldEarly}}+\beta_{\mathrm{post,oldMiddle}}+\beta_{\mathrm{post,oldLate}})-\beta_{\mathrm{post,new}}$$",
        "interpretation": "POST response to all old images split by encoding time, averaged across early/middle/late, > new images.",
        "weights": {
            "post_oldEarly": 1/3,
            "post_oldMiddle": 1/3,
            "post_oldLate": 1/3,
            "post_new": -1.0,
        },
    }

    # All ordered pairwise contrasts between encoding-time bins.
    pairwise = [
        ("Early", "Middle"), ("Middle", "Early"),
        ("Early", "Late"), ("Late", "Early"),
        ("Middle", "Late"), ("Late", "Middle"),
    ]
    for a, b in pairwise:
        name = f"post_old{a}_gt_old{b}"
        contrasts[name] = {
            "priority": "08_pairwise_encoding_time_contrast",
            "formula": rf"$$\beta_{{\mathrm{{post,old{a}}}}}-\beta_{{\mathrm{{post,old{b}}}}}$$",
            "interpretation": f"POST response to old images encoded in {a.lower()} checkpoints > old images encoded in {b.lower()} checkpoints.",
            "weights": {f"post_old{a}": 1.0, f"post_old{b}": -1.0},
        }

    # Optional simple temporal trend controls.
    contrasts["post_oldLate_gt_oldEarly_linear_control"] = {
        "priority": "08_temporal_trend_control",
        "formula": r"$$\beta_{\mathrm{post,oldLate}}-\beta_{\mathrm{post,oldEarly}}$$",
        "interpretation": "Simple late > early encoding-time trend control.",
        "weights": {"post_oldLate": 1.0, "post_oldEarly": -1.0},
    }

    contrasts["post_oldEarly_gt_oldLate_linear_control"] = {
        "priority": "08_temporal_trend_reverse_control",
        "formula": r"$$\beta_{\mathrm{post,oldEarly}}-\beta_{\mathrm{post,oldLate}}$$",
        "interpretation": "Simple early > late encoding-time trend control.",
        "weights": {"post_oldEarly": 1.0, "post_oldLate": -1.0},
    }

    return contrasts




def make_post_episode_encoding_time_contrasts() -> dict[str, dict]:
    """POST-only episode x encoding-time contrasts.

    Expected regressors:
    post_oldE1Early, post_oldE1Middle, post_oldE1Late,
    post_oldE2Early, post_oldE2Middle, post_oldE2Late,
    post_oldE3Early, post_oldE3Middle, post_oldE3Late,
    post_new.
    """
    contrasts = {}

    episodes = ["E1", "E2", "E3"]
    time_bins = ["Early", "Middle", "Late"]
    episode_pairs = [
        ("E1", "E2"), ("E2", "E1"),
        ("E1", "E3"), ("E3", "E1"),
        ("E2", "E3"), ("E3", "E2"),
    ]
    time_pairs = [
        ("Early", "Middle"), ("Middle", "Early"),
        ("Early", "Late"), ("Late", "Early"),
        ("Middle", "Late"), ("Late", "Middle"),
    ]

    # --------------------------------------------------------
    # Cell-wise old-image responses versus new controls.
    # --------------------------------------------------------
    for ep in episodes:
        for time_bin in time_bins:
            name = f"post_old{ep}{time_bin}_gt_new"
            contrasts[name] = {
                "priority": "09_cell_vs_new_control",
                "formula": rf"$$\beta_{{\mathrm{{post,old{ep}{time_bin}}}}}-\beta_{{\mathrm{{post,new}}}}$$",
                "interpretation": f"POST old images from {ep}, {time_bin.lower()} checkpoints > new images.",
                "weights": {f"post_old{ep}{time_bin}": 1.0, "post_new": -1.0},
            }

    contrasts["post_oldEpisodeEncoding_avg_gt_new"] = {
        "priority": "09_all_old_cells_vs_new_average",
        "formula": r"$$\frac{1}{9}\sum_{e,t}\beta_{\mathrm{post,old}_{e,t}}-\beta_{\mathrm{post,new}}$$",
        "interpretation": "POST response to all old images, averaged across episode and encoding-time cells, > new images.",
        "weights": {
            **{f"post_old{ep}{time_bin}": 1/9 for ep in episodes for time_bin in time_bins},
            "post_new": -1.0,
        },
    }

    # --------------------------------------------------------
    # Encoding-time effects within each episode.
    # --------------------------------------------------------
    for ep in episodes:
        ep_lower = ep.lower()
        for a, b in time_pairs:
            name = f"post_time{a}_gt_{b}_{ep_lower}"
            contrasts[name] = {
                "priority": "09_time_effect_within_episode",
                "formula": rf"$$\beta_{{\mathrm{{post,old{ep}{a}}}}}-\beta_{{\mathrm{{post,old{ep}{b}}}}}$$",
                "interpretation": f"Within {ep}: old images encoded in {a.lower()} checkpoints > {b.lower()} checkpoints.",
                "weights": {f"post_old{ep}{a}": 1.0, f"post_old{ep}{b}": -1.0},
            }

    # --------------------------------------------------------
    # Encoding-time effects averaged across episodes.
    # --------------------------------------------------------
    for a, b in time_pairs:
        name = f"post_time{a}_gt_{b}_avgEpisodes"
        contrasts[name] = {
            "priority": "09_average_time_effect_across_episodes",
            "formula": rf"$$\frac{{1}}{{3}}\sum_e \beta_{{\mathrm{{post,old}}e\mathrm{{{a}}}}}-\frac{{1}}{{3}}\sum_e \beta_{{\mathrm{{post,old}}e\mathrm{{{b}}}}}$$",
            "interpretation": f"Encoding-time effect {a.lower()} > {b.lower()}, averaged across E1/E2/E3.",
            "weights": {
                **{f"post_old{ep}{a}": 1/3 for ep in episodes},
                **{f"post_old{ep}{b}": -1/3 for ep in episodes},
            },
        }

    # --------------------------------------------------------
    # Episode effects within each encoding-time bin.
    # --------------------------------------------------------
    for time_bin in time_bins:
        for a, b in episode_pairs:
            name = f"post_episode{a}_gt_{b}_{time_bin}"
            contrasts[name] = {
                "priority": "09_episode_effect_within_time_bin",
                "formula": rf"$$\beta_{{\mathrm{{post,old{a}{time_bin}}}}}-\beta_{{\mathrm{{post,old{b}{time_bin}}}}}$$",
                "interpretation": f"Within {time_bin.lower()} checkpoints: old images from {a} > {b}.",
                "weights": {f"post_old{a}{time_bin}": 1.0, f"post_old{b}{time_bin}": -1.0},
            }

    # --------------------------------------------------------
    # Full episode x encoding-time interactions.
    # Example:
    # (E1 Early - E1 Late) - (E2 Early - E2 Late)
    # --------------------------------------------------------
    for ep_a, ep_b in episode_pairs:
        for time_a, time_b in time_pairs:
            name = f"post_ep{ep_a}_gt_{ep_b}_x_time{time_a}_gt_{time_b}"
            contrasts[name] = {
                "priority": "09_episode_by_encoding_time_interaction",
                "formula": rf"$$(\beta_{{\mathrm{{post,old{ep_a}{time_a}}}}}-\beta_{{\mathrm{{post,old{ep_a}{time_b}}}}})-(\beta_{{\mathrm{{post,old{ep_b}{time_a}}}}}-\beta_{{\mathrm{{post,old{ep_b}{time_b}}}}})$$",
                "interpretation": f"Encoding-time effect {time_a.lower()} > {time_b.lower()} is stronger in {ep_a} than {ep_b}.",
                "weights": {
                    f"post_old{ep_a}{time_a}": 1.0,
                    f"post_old{ep_a}{time_b}": -1.0,
                    f"post_old{ep_b}{time_a}": -1.0,
                    f"post_old{ep_b}{time_b}": 1.0,
                },
            }

    return contrasts


def make_pre_post_episode_contrasts() -> dict[str, dict]:
    """PRE/POST episode-control contrasts for e1/e2/e3.

    Regressors expected in the design matrix:
    pre_e1, pre_e2, pre_e3, pre_new,
    post_e1, post_e2, post_e3, post_new.

    The central contrasts are POST-PRE interactions. They test whether an
    episode contrast increased after Island navigation, controlling for
    pre-existing stimulus differences in the PRE viewing run.
    """
    contrasts = {}

    # Episode-specific old > new POST-PRE controls.
    for ep in ["e1", "e2", "e3"]:
        contrasts[f"{ep}_gt_new_post_gt_pre"] = {
            "priority": "10_episode_old_new_post_pre_control",
            "formula": rf"$$ (\beta_{{\mathrm{{post,{ep}}}}}-\beta_{{\mathrm{{post,new}}}}) - (\beta_{{\mathrm{{pre,{ep}}}}}-\beta_{{\mathrm{{pre,new}}}}) $$",
            "interpretation": f"Episode {ep.upper()} old-new effect is stronger at POST than PRE.",
            "weights": {
                f"post_{ep}": 1.0,
                "post_new": -1.0,
                f"pre_{ep}": -1.0,
                "pre_new": 1.0,
            },
        }

    contrasts["episode_old_gt_new_post_gt_pre"] = {
        "priority": "10_episode_average_old_new_post_pre_control",
        "formula": r"$$ \frac{1}{3}(\beta_{\mathrm{post,e1}}+\beta_{\mathrm{post,e2}}+\beta_{\mathrm{post,e3}})-\beta_{\mathrm{post,new}} - \left[\frac{1}{3}(\beta_{\mathrm{pre,e1}}+\beta_{\mathrm{pre,e2}}+\beta_{\mathrm{pre,e3}})-\beta_{\mathrm{pre,new}}\right] $$",
        "interpretation": "Average episode old-new effect is stronger at POST than PRE.",
        "weights": {
            "post_e1": 1/3,
            "post_e2": 1/3,
            "post_e3": 1/3,
            "post_new": -1.0,
            "pre_e1": -1/3,
            "pre_e2": -1/3,
            "pre_e3": -1/3,
            "pre_new": 1.0,
        },
    }

    # All ordered pairwise episode POST-PRE controls.
    pairwise_specs = [
        ("e1", "e2"),
        ("e2", "e1"),
        ("e1", "e3"),
        ("e3", "e1"),
        ("e2", "e3"),
        ("e3", "e2"),
    ]

    for a, b in pairwise_specs:
        name = f"{a}_gt_{b}_post_gt_pre"
        contrasts[name] = {
            "priority": "10_episode_pairwise_post_pre_control",
            "formula": rf"$$ (\beta_{{\mathrm{{post,{a}}}}}-\beta_{{\mathrm{{post,{b}}}}}) - (\beta_{{\mathrm{{pre,{a}}}}}-\beta_{{\mathrm{{pre,{b}}}}}) $$",
            "interpretation": f"The {a.upper()} > {b.upper()} difference is stronger at POST than PRE, controlling for pre-existing image differences.",
            "weights": {
                f"post_{a}": 1.0,
                f"post_{b}": -1.0,
                f"pre_{a}": -1.0,
                f"pre_{b}": 1.0,
            },
        }

    # Descriptive PRE and POST pairwise contrasts, useful for checking whether
    # interaction effects are driven by POST emergence rather than PRE baseline.
    for phase in ["pre", "post"]:
        for a, b in pairwise_specs:
            name = f"{phase}_{a}_gt_{b}"
            contrasts[name] = {
                "priority": f"10_descriptive_{phase}_episode_pairwise",
                "formula": rf"$$\beta_{{\mathrm{{{phase},{a}}}}}-\beta_{{\mathrm{{{phase},{b}}}}}$$",
                "interpretation": f"Descriptive {phase.upper()} {a.upper()} > {b.upper()} episode difference.",
                "weights": {f"{phase}_{a}": 1.0, f"{phase}_{b}": -1.0},
            }

    return contrasts





def make_pre_post_encoding_space_contrasts() -> dict[str, dict]:
    """PRE/POST encoding-space-control contrasts for early/middle/late.

    Regressors expected in the design matrix:
    pre_oldEarly, pre_oldMiddle, pre_oldLate, pre_new,
    post_oldEarly, post_oldMiddle, post_oldLate, post_new.

    The central contrasts are POST-PRE interactions. They test whether a route
    position / encoding-time contrast increased after Island navigation,
    controlling for pre-existing image differences in the PRE viewing run.
    """
    contrasts = {}

    bins = [
        ("early", "Early"),
        ("middle", "Middle"),
        ("late", "Late"),
    ]

    # Encoding-time-specific old > new POST-PRE controls.
    for label, suffix in bins:
        contrasts[f"{label}_gt_new_post_gt_pre"] = {
            "priority": "11_encoding_time_old_new_post_pre_control",
            "formula": rf"$$ (\beta_{{\mathrm{{post,old{suffix}}}}}-\beta_{{\mathrm{{post,new}}}}) - (\beta_{{\mathrm{{pre,old{suffix}}}}}-\beta_{{\mathrm{{pre,new}}}}) $$",
            "interpretation": f"Old images from the {label} route segment show a stronger old-new effect at POST than PRE.",
            "weights": {
                f"post_old{suffix}": 1.0,
                "post_new": -1.0,
                f"pre_old{suffix}": -1.0,
                "pre_new": 1.0,
            },
        }

    contrasts["encoding_time_old_gt_new_post_gt_pre"] = {
        "priority": "11_encoding_time_average_old_new_post_pre_control",
        "formula": r"$$ \frac{1}{3}(\beta_{\mathrm{post,oldEarly}}+\beta_{\mathrm{post,oldMiddle}}+\beta_{\mathrm{post,oldLate}})-\beta_{\mathrm{post,new}} - \left[\frac{1}{3}(\beta_{\mathrm{pre,oldEarly}}+\beta_{\mathrm{pre,oldMiddle}}+\beta_{\mathrm{pre,oldLate}})-\beta_{\mathrm{pre,new}}\right] $$",
        "interpretation": "Average old-new effect across early/middle/late route positions is stronger at POST than PRE.",
        "weights": {
            "post_oldEarly": 1/3,
            "post_oldMiddle": 1/3,
            "post_oldLate": 1/3,
            "post_new": -1.0,
            "pre_oldEarly": -1/3,
            "pre_oldMiddle": -1/3,
            "pre_oldLate": -1/3,
            "pre_new": 1.0,
        },
    }

    # All ordered pairwise encoding-time POST-PRE controls.
    pairwise_specs = [
        ("early", "Early", "middle", "Middle"),
        ("middle", "Middle", "early", "Early"),
        ("early", "Early", "late", "Late"),
        ("late", "Late", "early", "Early"),
        ("middle", "Middle", "late", "Late"),
        ("late", "Late", "middle", "Middle"),
    ]

    for a_label, a_suffix, b_label, b_suffix in pairwise_specs:
        name = f"{a_label}_gt_{b_label}_post_gt_pre"
        contrasts[name] = {
            "priority": "11_encoding_time_pairwise_post_pre_control",
            "formula": rf"$$ (\beta_{{\mathrm{{post,old{a_suffix}}}}}-\beta_{{\mathrm{{post,old{b_suffix}}}}}) - (\beta_{{\mathrm{{pre,old{a_suffix}}}}}-\beta_{{\mathrm{{pre,old{b_suffix}}}}}) $$",
            "interpretation": f"The {a_label} > {b_label} route-position difference is stronger at POST than PRE, controlling for pre-existing image differences.",
            "weights": {
                f"post_old{a_suffix}": 1.0,
                f"post_old{b_suffix}": -1.0,
                f"pre_old{a_suffix}": -1.0,
                f"pre_old{b_suffix}": 1.0,
            },
        }

    # Descriptive PRE and POST pairwise contrasts.
    for phase in ["pre", "post"]:
        for a_label, a_suffix, b_label, b_suffix in pairwise_specs:
            name = f"{phase}_{a_label}_gt_{b_label}"
            contrasts[name] = {
                "priority": f"11_descriptive_{phase}_encoding_time_pairwise",
                "formula": rf"$$\beta_{{\mathrm{{{phase},old{a_suffix}}}}}-\beta_{{\mathrm{{{phase},old{b_suffix}}}}}$$",
                "interpretation": f"Descriptive {phase.upper()} {a_label} > {b_label} route-position difference.",
                "weights": {f"{phase}_old{a_suffix}": 1.0, f"{phase}_old{b_suffix}": -1.0},
            }

    return contrasts





def _time_space_regressor(phase: str, episode: str, space: str) -> str:
    """Return regressor name like post_oldE1Early."""
    ep_suffix = episode.upper()
    space_suffix = {
        "early": "Early",
        "middle": "Middle",
        "late": "Late",
    }[space]
    return f"{phase}_old{ep_suffix}{space_suffix}"


def _add_weight(weights: dict[str, float], key: str, value: float) -> None:
    weights[key] = weights.get(key, 0.0) + float(value)


def _post_pre_space_effect_weights(episode: str, space_a: str, space_b: str) -> dict[str, float]:
    """POST-PRE within-episode SPACE effect: (post A-B) - (pre A-B)."""
    weights = {}
    _add_weight(weights, _time_space_regressor("post", episode, space_a), 1.0)
    _add_weight(weights, _time_space_regressor("post", episode, space_b), -1.0)
    _add_weight(weights, _time_space_regressor("pre", episode, space_a), -1.0)
    _add_weight(weights, _time_space_regressor("pre", episode, space_b), 1.0)
    return weights


def _post_pre_time_effect_weights(episode_a: str, episode_b: str, space: str) -> dict[str, float]:
    """POST-PRE within-space TIME effect: (post Ea-Eb) - (pre Ea-Eb)."""
    weights = {}
    _add_weight(weights, _time_space_regressor("post", episode_a, space), 1.0)
    _add_weight(weights, _time_space_regressor("post", episode_b, space), -1.0)
    _add_weight(weights, _time_space_regressor("pre", episode_a, space), -1.0)
    _add_weight(weights, _time_space_regressor("pre", episode_b, space), 1.0)
    return weights


def _post_pre_time_x_space_weights(
    episode_a: str,
    episode_b: str,
    space_a: str,
    space_b: str,
) -> dict[str, float]:
    """Triple interaction: [(Ea Sa-Sb) - (Eb Sa-Sb)] POST-PRE."""
    weights = {}

    # POST interaction: (EaSa - EaSb) - (EbSa - EbSb)
    _add_weight(weights, _time_space_regressor("post", episode_a, space_a), 1.0)
    _add_weight(weights, _time_space_regressor("post", episode_a, space_b), -1.0)
    _add_weight(weights, _time_space_regressor("post", episode_b, space_a), -1.0)
    _add_weight(weights, _time_space_regressor("post", episode_b, space_b), 1.0)

    # subtract PRE interaction
    _add_weight(weights, _time_space_regressor("pre", episode_a, space_a), -1.0)
    _add_weight(weights, _time_space_regressor("pre", episode_a, space_b), 1.0)
    _add_weight(weights, _time_space_regressor("pre", episode_b, space_a), 1.0)
    _add_weight(weights, _time_space_regressor("pre", episode_b, space_b), -1.0)

    return weights


def make_pre_post_time_space_interaction_contrasts() -> dict[str, dict]:
    """PRE/POST TIME × ENCODING-SPACE interaction contrasts.

    Expected regressors:
    pre_oldE1Early ... pre_oldE3Late, pre_new,
    post_oldE1Early ... post_oldE3Late, post_new.

    Main interaction contrasts test whether the SPACE effect (early/middle/late)
    differs between TIME episodes (E1/E2/E3) after Island navigation, controlling
    for the same image-labelled difference at PRE.
    """
    contrasts = {}

    episodes = ["e1", "e2", "e3"]
    spaces = ["early", "middle", "late"]
    episode_pairwise = [
        ("e1", "e2"), ("e2", "e1"),
        ("e1", "e3"), ("e3", "e1"),
        ("e2", "e3"), ("e3", "e2"),
    ]
    space_pairwise = [
        ("early", "middle"), ("middle", "early"),
        ("early", "late"), ("late", "early"),
        ("middle", "late"), ("late", "middle"),
    ]

    episode_index = {"e1": 1, "e2": 2, "e3": 3}
    space_index = {"early": 1, "middle": 2, "late": 3}

    # --------------------------------------------------------
    # Old > new controls for each TIME × SPACE cell
    # --------------------------------------------------------
    for ep in episodes:
        for sp in spaces:
            ep_title = ep.upper()
            sp_title = sp.capitalize()
            name = f"{ep}_{sp}_gt_new_post_gt_pre"
            weights = {
                _time_space_regressor("post", ep, sp): 1.0,
                "post_new": -1.0,
                _time_space_regressor("pre", ep, sp): -1.0,
                "pre_new": 1.0,
            }
            contrasts[name] = {
                "priority": "12_cell_old_new_post_pre_control",
                "formula": rf"$$ (\beta_{{\mathrm{{post,{ep_title},{sp_title}}}}}-\beta_{{\mathrm{{post,new}}}}) - (\beta_{{\mathrm{{pre,{ep_title},{sp_title}}}}}-\beta_{{\mathrm{{pre,new}}}}) $$",
                "interpretation": f"Old-new effect for {ep_title} {sp} images is stronger at POST than PRE.",
                "weights": weights,
            }

    avg_weights = {"post_new": -1.0, "pre_new": 1.0}
    for ep in episodes:
        for sp in spaces:
            _add_weight(avg_weights, _time_space_regressor("post", ep, sp), 1/9)
            _add_weight(avg_weights, _time_space_regressor("pre", ep, sp), -1/9)
    contrasts["time_space_old_gt_new_post_gt_pre"] = {
        "priority": "12_average_old_new_post_pre_control",
        "formula": r"$$ \mathrm{mean}_{E,S}(\beta_{\mathrm{post,E,S}})-\beta_{\mathrm{post,new}} - [\mathrm{mean}_{E,S}(\beta_{\mathrm{pre,E,S}})-\beta_{\mathrm{pre,new}}] $$",
        "interpretation": "Average old-new effect across all episode × space bins is stronger at POST than PRE.",
        "weights": avg_weights,
    }

    # --------------------------------------------------------
    # Supporting within-episode SPACE effects, POST > PRE
    # --------------------------------------------------------
    for ep in episodes:
        for sp_a, sp_b in space_pairwise:
            name = f"{ep}_{sp_a}_gt_{sp_b}_post_gt_pre"
            contrasts[name] = {
                "priority": "12_within_episode_space_post_pre",
                "formula": rf"$$ (\beta_{{\mathrm{{post,{ep.upper()},{sp_a}}}}}-\beta_{{\mathrm{{post,{ep.upper()},{sp_b}}}}}) - (\beta_{{\mathrm{{pre,{ep.upper()},{sp_a}}}}}-\beta_{{\mathrm{{pre,{ep.upper()},{sp_b}}}}}) $$",
                "interpretation": f"Within {ep.upper()}, the {sp_a} > {sp_b} space effect is stronger at POST than PRE.",
                "weights": _post_pre_space_effect_weights(ep, sp_a, sp_b),
            }

    # --------------------------------------------------------
    # Supporting within-space TIME effects, POST > PRE
    # --------------------------------------------------------
    for sp in spaces:
        for ep_a, ep_b in episode_pairwise:
            name = f"{sp}_{ep_a}_gt_{ep_b}_post_gt_pre"
            contrasts[name] = {
                "priority": "12_within_space_time_post_pre",
                "formula": rf"$$ (\beta_{{\mathrm{{post,{ep_a.upper()},{sp}}}}}-\beta_{{\mathrm{{post,{ep_b.upper()},{sp}}}}}) - (\beta_{{\mathrm{{pre,{ep_a.upper()},{sp}}}}}-\beta_{{\mathrm{{pre,{ep_b.upper()},{sp}}}}}) $$",
                "interpretation": f"Within {sp} route positions, the {ep_a.upper()} > {ep_b.upper()} time effect is stronger at POST than PRE.",
                "weights": _post_pre_time_effect_weights(ep_a, ep_b, sp),
            }

    # --------------------------------------------------------
    # Main TIME × SPACE interactions, POST > PRE
    # All ordered episode pairs × all ordered space pairs.
    # --------------------------------------------------------
    for ep_a, ep_b in episode_pairwise:
        time_distance = abs(episode_index[ep_a] - episode_index[ep_b])
        time_label = "large_time" if time_distance == 2 else "small_time"

        for sp_a, sp_b in space_pairwise:
            space_distance = abs(space_index[sp_a] - space_index[sp_b])
            space_label = "large_space" if space_distance == 2 else "small_space"
            name = f"time_{ep_a}_gt_{ep_b}_x_space_{sp_a}_gt_{sp_b}_post_gt_pre"
            contrasts[name] = {
                "priority": f"12_time_space_interaction_{time_label}_{space_label}",
                "formula": rf"$$ [(\beta_{{\mathrm{{post,{ep_a.upper()},{sp_a}}}}}-\beta_{{\mathrm{{post,{ep_a.upper()},{sp_b}}}}}) - (\beta_{{\mathrm{{post,{ep_b.upper()},{sp_a}}}}}-\beta_{{\mathrm{{post,{ep_b.upper()},{sp_b}}}}})] - [(\beta_{{\mathrm{{pre,{ep_a.upper()},{sp_a}}}}}-\beta_{{\mathrm{{pre,{ep_a.upper()},{sp_b}}}}}) - (\beta_{{\mathrm{{pre,{ep_b.upper()},{sp_a}}}}}-\beta_{{\mathrm{{pre,{ep_b.upper()},{sp_b}}}}})] $$",
                "interpretation": f"TIME × SPACE interaction: the {sp_a} > {sp_b} space effect differs for {ep_a.upper()} > {ep_b.upper()} after Island navigation, controlling for PRE image baseline. This is a {time_label.replace('_', ' ')} × {space_label.replace('_', ' ')} contrast.",
                "time_distance": time_distance,
                "space_distance": space_distance,
                "weights": _post_pre_time_x_space_weights(ep_a, ep_b, sp_a, sp_b),
            }

    return contrasts




