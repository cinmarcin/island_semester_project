from enum import Enum
import numpy as np
import pandas as pd

_CONTRAST_MAP = {
    "old_vs_new": ("old", "new"),
    "faces_vs_objects": ("_h_", "_o_"),
    "relevance_vs_irrelevance": ("_r_", "_nr_"),
    "per_trial": (".jpg", "NONE"),
    "order": (".jpg", "NONE"),
    "new_faces_vs_objects": ("_h_v1_new", "_o_v1_new"),
}

def transform_trial_to_contrast(events, c1: str, c2: str):
    """Rename trial types to represent the contrast conditions."""
    if c1 == '.jpg':
        # dictionary to count repetitions per event
        counter = {}
        
        # function to generate trial_type with repetitions
        def trial_type_with_rep(event):
            counter[event] = counter.get(event, 0) + 1
            return f"rep{counter[event]}_{event}"
        
        # apply to create trial_type column
        events['trial_type'] = events['trial_type'].apply(trial_type_with_rep)
        return events

    mask1 = events['trial_type'].str.contains(c1, case=False, na=False)
    mask2 = events['trial_type'].str.contains(c2, case=False, na=False)
    mask_other = ~(mask1 | mask2)

    events['trial_type'] = np.where(mask1, c1, np.where(mask2, c2, events['trial_type']))

    if mask_other.any():
        unmatched = events.loc[mask_other, 'trial_type'].unique()
        print(f"Found trials that don't belong to '{c1}' or '{c2}': {unmatched}")

    return events


# Define category groups
HUMAN_CATEGORIES = ['pirate', 'viking', 'maya']
OBJECT_CATEGORIES = ['food', 'wealth', 'drink']


def categorize_and_replace(trial):
    """Convert strings like 'wealth_9_new' to '_o_v1_new.jpg' or '_h_v1_new.jpg'."""
    trial = trial.lower()
    if 'new' not in trial:
        return trial
    if any(h in trial for h in HUMAN_CATEGORIES):
        return '_h_v1_new.jpg'
    if any(o in trial for o in OBJECT_CATEGORIES):
        return '_o_v1_new.jpg'
    raise ValueError(f"⚠️ Unrecognized category in trial_type: {trial}")


class ContrastType(Enum):
    OLD_VS_NEW = "old_vs_new"
    FACES_VS_OBJECTS = "faces_vs_objects"
    RELEVANCE_VS_IRRELEVANCE = "relevance_vs_irrelevance"
    PER_TRIAL = "per_trial"
    ORDER = "order"
    NEW_FACES_VS_OBJECTS = "new_faces_vs_objects"

    # --- helper maps for contrasts ---

    def preprocess_stimuli(self, events: pd.DataFrame, run_label: str = None) -> pd.DataFrame:
        """Prepare events table according to contrast type."""
        if self == ContrastType.NEW_FACES_VS_OBJECTS:
            events['trial_type'] = events['trial_type'].astype(str).apply(categorize_and_replace)
        c1, c2 = _CONTRAST_MAP[self.value]
        events = transform_trial_to_contrast(events, c1, c2)

        if run_label:
            events['trial_type'] = f"{run_label}_" + events['trial_type']

        n1 = events['trial_type'].str.contains(c1, case=False, na=False).sum()
        n2 = events['trial_type'].str.contains(c2, case=False, na=False).sum()
        print(f"Found {n1} trials for '{c1}' and {n2} trials for '{c2}'")

        return events

    # --- helper to build weights ---
    @staticmethod
    def _build_weights(columns, include, exclude):
        """Return a weight vector for a simple contrast."""
        return np.array([
            1 if (include[0] in c and all(e not in c for e in exclude))
            else -1 if (include[1] in c and all(e not in c for e in exclude))
            else 0
            for c in columns
        ])

    def get_vector(self, design_matrix, combined=False):
        columns = design_matrix.columns.tolist()
        c1, c2 = _CONTRAST_MAP[self.value]
        exclude_terms = ['derivative', 'dispersion']

        def run_weights(run):
            # handle contrasts order separately
            if self == ContrastType.ORDER:
                # return empty weights order is used for RSA only
                return {}
            else:
                return self._build_weights(columns, (f"{run}_{c1}", f"{run}_{c2}"), exclude_terms)

        if combined:
            pre = run_weights('pre')
            post = run_weights('post')
            if self == ContrastType.ORDER:
                # return empty weights order is used for RSA only
                print("Returning empty weights for ORDER contrast")
                return {}
            else:
                return {
                    f"Pre_{self.value}": pre,
                    f"Post_{self.value}": post,
                    f"Pre_vs_Post_{self.value}": post - pre
                }
        else:
            return self._build_weights(columns, (c1, c2), exclude_terms)
