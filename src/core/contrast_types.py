from enum import Enum
import numpy as np
import pandas as pd

_CONTRAST_MAP = {
    "old_vs_new": ("old", "new"),
    "faces_vs_objects": ("_h_", "_o_"),
    "relevance_vs_irrelevance": ("_r_", "_nr_"),
    "per_trial": (".jpg", "NONE"),
    "order": ("_", "NONE"),
    "new_faces_vs_objects": ("_h_v1_new", "_o_v1_new"),
    "random": ("random1", "random2"),
}

def transform_trial_to_contrast(events, c1: str, c2: str):
    """Rename trial types to represent the contrast conditions."""
    if c1 == '_':
        # dictionary to count repetitions per event
        counter = {}
        
        # function to generate trial_type with repetitions
        def trial_type_with_rep(event):
            counter[event] = counter.get(event, 0) + 1
            return f"rep{counter[event]}_{event}"
        
        # take only the two first elements after splitting by '_'
        events['trial_type'] = events['trial_type'].astype(str).apply(lambda x: '_'.join(x.split('_')[:2]))
        
        # apply to create trial_type column
        events['trial_type'] = events['trial_type'].apply(trial_type_with_rep)
        return events
    
    elif c1 == 'random1' and c2 == 'random2':
        first_rep = events.head(64).copy()
        classes = controlled_randomization_of_trials(first_rep)
        mapping = dict(zip(classes['trial_type'], classes['group']))
        # apply to all events
        events['trial_type'] = events['trial_type'].map(mapping).fillna(events['trial_type'])
        # check nas 
        if events['trial_type'].isna().any():
            unmatched = events.loc[events['trial_type'].isna(), 'trial_type'].unique()
            print(f"Found trials that don't belong to 'random1' or 'random2': {unmatched}")
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
    RANDOM = "random"

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


def controlled_randomization_of_trials(events, n_bins=4, seed=42):
    import numpy as np

    # --- Extract categorical ---
    events['category'] = events['trial_type'].str.extract(r'^(maya|wealth|pirate|food|drink|viking)')
    events['r_nr'] = events['trial_type'].str.extract(r'_(r|nr)_')
    events['o_h'] = events['trial_type'].str.extract(r'_(o|h)_')
    events['old_new'] = events['trial_type'].apply(lambda x: 'old' if 'old' in x else 'new')

    # --- Bin continuous ---
    events['onset_bin'] = pd.qcut(events['onset'], q=min(n_bins, len(events)), duplicates='drop')
    events['duration_bin'] = pd.qcut(events['duration'], q=min(n_bins, len(events)), duplicates='drop')

    # --- Stratification code (categorical + binned continuous) ---
    events['stratum'] = (events['category'].astype(str) + "_" +
                         events['r_nr'].astype(str) + "_" +
                         events['o_h'].astype(str) + "_" +
                         events['old_new'].astype(str) + "_" +
                         events['onset_bin'].astype(str) + "_" +
                         events['duration_bin'].astype(str))

    # --- Shuffle globally ---
    events = events.sample(frac=1, random_state=seed).reset_index(drop=True)

    # --- Assign group by exact half ---
    n_trials = len(events)
    half = n_trials // 2
    events['group'] = ['random1']*half + ['random2']*(n_trials - half)

    # --- Verify ---
    print(events['group'].value_counts())
    print(events.groupby(['group', 'category', 'r_nr', 'o_h', 'old_new']).size().unstack(fill_value=0))

    return events
