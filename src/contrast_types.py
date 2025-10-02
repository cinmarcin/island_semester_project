from enum import Enum
import numpy as np
import pandas as pd

def transform_trial_to_contrast(events, c1 : str, c2: str):

    # Special case for 'order' contrast: rename trials to trial_1, trial_2, ...
    if c1 == 'order':
        events = events.sort_values(by='onset').reset_index(drop=True)
        events['trial_type'] = ['trial_' + str(i+1) for i in range(len(events))]
        return events
    
    # Create boolean masks for the two conditions
    mask_old = events['trial_type'].str.contains(c1, case=False, na=False)
    mask_new = events['trial_type'].str.contains(c2, case=False, na=False)
    mask_other = ~(mask_old | mask_new)

    events['trial_type'] = np.where(mask_old, c1, np.where(mask_new, c2, events['trial_type']))

    # Report any unmatched trials
    if mask_other.any():
        unmatched_trials = events['trial_type'][mask_other].unique()
        print(f"Found trials that don't belong to '{c1}' or '{c2}': {unmatched_trials}")

    return events

# Define your category groups
human_categories = ['pirate', 'viking', 'maya']
object_categories = ['food', 'wealth', 'drink']

def categorize_and_replace(trial):
    """Convert strings like 'wealth_9_new' to '_o_v1_new.jpg' or '_h_v1_new.jpg'."""
    trial_lower = trial.lower()
    if 'new' not in trial_lower:
        return trial  # leave non-new trials unchanged

    # Check if the trial contains any human category
    for h in human_categories:
        if h in trial_lower:
            return '_h_v1_new.jpg'

    # Check if the trial contains any object category
    for o in object_categories:
        if o in trial_lower:
            return '_o_v1_new.jpg'

    # If neither matches, throw an error
    raise ValueError(f"⚠️ Unrecognized category in trial_type: {trial}")

class ContrastType(Enum):
    OLD_VS_NEW = "old_vs_new"
    FACES_VS_OBJECTS = "faces_vs_objects"
    RELEVANCE_VS_IRRELEVANCE = "relevance_vs_irrelevance"
    PER_TRIAL = "per_trial"
    ORDER = "order"
    NEW_FACES_VS_OBJECTS = "new_faces_vs_objects"



    def preprocess_stimuli(self, events: pd.DataFrame, run_label: str = None) -> pd.DataFrame:
        """
        Handle stimuli differently for each contrast and encode pre/post run.
        
        Parameters
        ----------
        events : pd.DataFrame
            Event file with at least a 'trial_type' column.
        run_label : str
            Label for the run, e.g., 'pre' or 'post'.
            
        Returns
        -------
        pd.DataFrame
            Events DataFrame with updated 'trial_type' column.
        """

        # We need to handle the fact that new stimuli are not labeled as human/object
        events['trial_type'] = events['trial_type'].astype(str).apply(categorize_and_replace)

        if self == ContrastType.OLD_VS_NEW:
            c1, c2 = 'old', 'new'
        elif self == ContrastType.FACES_VS_OBJECTS:
            c1, c2 = '_h_', '_o_'
        elif self == ContrastType.RELEVANCE_VS_IRRELEVANCE:
            c1, c2 = '_r_', '_nr_'
        elif self == ContrastType.PER_TRIAL:
            c1, c2 = 'trial', 'trial'
        elif self == ContrastType.ORDER:
            c1, c2 = 'order', 'order'
        elif self == ContrastType.NEW_FACES_VS_OBJECTS:
            c1, c2 = '_h_v1_new', '_o_v1_new'
        else:
            raise ValueError(f"Unknown stimuli handling for {self}")

        events = transform_trial_to_contrast(events, c1, c2)

        # Encode run (pre/post) in trial_type
        if run_label is not None:
            events['trial_type'] = run_label + '_' + events['trial_type']

        # Print the number of trials per condition c1 and c2
        count_c1 = (events['trial_type'].str.contains(c1, case=False, na=False)).sum()
        count_c2 = (events['trial_type'].str.contains(c2, case=False, na=False)).sum()
        print(f"Found {count_c1} trials for condition '{c1}' and {count_c2} trials for condition '{c2}'")

        return events

    def get_vector(self, design_matrix, combined=False):
        """Return the contrast weights vector for a design matrix using 1, -1, 0"""
        columns = design_matrix.columns.tolist()
        
        if self == ContrastType.OLD_VS_NEW:
            if combined:
                # create three sets of weights to compare old vs new in each run + the difference between runs
                pre_weights = np.array([
                    1 if 'pre_old' in c else (-1 if 'pre_new' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post_old' in c else (-1 if 'post_new' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a dictionary
                weights = {
                    "Pre_Old_vs_New_Effect": pre_weights,
                    "Post_Old_vs_New_Effect": post_weights,
                    "Pre_vs_Post_Old_vs_New_Effect": pre_vs_post_weights
                }
            else:
                weights =  np.array([1 if 'old' in c else (-1 if 'new' in c else 0) for c in columns])

        elif self == ContrastType.FACES_VS_OBJECTS:
            if combined:
                # create three sets of weights to compare faces vs objects in each run + the difference between runs
                pre_weights = np.array([
                    1 if 'pre__h' in c else (-1 if 'pre__o' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post__h' in c else (-1 if 'post__o' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a single array (you can choose which one to return based on your needs)
                weights = {
                    "Pre_Faces_vs_Objects_Effect": pre_weights,
                    "Post_Faces_vs_Objects_Effect": post_weights,
                    "Pre_vs_Post_Faces_vs_Objects_Effect": pre_vs_post_weights
                }
            else:
                weights = np.array([1 if '_h_' in c else (-1 if '_o_' in c else 0) for c in columns])

        elif self == ContrastType.RELEVANCE_VS_IRRELEVANCE:
            if combined:
                # create three sets of weights to compare relevant vs irrelevant in each run + the difference between runs
                pre_weights = np.array([
                    1 if 'pre__r' in c else (-1 if 'pre__nr' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post__r' in c else (-1 if 'post__nr' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a single array (you can choose which one to return based on your needs)
                weights = {
                    "Pre_Relevance_vs_Irrelevance_Effect": pre_weights,
                    "Post_Relevance_vs_Irrelevance_Effect": post_weights,
                    "Pre_vs_Post_Relevance_vs_Irrelevance_Effect": pre_vs_post_weights
                }
            else:
                weights = np.array([1 if '_r_' in c else (-1 if '_nr_' in c else 0) for c in columns])
        elif self == ContrastType.PER_TRIAL:
            if combined : 
                weights = {'Trial_Effect': np.array([1 if 'trial' in c else 0 for c in columns])}
            else:
                weights = np.array([1 if 'trial' in c else 0 for c in columns])
        elif self == ContrastType.ORDER:
            if combined:
                weights = {'Baseline' : np.array([1 if 'trial' in c else 0 for c in columns])}
            else:
               weights = np.array([1 if 'trial' in c else 0 for c in columns])
        elif self == ContrastType.NEW_FACES_VS_OBJECTS:
            if combined:
                # create three sets of weights to compare faces vs objects in each run + the difference between runs
                pre_weights = np.array([
                    1 if 'pre__h_v1_new' in c else (-1 if 'pre__o_v1_new' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post__h_v1_new' in c else (-1 if 'post__o_v1_new' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a single array (you can choose which one to return based on your needs)
                weights = {
                    "Pre_New_Faces_vs_Objects_Effect": pre_weights,
                    "Post_New_Faces_vs_Objects_Effect": post_weights,
                    "Pre_vs_Post_New_Faces_vs_Objects_Effect": pre_vs_post_weights
                }
            else:
                weights = np.array([1 if '_h_v1_new' in c else (-1 if '_o_v1_new' in c else 0) for c in columns])
        else:
            raise ValueError(f"No weights defined for {self}")
                
        return weights
