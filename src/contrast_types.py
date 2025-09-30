from enum import Enum
import numpy as np
import pandas as pd

class ContrastType(Enum):
    OLD_VS_NEW = "old_vs_new"
    FACES_VS_OBJECTS = "faces_vs_objects"
    RELEVANCE_VS_IRRELEVANCE = "relevance_vs_irrelevance"
    PER_TRIAL = "per_trial"
    ORDER = "order"



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
        if self == ContrastType.OLD_VS_NEW:
            # Convert trial_type to old/new
            events['trial_type'] = np.where(events['trial_type'].str.contains('old'), 'old', 'new')
        elif self == ContrastType.FACES_VS_OBJECTS:
            # Convert trial_type to face/object
            events['trial_type'] = np.where(events['trial_type'].str.contains('_h_'), 'face', 'object')
        elif self == ContrastType.RELEVANCE_VS_IRRELEVANCE:
            # Convert trial_type to relevant/irrelevant
            events['trial_type'] = np.where(events['trial_type'].str.contains('_r_'), 'relevant', 'irrelevant')
        elif self == ContrastType.PER_TRIAL:
            events['trial_type'] = 'trial'
        elif self == ContrastType.ORDER:
            events = events.sort_values(by='onset').reset_index(drop=True)
            events['trial_type'] = ['trial_' + str(i+1) for i in range(len(events))]
        else:
            raise ValueError(f"Unknown stimuli handling for {self}")

        # Encode run (pre/post) in trial_type
        if run_label is not None:
            events['trial_type'] = run_label + '_' + events['trial_type']

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
                    1 if 'pre_face' in c else (-1 if 'pre_object' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post_face' in c else (-1 if 'post_object' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a single array (you can choose which one to return based on your needs)
                weights = {
                    "Pre_Faces_vs_Objects_Effect": pre_weights,
                    "Post_Faces_vs_Objects_Effect": post_weights,
                    "Pre_vs_Post_Faces_vs_Objects_Effect": pre_vs_post_weights
                }
            else:
                weights = np.array([1 if 'face' in c else (-1 if 'object' in c else 0) for c in columns])

        elif self == ContrastType.RELEVANCE_VS_IRRELEVANCE:
            if combined:
                # create three sets of weights to compare relevant vs irrelevant in each run + the difference between runs
                pre_weights = np.array([
                    1 if 'pre_relevant' in c else (-1 if 'pre_irrelevant' in c else 0) for c in columns
                ])
                post_weights = np.array([
                    1 if 'post_relevant' in c else (-1 if 'post_irrelevant' in c else 0) for c in columns
                ])
                pre_vs_post_weights = post_weights - pre_weights
                # Combine weights into a single array (you can choose which one to return based on your needs)
                weights = {
                    "Pre_Relevance_vs_Irrelevance_Effect": pre_weights,
                    "Post_Relevance_vs_Irrelevance_Effect": post_weights,
                    "Pre_vs_Post_Relevance_vs_Irrelevance_Effect": pre_vs_post_weights
                }
            else:
                weights = np.array([1 if 'relevant' in c else (-1 if 'irrelevant' in c else 0) for c in columns])
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
        else:
            raise ValueError(f"No weights defined for {self}")
                
        return weights
