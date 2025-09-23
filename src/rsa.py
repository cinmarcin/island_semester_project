from operator import sub
import numpy as np
import matplotlib.pyplot as plt
import os
def create_index_rdm(n_trials, repetitions, relative_distance=False, save_fig=False):
    """
    Create an index RDM where dissimilarity increases with the distance between trial indices.
    
    Parameters
    ----------
    n_trials : int
        Number of unique trials.
    repetitions : int
        Number of repetitions per trial.

    save_folder : str, optional
        Folder to save the RDM plot. If None, the plot is not saved.
    Returns
    -------
    np.ndarray
        An (n_trials * repetitions) x (n_trials * repetitions) RDM matrix.
    """
    total_trials = n_trials * repetitions
    rdm = np.zeros((total_trials, total_trials))
    
    if relative_distance:
        dist = 64
    else:
        dist = 0
    for i in range(total_trials):
        for j in range(total_trials):
            rdm[i, j] = min(abs((i % (n_trials)) - (j % (n_trials))), abs((i % (n_trials) + dist) - (j % (n_trials))), abs((i % (n_trials) - dist) - (j % (n_trials))))
    # plot
    if save_fig is not None:
        plt.imshow(rdm, cmap='viridis')
        plt.colorbar(label='Dissimilarity')
        plt.title(f'Index RDM for {n_trials} trials with {repetitions} repetitions each')
        plt.xlabel('Trial Index')
        plt.ylabel('Trial Index')
        rdm_plot_path = os.path.join('../figures/', f"index_{'relative' if relative_distance else 'absolute'}_rdm.png")
        plt.savefig(rdm_plot_path)
        plt.close()
        print(f"Saved index RDM plot to {rdm_plot_path}")
        
    return rdm