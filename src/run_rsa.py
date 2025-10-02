import os
from joblib import dump, load
from contrast_types import ContrastType
from tqdm import tqdm
from rsa import create_index_rdm, correlation_accross_repetitions, correlation_within_repetitions, extract_beta_maps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns


## Controls
CONTRAST_TYPE = ContrastType.ORDER
SMOOTHING_FWHM = None # in mm
HEIGHT_CONTROL = "bonferroni"  # "fdr" or "bonferroni"
P_VALUE = 0.05
HIPPOCAMPUS_ONLY = True  # If True, restrict analysis to hippocampus only

## Paths
list_of_subs = [
                'sub-P10',
                'sub-P12',
                'sub-P13',
                'sub-P14',
                'sub-P15',
                'sub-P16',
                'sub-P17',
                'sub-P18',
                'sub-P19',
            #    'sub-P20', Unattentive
                'sub-P21',
                'sub-P22',
                'sub-P23',
                'sub-P25',
        #        'sub-P26', missing event file
                'sub-P27',
        #        'sub-P30', incomplete data
                'sub-P31',
                'sub-P34',
                'sub-P35',
                'sub-P36',
                'sub-P37',
                'sub-P41',
                'sub-P42',
                'sub-P43',
                'sub-P44',
                'sub-P45',
                'sub-P47',
                'sub-P48',
        #        'sub-P49',   Unattentive
                'sub-P50',
                'sub-P51',
                ]

DATA_INPUT_PATH = f'data/processed/glm/{CONTRAST_TYPE.value}'
DATA_OUTPUT_PATH = f'data/processed/rsa/{CONTRAST_TYPE.value}'
FIGURE_OUTPUT_PATH = 'figures/rsa'
if not os.path.exists(FIGURE_OUTPUT_PATH):
    os.makedirs(FIGURE_OUTPUT_PATH)

group_level_correlation_within_repetitions = []
group_level_correlation_accross_repetitions = []
group_level_correlation_matrices = []
pre_avgs = []
post_avgs = []
for sub in tqdm(list_of_subs):
    print(f"--- Processing subject: {sub}")
    SUBJECT_FOLDER_IN = os.path.join(DATA_INPUT_PATH, sub)
    SUBJECT_FOLDER = os.path.join(DATA_OUTPUT_PATH, sub)
    PRE_GLM_MODEL_PATH = os.path.join(SUBJECT_FOLDER_IN, f'{sub}_pre_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    POST_GLM_MODEL_PATH = os.path.join(SUBJECT_FOLDER_IN, f'{sub}_post_smooth_{SMOOTHING_FWHM}_{CONTRAST_TYPE.value}_{"hippo_" if HIPPOCAMPUS_ONLY else ""}fitted_glm.pkl')
    if not os.path.exists(SUBJECT_FOLDER):
        os.makedirs(SUBJECT_FOLDER)

    pre_fmri_glm = load(PRE_GLM_MODEL_PATH)
    post_fmri_glm = load(POST_GLM_MODEL_PATH)
    print(f"Loaded GLM models from {PRE_GLM_MODEL_PATH} and {POST_GLM_MODEL_PATH}")

    pre_trial_data = extract_beta_maps(pre_fmri_glm)
    post_trial_data = extract_beta_maps(post_fmri_glm)

    # Reshape to (repetitions, tasks, n_voxels)
    pre_reshaped = pre_trial_data.reshape(6, 64, -1)
    post_reshaped = post_trial_data.reshape(6, 64, -1)

    # Average within repetitions
    pre_avg_rep = pre_reshaped.mean(axis=1)  
    post_avg_rep = post_reshaped.mean(axis=1) 

    # Average over voxels
    pre_avg = pre_avg_rep.mean(axis=1)
    post_avg = post_avg_rep.mean(axis=1)

    pre_avgs.append(pre_avg)
    post_avgs.append(post_avg)




    # Compute correlation matrix
    print("Computing correlation matrix...")
    pre_correlation_matrix = np.corrcoef(pre_trial_data)
    post_correlation_matrix = np.corrcoef(post_trial_data)
    # Save correlation matrix
    pre_correlation_matrix_path = os.path.join(SUBJECT_FOLDER, f'{sub}_pre_{"hippo_" if HIPPOCAMPUS_ONLY else ""}correlation_matrix.npy')
    post_correlation_matrix_path = os.path.join(SUBJECT_FOLDER, f'{sub}_post_{"hippo_" if HIPPOCAMPUS_ONLY else ""}correlation_matrix.npy')
    dump(pre_correlation_matrix, pre_correlation_matrix_path)
    dump(post_correlation_matrix, post_correlation_matrix_path)
    # Save correlation matrix figure
    # PRE
    plt.figure(figsize=(10, 8))
    plt.imshow(pre_correlation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.title(f'Correlation Matrix for {sub}')
    plt.xlabel('Trials')
    plt.ylabel('Trials')
    plt.savefig(pre_correlation_matrix_path.replace('.npy', '.png'))
    plt.close()
    # POST
    plt.figure(figsize=(10, 8))
    plt.imshow(post_correlation_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(label='Correlation')
    plt.title(f'Correlation Matrix for {sub}')
    plt.xlabel('Trials')
    plt.ylabel('Trials')
    plt.savefig(post_correlation_matrix_path.replace('.npy', '.png'))
    plt.close()

    print(f"Saved correlation matrices to {pre_correlation_matrix_path} and {post_correlation_matrix_path}")

    ## PRE 
    print("Computing average correlations for PRE...")
    # Compute average correlation within repetitions
    pre_correlation_w_rep = correlation_within_repetitions(pre_correlation_matrix, trial_per_repetition=64, repetitions=6)
    group_level_correlation_within_repetitions.append(pre_correlation_w_rep)
    # Compute average correlation across repetitions
    pre_correlation_ac_rep = correlation_accross_repetitions(pre_correlation_matrix, trial_per_repetition=64, repetitions=6)
    group_level_correlation_accross_repetitions.append(pre_correlation_ac_rep)
    group_level_correlation_matrices.append(pre_correlation_matrix)

    ## POST
    print("Computing average correlations for POST...")
    # Compute average correlation within repetitions
    post_correlation_w_rep = correlation_within_repetitions(post_correlation_matrix, trial_per_repetition=64, repetitions=6)
    group_level_correlation_within_repetitions.append(post_correlation_w_rep)
    # Compute average correlation across repetitions
    post_correlation_ac_rep = correlation_accross_repetitions(post_correlation_matrix, trial_per_repetition=64, repetitions=6)
    group_level_correlation_accross_repetitions.append(post_correlation_ac_rep)
    group_level_correlation_matrices.append(post_correlation_matrix)

    # # Create RDM
    # n_trials = 64 
    # repetitions = 6
    # index_rdm = create_index_rdm(n_trials, repetitions, relative_distance=False, save_fig=True)


## Group level analysis
group_level_correlation_within_repetitions = np.array(group_level_correlation_within_repetitions)
group_level_correlation_accross_repetitions = np.array(group_level_correlation_accross_repetitions)
group_level_correlation_matrices = np.array(group_level_correlation_matrices)

# Number of subjects
n_subjects = len(group_level_correlation_within_repetitions) // 2

plt.figure(figsize=(8, 6))

# Boxplots 
positions = [1, 2]
plt.boxplot([group_level_correlation_within_repetitions, group_level_correlation_accross_repetitions], positions=positions, widths=0.6)

# Individual subject dots + connecting lines 
cmap = plt.get_cmap('tab20')
for i in range(n_subjects):
    index = i * 2  # Each subject has two entries: pre and post
    plt.plot(positions, [group_level_correlation_within_repetitions[index], group_level_correlation_accross_repetitions[index]], 
             marker='o', linestyle='-', color=cmap(i % 20), alpha=0.7)
    plt.plot(positions, [group_level_correlation_within_repetitions[index + 1], group_level_correlation_accross_repetitions[index + 1]], 
             marker='o', linestyle='--', color=cmap(i % 20), alpha=0.7)

# Formatting 
plt.xticks(positions, ['Within Repetitions', 'Across Repetitions'])
plt.ylabel('Correlation')
plt.title('Group Level Correlation')
plt.grid(axis='y')

#  Custom legend 
solid_line = mpatches.Patch(color='black', label='Pre-Viewing', linestyle='-')
dashed_line = mpatches.Patch(color='black', label='Post-Viewing', linestyle='--')
plt.legend(handles=[solid_line, dashed_line], loc='upper left')


plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, f"group_level_correlation_within_vs_across{'_hippo' if HIPPOCAMPUS_ONLY else ''}.png"))
plt.show()

# T test between within and across
from scipy.stats import ttest_rel
t_stat, p_value = ttest_rel(group_level_correlation_within_repetitions, group_level_correlation_accross_repetitions, alternative='greater')
print(f"T-test between within and across repetitions: t={t_stat}, p={p_value}")

# Plot boxplots 
pre_avgs = np.array(pre_avgs)
# Assume pre_avgs is already a NumPy array of shape (N_participants, 6)
N, n_reps = pre_avgs.shape

# Convert to long format for Seaborn
df = pd.DataFrame(pre_avgs, columns=[f"Rep {r+1}" for r in range(n_reps)])
df_long = df.melt(var_name="Repetition", value_name="Beta")

# Map repetitions to numeric x-values for fitting
rep_mapping = {f"Rep {r+1}": r+1 for r in range(n_reps)}
df_long["Rep_num"] = df_long["Repetition"].map(rep_mapping)

plt.figure(figsize=(8, 6))

# Boxplot
sns.boxplot(x="Repetition", y="Beta", data=df_long, color="lightblue", fliersize=2)

# Overlay estimated linear fit (mean trend)
sns.regplot(x="Rep_num", y="Beta", data=df_long, scatter=True, color="blue", label="Estimated fit", ci=None)

plt.xlabel("Repetition")
plt.ylabel("Beta value")
plt.title("Pre — Beta values per repetition")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Save figure
plt.savefig(os.path.join(FIGURE_OUTPUT_PATH, f"pre_beta_values_per_repetition{'_hippo' if HIPPOCAMPUS_ONLY else ''}.png"))
plt.show()