import argparse
import yaml
from utils.data import get_model_rdm, get_rsa_matrix
from scipy.stats import spearmanr
import numpy as np

# ------------------------------
# Argument parser
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config",
    type=str,
    default="configs/order_hippocampus.yaml",
    help="Path to YAML config file"
)
args = parser.parse_args()
config_file = args.config

FIGURE_PATH = f"figures/rsa/{config_file.split('/')[-1].replace('.yaml', '')}/"

# ------------------------------
# Load configuration
# ------------------------------
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# ------------------------------
# Subjects
# ------------------------------
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
        # #        'sub-P49',   Unattentive
                'sub-P50',
                'sub-P51',
                ]
# ------------------------------
# Initialize results containers
# ------------------------------
inter_tempo_spatial_results = {}
temp_results = {}
spat_results = {}
# store high/low split results
temp_low_results = {}
temp_high_results = {}
spat_low_results = {}
spat_high_results = {}

# ------------------------------
# Loop through subjects
# ------------------------------
for sub in list_of_subs:
    # Load behavioral RDMs
    temporal_rdm = get_model_rdm(sub, spatial=False, config=config, config_file=config_file)
    spatial_rdm = get_model_rdm(sub, spatial=True, config=config, config_file=config_file)

    print(f"\n{'-'*70}")
    print(f"Running RSA for {sub}")
    print(f"Config file: {config_file}")
    print(f"RDM dimensions: {temporal_rdm.shape[0]} x {temporal_rdm.shape[1]}")
    print(f"{'-'*70}")

    # Sanity checks
    assert temporal_rdm.shape == spatial_rdm.shape, "Temporal and spatial RDMs must have the same shape"
    assert temporal_rdm.shape[0] == temporal_rdm.shape[1], "RDMs must be square matrices"

    # Extract upper triangle (excluding diagonal)
    iu = np.triu_indices(len(temporal_rdm), k=1)
    temporal_vals = temporal_rdm[iu]
    spatial_vals = spatial_rdm[iu]

    # Compute medians
    temp_median = np.median(temporal_vals)
    spat_median = np.median(spatial_vals)
    print(f"Median temporal distance: {temp_median:.4f}")
    print(f"Median spatial distance:  {spat_median:.4f}")

    # Correlation between temporal and spatial behavioral RDMs
    tempo_spatial_corr = spearmanr(temporal_vals, spatial_vals)
    print(f"Temporal vs. spatial RDM: ρ = {tempo_spatial_corr.correlation:.4f}, p = {tempo_spatial_corr.pvalue:.4f}")

    # Load neural RSA matrix (PS′)
    pre_rsa = get_rsa_matrix(sub, config=config, config_file=config_file, run_label='pre-post')
    rsa_vals = pre_rsa[iu]

    # Compute Spearman correlations between model and neural RDMs (full)
    corr_temp, p_temp = spearmanr(temporal_vals, rsa_vals)
    corr_spat, p_spat = spearmanr(spatial_vals, rsa_vals)

    print(f"Temporal RDM ↔ RSA (full): ρ = {corr_temp:.4f}, p = {p_temp:.4f}")
    print(f"Spatial RDM  ↔ RSA (full): ρ = {corr_spat:.4f}, p = {p_spat:.4f}")

    # Median split indices for high / low
    temp_low_mask = temporal_vals <= temp_median
    temp_high_mask = temporal_vals > temp_median
    spat_low_mask = spatial_vals <= spat_median
    spat_high_mask = spatial_vals > spat_median

    # Safely compute spearman on subsets (handle degenerate cases)
    def safe_spearman(x, y):
        # if constant array -> return nan p=1
        if np.all(x == x[0]) or np.all(y == y[0]):
            return (np.nan, np.nan)
        return spearmanr(x, y).correlation, spearmanr(x, y).pvalue
    # Temporal low/high correlations
    corr_temp_low, p_temp_low = safe_spearman(rsa_vals[temp_low_mask], temporal_vals[temp_low_mask])
    corr_temp_high, p_temp_high = safe_spearman(rsa_vals[temp_high_mask], temporal_vals[temp_high_mask])

    # Spatial low/high correlations
    corr_spat_low, p_spat_low = safe_spearman(rsa_vals[spat_low_mask], spatial_vals[spat_low_mask])
    corr_spat_high, p_spat_high = safe_spearman(rsa_vals[spat_high_mask], spatial_vals[spat_high_mask])

    print(f"Temporal RDM ↔ RSA (low dist):  ρ = {np.nan_to_num(corr_temp_low):.4f}, p = {np.nan_to_num(p_temp_low):.4f}")
    print(f"Temporal RDM ↔ RSA (high dist): ρ = {np.nan_to_num(corr_temp_high):.4f}, p = {np.nan_to_num(p_temp_high):.4f}")
    print(f"Spatial  RDM ↔ RSA (low dist):   ρ = {np.nan_to_num(corr_spat_low):.4f}, p = {np.nan_to_num(p_spat_low):.4f}")
    print(f"Spatial  RDM ↔ RSA (high dist):  ρ = {np.nan_to_num(corr_spat_high):.4f}, p = {np.nan_to_num(p_spat_high):.4f}")

    # Store results
    temp_results[sub] = (corr_temp, p_temp)
    spat_results[sub] = (corr_spat, p_spat)
    inter_tempo_spatial_results[sub] = (tempo_spatial_corr.correlation, tempo_spatial_corr.pvalue)

    temp_low_results[sub] = (corr_temp_low, p_temp_low)
    temp_high_results[sub] = (corr_temp_high, p_temp_high)
    spat_low_results[sub] = (corr_spat_low, p_spat_low)
    spat_high_results[sub] = (corr_spat_high, p_spat_high)

# ------------------------------
# Summary helpers
# ------------------------------
def mean_sem(array_of_vals):
    arr = np.array([v for v in array_of_vals if not np.isnan(v)])
    if arr.size == 0:
        return np.nan, np.nan
    mean = np.mean(arr)
    sem = np.std(arr, ddof=1) / np.sqrt(len(arr))
    return mean, sem

def summarize_results(name, results):
    corrs = np.array([results[sub][0] for sub in results])
    ps = np.array([results[sub][1] for sub in results])
    mean_corr, sem_corr = mean_sem(corrs)
    mean_p = np.nanmean(ps)
    print(f"\n{name} (N={len(results)}):")
    print(f"  Mean ρ = {mean_corr:.4f} ± {sem_corr:.4f} (SEM)")
    print(f"  Mean p = {mean_p:.4f}")

# ------------------------------
# Print summary
# ------------------------------
print("\n" + "="*70)
print("Summary of RSA results across subjects")
print("="*70)
summarize_results("Temporal RDM ↔ RSA (full)", temp_results)
summarize_results("Spatial RDM  ↔ RSA (full)", spat_results)
summarize_results("Temporal ↔ Spatial RDM", inter_tempo_spatial_results)

# summarize high/low splits
def summarize_split(name, results_low, results_high):
    mean_low, sem_low = mean_sem([results_low[s][0] for s in results_low])
    mean_high, sem_high = mean_sem([results_high[s][0] for s in results_high])
    print(f"\n{name} (median split):")
    print(f"  Low-dist mean ρ = {mean_low:.4f} ± {sem_low:.4f} (SEM)")
    print(f"  High-dist meanρ = {mean_high:.4f} ± {sem_high:.4f} (SEM)")

summarize_split("Temporal RDM ↔ RSA", temp_low_results, temp_high_results)
summarize_split("Spatial  RDM ↔ RSA", spat_low_results, spat_high_results)
print("="*70)


import os
import matplotlib.pyplot as plt

os.makedirs(FIGURE_PATH, exist_ok=True)

# ------------------------------
# Per-subject bar plots
# ------------------------------
for sub in list_of_subs:
    values = [
        temp_results[sub][0],
        spat_results[sub][0],
        inter_tempo_spatial_results[sub][0]
    ]
    labels = ['Temporal ↔ RSA', 'Spatial ↔ RSA', 'Temporal ↔ Spatial RDM']

    plt.figure(figsize=(6, 4))
    plt.bar(labels, values, color=['skyblue', 'salmon', 'lightgreen'])
    plt.ylim(-1, 1)
    plt.ylabel("Spearman ρ")
    plt.title(f"RSA correlations - {sub}")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{FIGURE_PATH}{sub}_rsa_barplot.png")
    plt.close()

# ------------------------------
# Group-level bar plot with SEM
# ------------------------------
# Compute means and SEM
mean_temp, sem_temp = mean_sem([temp_results[s][0] for s in list_of_subs])
mean_spat, sem_spat = mean_sem([spat_results[s][0] for s in list_of_subs])
mean_inter, sem_inter = mean_sem([inter_tempo_spatial_results[s][0] for s in list_of_subs])

values = [mean_temp, mean_spat, mean_inter]
errors = [sem_temp, sem_spat, sem_inter]
labels = ['Temporal ↔ RSA', 'Spatial ↔ RSA', 'Temporal ↔ Spatial RDM']

plt.figure(figsize=(6, 4))
plt.bar(labels, values, yerr=errors, capsize=5, color=['skyblue', 'salmon', 'lightgreen'])
plt.ylim(-1, 1)
plt.ylabel("Spearman ρ (mean ± SEM)")
plt.title("Group-level RSA correlations")
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}group_rsa_barplot.png")
plt.close()

# ------------------------------
# Per-subject median-split bar plots
# ------------------------------
for sub in list_of_subs:
    # Temporal
    temp_low = temp_low_results[sub][0]
    temp_high = temp_high_results[sub][0]
    # Spatial
    spat_low = spat_low_results[sub][0]
    spat_high = spat_high_results[sub][0]

    plt.figure(figsize=(8, 4))

    # Temporal
    plt.subplot(1, 2, 1)
    plt.bar(['Low', 'High'], [temp_low, temp_high], color=['skyblue', 'dodgerblue'])
    plt.ylim(-1, 1)
    plt.ylabel("Spearman ρ")
    plt.title(f"Temporal RDM ↔ RSA ({sub})")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Spatial
    plt.subplot(1, 2, 2)
    plt.bar(['Low', 'High'], [spat_low, spat_high], color=['salmon', 'red'])
    plt.ylim(-1, 1)
    plt.title(f"Spatial RDM ↔ RSA ({sub})")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    path = FIGURE_PATH + sub
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/{sub}_rsa_median_split.png")
    plt.close()

# ------------------------------
# Group-level median-split plot with SEM
# ------------------------------
def group_mean_sem(results_low, results_high):
    mean_low, sem_low = mean_sem([results_low[s][0] for s in list_of_subs])
    mean_high, sem_high = mean_sem([results_high[s][0] for s in list_of_subs])
    return [mean_low, mean_high], [sem_low, sem_high]

# Temporal
temp_vals, temp_err = group_mean_sem(temp_low_results, temp_high_results)
# Spatial
spat_vals, spat_err = group_mean_sem(spat_low_results, spat_high_results)

plt.figure(figsize=(8, 4))
from matplotlib.ticker import FuncFormatter
scale_factor = 1e1  # for 10^-1 axis

# Convert to NumPy arrays and handle NaNs
temp_vals = np.array(temp_vals)
temp_err  = np.array(temp_err)
spat_vals = np.array(spat_vals)
spat_err  = np.array(spat_err)

# Replace NaNs with 0 (or you can choose np.nan_to_num(..., nan=0) if you prefer)
temp_vals = np.nan_to_num(temp_vals, nan=0)
temp_err  = np.nan_to_num(temp_err, nan=0)
spat_vals = np.nan_to_num(spat_vals, nan=0)
spat_err  = np.nan_to_num(spat_err, nan=0)

# Formatter for 10^-1 axis
formatter = FuncFormatter(lambda y, _: f'{y:.0f}')

# Temporal
plt.subplot(1, 2, 1)
plt.bar(['Low', 'High'], temp_vals*scale_factor, yerr=temp_err*scale_factor, capsize=5, color=['skyblue', 'dodgerblue'])
plt.ylim(-1, 1)
plt.ylabel("Spearman ρ (×10⁻¹)")
plt.title("Temporal RDM ↔ RSA")
plt.gca().yaxis.set_major_formatter(formatter)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Spatial
plt.subplot(1, 2, 2)
plt.bar(['Low', 'High'], spat_vals*scale_factor, yerr=spat_err*scale_factor, capsize=5, color=['salmon', 'red'])
plt.ylim(-1, 1)
plt.ylabel("Spearman ρ (×10⁻¹)")
plt.title("Spatial RDM ↔ RSA")
plt.gca().yaxis.set_major_formatter(formatter)
plt.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f"{FIGURE_PATH}group_rsa_median_split.png")
plt.close()