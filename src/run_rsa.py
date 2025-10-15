import argparse
import yaml
from utils.data import get_model_rdm, get_rsa_matrix
from scipy.stats import spearmanr, pearsonr
import numpy as np

## Controls
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="configs/order_hippocampus.yaml", help="Path to YAML config file")
args = parser.parse_args()

config_file = args.config

# Load YAML
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

## Paths
list_of_subs = [
                'sub-P10',
                'sub-P12',
                'sub-P13',
        #         'sub-P14',
        #         'sub-P15',
        #         'sub-P16',
        #         'sub-P17',
        #         'sub-P18',
        #         'sub-P19',
        #     #    'sub-P20', Unattentive
        #         'sub-P21',
        #         'sub-P22',
        #         'sub-P23',
        #         'sub-P25',
        # #        'sub-P26', missing event file
        #         'sub-P27',
        # #        'sub-P30', incomplete data
                'sub-P31',
                'sub-P34',
        #         'sub-P35',
        #         'sub-P36',
        #         'sub-P37',
        #         'sub-P41',
        #         'sub-P42',
        #         'sub-P43',
        #         'sub-P44',
        #         'sub-P45',
        #         'sub-P47',
        #         'sub-P48',
        # #        'sub-P49',   Unattentive
        #         'sub-P50',
        #         'sub-P51',
                ]
inter_tempo_spatial_results= {}
temp_results = {}
spat_results = {}
for sub in list_of_subs:
    print(f"---------- Running RSA for {sub} with config {config_file}")

    # Load the Data
    temporal_rdm = get_model_rdm(sub, spatial=False, config=config, config_file=config_file)
    spatial_rdm = get_model_rdm(sub, spatial=True, config=config, config_file=config_file)

    tempo_spatial_corr = spearmanr(temporal_rdm[np.triu_indices(len(temporal_rdm), k=1)], spatial_rdm[np.triu_indices(len(spatial_rdm), k=1)])
    print(f"Spearman correlation between temporal and spatial RDM for {sub}: {tempo_spatial_corr.correlation:.4f} (p={tempo_spatial_corr.pvalue:.4f})")

    pre_rsa = get_rsa_matrix(sub, config=config, config_file=config_file, run_label='pre-post')


    # Compute spearman correlation
    corr_temp, p_temp = spearmanr(temporal_rdm[np.triu_indices(len(temporal_rdm), k=1)], pre_rsa[np.triu_indices(len(pre_rsa), k=1)])
    corr_spat, p_spat = spearmanr(spatial_rdm[np.triu_indices(len(spatial_rdm), k=1)], pre_rsa[np.triu_indices(len(pre_rsa), k=1)])

    print(f"Spearman correlation between temporal RDM and RSA for {sub}: {corr_temp:.4f} (p={p_temp:.4f})")
    print(f"Spearman correlation between spatial RDM and RSA for {sub}: {corr_spat:.4f} (p={p_spat:.4f})")

    # add correlation and p-vaule to dict
    temp_results[sub] = (corr_temp, p_temp)
    spat_results[sub] = (corr_spat, p_spat)
    inter_tempo_spatial_results[sub] = (tempo_spatial_corr.correlation, tempo_spatial_corr.pvalue)

# print results
print("------- Summary of results -------")
print("Mean Spearman correlation between temporal RDM and RSA: {:.4f} (p={:.4f})".format(np.mean([temp_results[sub][0] for sub in temp_results]), np.mean([temp_results[sub][1] for sub in temp_results])))
print("Mean Spearman correlation between spatial RDM and RSA: {:.4f} (p={:.4f})".format(np.mean([spat_results[sub][0] for sub in spat_results]), np.mean([spat_results[sub][1] for sub in spat_results])))
print("Mean Spearman correlation between temporal and spatial RDM: {:.4f} (p={:.4f})".format(np.mean([inter_tempo_spatial_results[sub][0] for sub in inter_tempo_spatial_results]), np.mean([inter_tempo_spatial_results[sub][1] for sub in inter_tempo_spatial_results])))