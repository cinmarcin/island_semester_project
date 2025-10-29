import argparse
import os
import json

from src.utils.utils import load_config, make_dir, json_safe
from src.data.paths import get_processed_data_folder, get_next_run_id
from src.core.rsa import (
    run_subject_level_rsa,
    run_subject_level_rsa_v2,
    second_level_sign_flip,
    plot_and_save_rsa_results,
)


# ------------------------------
# Main
# ------------------------------
def main():
    # --- Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/order_hippocampus.yaml", help="Path to YAML config file")
    parser.add_argument("--only_post_session", action="store_true", help="If set, only process post-session data and not the difference between pre and post.")
    args = parser.parse_args()
    config_file = args.config
    config = load_config(config_file)
    ONLY_POST_SESSION = args.only_post_session
    PRE_REP = False

    # --- Output directory
    figure_dir = f"figures/rsa/{os.path.basename(config_file).replace('.yaml', '')}/"
    make_dir(figure_dir)

    # --- Subjects
    subs = [
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
            # #        'sub-P26', missing event file
                     'sub-P27',
            # #        'sub-P30', incomplete data
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
    reps = ['all_reps', 'rep1', 'rep2', 'rep3', 'rep4', 'rep5', 'rep6']
    if not PRE_REP:
        reps = ['all_reps']
    for rep in reps:
        print(f"Running RSA analysis for repetition: {rep}")
        run_label = 'post' if ONLY_POST_SESSION else 'pre-post'
        figure_dir_rep = os.path.join(figure_dir, rep, run_label)
        make_dir(figure_dir_rep)
        # --- Results containers
        results = {
            "temp": {}, "spat": {}, "z_temp": {}, "z_spat": {}, "inter": {},
            "spat_low": {}, "spat_high": {},
            "pos_temp": {}, "pos_spat": {},
            "neg_temp": {}, "neg_spat": {}
        }

        # --- Run RSA per subject
        for sub in subs:
            corr_temp, z_temp, corr_spat, z_spat, tempo_spatial_corr, corr_spat_low, corr_spat_high = run_subject_level_rsa(sub, config, config_file, rep, ONLY_POST_SESSION)
            _pos_temp, _pos_spat, _neg_temp, _neg_spat = run_subject_level_rsa_v2(sub, config, config_file, rep, ONLY_POST_SESSION)

            results["temp"][sub] = corr_temp
            results["z_temp"][sub] = z_temp
            results["spat"][sub] = corr_spat
            results["z_spat"][sub] = z_spat
            results["spat_low"][sub] = corr_spat_low
            results["spat_high"][sub] = corr_spat_high
            results["pos_temp"][sub] = _pos_temp
            results["pos_spat"][sub] = _pos_spat
            results["neg_temp"][sub] = _neg_temp
            results["neg_spat"][sub] = _neg_spat


        # Save results as JSON
        save_folder = get_processed_data_folder(config_file, analysis='rsa') / 'group_level' / rep 
        make_dir(save_folder)
        save_path = get_next_run_id(save_folder, base_name="rsa_results_run", ext=".json")

        with open(save_path, 'w') as f:
            json.dump({k: {sub: json_safe(v[sub]) for sub in v} for k, v in results.items()}, f, indent=4)
        print(f"Saved group-level RSA results to {save_path}")


        # --- Permutation at the group level ---
        z_spat_values = [results['z_spat'][sub] for sub in subs]
        z_temp_values = [results['z_temp'][sub] for sub in subs]


        second_level_results_spatial = second_level_sign_flip(z_spat_values, random_state=42)
        second_level_results_temporal = second_level_sign_flip(z_temp_values, random_state=42)

        group_level_results = {
            "spatial": second_level_results_spatial,
            "temporal": second_level_results_temporal
        }
        # Save group-level permutation results
        group_save_path = save_path.with_name(save_path.stem + "_group_level_permutation.json")
        with open(group_save_path, 'w') as f:
            json.dump(json_safe(group_level_results), f, indent=4)
        print(f"Saved group-level permutation results to {group_save_path}")

        # --- Plot and save RSA results ---
        plot_and_save_rsa_results(results, figure_dir_rep, subs, second_level_results_spatial, second_level_results_temporal)



if __name__ == "__main__":
    main()
