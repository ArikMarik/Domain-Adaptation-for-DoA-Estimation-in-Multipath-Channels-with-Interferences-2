import argparse
import os
import time
from acoustic_doa import run_acoustic_doa

def get_scenario_config(scenario_name):
    if scenario_name == 'fig2-3-4':
        return {
            'k_interference_pos': 1,
            'k_train_points': 100,
            'k_art_points': 200,
            'k_test_points': 300,
            'beta_vec': [0.2, 0.3, 0.4, 0.5, 0.6],
            'snr_max_db': -20,
            'snr_min_db': -20,
            'sir_max_db': -20,
            'sir_min_db': -20,
            'is_interference_active': False,
            'is_interference_fixed': True,
            'sig_dim_4_music_pt': 1,
            'sig_dim_4_music': 1,
            'train_is_test': True,
            'is_only_interf_active_during_train': 0,
            'is_two_interference_sources_active': False,
            'k_inter_pos': 1,
            'dist_between_train': 0.2,
            'plot_figs_2_3_4': True,
        }
    elif scenario_name == 'fig5':
        return {
            'k_interference_pos': 20,
            'k_train_points': 100,
            'k_art_points': 200,
            'k_test_points': 200,
            'beta_vec': [0.2],
            'snr_max_db': -20,
            'snr_min_db': -20,
            'sir_max_db': -20,
            'sir_min_db': -20,
            'is_interference_active': True,
            'is_interference_fixed': True,
            'sig_dim_4_music_pt': 1,
            'sig_dim_4_music': 2,
            'train_is_test': True,
            'is_only_interf_active_during_train': 0,
            'is_two_interference_sources_active': False,
            'k_inter_pos': 1,
            'dist_between_train': 0.2,
            'plot_figs_5': True,
        }
    elif scenario_name == 'fig6':
        return {
            'k_interference_pos': 1,
            'k_train_points': 100,
            'k_art_points': 200,
            'k_test_points': 300,
            'beta_vec': [0.2],
            'snr_max_db': 0,
            'snr_min_db': 0,
            'sir_max_db': 0,
            'sir_min_db': 0,
            'is_interference_active': True,
            'is_interference_fixed': False,
            'sig_dim_4_music_pt': 1,
            'sig_dim_4_music': 3,
            'train_is_test': True,
            'is_only_interf_active_during_train': 0,
            'is_two_interference_sources_active': True,
            'k_inter_pos': 1,
            'dist_between_train': 0.2,
            'plot_figs_6': True,
        }
    elif scenario_name == 'table':
        return {
            'k_interference_pos': 20,
            'k_train_points': 100,
            'k_art_points': 200,
            'k_test_points': 200,
            'beta_vec': [0.2],
            'snr_max_db': 0,
            'snr_min_db': -20,
            'sir_max_db': 0,
            'sir_min_db': -20,
            'is_interference_active': True,
            'is_interference_fixed': True,
            'sig_dim_4_music_pt': 1,
            'sig_dim_4_music': 2,
            'train_is_test': True,
            'is_only_interf_active_during_train': 0,
            'is_two_interference_sources_active': False,
            'k_inter_pos': 1,
            'dist_between_train': 0.2,
        }
    elif scenario_name == 'fig2-3-4-riemannian':
        return {
            'k_interference_pos': 1,
            'k_train_points': 100,
            'k_art_points': 200,
            'k_test_points': 300,
            'beta_vec': [0.2, 0.3, 0.4, 0.5, 0.6],
            'snr_max_db': -20,
            'snr_min_db': -20,
            'sir_max_db': -20,
            'sir_min_db': -20,
            'is_interference_active': False,
            'is_interference_fixed': True,
            'sig_dim_4_music_pt': 1,
            'sig_dim_4_music': 1,
            'train_is_test': True,
            'is_only_interf_active_during_train': 0,
            'is_two_interference_sources_active': False,
            'k_inter_pos': 1,
            'dist_between_train': 0.2,
            'plot_figs_2_3_4': True,
            'covariance_averaging_method': 'riemannian',
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario_name}")

def main():
    parser = argparse.ArgumentParser(description='Run Acoustic DoA Estimation with Domain Adaptation')
    parser.add_argument('--scenario', type=str, default='all', 
                        choices=['all', 'fig2-3-4', 'fig5', 'fig6', 'table', 'fig2-3-4-riemannian'],
                        help='Scenario to run')
    args = parser.parse_args()
    
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    if args.scenario == 'all':
        # scenarios = ['fig2-3-4', 'fig5', 'fig6', 'table']
        scenarios = ['fig2-3-4', 'fig6', 'table']
    else:
        scenarios = [args.scenario]
    
    sim_dur_vec_min = []
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {scenario}")
        print(f"{'='*60}\n")
        
        config = get_scenario_config(scenario)
        
        start_time = time.time()
        results = run_acoustic_doa(config)
        elapsed_time = (time.time() - start_time) / 60
        
        sim_dur_vec_min.append(elapsed_time)
        
        print(f"\nScenario {scenario} completed in {elapsed_time:.2f} minutes")
        
        import pickle
        result_file = os.path.join(results_dir, f'results_{scenario}.pkl')
        with open(result_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"Results saved to {result_file}")
    
    print(f"\n{'='*60}")
    print(f"All scenarios completed!")
    print(f"Total time: {sum(sim_dur_vec_min):.2f} minutes")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
