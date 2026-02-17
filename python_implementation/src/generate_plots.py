import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pickle
import numpy as np
from visualization.plotting import plot_beta_sweep, plot_boxplots
from visualization.room_plot import plot_room_3d

def generate_all_plots(results_dir='../results'):
    results_dir = os.path.join(os.path.dirname(__file__), results_dir)
    
    print("Generating plots from results...")
    
    scenarios = ['fig2-3-4', 'fig5', 'fig6']
    
    for scenario in scenarios:
        result_file = os.path.join(results_dir, f'results_{scenario}.pkl')
        
        if not os.path.exists(result_file):
            print(f"  Skipping {scenario} - no results file found")
            continue
        
        print(f"\n  Processing {scenario}...")
        
        with open(result_file, 'rb') as f:
            results = pickle.load(f)
        
        config = results.get('config', {})
        positions = results.get('positions', {})
        
        if scenario == 'fig2-3-4':
            print("    - Generating Fig 2: 3D room plot")
            if positions['train'] is not None:
                plot_room_3d(
                    positions['train'],
                    positions['test'],
                    positions['interference'],
                    positions['mics'],
                    results['room_dim'],
                    config.get('is_interference_active', False),
                    results_dir,
                    'Fig2_AcousticRoom3D'
                )
            
            print("    - Generating Fig 4: Beta sweep plots")
            snr_ind = 0
            sir_ind = 0
            
            errors_ds = {
                'adapted': results['theta_est_ts_pt_err'],
                'standard': results['theta_est_ts_err']
            }
            plot_beta_sweep(results['beta_vec'], errors_ds, 'DS', snr_ind, sir_ind, results_dir)
            
            errors_mvdr = {
                'adapted': results['theta_est_mvdr_ts_pt_err'],
                'standard': results['theta_est_mvdr_ts_err']
            }
            plot_beta_sweep(results['beta_vec'], errors_mvdr, 'MVDR', snr_ind, sir_ind, results_dir)
            
            errors_music = {
                'adapted': results['theta_est_music_ts_pt_err'],
                'standard': results['theta_est_music_ts_err']
            }
            plot_beta_sweep(results['beta_vec'], errors_music, 'MUSIC', snr_ind, sir_ind, results_dir)
        
        elif scenario == 'fig5':
            print("    - Generating Fig 5: Box plots (multiple interference)")
            
            snr_ind = 0
            sir_ind = 0
            beta_ind = 0
            
            data = np.column_stack([
                results['theta_est_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_ts_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_mvdr_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_mvdr_ts_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_music_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_music_ts_err'][:, snr_ind, sir_ind, beta_ind]
            ])
            
            labels = [
                ['DS+DA', 'DS', 'MVDR+DA', 'MVDR', 'MUSIC+DA', 'MUSIC'],
                ['DS', 'DS', 'MVDR', 'MVDR', 'MUSIC', 'MUSIC']
            ]
            
            plot_boxplots(data, labels, 'DoA Errors', results_dir, 'Fig5_DoAErrAcoust', n_subplots=3)
        
        elif scenario == 'fig6':
            print("    - Generating Fig 6: 3D room plot and box plots (random interference)")
            
            if positions['train'] is not None:
                plot_room_3d(
                    positions['train'],
                    positions['test'],
                    positions['interference'],
                    positions['mics'],
                    results['room_dim'],
                    config.get('is_interference_active', True),
                    results_dir,
                    'Fig6_AcousticRoom3D'
                )
            
            snr_ind = 0
            sir_ind = 0
            beta_ind = 0
            
            data = np.column_stack([
                results['theta_est_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_ts_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_mvdr_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_mvdr_ts_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_music_ts_pt_err'][:, snr_ind, sir_ind, beta_ind],
                results['theta_est_music_ts_err'][:, snr_ind, sir_ind, beta_ind]
            ])
            
            labels = [
                ['DS+DA', 'DS', 'MVDR+DA', 'MVDR', 'MUSIC+DA', 'MUSIC'],
                ['DS', 'DS', 'MVDR', 'MVDR', 'MUSIC', 'MUSIC']
            ]
            
            plot_boxplots(data, labels, 'DoA Errors', results_dir, 'Fig6_DoAErrAcoust', n_subplots=3)
    
    print("\n✓ All plots generated!")
    print(f"  Plots saved to: {results_dir}")

if __name__ == '__main__':
    generate_all_plots()
