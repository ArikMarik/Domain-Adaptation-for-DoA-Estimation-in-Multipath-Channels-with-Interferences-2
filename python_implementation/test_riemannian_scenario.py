"""
Quick test of Riemannian mean integration with DOA estimation.
Uses reduced parameters for fast testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from acoustic_doa import run_acoustic_doa
import time

print("Testing Riemannian Mean Integration with DOA Estimation")
print("=" * 60)

# Small test configuration
config_arithmetic = {
    'k_interference_pos': 1,
    'k_train_points': 5,      # Reduced from 100
    'k_art_points': 10,       # Reduced from 200
    'k_test_points': 3,       # Reduced from 300
    'beta_vec': [0.3],        # Single beta value
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
    'plot_figs_2_3_4': False,  # Disable plotting
    'covariance_averaging_method': 'arithmetic',
}

config_riemannian = config_arithmetic.copy()
config_riemannian['covariance_averaging_method'] = 'riemannian'

print("\n1. Running with ARITHMETIC mean...")
start = time.time()
results_arith = run_acoustic_doa(config_arithmetic)
time_arith = time.time() - start
print(f"   Completed in {time_arith:.2f} seconds")

print("\n2. Running with RIEMANNIAN mean...")
start = time.time()
results_riem = run_acoustic_doa(config_riemannian)
time_riem = time.time() - start
print(f"   Completed in {time_riem:.2f} seconds")
print(f"   Slowdown: {time_riem/time_arith:.1f}x")

print("\n" + "=" * 60)
print("Comparing Results:")
print("=" * 60)

import numpy as np

# Compare errors for DS method
ds_err_arith = results_arith['theta_est_ts_pt_err'][:, 0, 0, 0]
ds_err_riem = results_riem['theta_est_ts_pt_err'][:, 0, 0, 0]

print(f"\nDS+DA Method:")
print(f"  Arithmetic - Median error: {np.median(ds_err_arith):.3f}°")
print(f"  Riemannian - Median error: {np.median(ds_err_riem):.3f}°")
print(f"  Difference: {np.median(ds_err_arith) - np.median(ds_err_riem):.3f}°")

# Compare errors for MVDR method
mvdr_err_arith = results_arith['theta_est_mvdr_ts_pt_err'][:, 0, 0, 0]
mvdr_err_riem = results_riem['theta_est_mvdr_ts_pt_err'][:, 0, 0, 0]

print(f"\nMVDR+DA Method:")
print(f"  Arithmetic - Median error: {np.median(mvdr_err_arith):.3f}°")
print(f"  Riemannian - Median error: {np.median(mvdr_err_riem):.3f}°")
print(f"  Difference: {np.median(mvdr_err_arith) - np.median(mvdr_err_riem):.3f}°")

# Compare errors for MUSIC method
music_err_arith = results_arith['theta_est_music_ts_pt_err'][:, 0, 0, 0]
music_err_riem = results_riem['theta_est_music_ts_pt_err'][:, 0, 0, 0]

print(f"\nMUSIC+DA Method:")
print(f"  Arithmetic - Median error: {np.median(music_err_arith):.3f}°")
print(f"  Riemannian - Median error: {np.median(music_err_riem):.3f}°")
print(f"  Difference: {np.median(music_err_arith) - np.median(music_err_riem):.3f}°")

print("\n" + "=" * 60)
print("✓ Integration test completed successfully!")
print("=" * 60)
print(f"\nNote: With full scenario parameters (100 train, 200 art, 300 test points),")
print(f"the Riemannian method will take approximately {time_riem/time_arith:.0f}x longer.")
print(f"Estimated full runtime: ~{time_arith * (100/5) * (200/10) * (300/3) / 60:.1f} min (arith) vs ~{time_riem * (100/5) * (200/10) * (300/3) / 60:.1f} min (riem)")
