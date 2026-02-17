import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing small run to verify singular matrix fix...")

config = {
    'k_interference_pos': 1,
    'k_train_points': 3,
    'k_art_points': 3,
    'k_test_points': 3,
    'beta_vec': [0.2],
    'snr_max_db': 20,
    'snr_min_db': 20,
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
}

try:
    from acoustic_doa import run_acoustic_doa
    print("Starting small test run...")
    results = run_acoustic_doa(config)
    print("✓ Test run completed successfully!")
    print(f"  Result keys: {list(results.keys())}")
    print(f"  Errors shape: {results['theta_est_ts_err'].shape}")
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()
