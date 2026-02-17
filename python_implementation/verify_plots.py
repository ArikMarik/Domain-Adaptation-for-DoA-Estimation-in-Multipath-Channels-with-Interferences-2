import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import numpy as np
import matplotlib.pyplot as plt

print("Verifying plot data...\n")

results_dir = 'results'
result_file = os.path.join(results_dir, 'results_fig2-3-4.pkl')

if not os.path.exists(result_file):
    print(f"❌ {result_file} not found!")
    sys.exit(1)

with open(result_file, 'rb') as f:
    results = pickle.load(f)

print("✓ Loaded results successfully")
print(f"  Shape: {results['theta_est_ts_err'].shape}")
print(f"  Beta values: {results['beta_vec']}")
print(f"  Test points: {results['config']['k_test_points']}")

snr_ind = 0
sir_ind = 0

print("\n--- Checking DS errors ---")
adapted = results['theta_est_ts_pt_err'][:, snr_ind, sir_ind, :]
standard = results['theta_est_ts_err'][:, snr_ind, sir_ind, :]

for b, beta in enumerate(results['beta_vec']):
    median_adapted = np.median(adapted[:, b])
    median_standard = np.median(standard[:, b])
    print(f"  Beta {beta}: Adapted={median_adapted:.2f}°, Standard={median_standard:.2f}°")

print("\n--- Creating test plot ---")
fig, ax = plt.subplots(figsize=(8, 6))

vec_adapted = np.median(adapted, axis=0)
vec_standard = np.median(standard, axis=0)

ax.plot(results['beta_vec'], vec_adapted, 'p-', linewidth=3, markersize=6, label='Our DS')
ax.plot(results['beta_vec'], vec_standard, 'p-', linewidth=3, markersize=6, label='DS')

ax.set_ylim([0, 10])
ax.set_xlabel('Beta [sec]')
ax.set_ylabel('Error [deg]')
ax.legend()
ax.grid(True)

test_file = os.path.join(results_dir, 'TEST_Fig4_DS.png')
plt.savefig(test_file, dpi=150, bbox_inches='tight')
print(f"✓ Saved test plot to {test_file}")
plt.close()

print("\n--- Checking MVDR errors ---")
adapted_mvdr = results['theta_est_mvdr_ts_pt_err'][:, snr_ind, sir_ind, :]
standard_mvdr = results['theta_est_mvdr_ts_err'][:, snr_ind, sir_ind, :]

for b, beta in enumerate(results['beta_vec']):
    median_adapted = np.median(adapted_mvdr[:, b])
    median_standard = np.median(standard_mvdr[:, b])
    print(f"  Beta {beta}: Adapted={median_adapted:.2f}°, Standard={median_standard:.2f}°")

print("\n✓ All checks passed!")
print("\nIf TEST_Fig4_DS.png looks good, the data is valid.")
print("If it looks empty, there may be a matplotlib display issue.")
