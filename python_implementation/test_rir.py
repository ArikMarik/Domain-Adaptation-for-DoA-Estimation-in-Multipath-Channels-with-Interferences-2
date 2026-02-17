import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
from signal_processing.rir import compute_rir

print("Testing RIR generator...")

source_pos = np.array([2.5, 3.0, 1.5])
receiver_pos = np.array([[2.0, 0.5, 1.5],
                         [2.12, 0.5, 1.5],
                         [2.24, 0.5, 1.5]])
fs = 12000
n_samples = 2048
beta = 0.3
room_dim = [5.2, 6.2, 3.5]
order = -1

try:
    h = compute_rir(source_pos, fs, receiver_pos, n_samples, beta, room_dim, order)
    print(f"✓ RIR computed successfully")
    print(f"  Shape: {h.shape}")
    print(f"  Expected: ({len(receiver_pos)}, {n_samples})")
    print(f"  Max value: {np.max(np.abs(h)):.6f}")
    print(f"  Energy per channel: {np.sum(h**2, axis=1)}")
    
    if h.shape == (len(receiver_pos), n_samples):
        print("✓ Shape matches expected")
    else:
        print("✗ Shape mismatch!")
        
except Exception as e:
    import traceback
    print(f"✗ Error: {e}")
    traceback.print_exc()

print("\nAll RIR tests completed!")
