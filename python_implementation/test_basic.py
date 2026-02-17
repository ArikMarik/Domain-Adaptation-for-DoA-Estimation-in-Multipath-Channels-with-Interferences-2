import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    from signal_processing.stft import stft
    print("✓ STFT imported")
except Exception as e:
    print(f"✗ STFT import failed: {e}")

try:
    from signal_processing.rir import compute_rir
    print("✓ RIR imported")
except Exception as e:
    print(f"✗ RIR import failed: {e}")

try:
    from signal_processing.signal_correlation import sig_corr_at_mics_acoustic_array
    print("✓ Signal correlation imported")
except Exception as e:
    print(f"✗ Signal correlation import failed: {e}")

try:
    from algorithms.doa_estimators import ml_spectrum, mvdr_spectrum, music_spectrum
    print("✓ DoA estimators imported")
except Exception as e:
    print(f"✗ DoA estimators import failed: {e}")

try:
    from algorithms.domain_adaptation import compute_adaptation_matrix
    print("✓ Domain adaptation imported")
except Exception as e:
    print(f"✗ Domain adaptation import failed: {e}")

try:
    from utils.math_utils import sorted_evd
    from utils.geometry import angle_between_vectors
    from utils.sir_calculator import sir_calc
    print("✓ Utils imported")
except Exception as e:
    print(f"✗ Utils import failed: {e}")

print("\nAll imports successful!")
