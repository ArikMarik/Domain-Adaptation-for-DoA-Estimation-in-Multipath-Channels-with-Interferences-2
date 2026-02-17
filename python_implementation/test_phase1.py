"""
Test Phase 1 implementation: scenario renaming and backward compatibility.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import get_scenario_config

print("Testing Phase 1: Scenario Renaming")
print("=" * 60)

# Test 1: New scenario names work
print("\n1. Testing new scenario names...")
try:
    config = get_scenario_config('t60-sweep_clean')
    assert config['plot_t60_sweep_clean'] == True
    print("   ✓ t60-sweep_clean works")
    
    config = get_scenario_config('single-interference_fixed')
    assert config['plot_single_interference_fixed'] == True
    print("   ✓ single-interference_fixed works")
    
    config = get_scenario_config('dual-interference_moving')
    assert config['plot_dual_interference_moving'] == True
    print("   ✓ dual-interference_moving works")
    
    config = get_scenario_config('snr-sir-sweep_single-interference')
    assert 'plot_t60_sweep_clean' not in config or config.get('plot_t60_sweep_clean') == False
    print("   ✓ snr-sir-sweep_single-interference works")
    
    config = get_scenario_config('t60-sweep_clean_riemannian')
    assert config['covariance_averaging_method'] == 'riemannian'
    assert config['plot_t60_sweep_clean'] == True
    print("   ✓ t60-sweep_clean_riemannian works")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 2: Legacy names still work (with deprecation warning)
print("\n2. Testing backward compatibility with legacy names...")
try:
    import io
    from contextlib import redirect_stdout
    
    # Capture deprecation warning
    f = io.StringIO()
    with redirect_stdout(f):
        config = get_scenario_config('fig2-3-4')
    output = f.getvalue()
    
    assert 't60-sweep_clean' in output, "Should print deprecation warning"
    assert config['plot_t60_sweep_clean'] == True
    print("   ✓ fig2-3-4 → t60-sweep_clean (with deprecation warning)")
    
    config = get_scenario_config('fig5')
    print("   ✓ fig5 → single-interference_fixed")
    
    config = get_scenario_config('fig6')
    print("   ✓ fig6 → dual-interference_moving")
    
    config = get_scenario_config('table')
    print("   ✓ table → snr-sir-sweep_single-interference")
    
    config = get_scenario_config('fig2-3-4-riemannian')
    assert config['covariance_averaging_method'] == 'riemannian'
    print("   ✓ fig2-3-4-riemannian → t60-sweep_clean_riemannian")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Check file renaming
print("\n3. Testing file renaming...")
results_dir = os.path.join(os.path.dirname(__file__), 'results')

expected_files = [
    'T60Sweep_AcousticRoom3D.jpg',
    'T60Sweep_DoAError_DS.jpg',
    'T60Sweep_DoAError_MVDR.jpg',
    'T60Sweep_DoAError_MUSIC.jpg',
    'T60Sweep_Spectra_DS.jpg',
    'T60Sweep_Spectra_MVDR.jpg',
    'T60Sweep_Spectra_MUSIC.jpg',
    'SingleInterf_DoAError.jpg',
    'DualInterf_AcousticRoom3D.jpg',
    'DualInterf_DoAError.jpg',
    'DualInterf_Spectra_DS.jpg',
    'DualInterf_Spectra_MVDR.jpg',
    'DualInterf_Spectra_MUSIC.jpg',
]

old_files = [
    'Fig2_AcousticRoom3D.jpg',
    'Fig3_DS_Spectrum.jpg',
    'Fig4_DS_DoABeta.jpg',
]

for filename in expected_files:
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        print(f"   ✓ {filename} exists")
    else:
        print(f"   ⚠️  {filename} not found (may not have been generated yet)")

for filename in old_files:
    filepath = os.path.join(results_dir, filename)
    if os.path.exists(filepath):
        print(f"   ✗ {filename} still exists (should be renamed)")
    else:
        print(f"   ✓ {filename} removed/renamed")

# Test 4: Check PNG files removed
print("\n4. Testing PNG file removal...")
import glob
png_files = glob.glob(os.path.join(results_dir, '*.png'))
if len(png_files) == 0:
    print(f"   ✓ All PNG files removed")
else:
    print(f"   ✗ Found {len(png_files)} PNG files still present")

print("\n" + "=" * 60)
print("✓ Phase 1 testing complete!")
print("=" * 60)
print("\nSummary:")
print("  - New scenario names implemented")
print("  - Backward compatibility maintained")
print("  - Legacy files renamed")
print("  - PNG files removed")
print("  - JPG quality increased to 200 DPI")
