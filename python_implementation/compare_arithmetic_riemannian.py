"""
Compare arithmetic vs Riemannian mean results and generate comparison plots.

Usage:
    python compare_arithmetic_riemannian.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from visualization.plotting import plot_arithmetic_vs_riemannian_comparison

# Paths to result files
results_dir = os.path.join(os.path.dirname(__file__), 'results')
arith_file = os.path.join(results_dir, 'results_t60-sweep_clean.pkl')
riem_file = os.path.join(results_dir, 'results_t60-sweep_clean_riemannian.pkl')

# Try legacy names if new ones don't exist
if not os.path.exists(arith_file):
    arith_file_legacy = os.path.join(results_dir, 'results_fig2-3-4.pkl')
    if os.path.exists(arith_file_legacy):
        print(f"⚠️  Using legacy file: results_fig2-3-4.pkl")
        arith_file = arith_file_legacy

if not os.path.exists(riem_file):
    riem_file_legacy = os.path.join(results_dir, 'results_fig2-3-4-riemannian.pkl')
    if os.path.exists(riem_file_legacy):
        print(f"⚠️  Using legacy file: results_fig2-3-4-riemannian.pkl")
        riem_file = riem_file_legacy

print("Comparing Arithmetic vs Riemannian Mean Results")
print("=" * 60)
print(f"Arithmetic results: {arith_file}")
print(f"Riemannian results: {riem_file}")
print()

# Check if files exist
if not os.path.exists(arith_file):
    print(f"ERROR: Arithmetic results file not found!")
    print(f"Please run: python src/main.py --scenario t60-sweep_clean")
    sys.exit(1)

if not os.path.exists(riem_file):
    print(f"ERROR: Riemannian results file not found!")
    print(f"Please run: python src/main.py --scenario t60-sweep_clean_riemannian")
    sys.exit(1)

# Generate comparison plots
plot_arithmetic_vs_riemannian_comparison(arith_file, riem_file, results_dir)

print("\n" + "=" * 60)
print("Comparison complete! Check results/ directory for plots:")
print("  - Comparison_ArithVsRiem_DS.jpg")
print("  - Comparison_ArithVsRiem_MVDR.jpg")
print("  - Comparison_ArithVsRiem_MUSIC.jpg")
print("  - Comparison_ArithVsRiem_Summary.jpg")
print("=" * 60)
