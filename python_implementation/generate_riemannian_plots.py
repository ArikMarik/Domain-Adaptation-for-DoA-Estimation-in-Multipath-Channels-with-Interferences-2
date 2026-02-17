#!/usr/bin/env python3
"""
Generate plots for Riemannian scenario from saved results.

This script loads the completed Riemannian results and generates:
1. Beta sweep DoA error plots (DS, MVDR, MUSIC)
2. Comparison with arithmetic mean (if available)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

def generate_beta_sweep_plots(results, output_dir='results'):
    """Generate beta sweep plots for Riemannian scenario."""
    from src.visualization.plotting import plot_beta_sweep
    
    print("Generating beta sweep plots...")
    
    config = results['config']
    beta_vec = results['beta_vec']
    
    # Extract errors for each method
    theta_est_ts_err = results['theta_est_ts_err']
    theta_est_ts_pt_err = results['theta_est_ts_pt_err']
    theta_est_mvdr_ts_err = results['theta_est_mvdr_ts_err']
    theta_est_mvdr_ts_pt_err = results['theta_est_mvdr_ts_pt_err']
    theta_est_music_ts_err = results['theta_est_music_ts_err']
    theta_est_music_ts_pt_err = results['theta_est_music_ts_pt_err']
    
    # Generate plots for each method
    methods = ['DS', 'MVDR', 'MUSIC']
    errors_standard = [theta_est_ts_err, theta_est_mvdr_ts_err, theta_est_music_ts_err]
    errors_adapted = [theta_est_ts_pt_err, theta_est_mvdr_ts_pt_err, theta_est_music_ts_pt_err]
    
    snr_ind = 0  # Use first SNR index
    sir_ind = 0  # Use first SIR index
    
    for method, err_std, err_ada in zip(methods, errors_standard, errors_adapted):
        print(f"  Plotting {method}...")
        errors_dict = {
            'standard': err_std,
            'adapted': err_ada
        }
        plot_beta_sweep(
            beta_vec=beta_vec,
            errors_dict=errors_dict,
            method_name=f'{method}_Riemannian',
            snr_ind=snr_ind,
            sir_ind=sir_ind,
            output_dir=output_dir
        )
        print(f"    ✓ Saved T60Sweep_DoAError_{method}_Riemannian.jpg")
    
    print("\n✅ All beta sweep plots generated!")


def compare_with_arithmetic(riem_results, arith_results, output_dir='results'):
    """Compare Riemannian with arithmetic mean results."""
    from src.visualization.plotting import plot_arithmetic_vs_riemannian_comparison
    
    print("\nGenerating comparison plots (Arithmetic vs Riemannian)...")
    
    beta_vec = riem_results['beta_vec']
    
    # Call comparison plotting function
    plot_arithmetic_vs_riemannian_comparison(
        arith_results, riem_results, beta_vec, output_dir
    )
    
    print("✅ Comparison plots generated!")


def main():
    print("=" * 60)
    print("Riemannian Scenario - Plot Generation")
    print("=" * 60)
    print()
    
    # Load Riemannian results
    riem_file = 'results/results_t60-sweep_clean_riemannian.pkl'
    if not os.path.exists(riem_file):
        print(f"❌ Error: {riem_file} not found!")
        return False
    
    print(f"Loading Riemannian results from {riem_file}...")
    with open(riem_file, 'rb') as f:
        riem_results = pickle.load(f)
    
    print(f"  ✓ Loaded results for {len(riem_results['beta_vec'])} beta values")
    print(f"    Beta values: {riem_results['beta_vec']}")
    print()
    
    # Generate beta sweep plots
    generate_beta_sweep_plots(riem_results)
    
    # Check if arithmetic results exist for comparison
    arith_file = 'results/results_t60-sweep_clean.pkl'
    if os.path.exists(arith_file):
        print(f"\nFound arithmetic mean results: {arith_file}")
        print("Loading for comparison...")
        with open(arith_file, 'rb') as f:
            arith_results = pickle.load(f)
        
        compare_with_arithmetic(riem_results, arith_results)
    else:
        print(f"\n⚠️  Arithmetic mean results not found ({arith_file})")
        print("   Run standard scenario to generate comparison:")
        print("   python src/main.py --scenario t60-sweep_clean")
    
    print("\n" + "=" * 60)
    print("✅ Plot generation complete!")
    print("=" * 60)
    print(f"\nPlots saved in: results/")
    print("  - T60Sweep_DoAError_DS_Riemannian.jpg")
    print("  - T60Sweep_DoAError_MVDR_Riemannian.jpg")
    print("  - T60Sweep_DoAError_MUSIC_Riemannian.jpg")
    if os.path.exists(arith_file):
        print("  - Comparison_ArithVsRiem_*.jpg (4 files)")
    print()
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
