import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import pickle

def plot_beta_sweep(beta_vec, errors_dict, method_name, snr_ind, sir_ind, output_dir):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    blue_col = [0, 0.4470, 0.7410]
    red_col = [0.8500, 0.3250, 0.0980]
    
    vec_adapted = np.median(errors_dict['adapted'][:, snr_ind, sir_ind, :], axis=0)
    vec_standard = np.median(errors_dict['standard'][:, snr_ind, sir_ind, :], axis=0)
    
    ax.plot(beta_vec, vec_adapted, 'p-', color=blue_col, linewidth=3, 
            markersize=6, markerfacecolor=blue_col, label=f'{method_name}+DA')
    ax.plot(beta_vec, vec_standard, 'p-', color=red_col, linewidth=3,
            markersize=6, markerfacecolor=red_col, label=method_name)
    
    y_max = max(np.max(vec_adapted), np.max(vec_standard)) * 1.1
    y_min = min(np.min(vec_adapted), np.min(vec_standard)) * 0.9
    ax.set_ylim([max(0, y_min), y_max])
    ax.set_xlabel(r'$\beta$ [sec]', fontsize=12)
    ax.set_ylabel('Error [deg]', fontsize=12)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True)
    ax.set_box_aspect(None)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'T60Sweep_DoAError_{method_name}.jpg'), dpi=200)
    plt.close()

def plot_polar_spectrum(theta_vec, spectrum_adapted, spectrum_standard, theta_true, 
                        theta_interference, is_interference_active, method_name, output_dir, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    theta_rad = theta_vec * np.pi / 180
    
    spectrum_adapted_db = 10 * np.log10(np.abs(spectrum_adapted) / np.max(np.abs(spectrum_adapted)))
    spectrum_standard_db = 10 * np.log10(np.abs(spectrum_standard) / np.max(np.abs(spectrum_standard)))
    
    ax.plot(theta_rad, spectrum_adapted_db, linewidth=3, label=f'{method_name}+DA')
    ax.plot(theta_rad, spectrum_standard_db, ':', linewidth=3, label=method_name)
    
    r_lim = [-15, 0]
    ax.set_ylim(r_lim)
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    theta_true_rad = theta_true * np.pi / 180
    ax.plot([theta_true_rad, theta_true_rad], r_lim, 'k-', linewidth=2, label='Desired')
    
    if is_interference_active:
        theta_inter_rad = theta_interference * np.pi / 180
        ax.plot([theta_inter_rad, theta_inter_rad], r_lim, 'k--', linewidth=2, label='Interference')
        ax.legend([f'{method_name}+DA', method_name, 'Desired', 'Interference'], fontsize=16)
    else:
        ax.legend([f'{method_name}+DA', method_name, 'Desired'], fontsize=16)
    
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=200)
    plt.close()

def plot_polar_spectrum_components(theta_vec, spectrum_e, spectrum_sigma_tr, spectrum_sigma_art,
                                   output_dir, filename):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='polar')
    
    theta_rad = theta_vec * np.pi / 180
    
    spectrum_e_db = 10 * np.log10(np.abs(spectrum_e) / np.max(np.abs(spectrum_e)))
    spectrum_sigma_tr_db = 10 * np.log10(np.abs(spectrum_sigma_tr) / np.max(np.abs(spectrum_sigma_tr)))
    spectrum_sigma_art_db = 10 * np.log10(np.abs(spectrum_sigma_art) / np.max(np.abs(spectrum_sigma_art)))
    
    ax.plot(theta_rad, spectrum_e_db, linewidth=3, label='E')
    ax.plot(theta_rad, spectrum_sigma_tr_db, '--', linewidth=3, label=r'$\Sigma_A^{-0.5}$')
    ax.plot(theta_rad, spectrum_sigma_art_db, '-.', linewidth=3, label=r'$\Sigma_S^{0.5}$')
    
    ax.set_ylim([-30, 0])
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    
    ax.legend(fontsize=16)
    ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=200)
    plt.close()

def plot_boxplots(data, labels, title, output_dir, filename, n_subplots=3):
    colors_arr = [[0, 0.4470, 0.7410], [0.8500, 0.3250, 0.0980], [0.9290, 0.6940, 0.1250]]
    
    labels = [[lbl.replace('Our\n', '').replace('+DA', '+DA') if 'Our' in lbl else lbl for lbl in labels[0]], labels[1]]
    
    fig, axes = plt.subplots(1, n_subplots, figsize=(15, 5))
    
    sp_size = data.shape[1] // n_subplots
    
    for i in range(n_subplots):
        ax = axes[i]
        start_idx = i * sp_size
        end_idx = (i + 1) * sp_size
        
        bp = ax.boxplot(data[:, start_idx:end_idx], 
                        labels=labels[0][start_idx:end_idx],
                        patch_artist=True,
                        widths=0.6,
                        whiskerprops=dict(linewidth=2),
                        capprops=dict(linewidth=2),
                        medianprops=dict(linewidth=2))
        
        for patch, color in zip(bp['boxes'], colors_arr[:sp_size]):
            patch.set_facecolor(color)
            patch.set_linewidth(2)
        
        if i == 0:
            ax.set_ylabel('[deg]', fontsize=18)
        else:
            ax.set_yticklabels([])
        
        ax.tick_params(labelsize=18)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    
    y_max = max([ax.get_ylim()[1] for ax in axes])
    y_min = min([ax.get_ylim()[0] for ax in axes])
    for ax in axes:
        ax.set_ylim([y_min, y_max])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=200)
    plt.close()


def plot_arithmetic_vs_riemannian_comparison(results_arith_file, results_riem_file, output_dir):
    """
    Compare arithmetic vs Riemannian mean methods.
    
    Creates side-by-side comparison plots for DS, MVDR, and MUSIC methods
    showing performance with and without domain adaptation for both averaging methods.
    
    Parameters
    ----------
    results_arith_file : str
        Path to pickle file with arithmetic mean results
    results_riem_file : str
        Path to pickle file with Riemannian mean results
    output_dir : str
        Directory to save comparison plots
    """
    # Load results
    with open(results_arith_file, 'rb') as f:
        results_arith = pickle.load(f)
    with open(results_riem_file, 'rb') as f:
        results_riem = pickle.load(f)
    
    beta_vec = results_arith['beta_vec']
    
    # Define colors
    blue_col = [0, 0.4470, 0.7410]
    red_col = [0.8500, 0.3250, 0.0980]
    orange_col = [0.9290, 0.6940, 0.1250]
    purple_col = [0.4940, 0.1840, 0.5560]
    
    methods = ['DS', 'MVDR', 'MUSIC']
    method_keys = [
        ('theta_est_ts_pt_err', 'theta_est_ts_err'),
        ('theta_est_mvdr_ts_pt_err', 'theta_est_mvdr_ts_err'),
        ('theta_est_music_ts_pt_err', 'theta_est_music_ts_err')
    ]
    
    for method_name, (adapted_key, standard_key) in zip(methods, method_keys):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract errors for SNR=0, SIR=0 (first indices)
        arith_adapted = np.median(results_arith[adapted_key][:, 0, 0, :], axis=0)
        arith_standard = np.median(results_arith[standard_key][:, 0, 0, :], axis=0)
        riem_adapted = np.median(results_riem[adapted_key][:, 0, 0, :], axis=0)
        riem_standard = np.median(results_riem[standard_key][:, 0, 0, :], axis=0)
        
        # Plot all four curves
        ax.plot(beta_vec, arith_adapted, 'o-', color=blue_col, linewidth=2.5, 
                markersize=8, markerfacecolor=blue_col, label=f'{method_name}+DA (Arithmetic)')
        ax.plot(beta_vec, arith_standard, 's--', color=red_col, linewidth=2.5,
                markersize=8, markerfacecolor=red_col, label=f'{method_name} (Arithmetic)')
        ax.plot(beta_vec, riem_adapted, '^-', color=orange_col, linewidth=2.5,
                markersize=8, markerfacecolor=orange_col, label=f'{method_name}+DA (Riemannian)')
        ax.plot(beta_vec, riem_standard, 'd--', color=purple_col, linewidth=2.5,
                markersize=8, markerfacecolor=purple_col, label=f'{method_name} (Riemannian)')
        
        # Calculate improvement percentages
        arith_improvement = (arith_standard[-1] - arith_adapted[-1]) / arith_standard[-1] * 100
        riem_improvement = (riem_standard[-1] - riem_adapted[-1]) / riem_standard[-1] * 100
        
        ax.set_xlabel(r'$\beta$ [sec]', fontsize=14)
        ax.set_ylabel('Median DoA Error [deg]', fontsize=14)
        ax.set_title(f'{method_name} Beamformer: Arithmetic vs Riemannian Mean\n' + 
                    f'DA Improvement: Arith={arith_improvement:.1f}%, Riem={riem_improvement:.1f}%',
                    fontsize=14)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'Comparison_ArithVsRiem_{method_name}.jpg'), dpi=200)
        plt.close()
        
        print(f"Saved: Comparison_ArithVsRiem_{method_name}.jpg")
    
    # Create summary comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (method_name, (adapted_key, standard_key)) in enumerate(zip(methods, method_keys)):
        ax = axes[idx]
        
        arith_adapted = np.median(results_arith[adapted_key][:, 0, 0, :], axis=0)
        riem_adapted = np.median(results_riem[adapted_key][:, 0, 0, :], axis=0)
        
        ax.plot(beta_vec, arith_adapted, 'o-', color=blue_col, linewidth=2.5, 
                markersize=8, label='Arithmetic Mean')
        ax.plot(beta_vec, riem_adapted, '^-', color=orange_col, linewidth=2.5,
                markersize=8, label='Riemannian Mean')
        
        improvement = (arith_adapted[-1] - riem_adapted[-1]) / arith_adapted[-1] * 100
        sign = '+' if improvement > 0 else ''
        
        ax.set_xlabel(r'$\beta$ [sec]', fontsize=12)
        ax.set_ylabel('Median DoA Error [deg]', fontsize=12)
        ax.set_title(f'{method_name}+DA\nRiem vs Arith: {sign}{improvement:.1f}%', fontsize=13)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)
    
    plt.suptitle('Domain Adaptation Methods: Arithmetic vs Riemannian Mean Comparison', 
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Comparison_ArithVsRiem_Summary.jpg'), dpi=200)
    plt.close()
    
    print(f"Saved: Comparison_ArithVsRiem_Summary.jpg")
    print("\nComparison plots generated successfully!")
