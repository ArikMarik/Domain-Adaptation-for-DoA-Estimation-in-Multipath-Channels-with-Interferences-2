import numpy as np
from tqdm import tqdm
import sys
import os
from scipy.linalg import sqrtm
sys.path.insert(0, os.path.dirname(__file__))
from signal_processing.signal_correlation import sig_corr_at_mics_acoustic_array
from algorithms.doa_estimators import (compute_steering_vector_acoustic, ml_spectrum, 
                                         mvdr_spectrum, music_spectrum, doa_from_spectrum,
                                         compute_steering_vector_array)
from algorithms.domain_adaptation import (compute_mean_covariance, compute_adaptation_matrix,
                                            compute_adaptation_matrix_inv, apply_adaptation)
from utils.geometry import angle_between_vectors, generate_random_positions, generate_grid_positions
from utils.sir_calculator import sir_calc

def run_acoustic_doa(config):
    # Check if wideband processing is enabled
    if config.get('is_wideband', False):
        return run_acoustic_doa_wideband(config)
    
    # Otherwise, run standard narrowband processing
    np.random.seed(10)
    
    c = 340
    room_dim = [5.2, 6.2, 3.5]
    fs = 12000
    sig_length = int(2.5 * fs)
    win_length = 2 * 1024
    n_samples_rir = 2048
    order_tr = -1
    
    n_mics = 9
    dist_between_mics = 0.12
    epsilon = 1e-7
    tol_deg = 0.1
    
    train_pos_x_start = 0.5
    train_pos_y_start = 3.5
    
    bin_ind = int(np.round(1/(2*dist_between_mics/c)/fs*win_length + 0.5)) - 1
    freq_bin_inter = fs/win_length * (bin_ind + 0.5)
    wavelength = c / freq_bin_inter
    
    snr_vec_db = np.arange(config['snr_min_db'], config['snr_max_db'] + 10, 10)
    sir_vec_db = np.arange(config['sir_min_db'], config['sir_max_db'] + 10, 10)
    
    array = np.arange(n_mics)
    
    first_mic_pos = np.array([2, 0.5, 1.5])
    dir_vec = np.array([1, 0, 0])
    mics_pos_mat = np.array([first_mic_pos + m*dist_between_mics*dir_vec for m in range(n_mics)])
    array_vec = mics_pos_mat[1, :] - mics_pos_mat[0, :]
    mid_array_pos = np.mean(mics_pos_mat, axis=0)
    
    n_test_points = config['k_test_points']
    n_train_points = config['k_train_points']
    n_art_points = config['k_art_points']
    n_interference_pos = config['k_interference_pos']
    
    lx = 4
    ly = 2
    lx_inter = 1
    ly_inter = 2
    
    theta_est_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_mvdr_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_mvdr_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_music_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_music_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    
    for int_pos in tqdm(range(n_interference_pos), desc=f"Processing ({n_test_points} test pts, {len(config['beta_vec'])} betas)"):
        if config['train_is_test']:
            train_pos_mat = generate_random_positions(n_train_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        else:
            train_pos_mat = generate_grid_positions([lx, ly], config['dist_between_train'], 
                                                     [train_pos_x_start, train_pos_y_start])
        
        art_pos_mat = generate_random_positions(n_art_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        test_pos_mat = generate_random_positions(n_test_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        
        if config['is_interference_fixed']:
            interf_pos_train = generate_random_positions(config['k_inter_pos'], [lx, ly], 
                                                          [train_pos_x_start, train_pos_y_start])
            interf_pos_test = interf_pos_train
        else:
            inter_start_point = np.array([0, 1])
            interf_pos_train = generate_random_positions(n_train_points, [lx_inter, ly_inter], inter_start_point)
            interf_pos_test = generate_random_positions(n_test_points, [lx_inter, ly_inter], inter_start_point)
        
        for sir_ind in range(len(sir_vec_db)):
            sir_lin_train = 1 / 10**(sir_vec_db[sir_ind]/20)
            sir_lin_test = sir_lin_train
            
            for snr_ind in range(len(snr_vec_db)):
                snr_train = snr_vec_db[snr_ind]
                snr_test = snr_train
                
                sig_train = np.random.randn(sig_length, n_train_points)
                sig_inter_train = np.random.randn(sig_length, n_train_points)
                sig_test = np.random.randn(sig_length, n_test_points)
                sig_inter_test = np.random.randn(sig_length, n_test_points)
                
                for b in tqdm(range(len(config['beta_vec'])), desc=f"  Beta values (SNR={snr_ind+1}/{len(snr_vec_db)})", leave=False):
                    beta = config['beta_vec'][b]
                    
                    corr_tr_vec = np.zeros((n_mics, n_mics, n_train_points), dtype=complex)
                    corr_tr_vec_inv = np.zeros((n_mics, n_mics, n_train_points), dtype=complex)
                    
                    for source_ind in range(n_train_points):
                        ztr, corr_tr = sig_corr_at_mics_acoustic_array(
                            train_pos_mat[source_ind, :], mics_pos_mat, n_mics, beta, fs, bin_ind,
                            room_dim, order_tr, n_samples_rir, win_length, sig_train[:, source_ind], snr_train
                        )
                        
                        if config['is_interference_active']:
                            inter_idx = np.random.randint(interf_pos_train.shape[0])
                            inter_sig_at_array, _ = sig_corr_at_mics_acoustic_array(
                                interf_pos_train[inter_idx, :], mics_pos_mat, n_mics, beta, fs, bin_ind,
                                room_dim, order_tr, n_samples_rir, win_length, sig_inter_train[:, source_ind], snr_train
                            )
                        else:
                            inter_sig_at_array = 0
                        
                        sig_tmp_train = (config['is_only_interf_active_during_train'] - 1) * ztr + sir_lin_train * inter_sig_at_array
                        corr_tr_vec[:, :, source_ind] = (1 / sig_tmp_train.shape[1]) * (sig_tmp_train @ sig_tmp_train.conj().T)
                        corr_tr_mat_reg = corr_tr_vec[:, :, source_ind] + epsilon * np.eye(n_mics)
                        corr_tr_vec_inv[:, :, source_ind] = np.linalg.inv(corr_tr_mat_reg)
                    
                    corr_ts_vec = np.zeros((n_mics, n_mics, n_test_points), dtype=complex)
                    
                    for source_ind in range(n_test_points):
                        zts, corr_ts = sig_corr_at_mics_acoustic_array(
                            test_pos_mat[source_ind, :], mics_pos_mat, n_mics, beta, fs, bin_ind,
                            room_dim, order_tr, n_samples_rir, win_length, sig_test[:, source_ind], snr_test
                        )
                        
                        if config['is_interference_active']:
                            inter_idx = np.random.randint(interf_pos_test.shape[0])
                            inter_sig_at_array, _ = sig_corr_at_mics_acoustic_array(
                                interf_pos_test[inter_idx, :], mics_pos_mat, n_mics, beta, fs, bin_ind,
                                room_dim, order_tr, n_samples_rir, win_length, sig_inter_test[:, source_ind], snr_test
                            )
                            
                            if config['is_two_interference_sources_active']:
                                inter_idx2 = np.random.randint(interf_pos_test.shape[0])
                                inter_sig_at_array2, _ = sig_corr_at_mics_acoustic_array(
                                    interf_pos_test[inter_idx2, :], mics_pos_mat, n_mics, beta, fs, bin_ind,
                                    room_dim, order_tr, n_samples_rir, win_length, sig_inter_test[:, source_ind], snr_test
                                )
                                inter_sig_at_array = inter_sig_at_array + inter_sig_at_array2
                        else:
                            inter_sig_at_array = 0
                        
                        sig_tmp_test = zts + sir_lin_test * inter_sig_at_array
                        corr_ts_vec[:, :, source_ind] = (1 / sig_tmp_test.shape[1]) * (sig_tmp_test @ sig_tmp_test.conj().T)
                    
                    cor_art_vec = np.zeros((n_mics, n_mics, n_art_points), dtype=complex)
                    cor_art_vec_inv = np.zeros((n_mics, n_mics, n_art_points), dtype=complex)
                    
                    for source_ind in range(n_art_points):
                        steer_vec = compute_steering_vector_acoustic(art_pos_mat[source_ind, :], mics_pos_mat, wavelength)
                        cor_art_vec[:, :, source_ind] = np.outer(steer_vec, steer_vec.conj()) + epsilon * np.eye(n_mics)
                        cor_art_mat_reg = cor_art_vec[:, :, source_ind] + epsilon * np.eye(n_mics)
                        cor_art_vec_inv[:, :, source_ind] = np.linalg.inv(cor_art_mat_reg)
                    
                    # Get covariance averaging method from config (default: arithmetic)
                    cov_method = config.get('covariance_averaging_method', 'arithmetic')
                    
                    corr_tr_mean = compute_mean_covariance(corr_tr_vec, method=cov_method)
                    cor_art_mean = compute_mean_covariance(cor_art_vec, method=cov_method)
                    corr_tr_mean_inv = compute_mean_covariance(corr_tr_vec_inv, method=cov_method)
                    cor_art_mean_inv = compute_mean_covariance(cor_art_vec_inv, method=cov_method)
                    
                    E_pt = compute_adaptation_matrix(cor_art_mean, corr_tr_mean, epsilon)
                    E_pt_inv = compute_adaptation_matrix_inv(cor_art_mean_inv, corr_tr_mean_inv, epsilon)
                    
                    theta_vec = np.linspace(0, 180, 1001)
                    
                    theta_test_deg = np.zeros(n_test_points)
                    theta_interference_deg = np.zeros(n_test_points)
                    theta_est_vec_ts = np.zeros(n_test_points)
                    theta_est_vec_ts_pt = np.zeros(n_test_points)
                    theta_est_mvdr_vec_ts = np.zeros(n_test_points)
                    theta_est_mvdr_vec_ts_pt = np.zeros(n_test_points)
                    theta_est_music_vec_ts = np.zeros(n_test_points)
                    theta_est_music_vec_ts_pt = np.zeros(n_test_points)
                    
                    for source_ind in range(n_test_points):
                        gamma1_ts = corr_ts_vec[:, :, source_ind]
                        gamma1_ts_adapted = apply_adaptation(gamma1_ts, E_pt)
                        
                        ml_spectrum_ts = ml_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics, wavelength)
                        ml_spectrum_ts_pt = ml_spectrum(gamma1_ts_adapted, theta_vec, n_mics, dist_between_mics, wavelength)
                        
                        mvdr_spectrum_ts = mvdr_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics, wavelength)
                        gamma1_ts_reg = gamma1_ts + epsilon * np.eye(n_mics)
                        gamma1_ts_adapted_mvdr = E_pt_inv @ np.linalg.inv(gamma1_ts_reg) @ E_pt_inv.conj().T
                        mvdr_spectrum_ts_pt = np.zeros(len(theta_vec), dtype=complex)
                        for i, theta in enumerate(theta_vec):
                            steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
                            mvdr_spectrum_ts_pt[i] = 1.0 / (steering_vec.conj().T @ gamma1_ts_adapted_mvdr @ steering_vec)
                        
                        music_spectrum_ts = music_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics, 
                                                            wavelength, config['sig_dim_4_music'])
                        music_spectrum_ts_pt = music_spectrum(gamma1_ts_adapted, theta_vec, n_mics, dist_between_mics,
                                                               wavelength, config['sig_dim_4_music_pt'])
                        
                        theta_test_deg[source_ind] = angle_between_vectors(test_pos_mat[source_ind, :] - mid_array_pos, array_vec)
                        
                        if interf_pos_test.shape[0] > 1:
                            theta_interference_deg[source_ind] = angle_between_vectors(
                                interf_pos_test[source_ind, :] - mid_array_pos, array_vec)
                        else:
                            theta_interference_deg[source_ind] = angle_between_vectors(
                                interf_pos_test[0, :] - mid_array_pos, array_vec)
                        
                        theta_est_vec_ts[source_ind] = doa_from_spectrum(theta_vec, ml_spectrum_ts)
                        theta_est_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, ml_spectrum_ts_pt)
                        theta_est_mvdr_vec_ts[source_ind] = doa_from_spectrum(theta_vec, mvdr_spectrum_ts)
                        theta_est_mvdr_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, mvdr_spectrum_ts_pt)
                        theta_est_music_vec_ts[source_ind] = doa_from_spectrum(theta_vec, music_spectrum_ts)
                        theta_est_music_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, music_spectrum_ts_pt)
                        
                        should_plot_spectra = (
                            (config.get('plot_t60_sweep_clean', False) and abs(beta - 0.4) < 0.01 and source_ind == 13) or
                            (config.get('plot_single_interference_fixed', False) and sir_ind == 0 and snr_ind == 0 and b == 0 and source_ind == 50) or
                            (config.get('plot_dual_interference_moving', False) and sir_ind == 0 and snr_ind == 0 and b == 0 and source_ind == 50)
                        )
                        
                        if should_plot_spectra:
                            import os
                            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                            os.makedirs(results_dir, exist_ok=True)
                            
                            from visualization.plotting import plot_polar_spectrum, plot_polar_spectrum_components
                            
                            # Determine figure prefix based on scenario
                            scenario_suffix = ''
                            if config.get('covariance_averaging_method', 'arithmetic') == 'riemannian':
                                scenario_suffix = '_Riemannian'
                            
                            if config.get('plot_t60_sweep_clean', False):
                                fig_prefix = f'T60Sweep_Spectra{scenario_suffix}'
                            elif config.get('plot_single_interference_fixed', False):
                                fig_prefix = f'SingleInterf_Spectra{scenario_suffix}'
                            elif config.get('plot_dual_interference_moving', False):
                                fig_prefix = f'DualInterf_Spectra{scenario_suffix}'
                            else:
                                fig_prefix = f'Spectra{scenario_suffix}'
                            
                            plot_polar_spectrum(
                                theta_vec, ml_spectrum_ts_pt, ml_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'DS', results_dir, f'{fig_prefix}_DS'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, mvdr_spectrum_ts_pt, mvdr_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'MVDR', results_dir, f'{fig_prefix}_MVDR'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, music_spectrum_ts_pt, music_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'MUSIC', results_dir, f'{fig_prefix}_MUSIC'
                            )
                            
                            if config.get('plot_t60_sweep_clean', False):
                                sigma_tr_half = sqrtm(corr_tr_mean)
                                sigma_art_half = sqrtm(cor_art_mean)
                                
                                spectrum_e = np.zeros(len(theta_vec), dtype=complex)
                                spectrum_sigma_tr = np.zeros(len(theta_vec), dtype=complex)
                                spectrum_sigma_art = np.zeros(len(theta_vec), dtype=complex)
                                
                                for i, theta in enumerate(theta_vec):
                                    steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
                                    spectrum_e[i] = np.abs(steering_vec.conj().T @ E_pt @ steering_vec)**2
                                    spectrum_sigma_tr[i] = np.abs(steering_vec.conj().T @ sigma_tr_half @ steering_vec)**2
                                    spectrum_sigma_art[i] = np.abs(steering_vec.conj().T @ sigma_art_half @ steering_vec)**2
                                
                                plot_polar_spectrum_components(
                                    theta_vec, spectrum_e, spectrum_sigma_tr, spectrum_sigma_art,
                                    results_dir, f'T60Sweep_Components_Spectrum{scenario_suffix}'
                                )
                    
                    inds = slice(int_pos*n_test_points, (int_pos+1)*n_test_points)
                    theta_est_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_vec_ts - theta_test_deg)
                    theta_est_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_vec_ts_pt - theta_test_deg)
                    theta_est_mvdr_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_mvdr_vec_ts - theta_test_deg)
                    theta_est_mvdr_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_mvdr_vec_ts_pt - theta_test_deg)
                    theta_est_music_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_music_vec_ts - theta_test_deg)
                    theta_est_music_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_music_vec_ts_pt - theta_test_deg)
    
    results = {
        'theta_est_ts_err': theta_est_ts_err_snr,
        'theta_est_ts_pt_err': theta_est_ts_pt_err_snr,
        'theta_est_mvdr_ts_err': theta_est_mvdr_ts_err_snr,
        'theta_est_mvdr_ts_pt_err': theta_est_mvdr_ts_pt_err_snr,
        'theta_est_music_ts_err': theta_est_music_ts_err_snr,
        'theta_est_music_ts_pt_err': theta_est_music_ts_pt_err_snr,
        'config': config,
        'positions': {
            'train': train_pos_mat if 'train_pos_mat' in locals() else None,
            'test': test_pos_mat if 'test_pos_mat' in locals() else None,
            'interference': interf_pos_test if 'interf_pos_test' in locals() else None,
            'mics': mics_pos_mat,
        },
        'room_dim': room_dim,
        'beta_vec': config['beta_vec'],
    }
    
    return results


def run_acoustic_doa_wideband(config):
    """
    Run wideband acoustic DoA estimation with domain adaptation.
    
    Processes multiple frequency bins and combines spectra coherently or incoherently.
    """
    from signal_processing.wideband import (select_frequency_bins, compute_frequency_dependent_wavelength,
                                             check_spatial_aliasing)
    from acoustic_doa_wideband import (process_frequency_bin_wideband, 
                                       compute_wideband_spectra_per_test_point)
    
    np.random.seed(10)
    
    wb_params = config['wideband_params']
    
    c = 340
    room_dim = [5.2, 6.2, 3.5]
    fs = 12000
    sig_length = int(2.5 * fs)
    win_length = 2 * 1024
    n_samples_rir = 2048
    order_tr = -1
    
    n_mics = 9
    dist_between_mics = 0.12
    epsilon = 1e-7
    tol_deg = 0.1
    
    if not check_spatial_aliasing(wb_params['f_max'], dist_between_mics, c):
        print(f"WARNING: f_max={wb_params['f_max']} Hz exceeds spatial aliasing limit "
              f"({c/(2*dist_between_mics):.2f} Hz) for mic spacing {dist_between_mics} m")
    
    train_pos_x_start = 0.5
    train_pos_y_start = 3.5
    
    snr_vec_db = np.arange(config['snr_min_db'], config['snr_max_db'] + 10, 10)
    sir_vec_db = np.arange(config['sir_min_db'], config['sir_max_db'] + 10, 10)
    
    array = np.arange(n_mics)
    
    first_mic_pos = np.array([2, 0.5, 1.5])
    dir_vec = np.array([1, 0, 0])
    mics_pos_mat = np.array([first_mic_pos + m*dist_between_mics*dir_vec for m in range(n_mics)])
    array_vec = mics_pos_mat[1, :] - mics_pos_mat[0, :]
    mid_array_pos = np.mean(mics_pos_mat, axis=0)
    
    n_test_points = config['k_test_points']
    n_train_points = config['k_train_points']
    n_art_points = config['k_art_points']
    n_interference_pos = config['k_interference_pos']
    
    lx = 4
    ly = 2
    lx_inter = 1
    ly_inter = 2
    
    theta_est_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_mvdr_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_mvdr_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_music_ts_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    theta_est_music_ts_pt_err_snr = np.zeros((n_interference_pos*n_test_points, len(snr_vec_db), len(sir_vec_db), len(config['beta_vec'])))
    
    print(f"\nWideband DoA Estimation:")
    print(f"  Frequency range: {wb_params['f_min']}-{wb_params['f_max']} Hz")
    print(f"  Number of frequency bins: {wb_params['n_freq_bins']}")
    print(f"  Frequency selection: {wb_params['freq_selection']}")
    print(f"  Combination method: {wb_params['combination_method']}\n")
    
    for int_pos in tqdm(range(n_interference_pos), desc=f"Processing ({n_test_points} test pts, {len(config['beta_vec'])} betas, {wb_params['n_freq_bins']} freqs)"):
        if config['train_is_test']:
            train_pos_mat = generate_random_positions(n_train_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        else:
            train_pos_mat = generate_grid_positions([lx, ly], config['dist_between_train'], 
                                                     [train_pos_x_start, train_pos_y_start])
        
        art_pos_mat = generate_random_positions(n_art_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        test_pos_mat = generate_random_positions(n_test_points, [lx, ly], [train_pos_x_start, train_pos_y_start])
        
        if config['is_interference_fixed']:
            interf_pos_train = generate_random_positions(config['k_inter_pos'], [lx, ly], 
                                                          [train_pos_x_start, train_pos_y_start])
            interf_pos_test = interf_pos_train
        else:
            inter_start_point = np.array([0, 1])
            interf_pos_train = generate_random_positions(n_train_points, [lx_inter, ly_inter], inter_start_point)
            interf_pos_test = generate_random_positions(n_test_points, [lx_inter, ly_inter], inter_start_point)
        
        for sir_ind in range(len(sir_vec_db)):
            sir_lin_train = 1 / 10**(sir_vec_db[sir_ind]/20)
            sir_lin_test = sir_lin_train
            
            for snr_ind in range(len(snr_vec_db)):
                snr_train = snr_vec_db[snr_ind]
                snr_test = snr_train
                
                sig_train = np.random.randn(sig_length, n_train_points)
                sig_inter_train = np.random.randn(sig_length, n_train_points)
                sig_test = np.random.randn(sig_length, n_test_points)
                sig_inter_test = np.random.randn(sig_length, n_test_points)
                
                freq_bins, freq_hz, freq_weights = select_frequency_bins(
                    sig_train[:, 0], fs, win_length,
                    wb_params['n_freq_bins'],
                    wb_params['f_min'],
                    wb_params['f_max'],
                    wb_params['freq_selection']
                )
                
                wavelengths = compute_frequency_dependent_wavelength(freq_hz, c)
                
                for b in tqdm(range(len(config['beta_vec'])), desc=f"  Beta values (SNR={snr_ind+1}/{len(snr_vec_db)})", leave=False):
                    beta = config['beta_vec'][b]
                    
                    corr_ts_vec_per_freq = []
                    E_pt_per_freq = []
                    E_pt_inv_per_freq = []
                    
                    for k, (freq_bin, freq, wavelength) in enumerate(zip(freq_bins, freq_hz, wavelengths)):
                        if k == 0 or k == len(freq_bins) - 1:
                            print(f"    Processing freq bin {k+1}/{len(freq_bins)}: {freq:.1f} Hz (beta={beta:.1f})")
                        
                        corr_ts_vec, E_pt, E_pt_inv = process_frequency_bin_wideband(
                            freq_bin, freq, wavelength, config,
                            train_pos_mat, art_pos_mat, test_pos_mat,
                            interf_pos_train, interf_pos_test,
                            sig_train, sig_inter_train, sig_test, sig_inter_test,
                            mics_pos_mat, n_mics, beta, room_dim, order_tr,
                            n_samples_rir, win_length, fs, snr_train, snr_test,
                            sir_lin_train, sir_lin_test, epsilon
                        )
                        
                        corr_ts_vec_per_freq.append(corr_ts_vec)
                        E_pt_per_freq.append(E_pt)
                        E_pt_inv_per_freq.append(E_pt_inv)
                    
                    theta_vec = np.linspace(0, 180, 1001)
                    
                    theta_test_deg = np.zeros(n_test_points)
                    theta_est_vec_ts = np.zeros(n_test_points)
                    theta_est_vec_ts_pt = np.zeros(n_test_points)
                    theta_est_mvdr_vec_ts = np.zeros(n_test_points)
                    theta_est_mvdr_vec_ts_pt = np.zeros(n_test_points)
                    theta_est_music_vec_ts = np.zeros(n_test_points)
                    theta_est_music_vec_ts_pt = np.zeros(n_test_points)
                    
                    for source_ind in range(n_test_points):
                        (ml_spectrum_adapted, ml_spectrum_standard,
                         mvdr_spectrum_adapted, mvdr_spectrum_standard,
                         music_spectrum_adapted, music_spectrum_standard) = compute_wideband_spectra_per_test_point(
                            source_ind, corr_ts_vec_per_freq, E_pt_per_freq, E_pt_inv_per_freq,
                            theta_vec, n_mics, dist_between_mics, wavelengths,
                            freq_weights, wb_params['combination_method'], config, epsilon
                        )
                        
                        theta_test_deg[source_ind] = angle_between_vectors(test_pos_mat[source_ind, :] - mid_array_pos, array_vec)
                        
                        theta_est_vec_ts[source_ind] = doa_from_spectrum(theta_vec, ml_spectrum_standard)
                        theta_est_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, ml_spectrum_adapted)
                        theta_est_mvdr_vec_ts[source_ind] = doa_from_spectrum(theta_vec, mvdr_spectrum_standard)
                        theta_est_mvdr_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, mvdr_spectrum_adapted)
                        theta_est_music_vec_ts[source_ind] = doa_from_spectrum(theta_vec, music_spectrum_standard)
                        theta_est_music_vec_ts_pt[source_ind] = doa_from_spectrum(theta_vec, music_spectrum_adapted)
                        
                        # Plot wideband spectra for beta=0.4, source_ind=13
                        should_plot_spectra = (abs(beta - 0.4) < 0.01 and source_ind == 13)
                        
                        if should_plot_spectra:
                            import os
                            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                            os.makedirs(results_dir, exist_ok=True)
                            
                            from visualization.plotting import plot_polar_spectrum
                            
                            # Wideband suffix
                            wb_suffix = f'Wideband_{wb_params["combination_method"].capitalize()}'
                            
                            # Dummy interference angle (not active for t60-sweep_clean)
                            theta_interference = 0.0
                            
                            plot_polar_spectrum(
                                theta_vec, ml_spectrum_adapted, ml_spectrum_standard,
                                theta_test_deg[source_ind], theta_interference,
                                False,  # is_interference_active
                                'DS', results_dir, f'T60Sweep_{wb_suffix}_DS'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, mvdr_spectrum_adapted, mvdr_spectrum_standard,
                                theta_test_deg[source_ind], theta_interference,
                                False,
                                'MVDR', results_dir, f'T60Sweep_{wb_suffix}_MVDR'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, music_spectrum_adapted, music_spectrum_standard,
                                theta_test_deg[source_ind], theta_interference,
                                False,
                                'MUSIC', results_dir, f'T60Sweep_{wb_suffix}_MUSIC'
                            )
                    
                    inds = slice(int_pos*n_test_points, (int_pos+1)*n_test_points)
                    theta_est_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_vec_ts - theta_test_deg)
                    theta_est_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_vec_ts_pt - theta_test_deg)
                    theta_est_mvdr_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_mvdr_vec_ts - theta_test_deg)
                    theta_est_mvdr_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_mvdr_vec_ts_pt - theta_test_deg)
                    theta_est_music_ts_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_music_vec_ts - theta_test_deg)
                    theta_est_music_ts_pt_err_snr[inds, snr_ind, sir_ind, b] = np.abs(theta_est_music_vec_ts_pt - theta_test_deg)
    
    results = {
        'theta_est_ts_err': theta_est_ts_err_snr,
        'theta_est_ts_pt_err': theta_est_ts_pt_err_snr,
        'theta_est_mvdr_ts_err': theta_est_mvdr_ts_err_snr,
        'theta_est_mvdr_ts_pt_err': theta_est_mvdr_ts_pt_err_snr,
        'theta_est_music_ts_err': theta_est_music_ts_err_snr,
        'theta_est_music_ts_pt_err': theta_est_music_ts_pt_err_snr,
        'config': config,
        'positions': {
            'train': train_pos_mat if 'train_pos_mat' in locals() else None,
            'test': test_pos_mat if 'test_pos_mat' in locals() else None,
            'interference': interf_pos_test if 'interf_pos_test' in locals() else None,
            'mics': mics_pos_mat,
        },
        'room_dim': room_dim,
        'beta_vec': config['beta_vec'],
        'wideband_params': wb_params,
        'freq_bins': freq_bins,
        'freq_hz': freq_hz,
    }
    
    return results
