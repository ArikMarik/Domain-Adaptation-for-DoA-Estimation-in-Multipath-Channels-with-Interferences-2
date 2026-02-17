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
                            (config.get('plot_figs_2_3_4', False) and abs(beta - 0.4) < 0.01 and source_ind == 13) or
                            (config.get('plot_figs_5', False) and sir_ind == 0 and snr_ind == 0 and b == 0 and source_ind == 50) or
                            (config.get('plot_figs_6', False) and sir_ind == 0 and snr_ind == 0 and b == 0 and source_ind == 50)
                        )
                        
                        if should_plot_spectra:
                            import os
                            results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
                            os.makedirs(results_dir, exist_ok=True)
                            
                            from visualization.plotting import plot_polar_spectrum, plot_polar_spectrum_components
                            
                            fig_prefix = 'Fig3' if config.get('plot_figs_2_3_4', False) else ('Fig5' if config.get('plot_figs_5', False) else 'Fig6')
                            
                            plot_polar_spectrum(
                                theta_vec, ml_spectrum_ts_pt, ml_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'DS', results_dir, f'{fig_prefix}_DS_Spectrum'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, mvdr_spectrum_ts_pt, mvdr_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'MVDR', results_dir, f'{fig_prefix}_MVDR_Spectrum'
                            )
                            
                            plot_polar_spectrum(
                                theta_vec, music_spectrum_ts_pt, music_spectrum_ts,
                                theta_test_deg[source_ind], theta_interference_deg[source_ind],
                                config.get('is_interference_active', False),
                                'MUSIC', results_dir, f'{fig_prefix}_MUSIC_Spectrum'
                            )
                            
                            if config.get('plot_figs_2_3_4', False):
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
                                    results_dir, 'Fig3_Components_Spectrum'
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
