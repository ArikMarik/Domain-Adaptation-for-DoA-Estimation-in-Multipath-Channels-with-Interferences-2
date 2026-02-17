import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from signal_processing.signal_correlation import sig_corr_at_mics_acoustic_array
from signal_processing.wideband import (select_frequency_bins, compute_frequency_dependent_wavelength,
                                        compute_wideband_spectrum, check_spatial_aliasing)
from algorithms.doa_estimators import (compute_steering_vector_acoustic, ml_spectrum, 
                                         mvdr_spectrum, music_spectrum, doa_from_spectrum,
                                         compute_steering_vector_array)
from algorithms.domain_adaptation import (compute_mean_covariance, compute_adaptation_matrix,
                                            compute_adaptation_matrix_inv, apply_adaptation)
from utils.geometry import angle_between_vectors

def process_frequency_bin_wideband(freq_bin_idx, freq_hz, wavelength, config, 
                                   train_pos_mat, art_pos_mat, test_pos_mat,
                                   interf_pos_train, interf_pos_test,
                                   sig_train, sig_inter_train, sig_test, sig_inter_test,
                                   mics_pos_mat, n_mics, beta, room_dim, order_tr,
                                   n_samples_rir, win_length, fs, snr_train, snr_test,
                                   sir_lin_train, sir_lin_test, epsilon):
    """
    Process a single frequency bin for wideband DoA estimation.
    
    Returns per-frequency spectra (complex for coherent combination).
    """
    n_train_points = train_pos_mat.shape[0]
    n_art_points = art_pos_mat.shape[0]
    n_test_points = test_pos_mat.shape[0]
    
    corr_tr_vec = np.zeros((n_mics, n_mics, n_train_points), dtype=complex)
    corr_tr_vec_inv = np.zeros((n_mics, n_mics, n_train_points), dtype=complex)
    
    for source_ind in range(n_train_points):
        ztr, corr_tr = sig_corr_at_mics_acoustic_array(
            train_pos_mat[source_ind, :], mics_pos_mat, n_mics, beta, fs, freq_bin_idx,
            room_dim, order_tr, n_samples_rir, win_length, sig_train[:, source_ind], snr_train
        )
        
        if config['is_interference_active']:
            inter_idx = np.random.randint(interf_pos_train.shape[0])
            inter_sig_at_array, _ = sig_corr_at_mics_acoustic_array(
                interf_pos_train[inter_idx, :], mics_pos_mat, n_mics, beta, fs, freq_bin_idx,
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
            test_pos_mat[source_ind, :], mics_pos_mat, n_mics, beta, fs, freq_bin_idx,
            room_dim, order_tr, n_samples_rir, win_length, sig_test[:, source_ind], snr_test
        )
        
        if config['is_interference_active']:
            inter_idx = np.random.randint(interf_pos_test.shape[0])
            inter_sig_at_array, _ = sig_corr_at_mics_acoustic_array(
                interf_pos_test[inter_idx, :], mics_pos_mat, n_mics, beta, fs, freq_bin_idx,
                room_dim, order_tr, n_samples_rir, win_length, sig_inter_test[:, source_ind], snr_test
            )
            
            if config['is_two_interference_sources_active']:
                inter_idx2 = np.random.randint(interf_pos_test.shape[0])
                inter_sig_at_array2, _ = sig_corr_at_mics_acoustic_array(
                    interf_pos_test[inter_idx2, :], mics_pos_mat, n_mics, beta, fs, freq_bin_idx,
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
    
    cov_method = config.get('covariance_averaging_method', 'arithmetic')
    
    corr_tr_mean = compute_mean_covariance(corr_tr_vec, method=cov_method)
    cor_art_mean = compute_mean_covariance(cor_art_vec, method=cov_method)
    corr_tr_mean_inv = compute_mean_covariance(corr_tr_vec_inv, method=cov_method)
    cor_art_mean_inv = compute_mean_covariance(cor_art_vec_inv, method=cov_method)
    
    E_pt = compute_adaptation_matrix(cor_art_mean, corr_tr_mean, epsilon)
    E_pt_inv = compute_adaptation_matrix_inv(cor_art_mean_inv, corr_tr_mean_inv, epsilon)
    
    return corr_ts_vec, E_pt, E_pt_inv


def compute_wideband_spectra_per_test_point(source_ind, corr_ts_vec_per_freq, E_pt_per_freq, E_pt_inv_per_freq,
                                             theta_vec, n_mics, dist_between_mics, wavelengths, 
                                             freq_weights, combination_method, config, epsilon):
    """
    Compute wideband spectra for a single test point by combining per-frequency spectra.
    
    Returns
    -------
    ml_spectrum_combined_adapted : np.ndarray
        Combined ML/DS spectrum (adapted)
    ml_spectrum_combined_standard : np.ndarray
        Combined ML/DS spectrum (standard)
    mvdr_spectrum_combined_adapted : np.ndarray
        Combined MVDR spectrum (adapted)
    mvdr_spectrum_combined_standard : np.ndarray
        Combined MVDR spectrum (standard)
    music_spectrum_combined_adapted : np.ndarray
        Combined MUSIC spectrum (adapted)
    music_spectrum_combined_standard : np.ndarray
        Combined MUSIC spectrum (standard)
    """
    n_freq = len(wavelengths)
    n_angles = len(theta_vec)
    
    ml_spectra_adapted = np.zeros((n_freq, n_angles), dtype=complex)
    ml_spectra_standard = np.zeros((n_freq, n_angles), dtype=complex)
    mvdr_spectra_adapted = np.zeros((n_freq, n_angles), dtype=complex)
    mvdr_spectra_standard = np.zeros((n_freq, n_angles), dtype=complex)
    music_spectra_adapted = np.zeros((n_freq, n_angles), dtype=complex)
    music_spectra_standard = np.zeros((n_freq, n_angles), dtype=complex)
    
    for k, wavelength in enumerate(wavelengths):
        gamma1_ts = corr_ts_vec_per_freq[k][:, :, source_ind]
        gamma1_ts_adapted = apply_adaptation(gamma1_ts, E_pt_per_freq[k])
        
        ml_spectra_standard[k, :] = ml_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics, wavelength)
        ml_spectra_adapted[k, :] = ml_spectrum(gamma1_ts_adapted, theta_vec, n_mics, dist_between_mics, wavelength)
        
        mvdr_spectra_standard[k, :] = mvdr_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics, wavelength)
        
        gamma1_ts_reg = gamma1_ts + epsilon * np.eye(n_mics)
        gamma1_ts_adapted_mvdr = E_pt_inv_per_freq[k] @ np.linalg.inv(gamma1_ts_reg) @ E_pt_inv_per_freq[k].conj().T
        mvdr_spectrum_adapted_freq = np.zeros(len(theta_vec), dtype=complex)
        for i, theta in enumerate(theta_vec):
            steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
            mvdr_spectrum_adapted_freq[i] = 1.0 / (steering_vec.conj().T @ gamma1_ts_adapted_mvdr @ steering_vec)
        mvdr_spectra_adapted[k, :] = mvdr_spectrum_adapted_freq
        
        music_spectra_standard[k, :] = music_spectrum(gamma1_ts, theta_vec, n_mics, dist_between_mics,
                                                       wavelength, config['sig_dim_4_music'])
        music_spectra_adapted[k, :] = music_spectrum(gamma1_ts_adapted, theta_vec, n_mics, dist_between_mics,
                                                      wavelength, config['sig_dim_4_music_pt'])
    
    ml_spectrum_combined_standard = compute_wideband_spectrum(ml_spectra_standard, None, freq_weights, combination_method)
    ml_spectrum_combined_adapted = compute_wideband_spectrum(ml_spectra_adapted, None, freq_weights, combination_method)
    mvdr_spectrum_combined_standard = compute_wideband_spectrum(mvdr_spectra_standard, None, freq_weights, combination_method)
    mvdr_spectrum_combined_adapted = compute_wideband_spectrum(mvdr_spectra_adapted, None, freq_weights, combination_method)
    music_spectrum_combined_standard = compute_wideband_spectrum(music_spectra_standard, None, freq_weights, combination_method)
    music_spectrum_combined_adapted = compute_wideband_spectrum(music_spectra_adapted, None, freq_weights, combination_method)
    
    return (ml_spectrum_combined_adapted, ml_spectrum_combined_standard,
            mvdr_spectrum_combined_adapted, mvdr_spectrum_combined_standard,
            music_spectrum_combined_adapted, music_spectrum_combined_standard)
