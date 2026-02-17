import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.math_utils import sorted_evd

def compute_steering_vector_array(theta_deg, n_mics, dist_between_mics, wavelength):
    theta_rad = theta_deg * np.pi / 180
    array = np.arange(n_mics)
    steering_vec = np.exp(2j * np.pi * array * (dist_between_mics / wavelength) * np.cos(theta_rad))
    return steering_vec

def compute_steering_vector_acoustic(source_pos, mics_pos_mat, wavelength):
    n_mics = mics_pos_mat.shape[0]
    dist_source_mic = np.zeros(n_mics)
    
    for m in range(n_mics):
        dist_source_mic[m] = np.linalg.norm(source_pos - mics_pos_mat[m, :])
    
    fs_atten = 1.0 / dist_source_mic
    steering_vec = fs_atten * np.exp(-2j * np.pi * (1/wavelength) * dist_source_mic)
    
    return steering_vec

def doa_from_spectrum(theta_vec, spectrum):
    idx = np.argmax(np.abs(spectrum))
    theta_est = theta_vec[idx]
    return theta_est

def ml_spectrum(cov_matrix, theta_vec, n_mics, dist_between_mics, wavelength):
    spectrum = np.zeros(len(theta_vec), dtype=complex)
    
    for i, theta in enumerate(theta_vec):
        steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
        spectrum[i] = steering_vec.conj().T @ cov_matrix @ steering_vec
    
    return spectrum

def mvdr_spectrum(cov_matrix, theta_vec, n_mics, dist_between_mics, wavelength, epsilon=1e-7):
    spectrum = np.zeros(len(theta_vec), dtype=complex)
    cov_matrix_reg = cov_matrix + epsilon * np.eye(n_mics)
    cov_inv = np.linalg.inv(cov_matrix_reg)
    
    for i, theta in enumerate(theta_vec):
        steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
        spectrum[i] = 1.0 / (steering_vec.conj().T @ cov_inv @ steering_vec)
    
    return spectrum

def music_spectrum(cov_matrix, theta_vec, n_mics, dist_between_mics, wavelength, n_sources):
    spectrum = np.zeros(len(theta_vec), dtype=complex)
    
    eigvec_mat, _ = sorted_evd(cov_matrix)
    u_noise = eigvec_mat[:, n_sources:]
    
    for i, theta in enumerate(theta_vec):
        steering_vec = compute_steering_vector_array(theta, n_mics, dist_between_mics, wavelength)
        projection = steering_vec.conj().T @ u_noise
        spectrum[i] = 1.0 / (projection @ projection.conj().T)
    
    return spectrum
