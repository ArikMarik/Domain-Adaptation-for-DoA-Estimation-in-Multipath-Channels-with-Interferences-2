import numpy as np
from scipy.linalg import sqrtm

def compute_mean_covariance(cov_matrices):
    return np.mean(cov_matrices, axis=2)

def compute_adaptation_matrix(cov_source_mean, cov_adapt_mean, epsilon=1e-7):
    cov_adapt_mean_reg = cov_adapt_mean + epsilon * np.eye(cov_adapt_mean.shape[0])
    cov_source_mean_reg = cov_source_mean + epsilon * np.eye(cov_source_mean.shape[0])
    
    sigma_adapt_inv_half = sqrtm(np.linalg.inv(cov_adapt_mean_reg))
    sigma_source_half = sqrtm(cov_source_mean_reg)
    
    E = sigma_source_half @ sigma_adapt_inv_half
    
    return E

def compute_adaptation_matrix_inv(cov_source_mean_inv, cov_adapt_mean_inv, epsilon=1e-7):
    cov_adapt_mean_inv_reg = cov_adapt_mean_inv + epsilon * np.eye(cov_adapt_mean_inv.shape[0])
    cov_source_mean_inv_reg = cov_source_mean_inv + epsilon * np.eye(cov_source_mean_inv.shape[0])
    
    sigma_adapt_inv_inv_half = sqrtm(np.linalg.inv(cov_adapt_mean_inv_reg))
    sigma_source_inv_half = sqrtm(cov_source_mean_inv_reg)
    
    E_inv = sigma_source_inv_half @ sigma_adapt_inv_inv_half
    
    return E_inv

def apply_adaptation(cov_matrix, E_matrix):
    return E_matrix @ cov_matrix @ E_matrix.conj().T
