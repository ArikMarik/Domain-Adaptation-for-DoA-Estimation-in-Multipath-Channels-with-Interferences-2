import numpy as np
from scipy.linalg import sqrtm

def compute_mean_covariance(cov_matrices, method='arithmetic'):
    """
    Compute mean of covariance matrices.
    
    Parameters
    ----------
    cov_matrices : np.ndarray
        Array of covariance matrices of shape (n_mics, n_mics, n_matrices)
    method : str, optional
        Averaging method: 'arithmetic' or 'riemannian' (default: 'arithmetic')
        
    Returns
    -------
    mean_cov : np.ndarray
        Mean covariance matrix of shape (n_mics, n_mics)
        
    Notes
    -----
    - 'arithmetic': Simple arithmetic mean $\bar{\Sigma} = \frac{1}{N}\sum_{i=1}^N \Sigma_i$
    - 'riemannian': Riemannian mean on the manifold of SPD matrices,
      which is the Fréchet mean with respect to the affine-invariant metric.
      More geometrically meaningful for covariance matrices.
    """
    if method == 'arithmetic':
        return np.mean(cov_matrices, axis=2)
    elif method == 'riemannian':
        from algorithms.riemannian_geometry import riemannian_mean
        return riemannian_mean(cov_matrices)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'arithmetic' or 'riemannian'.")

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
