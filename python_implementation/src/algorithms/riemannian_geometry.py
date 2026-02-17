"""
Riemannian geometry operations for covariance matrices.

This module implements Riemannian mean computation for positive definite matrices,
which provides a geometrically meaningful average on the manifold of SPD matrices.
"""

import numpy as np
from scipy.linalg import sqrtm, logm, expm


def riemannian_mean(cov_matrices, max_iter=20, tol=1e-6):
    """
    Compute Riemannian mean of covariance matrices.
    
    The Riemannian mean is the Fréchet mean on the manifold of symmetric
    positive definite (SPD) matrices with the affine-invariant Riemannian metric.
    
    Parameters
    ----------
    cov_matrices : np.ndarray
        Array of covariance matrices of shape (n_mics, n_mics, n_matrices)
    max_iter : int, optional
        Maximum number of iterations (default: 20)
    tol : float, optional
        Convergence tolerance based on Frobenius norm of S (default: 1e-6)
        
    Returns
    -------
    M : np.ndarray
        Riemannian mean covariance matrix of shape (n_mics, n_mics)
        
    Notes
    -----
    The Riemannian mean $M$ minimizes:
    
    $$M = \arg\min_X \sum_{i=1}^{N} d_R(X, C_i)^2$$
    
    where $d_R(X,Y) = \|\log(X^{-1/2}YX^{-1/2})\|_F$ is the Riemannian distance.
    
    Algorithm:
    1. Initialize: $M_0 = \frac{1}{N}\sum_{i=1}^N C_i$ (arithmetic mean)
    2. Iterate until convergence:
       - $A = M_k^{1/2}$
       - $B = M_k^{-1/2}$
       - $S = \frac{1}{N}\sum_{i=1}^N A \log(B C_i B) A$
       - $M_{k+1} = A \exp(B S B) A$
       - If $\|S\|_F < \epsilon$, stop
    
    References
    ----------
    M. Moakher, "A Differential Geometric Approach to the Geometric Mean 
    of Symmetric Positive-Definite Matrices", SIAM J. Matrix Anal. Appl., 2005
    """
    n_matrices = cov_matrices.shape[2]
    
    # Initialize with arithmetic mean
    M = np.mean(cov_matrices, axis=2)
    
    for iteration in range(max_iter):
        # Compute matrix square root and its inverse
        # A = M^(1/2), B = M^(-1/2)
        A = sqrtm(M)
        M_inv = np.linalg.inv(M)
        B = sqrtm(M_inv)
        
        # Compute sum of log-mapped covariances
        S = np.zeros_like(M)
        for i in range(n_matrices):
            C = cov_matrices[:, :, i]
            # S += A * log(B * C * B) * A
            log_term = logm(B @ C @ B)
            S += A @ log_term @ A
        
        S = S / n_matrices
        
        # Update: M = A * exp(B * S * B) * A
        exp_term = expm(B @ S @ B)
        M = A @ exp_term @ A
        
        # Check convergence based on Frobenius norm of S
        eps = np.linalg.norm(S, 'fro')
        if eps < tol:
            break
    
    return M


def riemannian_distance(cov1, cov2):
    """
    Compute Riemannian distance between two SPD matrices.
    
    Parameters
    ----------
    cov1 : np.ndarray
        First covariance matrix of shape (n, n)
    cov2 : np.ndarray
        Second covariance matrix of shape (n, n)
        
    Returns
    -------
    distance : float
        Riemannian distance between cov1 and cov2
        
    Notes
    -----
    The affine-invariant Riemannian distance is:
    
    $$d_R(C_1, C_2) = \|\log(C_1^{-1/2} C_2 C_1^{-1/2})\|_F$$
    """
    cov1_inv_sqrt = sqrtm(np.linalg.inv(cov1))
    log_term = logm(cov1_inv_sqrt @ cov2 @ cov1_inv_sqrt)
    distance = np.linalg.norm(log_term, 'fro')
    
    return distance
