import numpy as np
from scipy.linalg import eig, sqrtm

def sorted_evd(mat):
    eigvals, eigvecs = eig(mat)
    idx = np.argsort(np.real(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs, eigvals

def stable_matrix_sqrt(A, epsilon=1e-7):
    A_reg = A + epsilon * np.eye(A.shape[0])
    return sqrtm(A_reg)

def stable_inv(A, epsilon=1e-7):
    A_reg = A + epsilon * np.eye(A.shape[0])
    return np.linalg.inv(A_reg)
