import numpy as np

def sir_calc(theta_vec, theta_tar_deg, theta_tar_inter, tol_deg, spectrum):
    ind_spec_tar = np.where(np.abs(theta_vec - theta_tar_deg) < tol_deg)[0]
    ind_spec_inter = np.where(np.abs(theta_vec - theta_tar_inter) < tol_deg)[0]
    
    spectrum_norm = spectrum / np.max(np.abs(spectrum))
    
    sir = 10 * np.log10(np.max(np.abs(spectrum_norm[ind_spec_tar]))) - \
          10 * np.log10(np.max(np.abs(spectrum_norm[ind_spec_inter])))
    
    return sir
