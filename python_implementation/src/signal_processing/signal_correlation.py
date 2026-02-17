import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from signal_processing.rir import compute_rir
from signal_processing.stft import stft

def sig_corr_at_mics_acoustic_array(source_pos, mics_pos_mat, n_mics, beta, fs, bin_ind, 
                                     room_dim, order, n_samples_rir, win_length, signal, snr):
    sig_length = len(signal)
    
    hr = compute_rir(source_pos, fs, mics_pos_mat, n_samples_rir, beta, room_dim, order)
    
    sig_at_mic = np.zeros((sig_length, n_mics))
    for m in range(n_mics):
        sig_at_mic[:, m] = np.convolve(signal, hr[m, :], mode='same')
    
    nl = np.random.randn(sig_length, n_mics)
    varn = np.mean(np.var(nl, axis=0))
    vars = np.mean(np.var(sig_at_mic, axis=0))
    Gi_Tr = np.sqrt(vars / varn * 10**(-snr/10))
    sig_noise = sig_at_mic + Gi_Tr * nl
    
    sig_tr_per_mic_stft = stft(sig_noise[:, 0], win_length)
    n_time_frames = sig_tr_per_mic_stft.shape[1]
    
    sig_tr_per_mic_stft_freq = np.zeros((n_mics, n_time_frames), dtype=complex)
    for m in range(n_mics):
        sig_tr_per_mic_stft = stft(sig_noise[:, m], win_length)
        sig_tr_per_mic_stft_freq[m, :] = sig_tr_per_mic_stft[bin_ind, :]
    
    sig = sig_tr_per_mic_stft_freq.conj()
    corr = (1 / sig.shape[1]) * (sig.conj().T @ sig)
    
    return sig, corr
