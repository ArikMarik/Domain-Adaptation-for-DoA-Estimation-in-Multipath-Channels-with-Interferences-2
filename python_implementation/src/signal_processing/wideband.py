import numpy as np
from typing import Tuple
from .stft import stft

def select_frequency_bins(signal: np.ndarray, fs: int, win_length: int, 
                          n_bins: int, f_min: float, f_max: float, 
                          method: str = 'energy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Select frequency bins for wideband processing.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    fs : int
        Sampling frequency (Hz)
    win_length : int
        STFT window length
    n_bins : int
        Number of frequency bins to select (K)
    f_min : float
        Minimum frequency (Hz)
    f_max : float
        Maximum frequency (Hz)
    method : str
        'energy' for energy-based selection, 'uniform' for uniform spacing
        
    Returns
    -------
    freq_bins : np.ndarray
        Selected frequency bin indices (shape: (n_bins,))
    freq_hz : np.ndarray
        Frequencies in Hz (shape: (n_bins,))
    weights : np.ndarray
        Normalized weights for each bin (shape: (n_bins,))
        
    Notes
    -----
    Energy-based selection: $E_f = \frac{1}{T}\sum_{t} |X(f,t)|^2$
    Uniform selection: $f_k = f_{min} + k \cdot \frac{f_{max} - f_{min}}{K-1}$
    """
    freq_resolution = fs / win_length
    min_bin = int(np.ceil(f_min / freq_resolution))
    max_bin = int(np.floor(f_max / freq_resolution))
    
    if method == 'energy':
        X = stft(signal, win_length)
        
        energy_per_bin = np.mean(np.abs(X) ** 2, axis=1)
        
        valid_bins = np.arange(min_bin, max_bin + 1)
        valid_energy = energy_per_bin[valid_bins]
        
        top_k_indices = np.argsort(valid_energy)[-n_bins:][::-1]
        freq_bins = valid_bins[top_k_indices]
        freq_bins = np.sort(freq_bins)
        
        weights = energy_per_bin[freq_bins]
        weights = np.sqrt(weights)
        weights = weights / np.sum(weights)
        
    elif method == 'uniform':
        freq_centers = np.linspace(f_min, f_max, n_bins)
        freq_bins = np.round(freq_centers / freq_resolution).astype(int)
        freq_bins = np.unique(freq_bins)
        
        if len(freq_bins) < n_bins:
            freq_bins = np.linspace(min_bin, max_bin, n_bins).astype(int)
            freq_bins = np.unique(freq_bins)
        
        weights = np.ones(len(freq_bins)) / len(freq_bins)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'energy' or 'uniform'.")
    
    freq_hz = freq_bins * freq_resolution
    
    return freq_bins, freq_hz, weights


def extract_frequency_signal(signal: np.ndarray, fs: int, win_length: int, 
                             bin_ind: int) -> np.ndarray:
    """
    Extract signal at specific frequency bin from STFT.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal (1D array)
    fs : int
        Sampling frequency (Hz)
    win_length : int
        STFT window length
    bin_ind : int
        Frequency bin index to extract
        
    Returns
    -------
    signal_freq : np.ndarray
        Complex-valued signal at frequency bin (1D array)
        
    Notes
    -----
    Extracts time-series at specific frequency: $X_k(t) = \text{STFT}(x)[k, :]$
    """
    X = stft(signal, win_length)
    signal_freq = X[bin_ind, :]
    
    return signal_freq


def compute_wideband_spectrum(spectra_per_freq: np.ndarray, 
                              steering_vectors_per_freq: np.ndarray,
                              weights: np.ndarray, 
                              combination_method: str) -> np.ndarray:
    """
    Combine per-frequency spectra for wideband beamforming.
    
    Parameters
    ----------
    spectra_per_freq : np.ndarray
        Per-frequency spectrum values (complex), shape (n_freq, n_angles)
        For coherent: complex spectrum values $a_k^H(\theta) \tilde{\Sigma}_k a_k(\theta)$
        For incoherent: real spectrum magnitudes
    steering_vectors_per_freq : np.ndarray or None
        Steering vectors per frequency (only needed for coherent), shape (n_freq, n_angles)
        Can be None for incoherent method
    weights : np.ndarray
        Weights for each frequency (shape: (n_freq,))
    combination_method : str
        'coherent' or 'incoherent'
        
    Returns
    -------
    combined_spectrum : np.ndarray
        Combined spectrum (real values, shape: (n_angles,))
        
    Notes
    -----
    Coherent: $P_{coherent}(\theta) = \left| \sum_{k=1}^{K} w_k \cdot s_k(\theta) \right|^2$
    where $s_k(\theta) = a_k^H(\theta) \tilde{\Sigma}_k a_k(\theta)$
    
    Incoherent: $P_{incoherent}(\theta) = \sum_{k=1}^{K} w_k \cdot |s_k(\theta)|^2$
    """
    n_freq, n_angles = spectra_per_freq.shape
    
    if combination_method == 'coherent':
        weighted_sum = np.zeros(n_angles, dtype=complex)
        for k in range(n_freq):
            weighted_sum += weights[k] * spectra_per_freq[k, :]
        
        combined_spectrum = np.abs(weighted_sum) ** 2
        
    elif combination_method == 'incoherent':
        combined_spectrum = np.zeros(n_angles)
        for k in range(n_freq):
            combined_spectrum += weights[k] * np.abs(spectra_per_freq[k, :]) ** 2
            
    else:
        raise ValueError(f"Unknown combination method: {combination_method}. "
                        "Use 'coherent' or 'incoherent'.")
    
    return combined_spectrum


def compute_frequency_dependent_wavelength(freq_hz: np.ndarray, c: float = 340.0) -> np.ndarray:
    """
    Compute wavelength for each frequency.
    
    Parameters
    ----------
    freq_hz : np.ndarray
        Frequencies in Hz
    c : float
        Speed of sound (m/s), default 340 m/s
        
    Returns
    -------
    wavelengths : np.ndarray
        Wavelength for each frequency: $\lambda_k = c / f_k$
    """
    return c / freq_hz


def check_spatial_aliasing(f_max: float, mic_spacing: float, c: float = 340.0) -> bool:
    """
    Check if maximum frequency satisfies spatial aliasing constraint.
    
    Parameters
    ----------
    f_max : float
        Maximum frequency (Hz)
    mic_spacing : float
        Microphone spacing (m)
    c : float
        Speed of sound (m/s)
        
    Returns
    -------
    is_valid : bool
        True if f_max satisfies constraint, False otherwise
        
    Notes
    -----
    Spatial aliasing constraint: $f_{max} \leq \frac{c}{2d}$
    For d = 0.12 m and c = 340 m/s: $f_{max} \leq 1416.67$ Hz
    """
    f_max_allowed = c / (2 * mic_spacing)
    return f_max <= f_max_allowed
