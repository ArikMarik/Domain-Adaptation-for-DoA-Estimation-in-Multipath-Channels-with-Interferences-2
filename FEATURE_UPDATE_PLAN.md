# Feature Update Plan: DOA Estimation Enhancement

**Date:** February 17, 2026  
**Project:** Acoustic DOA Estimation with Domain Adaptation  
**Version:** 2.0

---

## Table of Contents

1. [Overview](#overview)
2. [Task 1: Scenario Naming Convention](#task-1-scenario-naming-convention)
3. [Task 2: Riemannian Mean Implementation](#task-2-riemannian-mean-implementation)
4. [Task 3: Wideband Signal Processing](#task-3-wideband-signal-processing)
5. [Task 4: Audio Signal Scenarios](#task-4-audio-signal-scenarios)
6. [Task 5: File Format Cleanup](#task-5-file-format-cleanup)
7. [Implementation Order](#implementation-order)
8. [Testing Strategy](#testing-strategy)
9. [Appendix: File Structure Changes](#appendix-file-structure-changes)

---

## Overview

This document outlines a comprehensive feature update to the DOA estimation system, introducing:
- Improved naming conventions for better clarity
- Riemannian mean as an alternative covariance averaging method
- Wideband signal processing with frequency-selective beamforming
- Audio signal processing with anti-aliasing filters
- Streamlined output file formats

**Estimated Implementation Complexity:** High  
**Backward Compatibility:** Maintained through configuration flags

---

## Task 1: Scenario Naming Convention

### 1.1 Current State Analysis

**Current scenario names:**
- `fig2-3-4`: Beta sweep scenario (β = 0.2-0.6), SNR/SIR = -20dB, no interference
- `fig5`: Fixed beta (β = 0.2), SNR/SIR = -20dB, single interference source
- `fig6`: Fixed beta (β = 0.2), SNR/SIR = 0dB, two interference sources, non-fixed positions
- `table`: Table scenario with β = 0.2, SNR/SIR varying from -20dB to 0dB, single interference

**Problems:**
- Non-descriptive names that reference figure numbers
- Difficult to understand scenario purpose without reading config
- Hard to extend with new scenarios

### 1.2 New Naming Convention

**Naming Pattern:** `<parameter>-<condition>_<feature>`

**New scenario names:**

| Old Name    | New Name                          | Description |
|-------------|-----------------------------------|-------------|
| `fig2-3-4`  | `beta-sweep_clean`                | Beta sweep (0.2-0.6s) with clean conditions (-20dB SNR/SIR, no active interference) |
| `fig5`      | `single-interference_fixed`       | Single fixed interference source (β=0.2, -20dB SNR/SIR) |
| `fig6`      | `dual-interference_moving`        | Two interference sources with moving positions (β=0.2, 0dB SNR/SIR) |
| `table`     | `snr-sir-sweep_single-interference` | SNR/SIR sweep (-20 to 0 dB) with single fixed interference (β=0.2) |

### 1.3 Implementation Steps

1. **Update `main.py`:**
   - Rename all scenario keys in `get_scenario_config()`
   - Update `argparse` choices list
   - Add mapping dictionary for backward compatibility (optional)
   - Update default scenario list when `--scenario all` is used

2. **Update `acoustic_doa.py`:**
   - Change plotting condition checks:
     - `plot_figs_2_3_4` → `plot_beta_sweep_clean`
     - `plot_figs_5` → `plot_single_interference_fixed`
     - `plot_figs_6` → `plot_dual_interference_moving`
   - Update figure prefix naming logic

3. **Update result file naming:**
   - Change pickle file pattern from `results_fig*.pkl` to `results_<new_name>.pkl`

4. **Documentation:**
   - Update README with new scenario names
   - Add scenario comparison table
   - Document backward compatibility approach

---

## Task 2: Riemannian Mean Implementation

### 2.1 Background

**Current Implementation:**
- Uses simple arithmetic mean: `np.mean(cov_matrices, axis=2)`
- Located in: `algorithms/domain_adaptation.py::compute_mean_covariance()`

**MATLAB Reference:**
- File: `matlab_code/.../RiemannianMean.m`
- **NOT used in baseline MATLAB code** (uses simple `mean()` instead)
- Available as an optional alternative method

**Mathematical Definition:**

Riemannian mean $M$ of covariance matrices $\{C_1, ..., C_N\}$ is the solution to:

$$M = \arg\min_X \sum_{i=1}^{N} \|\log(M^{-1/2} C_i M^{-1/2})\|_F^2$$

**Iterative Algorithm:**
1. Initialize: $M_0 = \frac{1}{N}\sum_{i=1}^N C_i$ (arithmetic mean)
2. For $k = 1$ to $20$ (or until convergence):
   - $A = M_k^{1/2}$
   - $B = M_k^{-1/2}$
   - $S = \frac{1}{N}\sum_{i=1}^N A \log(B C_i B) A$
   - $M_{k+1} = A \exp(B S B) A$
   - If $\|S\|_F < 10^{-6}$, break

### 2.2 Implementation Details

#### 2.2.1 Create New Module: `algorithms/riemannian_geometry.py`

```python
import numpy as np
from scipy.linalg import sqrtm, logm, expm

def riemannian_mean(cov_matrices, max_iter=20, tol=1e-6):
    """
    Compute Riemannian mean of covariance matrices.
    
    Parameters
    ----------
    cov_matrices : np.ndarray
        Array of covariance matrices of shape (n_mics, n_mics, n_matrices)
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance based on Frobenius norm of S
        
    Returns
    -------
    M : np.ndarray
        Riemannian mean covariance matrix of shape (n_mics, n_mics)
        
    Notes
    -----
    Implements algorithm from:
    M. Moakher, "A Differential Geometric Approach to the Geometric Mean 
    of Symmetric Positive-Definite Matrices", SIAM J. Matrix Anal. Appl., 2005
    
    The Riemannian mean is the solution to:
    $$M = \arg\min_X \sum_{i=1}^{N} d_R(X, C_i)^2$$
    where $d_R(X,Y) = \|\log(X^{-1/2}YX^{-1/2})\|_F$ is the Riemannian distance.
    """
    n_matrices = cov_matrices.shape[2]
    
    # Initialize with arithmetic mean
    M = np.mean(cov_matrices, axis=2)
    
    for iteration in range(max_iter):
        # A = M^(1/2), B = M^(-1/2)
        A = sqrtm(M)
        B = sqrtm(np.linalg.inv(M))
        
        # Compute sum of log-mapped covariances
        S = np.zeros_like(M)
        for i in range(n_matrices):
            C = cov_matrices[:, :, i]
            # S += A * log(B * C * B) * A
            S += A @ logm(B @ C @ B) @ A
        S = S / n_matrices
        
        # Update: M = A * exp(B * S * B) * A
        M = A @ expm(B @ S @ B) @ A
        
        # Check convergence
        eps = np.linalg.norm(S, 'fro')
        if eps < tol:
            break
    
    return M
```

#### 2.2.2 Update `algorithms/domain_adaptation.py`

Add new function:
```python
def compute_mean_covariance(cov_matrices, method='arithmetic'):
    """
    Compute mean of covariance matrices.
    
    Parameters
    ----------
    cov_matrices : np.ndarray
        Array of shape (n_mics, n_mics, n_matrices)
    method : str
        'arithmetic' for np.mean, 'riemannian' for Riemannian mean
        
    Returns
    -------
    mean_cov : np.ndarray
        Mean covariance matrix of shape (n_mics, n_mics)
    """
    if method == 'arithmetic':
        return np.mean(cov_matrices, axis=2)
    elif method == 'riemannian':
        from algorithms.riemannian_geometry import riemannian_mean
        return riemannian_mean(cov_matrices)
    else:
        raise ValueError(f"Unknown method: {method}")
```

#### 2.2.3 Update `acoustic_doa.py`

Add configuration parameter:
- `covariance_averaging_method`: `'arithmetic'` or `'riemannian'`

Modify lines 162-165:
```python
corr_tr_mean = compute_mean_covariance(corr_tr_vec, 
                                       method=config.get('covariance_averaging_method', 'arithmetic'))
cor_art_mean = compute_mean_covariance(cor_art_vec,
                                       method=config.get('covariance_averaging_method', 'arithmetic'))
corr_tr_mean_inv = compute_mean_covariance(corr_tr_vec_inv,
                                           method=config.get('covariance_averaging_method', 'arithmetic'))
cor_art_mean_inv = compute_mean_covariance(cor_art_vec_inv,
                                           method=config.get('covariance_averaging_method', 'arithmetic'))
```

### 2.3 New Scenario: Beta Sweep with Riemannian Mean

Create new scenario: `beta-sweep_clean_riemannian`

**Configuration:**
```python
{
    # Same as beta-sweep_clean
    'k_interference_pos': 1,
    'k_train_points': 100,
    'k_art_points': 200,
    'k_test_points': 300,
    'beta_vec': [0.2, 0.3, 0.4, 0.5, 0.6],
    'snr_max_db': -20,
    'snr_min_db': -20,
    'sir_max_db': -20,
    'sir_min_db': -20,
    'is_interference_active': False,
    'is_interference_fixed': True,
    'sig_dim_4_music_pt': 1,
    'sig_dim_4_music': 1,
    'train_is_test': True,
    'is_only_interf_active_during_train': 0,
    'is_two_interference_sources_active': False,
    'k_inter_pos': 1,
    'dist_between_train': 0.2,
    'plot_beta_sweep_clean': True,
    # NEW PARAMETER
    'covariance_averaging_method': 'riemannian',
}
```

### 2.4 Comparison Plotting

Create new plotting function: `visualization/plotting.py::plot_method_comparison()`

**Purpose:** Compare arithmetic vs. Riemannian mean results

**Inputs:**
- `results_arithmetic`: Results dictionary from arithmetic mean scenario
- `results_riemannian`: Results dictionary from Riemannian mean scenario
- `beta_vec`: Vector of beta values
- `output_dir`: Output directory path

**Outputs:**
- Side-by-side box plots for each method (DS, MVDR, MUSIC)
- Each plot shows 4 bars: Arithmetic, Arithmetic+DA, Riemannian, Riemannian+DA
- Filename: `BetaSweep_ArithmeticVsRiemannian_<method>.jpg`

**Implementation:**
```python
def plot_method_comparison(results_arithmetic, results_riemannian, 
                          beta_vec, methods, output_dir):
    """
    Plot comparison between arithmetic and Riemannian mean methods.
    
    Creates side-by-side comparison for each beamforming method showing:
    - Arithmetic mean (standard)
    - Arithmetic mean + DA
    - Riemannian mean (standard)
    - Riemannian mean + DA
    """
    # Implementation details...
```

### 2.5 Analysis Requirements

After running both scenarios:
1. Generate comparison plots for DS, MVDR, MUSIC
2. Compute statistical metrics:
   - Median error for each method/beta
   - 75th percentile error
   - Error reduction percentage: $\frac{E_{arith} - E_{riem}}{E_{arith}} \times 100\%$
3. Save comparison table as CSV: `results/method_comparison.csv`

---

## Task 3: Wideband Signal Processing

### 3.1 Architecture Overview

**Current (Narrowband):**
- Single frequency bin processing (bin_ind = 1429 Hz)
- One steering vector per position
- Single covariance matrix per source

**New (Wideband):**
- Multiple frequency bins (f_max = 1429 Hz to avoid spatial aliasing)
- Per-frequency steering vectors and covariance matrices
- Per-frequency domain adaptation
- Coherent/non-coherent combination across frequencies

**Spatial Aliasing Constraint:**
$$f_{max} = \frac{c}{2d} = \frac{340}{2 \times 0.12} = 1416.67 \text{ Hz}$$

Round to 1429 Hz (existing bin frequency).

### 3.2 Configuration Parameters

Add to scenario configs:
```python
{
    'is_wideband': False,  # Enable wideband processing
    'wideband_params': {
        'n_freq_bins': 10,          # Number of frequency bins (K)
        'f_max': 1429,              # Maximum frequency (Hz)
        'f_min': 200,               # Minimum frequency (Hz)
        'freq_selection': 'energy', # 'energy' or 'uniform'
        'combination_method': 'coherent',  # 'coherent' or 'incoherent'
    }
}
```

### 3.3 Frequency Selection Strategy

#### 3.3.1 Energy-Based Selection

1. Compute STFT of source signal
2. Compute energy per frequency bin: $E_f = \frac{1}{T}\sum_{t} |X(f,t)|^2$
3. Select top K bins with highest energy within $[f_{min}, f_{max}]$

#### 3.3.2 Uniform Selection

1. Linearly space K frequencies: $f_k = f_{min} + k \cdot \frac{f_{max} - f_{min}}{K-1}$
2. Map to nearest FFT bin indices

### 3.4 Per-Frequency Processing

For each selected frequency $f_k$:

1. **Extract frequency-specific signal:**
   - Apply band-pass filter or STFT extraction
   - Extract bin $k$ from STFT: $X_k(t) = \text{STFT}(x)[k, :]$

2. **Compute steering vectors:**
   - Wavelength: $\lambda_k = c / f_k$
   - Steering vector: $a_k(\theta) = [e^{j2\pi d \sin\theta/\lambda_k}]_{m=0}^{M-1}$

3. **Compute covariance matrices:**
   - Source domain: $\Sigma_{S,k} = \frac{1}{T}X_k X_k^H$
   - Adaptation domain: $\Sigma_{A,k} = \frac{1}{N}\sum_{n=1}^N a_k(\theta_n) a_k^H(\theta_n)$

4. **Domain adaptation per frequency:**
   - $E_k = \Sigma_{S,k}^{1/2} \Sigma_{A,k}^{-1/2}$
   - Apply: $\tilde{\Sigma}_{k} = E_k \Sigma_{test,k} E_k^H$

5. **Beamforming per frequency:**
   - Compute DS/MVDR/MUSIC spectra using $\tilde{\Sigma}_k$

### 3.5 Cross-Frequency Combination

#### 3.5.1 Coherent Combination

Assumes phase relationship across frequencies:

$$P_{coherent}(\theta) = \left| \sum_{k=1}^{K} w_k \cdot a_k^H(\theta) \tilde{\Sigma}_k a_k(\theta) \right|^2$$

Where weights: $w_k = \sqrt{E_{f_k}}$ (energy-based) or $w_k = 1$ (uniform)

**Pros:** Higher resolution when signal is truly coherent  
**Cons:** Phase errors can degrade performance

#### 3.5.2 Incoherent Combination

No phase assumption:

$$P_{incoherent}(\theta) = \sum_{k=1}^{K} w_k \cdot |a_k^H(\theta) \tilde{\Sigma}_k a_k(\theta)|^2$$

**Pros:** Robust to phase errors, frequency-selective fading  
**Cons:** Lower resolution compared to coherent when signal is actually coherent

**Recommendation:** Support both, default to incoherent for robustness

### 3.6 Implementation Structure

#### 3.6.1 New Module: `signal_processing/wideband.py`

```python
def select_frequency_bins(signal, fs, win_length, n_bins, f_min, f_max, method='energy'):
    """
    Select frequency bins for wideband processing.
    
    Returns
    -------
    freq_bins : np.ndarray
        Selected frequency bin indices
    freq_hz : np.ndarray
        Frequencies in Hz
    weights : np.ndarray
        Energy-based weights for each bin
    """

def extract_frequency_signal(signal, fs, win_length, bin_ind):
    """
    Extract signal at specific frequency bin.
    
    Returns
    -------
    signal_freq : np.ndarray
        Complex-valued signal at frequency bin
    """

def compute_wideband_spectrum(spectra_per_freq, weights, combination_method):
    """
    Combine per-frequency spectra.
    
    Parameters
    ----------
    spectra_per_freq : np.ndarray
        Array of shape (n_freq, n_angles)
    weights : np.ndarray
        Weights for each frequency
    combination_method : str
        'coherent' or 'incoherent'
        
    Returns
    -------
    combined_spectrum : np.ndarray
        Combined spectrum of shape (n_angles,)
    """
```

#### 3.6.2 Update `acoustic_doa.py`

Add wideband processing branch:

```python
if config.get('is_wideband', False):
    # Wideband processing
    wb_params = config['wideband_params']
    
    # Select frequency bins based on energy
    freq_bins, freq_hz, freq_weights = select_frequency_bins(
        sig_train[:, 0], fs, win_length, 
        wb_params['n_freq_bins'], 
        wb_params['f_min'], 
        wb_params['f_max'],
        wb_params['freq_selection']
    )
    
    # Process each frequency
    spectra_per_freq = []
    for freq_idx, freq_bin in enumerate(freq_bins):
        # Compute wavelength
        wavelength_f = c / freq_hz[freq_idx]
        
        # [Training phase per frequency]
        # [Testing phase per frequency]
        # [Beamforming per frequency]
        
        spectra_per_freq.append(spectrum_adapted)
    
    # Combine spectra
    final_spectrum = compute_wideband_spectrum(
        np.array(spectra_per_freq),
        freq_weights,
        wb_params['combination_method']
    )
else:
    # Existing narrowband processing
    # ...
```

### 3.7 New Wideband Scenarios

Create wideband versions of each existing scenario:

| Narrowband Scenario               | Wideband Scenario                        |
|-----------------------------------|------------------------------------------|
| `beta-sweep_clean`                | `beta-sweep_clean_wideband-coherent`     |
|                                   | `beta-sweep_clean_wideband-incoherent`   |
| `single-interference_fixed`       | `single-interference_fixed_wideband-coherent` |
|                                   | `single-interference_fixed_wideband-incoherent` |
| `dual-interference_moving`        | `dual-interference_moving_wideband-coherent` |
|                                   | `dual-interference_moving_wideband-incoherent` |
| `snr-sir-sweep_single-interference` | `snr-sir-sweep_single-interference_wideband-coherent` |
|                                   | `snr-sir-sweep_single-interference_wideband-incoherent` |

**Configuration Template:**
```python
'beta-sweep_clean_wideband-coherent': {
    # Copy all parameters from beta-sweep_clean
    **get_scenario_config('beta-sweep_clean'),
    # Add wideband parameters
    'is_wideband': True,
    'wideband_params': {
        'n_freq_bins': 10,
        'f_max': 1429,
        'f_min': 200,
        'freq_selection': 'energy',
        'combination_method': 'coherent',
    },
}
```

---

## Task 4: Audio Signal Scenarios

### 4.1 Motivation

Current implementation uses white Gaussian noise. Real audio signals have:
- Spectral structure (speech formants, music harmonics)
- Time-varying envelopes
- Non-Gaussian statistics

### 4.2 Audio Signal Requirements

1. **Source Audio:**
   - Use speech or music samples
   - Sample rate: Must match fs = 12000 Hz or resample
   - Duration: At least 2.5 seconds per test

2. **Anti-Aliasing Filter:**
   - Low-pass filter with cutoff at 1429 Hz
   - Filter type: Butterworth, order 8
   - Zero-phase filtering (filtfilt) to avoid phase distortion

3. **Signal Normalization:**
   - Normalize to unit variance before applying SNR
   - Maintain SNR control: $\text{signal\_scaled} = \text{signal\_norm} \times \sqrt{P_{signal}/\text{var}(signal)}$

### 4.3 Implementation

#### 4.3.1 New Module: `signal_processing/audio_utils.py`

```python
import numpy as np
from scipy import signal as scipy_signal

def load_audio_signal(filepath, target_fs, duration):
    """
    Load and resample audio file.
    
    Parameters
    ----------
    filepath : str
        Path to audio file (wav, mp3, etc.)
    target_fs : int
        Target sample rate (12000 Hz)
    duration : float
        Duration in seconds
        
    Returns
    -------
    audio : np.ndarray
        Audio signal of length (duration * target_fs)
    """

def apply_antialiasing_filter(signal, fs, f_cutoff=1429, order=8):
    """
    Apply low-pass anti-aliasing filter.
    
    Parameters
    ----------
    signal : np.ndarray
        Input signal
    fs : int
        Sample rate
    f_cutoff : float
        Cutoff frequency in Hz
    order : int
        Filter order
        
    Returns
    -------
    filtered_signal : np.ndarray
        Filtered signal
        
    Notes
    -----
    Uses Butterworth filter with zero-phase filtering (filtfilt).
    Cutoff frequency set to $f_c = c/(2d) = 1416.67$ Hz to prevent
    spatial aliasing for array spacing d = 0.12 m.
    """
    nyquist = fs / 2
    normalized_cutoff = f_cutoff / nyquist
    
    b, a = scipy_signal.butter(order, normalized_cutoff, btype='low')
    filtered = scipy_signal.filtfilt(b, a, signal)
    
    return filtered

def generate_audio_signal_set(audio_source, n_signals, sig_length, fs):
    """
    Generate multiple audio signals from source.
    
    Parameters
    ----------
    audio_source : np.ndarray or str
        Source audio signal or path to audio file
    n_signals : int
        Number of signals to generate
    sig_length : int
        Length of each signal in samples
    fs : int
        Sample rate
        
    Returns
    -------
    signals : np.ndarray
        Array of shape (sig_length, n_signals)
        
    Notes
    -----
    Extracts random segments from source audio to create diverse test signals.
    """
```

#### 4.3.2 Update `acoustic_doa.py`

Add configuration parameter:
```python
{
    'signal_type': 'noise',  # 'noise' or 'audio'
    'audio_params': {
        'source_file': 'path/to/audio.wav',
        'apply_antialiasing': True,
        'f_cutoff': 1429,
    }
}
```

Modify signal generation section (lines 94-97):
```python
if config.get('signal_type', 'noise') == 'noise':
    sig_train = np.random.randn(sig_length, n_train_points)
    sig_inter_train = np.random.randn(sig_length, n_train_points)
    sig_test = np.random.randn(sig_length, n_test_points)
    sig_inter_test = np.random.randn(sig_length, n_test_points)
else:
    # Audio signal processing
    audio_params = config['audio_params']
    audio_source = load_audio_signal(
        audio_params['source_file'], 
        fs, 
        sig_length / fs * (n_train_points + n_test_points + 10)  # Load extra
    )
    
    if audio_params.get('apply_antialiasing', True):
        audio_source = apply_antialiasing_filter(
            audio_source, fs, audio_params.get('f_cutoff', 1429)
        )
    
    sig_train = generate_audio_signal_set(audio_source, n_train_points, sig_length, fs)
    sig_inter_train = generate_audio_signal_set(audio_source, n_train_points, sig_length, fs)
    sig_test = generate_audio_signal_set(audio_source, n_test_points, sig_length, fs)
    sig_inter_test = generate_audio_signal_set(audio_source, n_test_points, sig_length, fs)
```

### 4.4 New Audio Scenarios

Create audio versions with wideband processing:

| Base Scenario                     | Audio Scenario (Narrowband)           | Audio Scenario (Wideband)                    |
|-----------------------------------|---------------------------------------|----------------------------------------------|
| `beta-sweep_clean`                | `beta-sweep_clean_audio`              | `beta-sweep_clean_audio-wideband-coherent`   |
| `single-interference_fixed`       | `single-interference_fixed_audio`     | `single-interference_fixed_audio-wideband-coherent` |
| `dual-interference_moving`        | `dual-interference_moving_audio`      | `dual-interference_moving_audio-wideband-coherent` |
| `snr-sir-sweep_single-interference` | `snr-sir-sweep_single-interference_audio` | `snr-sir-sweep_single-interference_audio-wideband-coherent` |

**Configuration Example:**
```python
'beta-sweep_clean_audio-wideband-coherent': {
    # Base config from beta-sweep_clean
    'k_interference_pos': 1,
    'k_train_points': 100,
    # ... (all other parameters)
    
    # Audio parameters
    'signal_type': 'audio',
    'audio_params': {
        'source_file': 'data/audio/speech_sample.wav',
        'apply_antialiasing': True,
        'f_cutoff': 1429,
    },
    
    # Wideband parameters
    'is_wideband': True,
    'wideband_params': {
        'n_freq_bins': 10,
        'f_max': 1429,
        'f_min': 200,
        'freq_selection': 'energy',
        'combination_method': 'coherent',
    },
}
```

### 4.5 Audio Data Requirements

1. **Create data directory:** `data/audio/`
2. **Sample audio files:**
   - `speech_male.wav`: Male speech sample (> 30 seconds)
   - `speech_female.wav`: Female speech sample
   - `music_classical.wav`: Classical music
   - `music_jazz.wav`: Jazz music
3. **Format:** WAV, 16-bit PCM, any sample rate (will be resampled)

---

## Task 5: File Format Cleanup

### 5.1 Current State

**Problem:** Duplicate file generation
- Every plot saved as both `.jpg` (150 DPI) and `.png` (150 DPI)
- Wastes disk space (approximately 2x storage)
- No functional difference at same DPI

**Current output:** ~32 files in results/ directory

### 5.2 Strategy

**Keep:** JPG format only
- Reason: Smaller file size for plots, sufficient quality at 150 DPI
- Use 200 DPI for publication-quality outputs

**Remove:** PNG files

### 5.3 Implementation Steps

#### 5.3.1 Update `visualization/plotting.py`

**Before:**
```python
plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=150)
plt.savefig(os.path.join(output_dir, f'{filename}.png'), dpi=150)
```

**After:**
```python
plt.savefig(os.path.join(output_dir, f'{filename}.jpg'), dpi=200, quality=95)
```

**Files to update:**
- `plot_beta_sweep()`: Lines 30-31
- `plot_polar_spectrum()`: Lines 65-66
- `plot_polar_spectrum_components()`: Lines 92-93
- `plot_boxplots()`: Lines 137-138

#### 5.3.2 Delete Existing PNG Files

```bash
find /home/user/hw/audioHW/final_project2/python_implementation/results/ -name "*.png" -delete
```

#### 5.3.3 Rename Existing Files with New Naming Convention

Create migration script: `scripts/rename_legacy_files.py`

```python
import os
import re

RESULTS_DIR = '../python_implementation/results/'

# Mapping: old_prefix -> new_prefix
RENAME_MAP = {
    'Fig2': 'BetaSweep',
    'Fig3': 'BetaSweep_Spectra',
    'Fig4': 'BetaSweep_DoAError',
    'Fig5': 'SingleInterference',
    'Fig6': 'DualInterference',
}

def rename_files():
    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith('.jpg'):
            continue
        
        for old_prefix, new_prefix in RENAME_MAP.items():
            if filename.startswith(old_prefix):
                new_filename = filename.replace(old_prefix, new_prefix)
                old_path = os.path.join(RESULTS_DIR, filename)
                new_path = os.path.join(RESULTS_DIR, new_filename)
                
                os.rename(old_path, new_path)
                print(f"Renamed: {filename} -> {new_filename}")
                break
```

**New naming examples:**
- `Fig2_AcousticRoom3D.jpg` → `BetaSweep_AcousticRoom3D.jpg`
- `Fig3_DS_Spectrum.jpg` → `BetaSweep_Spectra_DS.jpg`
- `Fig4_MVDR_DoABeta.jpg` → `BetaSweep_DoAError_MVDR.jpg`
- `Fig5_DoAErrAcoust.jpg` → `SingleInterference_DoAError.jpg`
- `Fig6_MUSIC_Spectrum.jpg` → `DualInterference_Spectra_MUSIC.jpg`

#### 5.3.4 Update Figure Prefix Logic in `acoustic_doa.py`

**Current (Line 230):**
```python
fig_prefix = 'Fig3' if config.get('plot_figs_2_3_4', False) else ('Fig5' if config.get('plot_figs_5', False) else 'Fig6')
```

**New:**
```python
# Map scenario type to descriptive prefix
if config.get('plot_beta_sweep_clean', False):
    fig_prefix = 'BetaSweep_Spectra'
elif config.get('plot_single_interference_fixed', False):
    fig_prefix = 'SingleInterference_Spectra'
elif config.get('plot_dual_interference_moving', False):
    fig_prefix = 'DualInterference_Spectra'
else:
    fig_prefix = 'Spectrum'  # Default fallback
```

---

## Implementation Order

### Phase 1: Foundation (Priority: High)
**Estimated Time:** 2-3 hours

1. ✅ **Task 1.3:** Rename scenarios in `main.py` and `acoustic_doa.py`
2. ✅ **Task 5.3.1:** Update plotting functions to save JPG only
3. ✅ **Task 5.3.2:** Delete existing PNG files
4. ✅ **Task 5.3.3:** Rename legacy figure files
5. ✅ **Test:** Run one existing scenario to verify basic functionality

### Phase 2: Riemannian Mean (Priority: Medium)
**Estimated Time:** 3-4 hours

1. ✅ **Task 2.2.1:** Implement `riemannian_geometry.py` module
2. ✅ **Task 2.2.2:** Update `domain_adaptation.py`
3. ✅ **Task 2.2.3:** Add covariance method parameter to `acoustic_doa.py`
4. ✅ **Task 2.3:** Create `beta-sweep_clean_riemannian` scenario
5. ✅ **Task 2.4:** Implement comparison plotting function
6. ✅ **Test:** Run both arithmetic and Riemannian scenarios, generate comparison plots

### Phase 3: Wideband Processing (Priority: High)
**Estimated Time:** 6-8 hours

1. ✅ **Task 3.6.1:** Implement `signal_processing/wideband.py` module
   - Frequency bin selection
   - Per-frequency signal extraction
   - Spectrum combination (coherent/incoherent)
2. ✅ **Task 3.6.2:** Add wideband branch to `acoustic_doa.py`
   - Conditional processing based on `is_wideband` flag
   - Per-frequency covariance computation
   - Per-frequency domain adaptation
   - Per-frequency beamforming
3. ✅ **Task 3.7:** Create all wideband scenarios (8 new scenarios)
4. ✅ **Test:** Run narrowband vs. wideband comparison for one scenario

### Phase 4: Audio Signal Processing (Priority: Medium)
**Estimated Time:** 4-5 hours

1. ✅ **Task 4.3.1:** Implement `signal_processing/audio_utils.py` module
   - Audio loading and resampling
   - Anti-aliasing filter
   - Signal set generation
2. ✅ **Task 4.3.2:** Add audio signal branch to `acoustic_doa.py`
3. ✅ **Task 4.5:** Prepare audio data files
4. ✅ **Task 4.4:** Create all audio scenarios (8 new scenarios)
5. ✅ **Test:** Run noise vs. audio comparison

### Phase 5: Integration & Validation (Priority: High)
**Estimated Time:** 4-6 hours

1. ✅ **Test all scenarios:** Run full scenario suite
2. ✅ **Generate comparison plots:**
   - Arithmetic vs. Riemannian
   - Narrowband vs. Wideband
   - Coherent vs. Incoherent
   - Noise vs. Audio
3. ✅ **Documentation:**
   - Update README with new scenario descriptions
   - Add usage examples for each feature
   - Document configuration parameters
4. ✅ **Code review & cleanup:**
   - Remove commented-out code
   - Add docstrings
   - Verify PEP 8 compliance

**Total Estimated Time:** 19-26 hours

---

## Testing Strategy

### Unit Tests

Create `tests/` directory with:

1. **`test_riemannian_geometry.py`:**
   - Test convergence of Riemannian mean
   - Verify positive definiteness of result
   - Compare with arithmetic mean for identical matrices
   - Test edge cases (singular matrices with regularization)

2. **`test_wideband.py`:**
   - Test frequency bin selection
   - Verify energy-based vs. uniform selection
   - Test coherent vs. incoherent combination
   - Validate spectrum shapes

3. **`test_audio_utils.py`:**
   - Test anti-aliasing filter (verify frequency response)
   - Test audio loading and resampling
   - Test signal normalization

### Integration Tests

1. **Scenario execution:**
   - Run each scenario for 1 interference position, 10 test points (reduced load)
   - Verify output file generation
   - Check result dictionary structure

2. **Cross-method comparison:**
   - Verify adapted methods perform better than standard
   - Check that wideband coherent resolution ≥ narrowband
   - Validate audio results are comparable to noise results

### Validation Metrics

For each scenario, track:
- Median DOA error [degrees]
- 75th percentile error [degrees]
- Computation time [minutes]
- Memory usage [MB]

Expected relationships:
- Adapted < Standard (error)
- Wideband-coherent ≤ Narrowband (error, when phase is preserved)
- Wideband-incoherent < Wideband-coherent (robustness)
- Riemannian ≈ Arithmetic (similar performance, different geometry)

---

## Appendix: File Structure Changes

### New Files

```
python_implementation/
├── src/
│   ├── algorithms/
│   │   └── riemannian_geometry.py          # NEW: Riemannian mean implementation
│   ├── signal_processing/
│   │   ├── wideband.py                     # NEW: Wideband processing utilities
│   │   └── audio_utils.py                  # NEW: Audio signal utilities
│   └── visualization/
│       └── plotting.py                     # MODIFIED: Comparison plots
├── data/
│   └── audio/                              # NEW: Audio sample directory
│       ├── speech_male.wav
│       ├── speech_female.wav
│       ├── music_classical.wav
│       └── music_jazz.wav
├── tests/                                  # NEW: Unit tests directory
│   ├── test_riemannian_geometry.py
│   ├── test_wideband.py
│   └── test_audio_utils.py
├── scripts/
│   └── rename_legacy_files.py              # NEW: File migration script
└── FEATURE_UPDATE_PLAN.md                  # THIS FILE

```

### Modified Files

```
python_implementation/
├── src/
│   ├── main.py                             # Scenario naming, new scenarios
│   ├── acoustic_doa.py                     # Wideband/audio branches, config params
│   └── algorithms/
│       └── domain_adaptation.py            # Method parameter for mean computation
```

### Renamed Result Files

```
results/
├── BetaSweep_AcousticRoom3D.jpg            # Was: Fig2_AcousticRoom3D.jpg
├── BetaSweep_Spectra_DS.jpg                # Was: Fig3_DS_Spectrum.jpg
├── BetaSweep_DoAError_MVDR.jpg             # Was: Fig4_MVDR_DoABeta.jpg
├── SingleInterference_DoAError.jpg         # Was: Fig5_DoAErrAcoust.jpg
├── DualInterference_Spectra_MUSIC.jpg      # Was: Fig6_MUSIC_Spectrum.jpg
└── ...
```

---

## Summary of New Scenarios

**Total New Scenarios:** 20

| Category | Scenario Name | Description |
|----------|---------------|-------------|
| **Baseline (Renamed)** | `beta-sweep_clean` | Beta sweep, clean conditions |
| | `single-interference_fixed` | Single fixed interference |
| | `dual-interference_moving` | Dual moving interference |
| | `snr-sir-sweep_single-interference` | SNR/SIR table sweep |
| **Riemannian** | `beta-sweep_clean_riemannian` | Beta sweep with Riemannian mean |
| **Wideband (Noise)** | `beta-sweep_clean_wideband-coherent` | Wideband coherent |
| | `beta-sweep_clean_wideband-incoherent` | Wideband incoherent |
| | `single-interference_fixed_wideband-coherent` | |
| | `single-interference_fixed_wideband-incoherent` | |
| | `dual-interference_moving_wideband-coherent` | |
| | `dual-interference_moving_wideband-incoherent` | |
| | `snr-sir-sweep_single-interference_wideband-coherent` | |
| | `snr-sir-sweep_single-interference_wideband-incoherent` | |
| **Audio (Narrowband)** | `beta-sweep_clean_audio` | Audio signal, narrowband |
| | `single-interference_fixed_audio` | |
| | `dual-interference_moving_audio` | |
| | `snr-sir-sweep_single-interference_audio` | |
| **Audio (Wideband)** | `beta-sweep_clean_audio-wideband-coherent` | Audio + wideband |
| | `single-interference_fixed_audio-wideband-coherent` | |
| | `dual-interference_moving_audio-wideband-coherent` | |
| | `snr-sir-sweep_single-interference_audio-wideband-coherent` | |

---

**End of Plan**

---

## Notes for Implementation

1. **Backward Compatibility:**
   - Old scenario names can be supported through a deprecation mapping
   - Add warning message for deprecated names

2. **Performance Considerations:**
   - Wideband processing will be ~K times slower (K = number of frequencies)
   - Consider parallel processing for per-frequency operations
   - Riemannian mean adds ~15-20 iterations, watch for convergence

3. **Memory Management:**
   - Wideband requires storing K covariance matrices simultaneously
   - Consider processing frequencies in batches if memory is constrained

4. **Validation Against MATLAB:**
   - After Phase 1-2, compare Python results with MATLAB for baseline scenarios
   - Ensure numerical consistency (< 0.1 degree error difference)

5. **Documentation:**
   - Add mathematical definitions to docstrings using LaTeX
   - Include references to relevant papers
   - Provide usage examples in README

---

**Plan Version:** 1.0  
**Last Updated:** February 17, 2026  
**Author:** AI Assistant  
**Status:** Ready for Implementation
