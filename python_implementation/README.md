# Acoustic Domain Adaptation for DoA Estimation - Python Implementation

Complete Python reproduction of the MATLAB acoustic DoA estimation code with domain adaptation.

## Quick Start

### 1. Setup Environment

```bash
# Activate virtual environment
source ../venv/bin/activate

# Verify installation
python -c "import numpy, scipy, matplotlib, rir_generator, tqdm; print('✓ All dependencies installed')"
```

### 2. Run All Scenarios

```bash
# Run all scenarios (fig2-3-4, fig5, fig6, table)
cd src
python main.py --scenario all

# Or run individual scenarios
python main.py --scenario fig2-3-4
python main.py --scenario fig5
python main.py --scenario fig6
python main.py --scenario table
```

### 3. Generate Plots

```bash
# After running scenarios, generate all plots
python generate_plots.py
```

Results will be saved in `results/` directory.

---

## Project Structure

```
python_implementation/
├── src/
│   ├── signal_processing/
│   │   ├── stft.py              # Custom STFT with biorthogonal windowing
│   │   ├── rir.py               # RIR using rir-generator (MATLAB port)
│   │   └── signal_correlation.py # Signal processing pipeline
│   ├── algorithms/
│   │   ├── doa_estimators.py    # ML/DS, MVDR, MUSIC
│   │   └── domain_adaptation.py # E matrix computation
│   ├── utils/
│   │   ├── math_utils.py        # Matrix operations, EVD
│   │   ├── geometry.py          # Position generation, angles
│   │   └── sir_calculator.py    # SIR computation
│   ├── visualization/
│   │   ├── plotting.py          # Beta sweep, box plots, polar plots
│   │   └── room_plot.py         # 3D room visualization
│   ├── acoustic_doa.py          # Main algorithm
│   ├── main.py                  # Entry point with scenarios
│   └── generate_plots.py        # Plot generation script
├── results/                     # Output directory
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Algorithm Overview

### Domain Adaptation for DoA Estimation

**Key Idea**: Adapt between simulated (source) and real (adaptation) domains to improve DoA estimation in reverberant environments.

**Adaptation Matrix**:
$$E = \Sigma_S^{1/2} \Sigma_A^{-1/2}$$

Where:
- $\Sigma_S$ = Mean covariance from artificial (free-space) steering vectors
- $\Sigma_A$ = Mean covariance from training positions (with reverberation)

**Adapted Covariance**:
$$\hat{\Sigma}_{test} = E \Sigma_{test} E^H$$

### DoA Methods Implemented

1. **ML / Delay-and-Sum (DS)**:
   - $P_{ML}(\theta) = \mathbf{a}^H(\theta) \Sigma \mathbf{a}(\theta)$
   - Non-adaptive, optimal for AWGN

2. **MVDR (Minimum Variance Distortionless Response)**:
   - $P_{MVDR}(\theta) = \frac{1}{\mathbf{a}^H(\theta) \Sigma^{-1} \mathbf{a}(\theta)}$
   - Adaptive, nulls interferers

3. **MUSIC (Multiple Signal Classification)**:
   - $P_{MUSIC}(\theta) = \frac{1}{\mathbf{a}^H(\theta) U_N U_N^H \mathbf{a}(\theta)}$
   - Subspace-based, high resolution

---

## Scenarios

### Fig 2-3-4: Beta Sweep (No Interference)
- Varies reverberation time (Beta): 0.2 to 0.6 seconds
- Fixed positions, no interference
- Generates: 3D room plot, polar spectra, beta sweep plots

### Fig 5: Multiple Interference Positions
- 20 random interference positions
- Fixed Beta = 0.2 seconds
- Generates: Box plots comparing methods

### Fig 6: Random Interference Positions
- Random interference positions per test point
- Two simultaneous interference sources
- Generates: 3D room plot, box plots

### Table: Full Performance Evaluation
- SNR sweep: -20 to 0 dB
- SIR sweep: -20 to 0 dB
- 20 interference positions
- Generates: Performance table (LaTeX)

---

## Configuration Parameters

Key parameters (defined in `acoustic_doa.py`):

```python
n_mics = 9                    # Number of microphones
dist_between_mics = 0.12      # Microphone spacing [m]
fs = 12000                    # Sampling frequency [Hz]
win_length = 2048             # STFT window length
room_dim = [5.2, 6.2, 3.5]    # Room dimensions [m]
c = 340                       # Sound velocity [m/s]
```

---

## Key Implementation Details

### 1. STFT with Biorthogonal Windowing

**Why custom STFT?**
- MATLAB uses biorthogonal analysis/synthesis window pair
- Essential for exact frequency bin matching
- librosa.stft() would give different results

**Implementation**: `stft.py` with `biorwin()`, window caching

### 2. RIR Generation

**Using**: `rir-generator` package (direct MATLAB port)
- Identical algorithm to MATLAB's `rir_generator`
- Better accuracy than pyroomacoustics

### 3. Random Seed

Set to 10 for reproducibility:
```python
np.random.seed(10)
```

---

## Expected Output

### Console Output
```
Running scenario: fig2-3-4
Interference positions: 100%|████████| 1/1
Scenario fig2-3-4 completed in 45.23 minutes
Results saved to results/results_fig2-3-4.pkl

...

All scenarios completed!
Total time: 120.45 minutes
```

### Generated Files

**Results** (in `results/`):
- `results_fig2-3-4.pkl`
- `results_fig5.pkl`
- `results_fig6.pkl`
- `results_table.pkl`

**Figures** (in `results/`):
- `Fig2_AcousticRoom3D.jpg` - 3D room visualization
- `Fig4_DS_DoABeta.jpg` - Beta sweep for DS
- `Fig4_MVDR_DoABeta.jpg` - Beta sweep for MVDR
- `Fig4_MUSIC_DoABeta.jpg` - Beta sweep for MUSIC
- `Fig5_DoAErrAcoust.jpg` - Box plots (multiple interference)
- `Fig6_AcousticRoom3D.jpg` - 3D room with random interference
- `Fig6_DoAErrAcoust.jpg` - Box plots (random interference)

---

## Performance Notes

**Runtime Estimates** (on standard laptop):
- Fig 2-3-4: ~45-60 minutes (5 beta values, 300 test points)
- Fig 5: ~30-45 minutes (20 interference positions, 200 test points)
- Fig 6: ~45-60 minutes (300 test points, random interference)
- Table: ~45-60 minutes (SIR/SNR sweep)

**Total**: ~3-4 hours for all scenarios

**Tips**:
- Run scenarios individually for faster iteration
- Use fewer test points for quick testing (modify config)
- Results are cached as .pkl files for replotting

---

## Troubleshooting

**Issue**: Import errors
```bash
# Make sure you're in the src/ directory
cd src
python main.py
```

**Issue**: RIR errors
```bash
# Verify rir-generator installation
python -c "import rir_generator; print(rir_generator.__version__)"
```

**Issue**: Out of memory
- Reduce `k_test_points`, `k_train_points` in scenario configs
- Run scenarios sequentially rather than all at once

---

## Development

**Running tests**:
```bash
python test_basic.py          # Test imports
python test_rir.py            # Test RIR generation
python test_acoustic_doa.py   # Test full module
```

**Code structure**:
- No inline documentation (added later)
- Modular design for maintainability
- Type hints minimal (focus on functionality)
