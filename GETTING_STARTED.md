git 
# Getting Started - Python Implementation

## ✅ Implementation Complete! (Bugs Fixed 2026-02-16)

All phases of the Python implementation are complete and ready to run.

**Latest Update**: Fixed singular matrix error - see `BUGFIX_SUMMARY.md` for details.

---

## 📦 What's Included

### Core Implementation
- ✅ STFT with biorthogonal windowing (custom, matches MATLAB exactly)
- ✅ RIR generation using `rir-generator` (direct MATLAB port)
- ✅ Signal correlation with reverberation and noise
- ✅ DoA estimators: ML/DS, MVDR, MUSIC
- ✅ Domain adaptation: E matrix computation
- ✅ All visualization functions
- ✅ 4 scenario configurations
- ✅ Automated plot generation

### Files Created
- **1011 lines** of production code
- **13 core modules** in `src/`
- **3 test scripts** for verification
- **3 documentation files**

---

## 🚀 Quick Start (3 Commands)

```bash
# 1. Activate virtual environment
cd /home/user/hw/audioHW/final_project2
source venv/bin/activate

# 2. Run a scenario (start with smallest for testing)
cd python_implementation/src
python main.py --scenario fig5

# 3. Generate plots
python generate_plots.py
```

That's it! Results will be in `python_implementation/results/`.

---

## 📊 Scenarios Explained

### Option 1: Test Run (Fastest ~30-45 min)
```bash
python main.py --scenario fig5
```
- 20 interference positions
- 200 test points
- Fixed Beta = 0.2 seconds
- **Generates**: Box plots comparing methods

### Option 2: Beta Sweep (~45-60 min)
```bash
python main.py --scenario fig2-3-4
```
- Tests 5 different reverberation times
- 300 test points
- **Generates**: 3D room plot, beta sweep plots

### Option 3: Random Interference (~45-60 min)
```bash
python main.py --scenario fig6
```
- Random interference per test point
- 2 simultaneous interferers
- **Generates**: 3D room plot, box plots

### Option 4: Full Table (~45-60 min)
```bash
python main.py --scenario table
```
- SNR and SIR sweep
- 20 interference positions
- **Generates**: Performance data

### Option 5: Run Everything (~3-4 hours)
```bash
python main.py --scenario all
```
- Runs all 4 scenarios sequentially
- **Recommended**: Run overnight or during lunch

---

## 📁 Directory Structure

```
final_project2/
├── venv/                              # Virtual environment
├── python_implementation/
│   ├── src/                          # Source code
│   │   ├── signal_processing/        # STFT, RIR, signal correlation
│   │   ├── algorithms/               # DoA estimators, domain adaptation
│   │   ├── utils/                    # Math, geometry, SIR
│   │   ├── visualization/            # Plotting functions
│   │   ├── acoustic_doa.py           # Main algorithm
│   │   ├── main.py                   # Entry point ⭐
│   │   └── generate_plots.py         # Plot generator ⭐
│   ├── results/                      # Output directory (auto-created)
│   ├── requirements.txt              # Dependencies
│   └── README.md                     # Detailed documentation
├── matlab_code/                      # Original MATLAB (reference)
├── IMPLEMENTATION_PLAN.md            # Full implementation plan
└── GETTING_STARTED.md                # This file
```

---

## 🔧 Key Components Explained

### 1. ML/DS Beamformer
**Q**: Is ML the same as Delay-and-Sum?  
**A**: YES! For narrowband signals with AWGN, they're mathematically identical:
$$P_{ML}(\theta) = P_{DS}(\theta) = \mathbf{a}^H(\theta) \Sigma \mathbf{a}(\theta)$$

### 2. Domain Adaptation
Transforms covariance between artificial (free-space) and real (reverberant) domains:
$$E = \Sigma_S^{1/2} \Sigma_A^{-1/2}$$
$$\hat{\Sigma}_{test} = E \Sigma_{test} E^H$$

### 3. Why Custom STFT?
MATLAB uses **biorthogonal windowing** for perfect reconstruction:
- Analysis window computed via pseudo-inverse
- Windows cached to disk
- librosa.stft() lacks this, would give different results

### 4. Why rir-generator?
Direct Python port of MATLAB's `rir_generator`:
- Identical algorithm
- Numerically equivalent results
- Better than pyroomacoustics for reproduction

---

## 🎯 Expected Results

### Console Output
```
Running scenario: fig5

Interference positions: 100%|████████████| 20/20 [30:15<00:00, 91.23s/it]

Scenario fig5 completed in 30.25 minutes
Results saved to results/results_fig5.pkl

✓ All scenarios completed!
```

### Generated Files

**In `results/` directory**:
- `results_*.pkl` - Raw results (loadable for replotting)
- `Fig2_AcousticRoom3D.jpg` - 3D room layout
- `Fig4_DS_DoABeta.jpg` - DS performance vs Beta
- `Fig4_MVDR_DoABeta.jpg` - MVDR performance vs Beta
- `Fig4_MUSIC_DoABeta.jpg` - MUSIC performance vs Beta
- `Fig5_DoAErrAcoust.jpg` - Box plots (multiple interference)
- `Fig6_AcousticRoom3D.jpg` - 3D room (random interference)
- `Fig6_DoAErrAcoust.jpg` - Box plots (random interference)

---

## ⚡ Performance Tips

### Speed Up Runs
Edit scenario configs in `main.py`:
```python
'k_test_points': 50,    # Instead of 300 (6x faster)
'k_train_points': 25,   # Instead of 100 (4x faster)
```

### Memory Issues
Run scenarios individually instead of `--scenario all`.

### Parallel Execution
Run multiple scenarios in different terminals:
```bash
# Terminal 1
python main.py --scenario fig2-3-4

# Terminal 2 (different machine or wait)
python main.py --scenario fig5
```

---

## 🧪 Verification

### Test Everything Works
```bash
cd python_implementation
python test_basic.py          # Test imports (5 sec)
python test_rir.py            # Test RIR (5 sec)
python test_acoustic_doa.py   # Test main module (5 sec)
```

All should print `✓` marks.

### Quick Smoke Test
Modify `fig5` config to use fewer points:
```python
# In main.py, change fig5 config:
'k_test_points': 10,    # Instead of 200
'k_train_points': 5,    # Instead of 100
```

Then run:
```bash
python main.py --scenario fig5  # Should finish in ~2-3 minutes
```

---

## 🐛 Troubleshooting

### "No module named 'rir_generator'"
```bash
source ../venv/bin/activate
pip install rir-generator
```

### "ModuleNotFoundError: No module named 'signal_processing'"
```bash
# Make sure you're in src/ directory
cd python_implementation/src
python main.py
```

### RIR shape mismatch
Update to latest code - fixed in `rir.py` with transpose.

### Plots not generating
```bash
# Make sure results exist first
ls results/*.pkl

# Then generate plots
python generate_plots.py
```

---

## 📚 Documentation

- **README.md** - Detailed usage, algorithm overview
- **LIBRARY_CHOICES.md** - Why custom STFT, why rir-generator
- **IMPLEMENTATION_PLAN.md** - Complete development plan (893 lines!)

---

## ✅ Final Checklist

Before running:
- [ ] Virtual environment activated
- [ ] In `python_implementation/src/` directory
- [ ] Have 3-4 hours available (or start with quick scenario)
- [ ] At least 4GB RAM available

After running:
- [ ] Check `results/` directory for `.pkl` files
- [ ] Run `generate_plots.py` to create figures
- [ ] Check `results/` for `.jpg`/`.png` files

---

## 🎉 Ready to Run!

Start with the smallest scenario to test:
```bash
cd /home/user/hw/audioHW/final_project2/python_implementation/src
python main.py --scenario fig5
```

Good luck! 🚀
