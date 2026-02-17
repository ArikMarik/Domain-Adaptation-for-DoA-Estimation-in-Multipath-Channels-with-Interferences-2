# Plot Improvements - 2026-02-16

## Changes Made

### 1. ✅ Fixed Y-Axis Scaling
**Problem**: All Fig4 plots appeared empty because y-axis was [0, 10] but data was 14-33°

**Fixed**:
```python
# Before: Fixed range
ax.set_ylim([0, 10])

# After: Auto-scale to data
y_max = max(np.max(vec_adapted), np.max(vec_standard)) * 1.1
y_min = min(np.min(vec_adapted), np.min(vec_standard)) * 0.9
ax.set_ylim([max(0, y_min), y_max])
```

---

### 2. ✅ Changed Label Format
**Problem**: Labels said "Our DS" instead of desired format

**Changed**:
- ❌ Before: "Our DS", "Our MVDR", "Our MUSIC"
- ✅ After: "DS+DA", "MVDR+DA", "MUSIC+DA"

**Affected plots**:
- Beta sweep plots (Fig4)
- Polar spectrum plots (Fig3, Fig5, Fig6 spectra)
- Box plots (Fig5, Fig6)

---

### 3. ✅ Added Spectrum Plots for All Beamformers

**For fig2-3-4** (at beta=0.4, test point 14):
- `Fig3_DS_Spectrum.jpg/png` - DS spectrum (adapted vs standard)
- `Fig3_MVDR_Spectrum.jpg/png` - MVDR spectrum (adapted vs standard)
- `Fig3_MUSIC_Spectrum.jpg/png` - MUSIC spectrum (adapted vs standard)
- `Fig3_Components_Spectrum.jpg/png` - E, Σ_Tr^-0.5, Σ_Art^0.5

**For fig5** (at interference pos 0, test point 50):
- `Fig5_DS_Spectrum.jpg/png`
- `Fig5_MVDR_Spectrum.jpg/png`
- `Fig5_MUSIC_Spectrum.jpg/png`

**For fig6** (at test point 50):
- `Fig6_DS_Spectrum.jpg/png`
- `Fig6_MVDR_Spectrum.jpg/png`
- `Fig6_MUSIC_Spectrum.jpg/png`

---

## How to Generate These New Plots

### For Fig3 (requires re-run):
```bash
cd /home/user/hw/audioHW/final_project2/python_implementation/src

# Re-run fig2-3-4 to generate spectrum plots
../../venv/bin/python main.py --scenario fig2-3-4

# This will now create:
# - Fig3_DS_Spectrum.jpg
# - Fig3_MVDR_Spectrum.jpg
# - Fig3_MUSIC_Spectrum.jpg
# - Fig3_Components_Spectrum.jpg
```

### For Fig5 and Fig6 Spectra (requires re-run):
```bash
# Re-run fig5
../../venv/bin/python main.py --scenario fig5
# Creates: Fig5_DS/MVDR/MUSIC_Spectrum.jpg

# Re-run fig6
../../venv/bin/python main.py --scenario fig6
# Creates: Fig6_DS/MVDR/MUSIC_Spectrum.jpg
```

### For Updated Labels (already done):
```bash
# Just regenerate from existing results
../../venv/bin/python generate_plots.py

# This updates:
# - Fig4 plots (all 3)
# - Fig5 box plots
# - Fig6 box plots
```

---

## Complete File List

After running all scenarios, you'll have:

**Fig2-3-4 outputs**:
- `Fig2_AcousticRoom3D.jpg/png` - 3D room layout
- `Fig3_DS_Spectrum.jpg/png` - DS polar spectrum
- `Fig3_MVDR_Spectrum.jpg/png` - MVDR polar spectrum
- `Fig3_MUSIC_Spectrum.jpg/png` - MUSIC polar spectrum
- `Fig3_Components_Spectrum.jpg/png` - E matrix components
- `Fig4_DS_DoABeta.jpg/png` - DS error vs beta
- `Fig4_MVDR_DoABeta.jpg/png` - MVDR error vs beta
- `Fig4_MUSIC_DoABeta.jpg/png` - MUSIC error vs beta

**Fig5 outputs**:
- `Fig5_DS_Spectrum.jpg/png` - DS polar spectrum (representative)
- `Fig5_MVDR_Spectrum.jpg/png` - MVDR polar spectrum (representative)
- `Fig5_MUSIC_Spectrum.jpg/png` - MUSIC polar spectrum (representative)
- `Fig5_DoAErrAcoust.jpg/png` - Box plots

**Fig6 outputs**:
- `Fig6_AcousticRoom3D.jpg/png` - 3D room with random interference
- `Fig6_DS_Spectrum.jpg/png` - DS polar spectrum (representative)
- `Fig6_MVDR_Spectrum.jpg/png` - MVDR polar spectrum (representative)
- `Fig6_MUSIC_Spectrum.jpg/png` - MUSIC polar spectrum (representative)
- `Fig6_DoAErrAcoust.jpg/png` - Box plots

**Total**: 23 plot files (8 for Fig2-3-4, 8 for Fig5, 7 for Fig6)

---

## Summary of Changes

| Change | Status | Details |
|--------|--------|---------|
| Fix y-axis scaling | ✅ Done | Auto-scales to data range |
| Update labels | ✅ Done | "DS+DA" instead of "Our DS" |
| Add MVDR spectra | ✅ Done | Fig3_MVDR_Spectrum |
| Add MUSIC spectra | ✅ Done | Fig3_MUSIC_Spectrum |
| Add Fig5 spectra | ✅ Done | All 3 beamformers |
| Add Fig6 spectra | ✅ Done | All 3 beamformers |
| Regenerate existing | ✅ Done | Fig4, Fig5, Fig6 box plots |

---

## What's Already Updated

✅ **Without re-running** (just from existing data):
- Fig4_DS/MVDR/MUSIC_DoABeta - now show correct y-axis and labels
- Fig5_DoAErrAcoust - updated labels  
- Fig6_DoAErrAcoust - updated labels

⏳ **Requires re-running scenarios**:
- Fig3 spectrum plots (all beamformers)
- Fig5 spectrum plots (all beamformers)
- Fig6 spectrum plots (all beamformers)

---

## Quick Commands

```bash
cd /home/user/hw/audioHW/final_project2/python_implementation/src

# Re-run to get spectrum plots
../../venv/bin/python main.py --scenario fig2-3-4  # ~45-60 min
../../venv/bin/python main.py --scenario fig5      # ~30-45 min
../../venv/bin/python main.py --scenario fig6      # ~45-60 min

# Or run all at once
../../venv/bin/python main.py --scenario all       # ~3-4 hours
```

The spectrum plots will be generated automatically during the run!
