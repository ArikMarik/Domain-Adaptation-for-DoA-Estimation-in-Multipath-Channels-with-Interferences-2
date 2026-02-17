# How to Run - Step by Step

## ✅ Ready to Execute!

The code is tested and working. Follow these steps:

---

## Step 1: Activate Virtual Environment

```bash
cd /home/user/hw/audioHW/final_project2
source venv/bin/activate
```

You should see `(venv)` in your prompt.

---

## Step 2: Navigate to Source Directory

```bash
cd python_implementation/src
```

---

## Step 3: Run a Scenario

### Option A: Quick Test (Recommended First Run)
Test with the smallest scenario (~30-45 minutes):
```bash
python main.py --scenario fig5
```

### Option B: Individual Scenarios
```bash
# Beta sweep (no interference) - 45-60 min
python main.py --scenario fig2-3-4

# Multiple interference positions - 30-45 min  
python main.py --scenario fig5

# Random interference - 45-60 min
python main.py --scenario fig6

# Full table - 45-60 min
python main.py --scenario table
```

### Option C: Run Everything (~3-4 hours)
```bash
python main.py --scenario all
```

---

## Step 4: Generate Plots

After scenarios complete:
```bash
python generate_plots.py
```

---

## Step 5: View Results

Results are saved in:
```bash
cd ../results
ls -lh
```

You should see:
- `results_*.pkl` - Raw results (for replotting)
- `Fig*.jpg` / `Fig*.png` - Generated figures

---

## Expected Console Output

```
============================================================
Running scenario: fig5
============================================================

Interference positions: 100%|████████████| 20/20 [30:15<00:00, 91.23s/it]

Scenario fig5 completed in 30.25 minutes
Results saved to results/results_fig5.pkl

============================================================
All scenarios completed!
Total time: 30.25 minutes
============================================================
```

---

## Monitoring Progress

The code uses `tqdm` for progress bars:
- Shows which interference position is being processed
- Shows estimated time remaining
- Updates in real-time

---

## If Something Goes Wrong

### "Singular matrix" error
✅ **Fixed!** This was resolved on 2026-02-16. Make sure you have the latest code.

### Import errors
```bash
# Make sure you're in the right directory
pwd  # Should show: .../final_project2/python_implementation/src

# If not, navigate there
cd /home/user/hw/audioHW/final_project2/python_implementation/src
```

### Out of memory
Reduce test points in `main.py`:
```python
'k_test_points': 50,    # Instead of 200-300
'k_train_points': 25,   # Instead of 100
```

### Takes too long
Start with `fig5` which is the quickest scenario.

---

## Verification Before Running

Quick sanity check (takes ~10 seconds):
```bash
python test_run_small.py
```

Should output:
```
✓ Test run completed successfully!
```

---

## After Completion

### Check Results
```bash
ls ../results/
```

### View a Figure
```bash
# On WSL with Windows
explorer.exe ../results/Fig5_DoAErrAcoust.jpg

# Or copy to Windows location
cp ../results/*.jpg /mnt/c/Users/YourName/Desktop/
```

### Load Results in Python
```python
import pickle

with open('../results/results_fig5.pkl', 'rb') as f:
    results = pickle.load(f)

print(results.keys())
```

---

## Full Example Session

```bash
# Start from project root
cd /home/user/hw/audioHW/final_project2

# Activate venv
source venv/bin/activate

# Go to source
cd python_implementation/src

# Quick test
python test_run_small.py

# Run scenario
python main.py --scenario fig5

# Generate plots
python generate_plots.py

# View results
ls -lh ../results/
```

---

## Performance Notes

| Scenario | Points | Time | Memory |
|----------|--------|------|--------|
| fig5 | 200 test | ~30-45 min | ~2-3 GB |
| fig2-3-4 | 300 test, 5 beta | ~45-60 min | ~3-4 GB |
| fig6 | 300 test | ~45-60 min | ~3-4 GB |
| table | 200 test, sweeps | ~45-60 min | ~2-3 GB |
| **all** | **Combined** | **~3-4 hours** | **~4 GB** |

---

## Tips

1. **Run overnight**: `--scenario all` takes 3-4 hours
2. **Start small**: Test with `test_run_small.py` first
3. **Monitor progress**: Progress bars show ETA
4. **Save results**: `.pkl` files let you replot without recomputing
5. **One at a time**: Run scenarios individually for faster iteration

---

**Ready to go! Start with:**
```bash
cd /home/user/hw/audioHW/final_project2/python_implementation/src
python main.py --scenario fig5
```

Good luck! 🚀
