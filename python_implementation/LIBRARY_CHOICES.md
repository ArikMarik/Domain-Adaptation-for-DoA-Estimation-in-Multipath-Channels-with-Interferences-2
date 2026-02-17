# Library Choice Decisions

## STFT Implementation

**Decision**: ❌ **NOT using librosa**  
**Reason**: Custom implementation required for exact MATLAB reproduction

### Why NOT librosa?

The MATLAB code uses a **biorthogonal windowing scheme** that is essential for the algorithm:

1. **MATLAB's `stft.m`** computes a special analysis window using `biorwin()`
2. The analysis window is computed via **pseudo-inverse** of a matrix to ensure perfect reconstruction
3. **Analysis windows are cached** for efficiency (saved to disk in `Win4Stft/` directory)
4. **librosa.stft()** uses standard windowing (Hann, Hamming, etc.) without biorthogonal analysis

**Result**: Using librosa would produce **different numerical results** at the frequency bin level, causing the entire DoA estimation to differ from MATLAB.

### Our Implementation

- Custom `stft()` function with biorthogonal windowing
- Functions: `biorwin()`, `shiftcir()`, `lnshift()`
- Window caching mechanism (saved as `.pkl` files)
- **Exact match to MATLAB's STFT output**

---

## RIR (Room Impulse Response) Implementation

**Decision**: ✅ **Using rir-generator** (formerly considered pyroomacoustics)  
**Package**: `rir-generator>=0.3.0`

### Why rir-generator?

1. **Direct Python port** of MATLAB's `rir_generator` function
2. Uses **identical algorithm** to MATLAB implementation
3. Produces **numerically identical** impulse responses
4. Maintained and tested for MATLAB compatibility

### API Usage

```python
import rir_generator as rir

h = rir.generate(
    c=342.0,                              # Sound velocity
    fs=12000,                             # Sampling frequency
    r=receiver_positions,                  # (N, 3) array
    s=source_pos,                         # (3,) array
    L=room_dim,                           # [Lx, Ly, Lz]
    reverberation_time=beta,              # T60 in seconds
    nsample=2048,                         # Number of samples
    mtype=rir.mtype.omnidirectional,      # Microphone type
    order=-1,                             # Max reflection order
    dim=3,                                # 3D room
    hp_filter=True                        # High-pass filter
)
```

### Output Format

- MATLAB: Returns `(n_mics, n_samples)` array
- rir-generator: Returns `(n_samples, n_mics)` array
- **Solution**: Transpose with `h.T` to match MATLAB

---

## Summary

| Component | Library | Reason |
|-----------|---------|--------|
| **STFT** | Custom implementation | Required for biorthogonal windowing |
| **RIR** | `rir-generator` | Direct MATLAB port, exact match |
| **Matrix ops** | `numpy`, `scipy` | Standard, well-tested |
| **Visualization** | `matplotlib` | Standard Python plotting |

Both choices prioritize **exact reproduction** of MATLAB results over convenience.
