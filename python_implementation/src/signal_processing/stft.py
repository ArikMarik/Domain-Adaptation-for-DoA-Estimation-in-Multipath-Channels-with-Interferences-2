import numpy as np
import os
import pickle

def lnshift(x, n):
    N = len(x)
    y = np.zeros(N)
    if n >= 0:
        y[n:] = x[:N-n]
    else:
        y[:N+n] = x[-n:]
    return y

def shiftcir(x, n):
    N = len(x)
    n = int(n % N)
    if n == 0:
        return x.copy()
    y = np.concatenate([x[N-n:], x[:N-n]])
    return y

def biorwin(wins, dM, dN):
    wins = wins.flatten()
    L = len(wins)
    N = L // dN
    win = np.zeros(L)
    mu = np.zeros(2*dN-1)
    mu[0] = 1
    
    for k in range(1, dM+1):
        H = np.zeros((2*dN-1, int(np.ceil((L-k+1)/dM))))
        for q in range(2*dN-1):
            h = shiftcir(wins, q*N)
            H[q, :] = h[k-1:L:dM]
        win[k-1:L:dM] = np.linalg.pinv(H) @ mu
    
    return win

def stft(x, nfft=None, dM=None, dN=None, wintype='hanning'):
    x = x.flatten()
    nx = len(x)
    
    if nfft is None:
        nfft = min(nx, 256)
    if dM is None:
        dM = int(0.5 * nfft)
        dN = 1
    
    if wintype.lower() == 'hanning':
        wins = np.hanning(nfft)
    elif wintype.lower() == 'hamming':
        wins = np.hamming(nfft)
    else:
        wins = np.hanning(nfft)
    
    win_cache_dir = os.path.join(os.path.dirname(__file__), 'Win4Stft')
    os.makedirs(win_cache_dir, exist_ok=True)
    win_file = os.path.join(win_cache_dir, f'Win_dM{dM}_dN{dN}.pkl')
    
    if os.path.exists(win_file):
        with open(win_file, 'rb') as f:
            win = pickle.load(f)
    else:
        win = biorwin(wins, dM, dN)
        with open(win_file, 'wb') as f:
            pickle.dump(win, f)
    
    ncol = int(np.fix((nx - nfft) / dM + 1))
    y = np.zeros((nfft, ncol))
    
    colindex = 1 + np.arange(ncol) * dM
    rowindex = np.arange(1, nfft+1)
    
    indices = rowindex[:, np.newaxis] + colindex[np.newaxis, :] - 1
    indices = indices.astype(int)
    indices = np.clip(indices, 0, nx-1)
    
    y = x[indices]
    y = win[:, np.newaxis] * y
    
    N = nfft // dN
    for k in range(1, dN):
        y[:N, :] = y[:N, :] + y[k*N:(k+1)*N, :]
    
    y = y[:N, :]
    y = np.fft.fft(y, axis=0)
    
    if not np.any(np.imag(x)):
        y = y[:N//2+1, :]
    
    return y
