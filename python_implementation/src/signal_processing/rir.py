import numpy as np
import rir_generator as rir

def compute_rir(source_pos, fs, receiver_positions, n_samples, reverberation_time, room_dim, order=-1):
    sound_velocity = 342.0
    room_dimension = 3
    hp_filter = True
    
    if receiver_positions.ndim == 1:
        receiver_positions = receiver_positions.reshape(1, -1)
    
    h = rir.generate(
        c=sound_velocity,
        fs=fs,
        r=receiver_positions,
        s=source_pos,
        L=room_dim,
        reverberation_time=reverberation_time,
        nsample=n_samples,
        mtype=rir.mtype.omnidirectional,
        order=order,
        dim=room_dimension,
        hp_filter=hp_filter
    )
    
    h = h.T
    
    if h.ndim == 1:
        h = h.reshape(1, -1)
    
    if h.shape[1] < n_samples:
        h_padded = np.zeros((h.shape[0], n_samples))
        h_padded[:, :h.shape[1]] = h
        h = h_padded
    elif h.shape[1] > n_samples:
        h = h[:, :n_samples]
    
    return h
