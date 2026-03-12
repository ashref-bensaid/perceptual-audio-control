import numpy as np
from scipy.ndimage import gaussian_filter1d

def apply_multiaxis_transform(M, f, Theta, f_ref=1000):
    M_mod = np.copy(M)
    eps = 1e-8
    for t in range(M.shape[1]):
        alpha, gamma, eta = Theta[t]
        # Brightness tilt
        tilt = ((f + eps) / f_ref) ** alpha
        # Bandwidth concentration
        #frame_sum = np.sum(M[:, t])
        frame_sum = np.sum(M[:, t]) + eps
        f_c = np.sum(f * M[:, t]) / frame_sum
        # if frame_sum > eps:
        #     f_c = np.sum(f * M[:, t]) / frame_sum
        # else:
        #     f_c = f_ref
        bw = np.exp(-gamma * ((f - f_c) / 1000)**2)
        # Combine
        combined = M[:, t] * tilt * (1 + gamma * (bw - 1))
        smoothed = gaussian_filter1d(combined, sigma=2)
        frame_mod = np.nan_to_num((1 - eta) * combined + eta * smoothed)

        # --- energy compensation ---
        rms_original = np.sqrt(np.mean(M[:, t] ** 2))
        rms_mod = np.sqrt(np.mean(frame_mod ** 2))

        if rms_mod > 1e-8:
            frame_mod *= rms_original / rms_mod

        M_mod[:, t] = frame_mod
    return M_mod