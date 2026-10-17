"""Shared config, loading, preprocessing and feature extraction.

To use REAL data: replace `load_recording` so it returns a (n_channels, n_samples)
float array in microvolts at FS Hz with channels ordered as CHANNELS. For .edf / .set
files, use `mne.io.read_raw_edf(...).pick(CHANNELS).resample(FS).get_data() * 1e6`.
"""
import numpy as np
from scipy import signal

# np.trapezoid was introduced in NumPy 2.0; fall back to np.trapz on older installs
_trapezoid = getattr(np, "trapezoid", np.trapz)

FS = 250
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
            "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
EPOCH_SEC = 4
STEP_SEC = 2


def load_recording(path):
    """Load .npy (channels x samples) or .csv (samples x channels, header optional)."""
    path = str(path)
    if path.endswith(".npy"):
        x = np.load(path)
    elif path.endswith(".csv"):
        x = np.genfromtxt(path, delimiter=",")
        if np.isnan(x[0]).all():
            x = x[1:]
        x = x.T
    else:
        raise ValueError("Unsupported file type. Use .npy or .csv")
    if x.ndim != 2:
        raise ValueError("Recording must be 2-D")
    if x.shape[0] != len(CHANNELS):
        if x.shape[1] == len(CHANNELS):
            x = x.T
        else:
            raise ValueError(f"Expected {len(CHANNELS)} channels, got shape {x.shape}")
    return x.astype(np.float64)


def preprocess(x):
    """Band-pass 1-45 Hz, 50 Hz notch, common average reference."""
    sos = signal.butter(4, [1, 45], btype="bandpass", fs=FS, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=1)
    b, a = signal.iirnotch(50, 30, fs=FS)
    x = signal.filtfilt(b, a, x, axis=1)
    return x - x.mean(axis=0, keepdims=True)


def make_epochs(x):
    n, step = EPOCH_SEC * FS, STEP_SEC * FS
    starts = range(0, x.shape[1] - n + 1, step)
    return np.stack([x[:, s:s + n] for s in starts])  # (n_epochs, ch, samples)


def band_power(epochs):
    """Absolute band power, shape (n_epochs, n_channels, n_bands)."""
    f, pxx = signal.welch(epochs, fs=FS, nperseg=FS * 2, axis=-1)
    out = []
    for lo, hi in BANDS.values():
        m = (f >= lo) & (f < hi)
        out.append(_trapezoid(pxx[..., m], f[m], axis=-1))
    return np.stack(out, axis=-1)


def feature_names():
    names = [f"{b}_{c}" for b in BANDS for c in CHANNELS]
    return names + ["alpha_asym_F4-F3", "alpha_asym_F8-F7", "theta_beta_ratio_frontal"]


def extract_features(epochs):
    bp = band_power(epochs)                       # (E, C, B)
    rel = bp / bp.sum(axis=-1, keepdims=True)     # relative power
    feats = [rel[:, :, i] for i in range(len(BANDS))]
    feats = np.concatenate(feats, axis=1)         # (E, B*C) ordered band-major
    a = list(BANDS).index("alpha")
    t = list(BANDS).index("theta")
    b = list(BANDS).index("beta")
    ix = {c: i for i, c in enumerate(CHANNELS)}
    asym1 = np.log(bp[:, ix["F4"], a]) - np.log(bp[:, ix["F3"], a])
    asym2 = np.log(bp[:, ix["F8"], a]) - np.log(bp[:, ix["F7"], a])
    frontal = [ix[c] for c in ("Fp1", "Fp2", "F3", "F4", "Fz", "F7", "F8")]
    tbr = np.log(bp[:, frontal, t].mean(1)) - np.log(bp[:, frontal, b].mean(1))
    return np.column_stack([feats, asym1, asym2, tbr])


def recording_to_features(x):
    return extract_features(make_epochs(preprocess(x)))
