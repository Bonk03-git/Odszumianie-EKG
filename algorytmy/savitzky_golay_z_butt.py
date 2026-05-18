"""
Implementacja algorytmu savitzky-golay do odszumiania sygnału EKG. 
Ten moduł zawiera funkcje do zastosowania filtru Savitzky-Golay, 
jednak z dodatkiem wstępnego filtrowania pasmowego (filtru butterwortha), aby lepiej radzić sobie z różnymi typami szumów.
"""

import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt


def _ensure_odd(value: int) -> int:
    """Zwraca najbliższą większą nieparzystą wartość >=1."""
    v = max(1, int(value))
    return v if v % 2 == 1 else v + 1


def bandpass_filter(x, fs=360, low=0.5, high=40, order=4):
    """Prosty filtr pasmowoprzepustowy Butterwortha (zero-phase)."""
    nyq = fs / 2.0
    low_n = low / nyq
    high_n = high / nyq
    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, x)


def savitzky_golay_filter(signal: np.ndarray, fs: float = 360.0, window_length: int = None, polyorder: int = 3) -> np.ndarray:
    """Zastosowanie filtru Savitzky-Golay do wektora sygnału.
    """
    if window_length is None:
        approx = int(round(0.08 * fs))
        window_length = _ensure_odd(max(5, approx))

    if window_length <= polyorder:
        window_length = _ensure_odd(polyorder + 2)

    if polyorder < 0:
        raise ValueError("polyorder musi być >= 0")

    filtered = savgol_filter(signal, window_length, polyorder, mode='mirror')
    return filtered


def odszum_sygnal(sygnal: np.ndarray, noise_type: str = 'em') -> np.ndarray:
    """
    Odszumienie sygnału przy użyciu filtru Savitzky-Golay, z dodatkowym filtrowaniem pasmowym Butterwortha,
    """
    sygnal_bp = bandpass_filter(sygnal, fs=360, low=0.5, high=40)

    Np = len(sygnal_bp)

    if noise_type == 'bw':
        window = _ensure_odd(int(0.02 * 360)) 
        poly = 2
    elif noise_type == 'ma':
        window = _ensure_odd(int(0.04 * 360)) 
        poly = 2
    else:
        window = _ensure_odd(int(0.05 * 360))  
        poly = 2

    if window >= Np:
        window = _ensure_odd(max(3, Np - 1))

    odszumiony = savitzky_golay_filter(sygnal_bp, fs=360.0, window_length=window, polyorder=poly)
    return odszumiony[:Np]

