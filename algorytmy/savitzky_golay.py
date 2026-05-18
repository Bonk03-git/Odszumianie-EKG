"""
Prosta implementacja filtru savitzky-golay do odszumiania sygnału EKG, z wykorzystaniem funkcji `savgol_filter` z biblioteki SciPy.
"""

import numpy as np
from scipy.signal import savgol_filter

def _ensure_odd(value: int) -> int:
    """Zwraca najbliższą większą nieparzystą wartość >=1."""
    v = max(1, int(value))
    return v if v % 2 == 1 else v + 1


def savitzky_golay_filter(signal: np.ndarray, fs: float = 360.0, window_length: int = None, polyorder: int = 3) -> np.ndarray:
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
    Odszumia sygnał przy użyciu wyłącznie czystego filtru Savitzky-Golay.
    """
    Np = len(sygnal)

    # Parametry dobrane pod konkretne typy szumów dla samego filtra SG
    if noise_type == 'bw':
        window = _ensure_odd(int(0.06 * 360))  # ok. 21 próbek
        poly = 4
    elif noise_type == 'ma':
        window = _ensure_odd(int(0.055 * 360)) # ok. 19 próbek
        poly = 4
    else:
        window = _ensure_odd(int(0.04 * 360))  # ok. 15 próbek
        poly = 3

    if window >= Np:
        window = _ensure_odd(max(3, Np - 1))

    odszumiony = savitzky_golay_filter(sygnal, fs=360.0, window_length=window, polyorder=poly)
    return odszumiony[:Np]