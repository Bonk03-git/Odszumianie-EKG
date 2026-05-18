import numpy as np


def lms_filter(desired_signal: np.ndarray, reference_signal: np.ndarray, mu: float = 0.01, filter_length: int = 64) -> np.ndarray:
    """Prosta normalizowana implementacja adaptacyjnego algorytmu LMS.

    Args:
        desired_signal: sygnał wejściowy zawierający czysty sygnał + szum.
        reference_signal: sygnał referencyjny skorelowany z szumem.
        mu: współczynnik uczenia.
        filter_length: liczba wag filtru adaptacyjnego.

    Returns:
        Odszumiony sygnał uzyskany jako błąd adaptacyjnego filtru.
    """
    desired = np.asarray(desired_signal, dtype=float)
    reference = np.asarray(reference_signal, dtype=float)

    if desired.shape != reference.shape:
        min_len = min(len(desired), len(reference))
        desired = desired[:min_len]
        reference = reference[:min_len]

    N = len(desired)
    weights = np.zeros(filter_length, dtype=float)
    output = np.zeros(N, dtype=float)
    error = np.zeros(N, dtype=float)
    buffer = np.zeros(filter_length, dtype=float)

    for n in range(N):
        buffer[1:] = buffer[:-1]
        buffer[0] = reference[n]

        output[n] = np.dot(weights, buffer)
        error[n] = desired[n] - output[n]

        norm = np.dot(buffer, buffer) + 1e-8
        #weights += (mu / norm) * error[n] * buffer     # Normalizowana aktualizacja wag (NLMS)
        weights += mu * error[n] * buffer

    return error


def odszum_sygnal(zaszumiony: np.ndarray, referencyjny: np.ndarray, noise_type: str = 'em') -> np.ndarray:
    """Odszumia sygnał przy użyciu LMS z referencyjnym kanałem szumu."""
    if noise_type == 'bw':
        mu = 0.005
        filter_length = 64
    elif noise_type == 'ma':
        mu = 0.01
        filter_length = 48
    else:
        mu = 0.01
        filter_length = 48

    return lms_filter(zaszumiony, referencyjny, mu=mu, filter_length=filter_length)
