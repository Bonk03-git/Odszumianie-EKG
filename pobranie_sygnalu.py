import wfdb
import numpy as np
import os
from scipy.signal import butter, sosfiltfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Aplikuje filtr pasmowoprzepustowy Butterwortha bez przesunięcia fazowego.
    """
    nyq = 0.5 * fs  # Częstotliwość Nyquista
    low = lowcut / nyq
    high = highcut / nyq
    
    # Tworzymy filtr w formacie SOS (Second-Order Sections) - jest stabilniejszy numerycznie
    sos = butter(order, [low, high], btype='band', output='sos')
    
    # Filtrujemy dwukierunkowo (zero-phase filter), żeby zachować idealne pozycje pików R
    filtered_data = sosfiltfilt(sos, data)
    return filtered_data

def load_ecg_signal(record_id, channel=0):
    """Pobiera konkretny rekord z bazy MIT-BIH i wstępnie go oczyszcza."""
    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database-1.0.0/{record_id}')
    raw_signal = record.p_signal[:, channel]
    fs = record.fs
    
    # PRE-PROCESSING:
    # Odcinamy wszystko poniżej 0.5 Hz (pływanie izolinii) i powyżej 40 Hz (szum sieci i mięśniowy)
    clean_signal = butter_bandpass_filter(raw_signal, lowcut=0.5, highcut=40.0, fs=fs, order=4)
    
    return clean_signal, fs

def add_noise(clean_signal, noise_type, snr_db=5):
    """Nakłada szum z bazy MIT-BIH Noise Stress Test Database i zwraca kanał referencyjny.

    W celu zastosowania filtru LMS sygnał zaszumiony tworzony jest na podstawie
    pierwszego kanału szumu, a sygnał referencyjny pobierany jest z drugiego kanału
    tego samego rekordu.
    """
    noise_record = wfdb.rdrecord(f'mit-bih-noise-stress-test-database-1.0.0/{noise_type}')
    noise_data = noise_record.p_signal
    
    if len(noise_data) < len(clean_signal):
        repeats = int(np.ceil(len(clean_signal) / len(noise_data)))
        noise_data = np.tile(noise_data, (repeats, 1)) if noise_data.ndim == 2 else np.tile(noise_data, repeats)
    noise_data = noise_data[:len(clean_signal)]

    if noise_data.ndim == 1:
        noisy_noise = noise_data
        reference_noise = noise_data.copy()
    else:
        noisy_noise = noise_data[:, 0]
        reference_noise = noise_data[:, 1] if noise_data.shape[1] > 1 else noise_data[:, 0]

    combined_signal = clean_signal + noisy_noise
    return combined_signal, reference_noise

def get_processed_data(record_id, noise_type):
    """Funkcja-skrót algorytmów."""
    clean, fs = load_ecg_signal(record_id)
    noisy, reference = add_noise(clean, noise_type)
    return clean, noisy, reference, fs

if __name__ == "__main__":
    c, n, f = get_processed_data('100', 'bw')
    print(f"Załadowano i oczyszczono rekord 100. Dodano szum Baseline Wander. Częstotliwość: {f}Hz")