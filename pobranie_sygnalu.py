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
    """Nakłada szum z bazy MIT-BIH Noise Stress Test Database."""
    noise_record = wfdb.rdrecord(f'mit-bih-noise-stress-test-database-1.0.0/{noise_type}')
    noise_signal = noise_record.p_signal[:, 0] 
    
    # Dopasowanie długości szumu do sygnału
    if len(noise_signal) < len(clean_signal):
        noise_signal = np.tile(noise_signal, int(np.ceil(len(clean_signal) / len(noise_signal))))
    noise_signal = noise_signal[:len(clean_signal)]
    
    combined_signal = clean_signal + noise_signal
    return combined_signal

def get_processed_data(record_id, noise_type):
    """Funkcja-skrót algorytmów."""
    clean, fs = load_ecg_signal(record_id)
    noisy = add_noise(clean, noise_type)
    return clean, noisy, fs

if __name__ == "__main__":
    c, n, f = get_processed_data('100', 'bw')
    print(f"Załadowano i oczyszczono rekord 100. Dodano szum Baseline Wander. Częstotliwość: {f}Hz")