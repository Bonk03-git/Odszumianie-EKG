import wfdb
import numpy as np
import os

def load_ecg_signal(record_id, channel=0):
    """Pobiera konkretny rekord z bazy MIT-BIH Arrhythmia Database."""
    # record_id to np. '100', '115', '122'
    record = wfdb.rdrecord(f'mit-bih-arrhythmia-database-1.0.0/{record_id}')
    signal = record.p_signal[:, channel]
    return signal, record.fs

def add_noise(clean_signal, noise_type, snr_db=5):
    """
    Nakłada szum z bazy MIT-BIH Noise Stress Test Database.
    noise_type: 'ma' (muscle artifact), 'em' (electrode motion), 'bw' (baseline wander)
    """
    noise_record = wfdb.rdrecord(f'mit-bih-noise-stress-test-database-1.0.0/{noise_type}')
    noise_signal = noise_record.p_signal[:, 0] # bierzemy pierwszy kanał szumu
    
    # Dopasowanie długości szumu do sygnału (powtarzamy szum jeśli jest za krótki)
    if len(noise_signal) < len(clean_signal):
        noise_signal = np.tile(noise_signal, int(np.ceil(len(clean_signal) / len(noise_signal))))
    noise_signal = noise_signal[:len(clean_signal)]
    
    # Obliczanie mocy sygnałów dla zadanej proporcji SNR (opcjonalnie)
    # Na początek po prostu dodajemy sygnały
    combined_signal = clean_signal + noise_signal
    return combined_signal

def get_processed_data(record_id, noise_type):
    """Funkcja-skrót dla Twoich algorytmów."""
    clean, fs = load_ecg_signal(record_id)
    noisy = add_noise(clean, noise_type)
    return clean, noisy, fs

# Przykład użycia (możesz to zostawić w pliku dla testu)
if __name__ == "__main__":
    c, n, f = get_processed_data('100', 'bw')
    print(f"Załadowano rekord 100 z szumem Baseline Wander. Częstotliwość: {f}Hz")