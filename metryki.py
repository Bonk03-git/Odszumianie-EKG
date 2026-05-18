import importlib.util
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

try:
    from pobranie_sygnalu import get_processed_data
    from algorytmy import transformata_falkowa as dwt
    from algorytmy import savitzky_golay as savgol  
except ImportError as e:
    print(f"Błąd krytyczny importu: {e}")
    print("Upewnij się, że uruchamiasz skrypt z głównego folderu projektu używając: python metryki.py")
    exit(1)

def oblicz_mse(czysty, odszumiony):
    """Średni błąd kwadratowy (Mean Squared Error)."""
    return np.mean((czysty - odszumiony) ** 2)

def oblicz_snr(czysty, odszumiony):
    """Stosunek sygnału do szumu (Signal-to-Noise Ratio) wyrażony w decybelach (dB)."""
    energia_sygnalu = np.sum(czysty ** 2)
    energia_szumu = np.sum((czysty - odszumiony) ** 2)
    
    if energia_szumu == 0:
        return float('inf')
        
    return 10 * np.log10(energia_sygnalu / energia_szumu)

def oblicz_prd(czysty, odszumiony):
    """
    Poprawna miara PRD dla EKG.
    Centruje sygnał czysty i odnosi błąd bezpośrednio do niego.
    """
    # 1. Obliczamy średnią tylko z czystego sygnału
    czysty_mean_val = np.mean(czysty)
    
    # 2. Odejmujemy tę samą średnią od obu sygnałów
    czysty_centered = czysty - czysty_mean_val
    odszumiony_centered = odszumiony - czysty_mean_val
    
    # 3. Licznik to po prostu suma kwadratów różnic między oryginalnymi sygnałami
    # (czysty_centered - odszumiony_centered) to to samo co (czysty - odszumiony)
    licznik = np.sum((czysty - odszumiony) ** 2)
    mianownik = np.sum(czysty_centered ** 2)
    
    if mianownik == 0:
        return float('inf')
        
    return np.sqrt(licznik / mianownik) * 100

def uruchom_analize_porownawcza(record_id='100', noise_type='bw', probki=3600, algorytm='dwt'):
    """Pobiera sygnał, odszumia go wybranym algorytmem i wyświetla obliczone metryki.

    Parametry:
        algorytm: 'dwt' dla transformacji falkowej lub 'sg'/'savgol' dla filtru Savitzky-Golay.
    """
    print("-" * 50)
    print(f"ANALIZA METRYK DLA REKORDU {record_id} (Szum: {noise_type.upper()})")
    print("-" * 50)
    
    # 1. Pobranie danych
    czysty, zaszumiony, fs = get_processed_data(record_id, noise_type)
    
    czysty_frg = czysty[:probki]
    zaszumiony_frg = zaszumiony[:probki]
    
    # Obliczamy początkowy SNR sygnału zaszumionego (przed filtracją)
    snr_przed = oblicz_snr(czysty_frg, zaszumiony_frg)
    print(f"Początkowy SNR (przed odszumianiem): {snr_przed:.2f} dB")
    
    # 2. Wywołanie wybranego algorytmu odszumiania
    if algorytm == 'dwt':
        odszumiony_frg = dwt.odszum_sygnal(zaszumiony_frg, noise_type=noise_type)
        algorytm_label = 'falkowy DWT'
    elif algorytm == 'sg':
        odszumiony_frg = savgol.odszum_sygnal(zaszumiony_frg, noise_type=noise_type)
        algorytm_label = 'Savitzky-Golay'
    else:
        raise ValueError("Nieznany algorytm. Wybierz 'dwt' lub 'sg'.")

    # 3. Obliczanie metryk po odszumieniu
    mse = oblicz_mse(czysty_frg, odszumiony_frg)
    snr_po = oblicz_snr(czysty_frg, odszumiony_frg)
    prd = oblicz_prd(czysty_frg, odszumiony_frg)
    
    # 4. Wyświetlenie wyników
    print(f"\nWyniki po zastosowaniu filtra: {algorytm_label}")
    print(f"  * MSE (im bliżej 0, tym lepiej):   {mse:.6f}")
    print(f"  * SNR (im wyższy, tym lepiej):     {snr_po:.2f} dB  (Zysk: {snr_po - snr_przed:+.2f} dB)")
    print(f"  * PRD (im bliżej 0%, tym lepiej):  {prd:.2f}%")
    print("-" * 50)

    # 5. Tworzenie wizualizacji w Matplotlib
    os_czasu = np.arange(len(czysty_frg)) / fs
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Analiza odszumiania EKG ({algorytm_label}, Rekord {record_id}, Szum: {noise_type.upper()})", fontsize=14, fontweight='bold')
    
    # Wykres 1: Sygnał oryginalny
    axs[0].plot(os_czasu, czysty_frg, color='green', linewidth=1.2, label='Oryginalny EKG')
    axs[0].set_title("1. Czysty sygnał referencyjny (Baza MIT-BIH)", fontsize=11, loc='left')
    axs[0].set_ylabel("Amplituda [mV]")
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[0].legend(loc='upper right')
    
    # Wykres 2: Sygnał zaszumiony
    axs[1].plot(os_czasu, zaszumiony_frg, color='crimson', linewidth=1.2, label=f'Zaszumiony ({noise_type.upper()})')
    axs[1].set_title(f"2. Sygnał z nałożonym szumem (Początkowy SNR: {snr_przed:.2f} dB)", fontsize=11, loc='left')
    axs[1].set_ylabel("Amplituda [mV]")
    axs[1].grid(True, linestyle='--', alpha=0.5)
    axs[1].legend(loc='upper right')
    
    # Wykres 3: Sygnał po filtracji
    axs[2].plot(os_czasu, odszumiony_frg, color='royalblue', linewidth=1.2, label=f'Odszumiony ({algorytm_label})')
    axs[2].set_title(f"3. Sygnał po odszumieniu ({algorytm_label}) (Końcowy SNR: {snr_po:.2f} dB, PRD: {prd:.2f}%)", fontsize=11, loc='left')
    axs[2].set_xlabel("Czas [s]")
    axs[2].set_ylabel("Amplituda [mV]")
    axs[2].grid(True, linestyle='--', alpha=0.5)
    axs[2].legend(loc='upper right')

    # Wykres 4: Porównanie sygnału oryginalnego i odszumionego
    axs[3].plot(
        os_czasu,
        czysty_frg,
        color='green',
        linewidth=1.2,
        label='Oryginalny EKG'
    )

    axs[3].plot(
        os_czasu,
        odszumiony_frg,
        color='royalblue',
        linewidth=1.0,
        alpha=0.8,
        label=f'Odszumiony ({algorytm_label})'
    )

    axs[3].set_title(
        "4. Porównanie sygnału oryginalnego i odszumionego",
        fontsize=11,
        loc='left'
    )

    axs[3].set_xlabel("Czas [s]")
    axs[3].set_ylabel("Amplituda [mV]")
    axs[3].grid(True, linestyle='--', alpha=0.5)
    axs[3].legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test. Opcje algorytmu: 'dwt' lub 'sg' (Savitzky-Golay). Opcje szumu: 'bw', 'ma', 'em'.
    uruchom_analize_porownawcza(record_id='100', noise_type='ma', algorytm='sg')
    

