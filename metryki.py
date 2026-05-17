import numpy as np
import os
import sys
import matplotlib.pyplot as plt

try:
    from pobranie_sygnalu import get_processed_data
except ImportError:
    print("Błąd: Nie można zaimportować 'pobranie_sygnalu'")

# Bezpieczny import Twojego algorytmu z podfolderu 'algorytmy'
sys.path.append(os.path.join(os.path.dirname(__file__), 'algorytmy'))
try:
    import transformata_falkowa as dwt
except ImportError:
    print("Błąd: Nie można znaleźć skryptu falkowego w folderze 'algorytmy'.")

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
    Znormalizowana wersja PRD (często oznaczana jako PRD_1).
    Odejmuje średnią wartość (składową stałą), dzięki czemu przesunięcie 
    linii bazowej nie fałszuje wyniku zniekształcenia morfologii fal.
    """
    czysty_mean = czysty - np.mean(czysty)
    odszumiony_mean = odszumiony - np.mean(odszumiony)
    
    licznik = np.sum((czysty_mean - odszumiony_mean) ** 2)
    mianownik = np.sum(czysty_mean ** 2)
    
    if mianownik == 0:
        return float('inf')
        
    return np.sqrt(licznik / mianownik) * 100

def uruchom_analize_porownawcza(record_id='100', noise_type='bw', probki=3600, algorytm='dwt'):
    """Pobiera sygnał, odszumia go i wyświetla obliczone metryki."""
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
    
    # 2. Wywołanie Twojego algorytmu falkowego
    # Automatycznie przekazujemy noise_type, aby algorytm wiedział czy zerować aproksymację
    #try:
   #     odszumiony_frg = dwt.odszum_sygnal(zaszumiony_frg, poziomy=6, noise_type=noise_type)
  #  except AttributeError:
 #       print("[⚠️] Uwaga: Twoja funkcja odszum_sygnal nie obsługuje jeszcze parametru noise_type. Uruchamiam wersję domyślną.")
    #       odszumiony_frg = dwt.odszum_sygnal(zaszumiony_frg, poziomy=6)
    if algorytm == 'dwt':
        odszumiony_frg = dwt.odszum_sygnal(zaszumiony_frg, noise_type=noise_type)

    # 3. Obliczanie metryk po odszumieniu
    mse = oblicz_mse(czysty_frg, odszumiony_frg)
    snr_po = oblicz_snr(czysty_frg, odszumiony_frg)
    prd = oblicz_prd(czysty_frg, odszumiony_frg)
    
    # 4. Wyświetlenie wyników
    print(f"\nWyniki po zastosowaniu transformaty falkowej DWT:")
    print(f"  * MSE (im bliżej 0, tym lepiej):   {mse:.6f}")
    print(f"  * SNR (im wyższy, tym lepiej):     {snr_po:.2f} dB  (Zysk: {snr_po - snr_przed:+.2f} dB)")
    print(f"  * PRD (im bliżej 0%, tym lepiej):  {prd:.2f}%")
    print("-" * 50)

    # 5. Tworzenie wizualizacji w Matplotlib
    os_czasu = np.arange(len(czysty_frg)) / fs
    
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Analiza odszumiania EKG falką db4 (Rekord {record_id}, Szum: {noise_type.upper()})", fontsize=14, fontweight='bold')
    
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
    
    # Wykres 3: Sygnał po filtracji DWT
    axs[2].plot(os_czasu, odszumiony_frg, color='royalblue', linewidth=1.2, label='Odszumiony DWT')
    axs[2].set_title(f"3. Sygnał po odszumieniu falkowym (Końcowy SNR: {snr_po:.2f} dB, PRD: {prd:.2f}%)", fontsize=11, loc='left')
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
        label='Odszumiony DWT'
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
    # Test dla szumu typu Baseline Wander (Pływanie izolinii)
    uruchom_analize_porownawcza(record_id='100', noise_type='em', algorytm='dwt')
    
    # Możesz odkomentować poniższą linię, aby od razu przetestować szum mięśniowy (Muscle Artifact)
    # uruchom_analize_porownawcza(record_id='100', noise_type='ma')