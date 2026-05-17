import numpy as np
import pywt
import os
import matplotlib.pyplot as plt

try:
    from pobranie_sygnalu import get_processed_data
except ImportError:
    print("Nie można zaimportować 'pobranie_sygnalu'.")

def pobierz_filtry_db4():

    wavelet = pywt.Wavelet('db4')
    filtr_aproksymacji = np.array(wavelet.dec_lo) # filtr dolnoprzepustowy 
    filtr_detali = np.array(wavelet.dec_hi)       # filtr górnoprzepustowy 
    
    return filtr_aproksymacji, filtr_detali

def dekompozycja_dwt(x, h, g, poziomy_L=4):

    a_aktualne = x
    lista_detali = [] # lista detali d_i dla każdego poziomu i=1,...,L
    
    for _ in range(poziomy_L):
        # Używamy np.pad, aby sztucznie wydłużyć sygnał na brzegach (odbicie lustrzane)
        # To zapobiega drastycznym błędom na krawędziach wykresu
        pad_len = len(h) - 1
        x_padded = np.pad(a_aktualne, pad_len, mode='reflect')
        
        # Splot (wzory ze slajdu 2)
        a_splot = np.convolve(x_padded, h, mode='valid')
        d_splot = np.convolve(x_padded, g, mode='valid')
        
        # Down-sampling (wybieramy co drugą próbkę z przesunięciem naprawiającym fazę)
        a_nast = a_splot[::2]
        d_nast = d_splot[::2]
        
        lista_detali.append(d_nast)
        a_aktualne = a_nast
        
    return a_aktualne, lista_detali

def progowanie(lista_detali, N_p):
    
    # Standardowo w analizie falkowej wyznacza się je z mediany pierwszego poziomu detali (MAD).
    sigma = np.median(np.abs(lista_detali[0])) / 0.6745
    
    T = sigma * np.sqrt(2 * np.log(N_p))
    
    lista_detali_po_progowaniu = []

    for d in lista_detali:
        d_prog = np.where(np.abs(d) < T, 0, d)
        lista_detali_po_progowaniu.append(d_prog)
        
    return lista_detali_po_progowaniu

def rekonstrukcja_dwt(a_L, lista_detali_po_progowaniu, h_approx, g_detail):
  
    h_tilde = np.flip(h_approx) #zamiast wzoru korzystamy z wiedzy co on oznacza i odwracamy filtry
    g_tilde = np.flip(g_detail)
    
    #a_aktualne = a_L
    a_aktualne = np.zeros_like(a_L)  # dla szumu bw  
    # Wchodzimy w rekurencję od końca (od najwyższego poziomu L do 1)
    for d_aktualne in reversed(lista_detali_po_progowaniu):

        #Up-sampling
        a_up = np.zeros(2 * len(a_aktualne))
        a_up[0::2] = a_aktualne
        
        d_up = np.zeros(2 * len(d_aktualne))
        d_up[0::2] = d_aktualne
        
        # Wyrównanie długości wektorów
        min_len = min(len(a_up), len(d_up))
        a_up, d_up = a_up[:min_len], d_up[:min_len]
        
        #Splot up-samplowanych sygnałów z odwróconymi filtrami
        a_splot = np.convolve(a_up, h_tilde, mode='valid')
        d_splot = np.convolve(d_up, g_tilde, mode='valid')
        
        # Wynik
        a_aktualne = a_splot + d_splot
        
    return a_aktualne

def odszum_sygnal(sygnal, poziomy=8):
    """Główna funkcja spinająca cały process DWT."""
    Np = len(sygnal)
    h, g = pobierz_filtry_db4()
    
    # 1. Dekompozycja
    a_L, detale = dekompozycja_dwt(sygnal, h, g, poziomy_L=poziomy)
    
    # 2. Progowanie detali
    detale_po_progowaniu = progowanie(detale, Np)
    
    # 3. Rekonstrukcja
    sygnal_odszumiony = rekonstrukcja_dwt(a_L, detale_po_progowaniu, h, g)
    
    # Ponieważ po up-samplingu i splotach długość może nieznacznie różnić się od oryginału
    # dociągamy ją do długości wejściowej Np.
    return sygnal_odszumiony[:Np]

if __name__ == "__main__":

    rekord_id = '100'
    czysty, zaszumiony, fs = get_processed_data(rekord_id, 'em')
    
    # Bierzemy fragment sygnału do testów (np. pierwsze 10 sekund = 3600 próbek)
    probki = 3600
    zaszumiony_fragment = zaszumiony[:probki]
    czysty_fragment = czysty[:probki]
    
    # Przepuszczenie przez algorytm DWT
    poziomy = 6
    odszumiony_fragment = odszum_sygnal(zaszumiony_fragment, poziomy)
    
    # Wizualizacja
    czas = np.arange(probki) / fs
    
    plt.figure(figsize=(14, 8))
    plt.subplot(3, 1, 1)
    plt.title(f"Oryginalny EKG (Rekord {rekord_id})")
    plt.plot(czas, czysty_fragment, 'g')
    
    plt.subplot(3, 1, 2)
    plt.title("Zaszumiony EKG")
    plt.plot(czas, zaszumiony_fragment, 'r')
    
    plt.subplot(3, 1, 3)
    plt.title("Odszumiony algorytmem DWT")
    plt.plot(czas, odszumiony_fragment, 'b')
    
    plt.tight_layout()
    plt.show()