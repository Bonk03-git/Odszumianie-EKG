import numpy as np
import pywt
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

try:
    from pobranie_sygnalu import get_processed_data
except ImportError:
    print("Nie można zaimportować 'pobranie_sygnalu'.")

def bandpass_filter(x, fs=360, low=0.5, high=40):
    b, a = butter(4, [low/(fs/2), high/(fs/2)], btype='band')
    return filtfilt(b, a, x)

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

def progowanie(lista_detali, N_p, noise_type):

    lista_detali_po_progowaniu = []

    # 🔥 GLOBALNE sigma tylko dla BW (jak w Twojej starej wersji)
    sigma_global = np.median(np.abs(lista_detali[0])) / 0.6745
    T_global = sigma_global * np.sqrt(2 * np.log(N_p))

    for i, d in enumerate(lista_detali):
        poziom = i + 1

        # =========================
        # BW — IDENTYCZNE JAK WCZEŚNIEJ
        # =========================
        if noise_type == 'bw':
            T = T_global
            d_prog = np.where(np.abs(d) < T, 0, d)

        # =========================
        # MA — lokalne sigma (lepsze)
        # =========================
        elif noise_type == 'ma':

            sigma = np.median(np.abs(d)) / 0.6745
            T_baza = sigma * np.sqrt(2 * np.log(len(d)))

            if poziom == 1:
                T = 0.8 * T_baza
            elif poziom == 2:
                T = 0.6 * T_baza
            elif poziom == 3:
                T = 0.35 * T_baza
            elif poziom == 4:
                T = 0.2 * T_baza
            else:
                T = 0

            res = np.sign(d) * (np.abs(d) - T)
            d_prog = np.where(np.abs(d) < T, 0, res)

        # =========================
        # EM — jeszcze łagodniejsze
        # =========================
        else:

            sigma = np.median(np.abs(d)) / 0.6745
            T_baza = sigma * np.sqrt(2 * np.log(len(d)))

            if poziom == 1:
                T = 0.7 * T_baza
            elif poziom == 2:
                T = 0.5 * T_baza
            elif poziom == 3:
                T = 0.25 * T_baza
            elif poziom == 4:
                T = 0.1 * T_baza
            else:
                T = 0

            res = np.sign(d) * (np.abs(d) - T)
            d_prog = np.where(np.abs(d) < T, 0, res)

        lista_detali_po_progowaniu.append(d_prog)

    return lista_detali_po_progowaniu

def rekonstrukcja_dwt(a_L, lista_detali_po_progowaniu, h_approx, g_detail):
    h_tilde = np.flip(h_approx) 
    g_tilde = np.flip(g_detail)
    
    # POPRAWKA 1: Przywracamy zmienną wejściową. 
    # Teraz rekonstrukcja uszanuje to, co przygotowałeś w odszum_sygnal!
    a_aktualne = a_L
    
    # Wchodzimy w rekurencję od końca (od najwyższego poziomu L do 1)
    for d_aktualne in reversed(lista_detali_po_progowaniu):

        # Up-sampling
        a_up = np.zeros(2 * len(a_aktualne))
        a_up[0::2] = a_aktualne
        
        d_up = np.zeros(2 * len(d_aktualne))
        d_up[0::2] = d_aktualne
        
        # Wyrównanie długości wektorów
        min_len = min(len(a_up), len(d_up))
        a_up, d_up = a_up[:min_len], d_up[:min_len]
        
        # aby filtry rekonstrukcji poprawnie odbudowały pasma bez ucinania próbek
        a_splot = np.convolve(a_up, h_tilde, mode='valid')
        d_splot = np.convolve(d_up, g_tilde, mode='valid')
        
        # Wynik
        a_aktualne = a_splot + d_splot
        
    return a_aktualne

def odszum_sygnal(sygnal, noise_type):

    sygnal = bandpass_filter(sygnal, fs=360, low=0.5, high=40)

    Np = len(sygnal)
    h, g = pobierz_filtry_db4()
    
    if noise_type == 'bw':
        poziomy = 6
        a_L, detale = dekompozycja_dwt(sygnal, h, g, poziomy_L=poziomy)
        detale_po_progowaniu = progowanie(detale, Np, noise_type=noise_type)
        a_L_input = np.zeros_like(a_L)
        
    elif noise_type == 'ma':
        poziomy = 5 
        a_L, detale = dekompozycja_dwt(sygnal, h, g, poziomy_L=poziomy)
        detale_po_progowaniu = progowanie(detale, Np, noise_type=noise_type)
        a_L_input = a_L  # dla szumu mięśniowego zachowujemy aproksymację, bo zawiera ważne informacje o kształcie fali QRS
        
    elif noise_type == 'em':
        poziomy = 5
        a_L, detale = dekompozycja_dwt(sygnal, h, g, poziomy_L=poziomy)
        detale_po_progowaniu = progowanie(detale, Np, noise_type=noise_type)
        a_L_input = np.zeros_like(a_L)
        
    sygnal_odszumiony = rekonstrukcja_dwt(a_L_input, detale_po_progowaniu, h, g)
    return sygnal_odszumiony[:Np]