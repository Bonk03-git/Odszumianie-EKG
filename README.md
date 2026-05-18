# Odszumianie-EKG

## Przygotowanie danych

Projekt wykorzystuje bazy danych PhysioNet. Aby uruchomić algorytmy, należy pobrać i wypakować poniższe zestawy danych do głównego folderu projektu:

1.  **MIT-BIH Arrhythmia Database**: [Pobierz tutaj](https://physionet.org/content/mitdb/1.0.0/)
    *   Wymagane pliki dla rekordów: `100`, `115`, `122`.
    *   Folder powinien nazywać się: `mit-bih-arrhythmia-database-1.0.0`
2.  **MIT-BIH Noise Stress Test Database**: [Pobierz tutaj](https://physionet.org/content/nstdb/1.0.0/)
    *   Wymagane typy szumów: `bw` (Baseline Wander), `em` (Electrode Motion), `ma` (Muscle Artifact).
    *   Folder powinien nazywać się: `mit-bih-noise-stress-test-database-1.0.0`

Struktura plików powinna wyglądać następująco:

├── mit-bih-arrhythmia-database-1.0.0/

├── mit-bih-noise-stress-test-database-1.0.0/

├── pobranie_sygnalu.py

├── metrics.py

└── algorytmy/ (savitzky-golay, transformata_falkowa, LMS)

## Przygotowanie środowiska i instalacja

#### 1. Stworzenie środowiska z odpowiednią wersją Pythona
conda create -n odszumianie-ekg python=3.11 -y

#### 2. Aktywacja środowiska
conda activate odszumianie-ekg

#### 3. Instalacja wymaganych bibliotek
pip install -r requirements.txt

## Uruchamianie projektu

Projekty należy uruchamiać z poziomu głównego katalogu za pomocą flagi `-m`.

Przykład dla algorytmu Savitzky-Golay:
```bash
python -m algorytmy.savitzky-goolay

Metryki uruchamiamy po prostu: python metryki.py
