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
├── signal_loader.py
├── metrics.py
└── algorytmy/ (savitzky-golay, transformata_falkowa, LMS)