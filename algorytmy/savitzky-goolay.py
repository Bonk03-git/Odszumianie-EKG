from pobranie_sygnalu import get_processed_data
import matplotlib.pyplot as plt
import numpy as np

clean, noisy, fs = get_processed_data('115', 'em')

time_axis = np.arange(len(clean)) / fs

# Wykres
plt.plot(time_axis, noisy, label='Zaszumiony', linewidth=1)
plt.plot(time_axis, clean, label='Czysty', alpha=0.8, linewidth=1)
plt.xlim(0, 3) 
plt.grid(True, alpha=0.3)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda [mV]')
plt.legend()
plt.show()


