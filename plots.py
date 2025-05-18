import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

def smooth(data, window=10):
    data = np.array(data, dtype=float)  # Konwersja na tablicę float
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Znajdź wszystkie pasujące pliki CSV
files = glob.glob("iris_lr*.csv")

plt.figure(figsize=(10, 6))

for file in files:
    df = pd.read_csv(file, encoding='latin1')

    df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]

    # Czyszczenie kolumny z błędów
    df['AvgError'] = df['AvgError'].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)

    df['AvgError'] = pd.to_numeric(df['AvgError'], errors='coerce')

    epochs = df['Epoch']
    errors = df['AvgError']

    # Wygładź dane (opcjonalnie zmień window=5,10...)
    smoothed_errors = smooth(errors.values, window=10)

    # Dostosuj długość epok po wygładzeniu
    smoothed_epochs = epochs.values[:len(smoothed_errors)]

    label = os.path.splitext(os.path.basename(file))[0]
    plt.plot(smoothed_epochs, smoothed_errors, label=label)

plt.xlabel('Epoka')
plt.ylabel('Błąd Średniokwadratowy (MSE) - Wygładzony')
plt.title('MSE a Epoki Podczas Treningu')
plt.legend()
plt.grid(True)

# Zakres osi X na sztywno od 0 do 1000
plt.xlim(0, 1000)

# Podpisy co 100 epok
plt.xticks(ticks=range(0, 1001, 100))

plt.show()


