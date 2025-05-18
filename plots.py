import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


def smooth(data, window=10):
    data = np.array(data, dtype=float)
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_smoothed_mse(files, title, x_limit=1000, window=10):
    plt.figure(figsize=(10, 6))

    for file in files:
        df = pd.read_csv(file, encoding='latin1')

        # Upewnij się, że dane są liczbowe
        df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]
        df['AvgError'] = df['AvgError'].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
        df['AvgError'] = pd.to_numeric(df['AvgError'], errors='coerce')

        epochs = df['Epoch']
        errors = df['AvgError']

        smoothed_errors = smooth(errors.values, window=window)
        smoothed_epochs = epochs.values[:len(smoothed_errors)]

        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(smoothed_epochs, smoothed_errors, label=label)

    plt.xlabel('Epoka')
    plt.ylabel('Błąd Średniokwadratowy (MSE) - Wygładzony')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, x_limit)

    # Poprawione ustawienie znaczników osi X
    tick_interval = 100 if x_limit <= 1000 else 500
    plt.xticks(np.arange(0, x_limit + 1, tick_interval))


def plot_smoothed_mse_comparison(file_on, file_off, title, x_limit=3000, window=20):
    plt.figure(figsize=(10, 6))

    for file in [file_on, file_off]:
        df = pd.read_csv(file, encoding='latin1')

        # Upewnij się, że dane są liczbowe
        df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]
        df['AvgError'] = df['AvgError'].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
        df['AvgError'] = pd.to_numeric(df['AvgError'], errors='coerce')

        epochs = df['Epoch']
        errors = df['AvgError']

        smoothed_errors = smooth(errors.values, window=window)
        smoothed_epochs = epochs.values[:len(smoothed_errors)]

        label = "Bias ON" if "bias_on" in file else "Bias OFF"
        plt.plot(smoothed_epochs, smoothed_errors, label=label)

    plt.xlabel('Epoka')
    plt.ylabel('Błąd Średniokwadratowy (MSE) - Wygładzony')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, x_limit)

    # Poprawione ustawienie znaczników osi X
    tick_interval = 100 if x_limit <= 1000 else 500
    plt.xticks(np.arange(0, x_limit + 1, tick_interval))

    plt.show()


# === WYKRES 1: IRYSY ===
iris_files = glob.glob("Iris_lr*.csv")
plot_smoothed_mse(iris_files, title="MSE a Epoki Podczas Treningu (Irys)", x_limit=1000, window=10)
plt.show()

# === WYKRES 2: AUTOENKODER ===
ae_files = glob.glob("Autoencoder_*.csv")
plot_smoothed_mse(ae_files, title="MSE a Epoki Podczas Treningu (Autoenkoder)", x_limit=3000, window=20)
plt.show()

# === WYKRES 3: PORÓWNANIE AUTOENKODERA BIAS ON VS OFF ===
file_on = "lr0.5_mom0.5_biasTrue.csv"
file_off = "lr0.5_mom0.5_biasFalse.csv"

plot_smoothed_mse_comparison(file_on, file_off, title="Porównanie Autoenkodera: Bias ON vs Bias OFF", x_limit=3000, window=20)