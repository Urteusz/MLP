import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from typing import Optional


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

FIG_DIR = os.path.join('figures')
ensure_dir(FIG_DIR)


def read_log_csv(file):
    df = pd.read_csv(file, encoding='latin1')
    if 'Epoch' not in df.columns or 'AvgError' not in df.columns:
        return None
    df = df[pd.to_numeric(df['Epoch'], errors='coerce').notnull()]
    df['AvgError'] = df['AvgError'].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
    df['AvgError'] = pd.to_numeric(df['AvgError'], errors='coerce')
    df = df[df['AvgError'].notnull()]
    return df


def smooth(data, window=10):
    data = np.array(data, dtype=float)
    if window is None or window < 2:
        return data
    if len(data) < window * 2:  # jeśli bardzo mało punktów, nie wygładzaj by nie zgubić wszystkiego
        return data
    return np.convolve(data, np.ones(window) / window, mode='valid')


def _short_label(fname: str):
    base = os.path.splitext(os.path.basename(fname))[0]
    # Skróć do formatu: lrX_momY_bias(T/F)
    parts = base.split('_')
    lr = next((p for p in parts if p.startswith('lr')), '')
    mom = next((p for p in parts if p.startswith('mom')), '')
    bias = 'on' if 'biasTrue' in base else ('off' if 'biasFalse' in base else '')
    return f"{lr}_{mom}_b{bias}" if lr and mom else base


def plot_smoothed_mse(files, title, x_limit=1000, window: Optional[int]=10, save_name=None):
    plt.figure(figsize=(11, 6))
    any_plotted = False
    max_epoch_seen = 0
    for idx, file in enumerate(files):
        df = read_log_csv(file)
        if df is None or df.empty:
            continue
        epochs = df['Epoch'].astype(int)
        errors = df['AvgError'].astype(float)
        max_epoch_seen = max(max_epoch_seen, epochs.max())
        series = smooth(errors.values, window) if (window and window >= 2) else errors.values
        # Dostosuj epoki do długości serii po wygładzaniu
        if len(series) != len(epochs):
            # Przytnij epoki do długości serii (zaczynając od początku)
            epochs_adj = epochs.values[:len(series)]
        else:
            epochs_adj = epochs.values
        label = _short_label(file)
        color = plt.get_cmap('tab20')(idx % 20)
        plt.plot(epochs_adj, series, label=label, linewidth=2.0, marker='o', markersize=4, alpha=0.85)
        any_plotted = True
    if not any_plotted:
        print(f"Brak danych do wykresu: {title}")
        plt.close()
        return
    # Dynamiczny x_limit wg max epoki (z lekkim marginesem)
    dyn_limit = max_epoch_seen + 5
    plt.xlim(0, dyn_limit)
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(True, alpha=0.35)
    # Ogranicz liczbę kolumn legendy
    plt.legend(fontsize=8, ncol=min(3, (len(files)+1)//3), frameon=True)
    plt.tight_layout()
    if save_name:
        out_path_png = os.path.join(FIG_DIR, save_name)
        plt.savefig(out_path_png, dpi=160)
        print(f"Zapisano wykres: {out_path_png}")
    else:
        plt.show()


def plot_smoothed_mse_comparison(file_on, file_off, title, x_limit=3000, window=20, save_name=None):
    plt.figure(figsize=(10, 6))
    max_epoch_seen = 0
    for idx, file in enumerate([file_on, file_off]):
        if not os.path.isfile(file):
            print(f"Brak pliku: {file}")
            continue
        df = read_log_csv(file)
        if df is None or df.empty:
            continue
        epochs = df['Epoch'].astype(int)
        errors = df['AvgError'].astype(float)
        max_epoch_seen = max(max_epoch_seen, epochs.max())
        series = smooth(errors.values, window) if (window and window >= 2) else errors.values
        if len(series) != len(epochs):
            epochs_adj = epochs.values[:len(series)]
        else:
            epochs_adj = epochs.values
        label = "Bias ON" if "biasTrue" in file or "bias_on" in file else "Bias OFF"
        color = plt.get_cmap('tab10')(idx % 10)
        plt.plot(epochs_adj, series, label=label, linewidth=2.4, marker='o', markersize=5)
    dyn_limit = max_epoch_seen + 5
    plt.xlim(0, dyn_limit)
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title(title)
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    if save_name:
        out_path_png = os.path.join(FIG_DIR, save_name)
        plt.savefig(out_path_png, dpi=160)
        print(f"Zapisano wykres: {out_path_png}")
    else:
        plt.show()


# --- Nowe funkcje ---

def gather_logs():
    iris_logs = glob.glob(os.path.join('logs', 'iris', 'Iris_lr*.csv'))
    ae_logs = glob.glob(os.path.join('logs', 'autoencoder', 'Autoencoder_lr*.csv'))
    # Dodatkowo wszystkie inne csv w logs/<tryb>
    other_iris = glob.glob(os.path.join('logs', 'iris', '*.csv'))
    other_auto = glob.glob(os.path.join('logs', 'autoencoder', '*.csv'))
    # Unikalne
    iris_all = sorted(set(iris_logs + other_iris))
    ae_all = sorted(set(ae_logs + other_auto))
    return iris_all, ae_all

def final_errors_table(files):
    rows = []
    for f in files:
        df = read_log_csv(f)
        if df is None or df.empty:
            continue
        last_epoch = int(df['Epoch'].iloc[-1])
        last_error = float(df['AvgError'].iloc[-1])
        rows.append((os.path.basename(f), last_epoch, last_error))
    rows.sort(key=lambda x: x[2])
    return rows

def save_errors_report(files, name):
    rows = final_errors_table(files)
    if not rows:
        return
    out_path = os.path.join(FIG_DIR, f"{name}_final_errors.txt")
    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write("Plik;OstatniaEpoka;OstatniMSE\n")
        for r in rows:
            fh.write(f"{r[0]};{r[1]};{r[2]:.8f}\n")
    print(f"Zapisano raport błędów: {out_path}")


# === WYKRES 1: IRYSY ===
iris_files, ae_files = gather_logs()
plot_smoothed_mse(iris_files, title="MSE a Epoki Podczas Treningu (Irys)", x_limit=1000, window=10, save_name='iris_all.png')
save_errors_report(iris_files, 'iris')

# === WYKRES 2: AUTOENKODER ===
plot_smoothed_mse(ae_files, title="MSE a Epoki Podczas Treningu (Autoenkoder)", x_limit=1000, window=20, save_name='autoencoder_all.png')
save_errors_report(ae_files, 'autoencoder')

# === WYKRES 3: PORÓWNANIE AUTOENKODERA BIAS ON VS OFF ===
file_on = glob.glob(os.path.join('logs', 'autoencoder', '*lr0.5*mom0.5*biasTrue*.csv'))
file_off = glob.glob(os.path.join('logs', 'autoencoder', '*lr0.5*mom0.5*biasFalse*.csv'))

if file_on and file_off:
    plot_smoothed_mse_comparison(file_on[0], file_off[0], title="Porównanie Autoenkodera: Bias ON vs Bias OFF", x_limit=3000, window=20, save_name='autoencoder_bias_compare.png')
else:
    print("Brak kompletnej pary do porównania bias ON/OFF.")

# Regeneracja wykresów (nadpisanie poprzednich plików) tylko prostych głównych
iris_files, ae_files = gather_logs()
plot_smoothed_mse(iris_files, title="Iris – przebieg MSE", window=5, save_name='iris_all.png')
plot_smoothed_mse(ae_files, title="Autoencoder – przebieg MSE", window=5, save_name='autoencoder_all.png')
file_on = glob.glob(os.path.join('logs', 'autoencoder', '*lr0.5*mom0.5*biasTrue*.csv'))
file_off = glob.glob(os.path.join('logs', 'autoencoder', '*lr0.5*mom0.5*biasFalse*.csv'))
if file_on and file_off:
    plot_smoothed_mse_comparison(file_on[0], file_off[0], title="Autoencoder – Bias ON vs OFF", window=5, save_name='autoencoder_bias_compare.png')
else:
    print("Brak kompletnej pary do porównania bias ON/OFF.")
