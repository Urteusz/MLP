import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os


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
    if len(data) < window:
        return data  # Za mało punktów do wygładzenia
    return np.convolve(data, np.ones(window) / window, mode='valid')


def plot_smoothed_mse(files, title, x_limit=1000, window=10, save_name=None):
    plt.figure(figsize=(10, 6))
    any_plotted = False
    for file in files:
        df = read_log_csv(file)
        if df is None or df.empty:
            continue
        epochs = df['Epoch']
        errors = df['AvgError']
        smoothed_errors = smooth(errors.values, window=window)
        smoothed_epochs = epochs.values[:len(smoothed_errors)]
        label = os.path.splitext(os.path.basename(file))[0]
        plt.plot(smoothed_epochs, smoothed_errors, label=label)
        any_plotted = True
    if not any_plotted:
        print(f"Brak danych do wykresu: {title}")
        plt.close()
        return
    plt.xlabel('Epoka')
    plt.ylabel('MSE (wygładzony)')
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x_limit)
    tick_interval = 100 if x_limit <= 1000 else 500
    plt.xticks(np.arange(0, x_limit + 1, tick_interval))
    if save_name:
        out_path = os.path.join(FIG_DIR, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Zapisano wykres: {out_path}")
    else:
        plt.show()


def plot_smoothed_mse_comparison(file_on, file_off, title, x_limit=3000, window=20, save_name=None):
    plt.figure(figsize=(10, 6))
    for file in [file_on, file_off]:
        if not os.path.isfile(file):
            print(f"Brak pliku: {file}")
            continue
        df = read_log_csv(file)
        if df is None or df.empty:
            continue
        epochs = df['Epoch']
        errors = df['AvgError']
        smoothed_errors = smooth(errors.values, window=window)
        smoothed_epochs = epochs.values[:len(smoothed_errors)]
        label = "Bias ON" if "biasTrue" in file or "bias_on" in file else "Bias OFF"
        plt.plot(smoothed_epochs, smoothed_errors, label=label)
    plt.xlabel('Epoka')
    plt.ylabel('MSE (wygładzony)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, x_limit)
    tick_interval = 100 if x_limit <= 1000 else 500
    plt.xticks(np.arange(0, x_limit + 1, tick_interval))
    if save_name:
        out_path = os.path.join(FIG_DIR, save_name)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        print(f"Zapisano wykres: {out_path}")
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
