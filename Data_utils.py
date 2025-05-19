# Data_utils.py
import numpy as np
from collections import defaultdict
import random

# Funkcja dzieląca zbiór danych na zbiór treningowy i testowy na podstawie etykiet klas.
def split_dataset(dataset, train_per_class=35, test_per_class=15, seed=42):
    if seed is not None:
        random.seed(seed)  # Ustawienie ziarna losowości dla powtarzalności wyników

    class_groups = defaultdict(list)
    # Grupowanie próbek według klasy (ostatni element próbki to etykieta)
    for sample in dataset:
        label = str(sample[-1])
        class_groups[label].append(sample)

    train_set = []
    test_set = []

    for label, samples in class_groups.items():
        # Sprawdzenie, czy w każdej klasie jest wystarczająco próbek
        if len(samples) < train_per_class + test_per_class:
            raise ValueError(f"Za mało próbek w klasie '{label}' (znaleziono {len(samples)}).")

        random.shuffle(samples)  # Losowe przetasowanie próbek danej klasy
        # Dodanie próbek do zbioru treningowego i testowego
        train_set.extend(samples[:train_per_class])
        test_set.extend(samples[train_per_class:train_per_class + test_per_class])

    # Przemieszanie całych zbiorów po zgrupowaniu po klasach
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set

# Funkcja wczytująca dane z pliku iris.data i konwertująca etykiety do formatu one-hot
def load_iris_data(filename="iris.data"):
    # Mapowanie nazw klas na reprezentację one-hot
    label_to_one_hot = {
        "Iris-setosa": [1.0, 0.0, 0.0],
        "Iris-versicolor": [0.0, 1.0, 0.0],
        "Iris-virginica": [0.0, 0.0, 1.0]
    }
    raw_features_list = []
    labels_list = []

    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Pominięcie pustych linii

            parts = line.split(',')
            try:
                # Konwersja cech z tekstu na liczby zmiennoprzecinkowe
                features = np.array([float(p) for p in parts[:-1]], dtype=float)
                label_str = parts[-1]

                # Dodanie cech i etykiety tylko jeśli etykieta jest rozpoznawalna
                if label_str in label_to_one_hot:
                    raw_features_list.append(features)
                    labels_list.append(np.array(label_to_one_hot[label_str], dtype=float))
            except ValueError:
                # Pominięcie błędnych linii
                print(f"Pominięto linię z powodu błędu konwersji: {line}")
                continue

    # Zwrócenie danych jako lista par (cechy, etykieta)
    dataset = list(zip(raw_features_list, labels_list))
    return dataset

# Funkcja normalizująca cechy do zakresu [0, 1]
def normalize_features(dataset):
    if not dataset:
        return []  # Obsługa przypadku pustego zbioru

    # Oddzielenie cech od etykiet
    features_only = np.array([item[0] for item in dataset])
    labels_only = [item[1] for item in dataset]

    # Wyznaczenie minimalnych i maksymalnych wartości dla każdej cechy
    min_vals = np.min(features_only, axis=0)
    max_vals = np.max(features_only, axis=0)

    # Unikanie dzielenia przez zero
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0

    # Normalizacja: (x - min) / (max - min)
    normalized_features = (features_only - min_vals) / ranges

    # Połączenie z powrotem z etykietami
    normalized_dataset = list(zip(normalized_features, labels_only))
    return normalized_dataset
