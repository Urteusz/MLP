# Data_utils.py
import numpy as np
import random


def split_dataset(dataset, train_ratio=0.7, seed=None):  # Zmieniono seed na None domyślnie
    if seed is not None:
        random.seed(seed)

    shuffled_dataset = list(dataset)  # Tworzymy kopię, żeby nie modyfikować oryginału
    random.shuffle(shuffled_dataset)

    split_point = int(len(shuffled_dataset) * train_ratio)
    train_set = shuffled_dataset[:split_point]
    test_set = shuffled_dataset[split_point:]
    return train_set, test_set


def load_iris_data(filename="iris.data"):
    data = []
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
                continue

            parts = line.split(',')
            try:
                # Pierwsze 4 cechy to floaty, ostatnia to etykieta klasy
                features = np.array([float(p) for p in parts[:-1]], dtype=float)
                label_str = parts[-1]

                if label_str in label_to_one_hot:
                    raw_features_list.append(features)
                    labels_list.append(np.array(label_to_one_hot[label_str], dtype=float))
            except ValueError:
                print(f"Pominięto linię z powodu błędu konwersji: {line}")
                continue

    # Połącz cechy i etykiety w krotki (cechy, etykieta)
    dataset = list(zip(raw_features_list, labels_list))
    return dataset


def normalize_features(dataset):
    # Rozdziel cechy i etykiety
    if not dataset:
        return []

    features_only = np.array([item[0] for item in dataset])
    labels_only = [item[1] for item in dataset]

    min_vals = np.min(features_only, axis=0)
    max_vals = np.max(features_only, axis=0)

    ranges = max_vals - min_vals
    # Zapobieganie dzieleniu przez zero, jeśli cecha ma stałą wartość
    ranges[ranges == 0] = 1.0

    normalized_features = (features_only - min_vals) / ranges

    # Połącz znormalizowane cechy z powrotem z etykietami
    normalized_dataset = list(zip(normalized_features, labels_only))
    return normalized_dataset


def load_patterns_from_file(filename):
    """
    Wczytuje wzorce z pliku. Format: ((wejscia),(wyjscia)) na linię.
    Np. ((1,0,0,0),(1,0,0,0))
    """
    patterns = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # Używamy eval, ale z ostrożnością - plik musi być zaufany
                pattern_tuple = eval(line)
                if isinstance(pattern_tuple, tuple) and len(pattern_tuple) == 2:
                    inputs = np.array(pattern_tuple[0], dtype=float)
                    outputs = np.array(pattern_tuple[1], dtype=float)
                    patterns.append((inputs, outputs))
                else:
                    print(f"Pominięto niepoprawny format linii: {line}")
            except Exception as e:
                print(f"Błąd podczas parsowania linii '{line}': {e}")
    return patterns