# Data_utils.py
import numpy as np
from collections import defaultdict
import random

def split_dataset(dataset, train_per_class=35, test_per_class=15, seed=42):
    """
    Zakłada, że dataset to lista przykładów, gdzie ostatni element to etykieta klasy.
    Zwraca dokładnie (train_per_class + test_per_class) próbek na klasę.
    """
    if seed is not None:
        random.seed(seed)

    # Grupowanie danych wg klasy
    class_groups = defaultdict(list)
    for sample in dataset:
        label = str(sample[-1])  # <-- KLUCZOWA POPRAWKA
        class_groups[label].append(sample)

    train_set = []
    test_set = []

    for label, samples in class_groups.items():
        if len(samples) < train_per_class + test_per_class:
            raise ValueError(f"Za mało próbek w klasie '{label}' (znaleziono {len(samples)}).")

        random.shuffle(samples)
        train_set.extend(samples[:train_per_class])
        test_set.extend(samples[train_per_class:train_per_class + test_per_class])

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set



def load_iris_data(filename="iris.data"):
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