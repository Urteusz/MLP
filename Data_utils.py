import numpy as np
import random

def split_dataset(dataset, train_ratio=0.7, seed=42):
    random.seed(seed)
    shuffled = dataset[:]
    random.shuffle(shuffled)
    split_point = int(len(shuffled) * train_ratio)
    return shuffled[:split_point], shuffled[split_point:]


def load_iris_data(filename="iris.data"):
    data = []
    labels = []
    label_to_vec = {
        "Iris-setosa": [1, 0, 0],
        "Iris-versicolor": [0, 1, 0],
        "Iris-virginica": [0, 0, 1]
    }

    with open(filename, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            features = list(map(float, parts[:4]))
            label = parts[4]
            if label not in label_to_vec:
                continue
            data.append(np.array(features))
            labels.append(np.array(label_to_vec[label]))

    dataset = list(zip(data, labels))
    return dataset


def normalize_features(dataset):

    features = np.array([x for x, _ in dataset])
    min_vals = features.min(axis=0)
    max_vals = features.max(axis=0)
    ranges = max_vals - min_vals

    normalized_dataset = [
        ((x - min_vals) / ranges, y) for x, y in dataset
    ]
    return normalized_dataset


