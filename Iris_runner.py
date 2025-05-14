from MLP import MLP
from Trainer import Trainer
from Data_utils import load_iris_data, split_dataset
import numpy as np
from collections import Counter

# 1. Wczytaj dane
dataset = load_iris_data("iris.data")
train_data, test_data = split_dataset(dataset, train_ratio=0.7)

# 2. Zbuduj sieć (np. 4 wejścia, 6 neuronów ukrytych, 3 wyjściowe)
mlp = MLP([4, 6, 3], use_bias=True)

# 3. Nauka
trainer = Trainer(mlp, train_data, learning_rate=0.6, momentum=0.0, shuffle=True)
trainer.train(max_epochs=300, error_threshold=0.01, log_interval=10, log_file="iris_training_log.txt")

# 4. Testowanie
def predict(x):
    output = mlp.forward(x)
    return np.argmax(output)

y_true = [np.argmax(y) for _, y in test_data]
y_pred = [predict(x) for x, _ in test_data]

# 5. Ocena
def confusion_matrix(true, pred, num_classes=3):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(true, pred):
        matrix[t][p] += 1
    return matrix

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

def precision_recall_f1(cm):
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(cm)):
        tp = cm[i][i]
        fp = sum(cm[:, i]) - tp
        fn = sum(cm[i, :]) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return precisions, recalls, f1s

precisions, recalls, f1s = precision_recall_f1(cm)

print("\nMetrics per class:")
for i in range(3):
    print(f"Class {i}: Precision={precisions[i]:.2f}, Recall={recalls[i]:.2f}, F1={f1s[i]:.2f}")

accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)
print(f"\nTotal Accuracy: {accuracy:.2%}")
