import random
import numpy as np
import json

class Trainer:
    def __init__(self, mlp, training_data, learning_rate=0.1, momentum=0.0):
        # Inicjalizacja trenera z siecią MLP, danymi treningowymi i parametrami uczenia
        self.mlp = mlp
        self.training_data = [(np.array(x, dtype=float), np.array(y, dtype=float)) for x, y in training_data]
        self.learning_rate = learning_rate
        self.momentum = momentum

    def train(self, max_epochs=10000, error_threshold=None, log_interval=10, log_file="training_log.txt"):
        # Trening sieci neuronowej metodą propagacji wstecznej
        print(f"\nRozpoczęcie treningu: max_epochs={max_epochs}, error_threshold={error_threshold}, "
              f"lr={self.learning_rate}, momentum={self.momentum}")

        with open(log_file, 'w') as log:
            log.write("Epoch,AvgError\n")  # Nagłówek pliku logowania

            for epoch in range(max_epochs):
                epoch_total_error = 0.0
                current_training_data = self.training_data[:]
                random.shuffle(current_training_data)  # Mieszamy dane co epokę

                for x_pattern, y_expected in current_training_data:
                    y_actual = self.mlp.forward(x_pattern)  # Przepuszczenie danych przez sieć
                    pattern_error = self.mlp.calculate_pattern_error(y_actual, y_expected)
                    epoch_total_error += pattern_error
                    self.mlp.backward(y_expected, self.learning_rate, self.momentum)  # Aktualizacja wag

                avg_epoch_error = epoch_total_error / len(self.training_data)

                # Logowanie błędu co log_interval epok
                if epoch % log_interval == 0 or epoch == max_epochs - 1:
                    log.write(f"{epoch},{avg_epoch_error:.8f}\n")
                    log.flush()

                # Sprawdzenie warunku zatrzymania
                if error_threshold is not None and avg_epoch_error <= error_threshold:
                    print(f"Trening zakończony: Osiągnięto próg błędu {avg_epoch_error:.8f} <= {error_threshold} w epoce {epoch}.")
                    log.write(f"Trening zakończony w epoce {epoch}, błąd: {avg_epoch_error:.8f}\n")
                    return

            # Komunikat jeśli osiągnięto maksymalną liczbę epok
            print(f"Trening zakończony: Osiągnięto maksymalną liczbę epok ({max_epochs}). Finalny błąd: {avg_epoch_error:.8f}")
            log.write(f"Trening zakończony po {max_epochs} epokach, błąd: {avg_epoch_error:.8f}\n")


    def calculate_classification_metrics(self, data_set, class_names=None):
        # Oblicza metryki klasyfikacji: Accuracy, Precision, Recall, F1
        y_true_labels = []
        y_pred_labels = []

        processed_data_set = [(np.array(x, dtype=float), np.array(y, dtype=float)) for x, y in data_set]

        for x_pattern, y_expected_one_hot in processed_data_set:
            y_actual_probs = self.mlp.forward(x_pattern)
            true_label = np.argmax(y_expected_one_hot)
            pred_label = np.argmax(y_actual_probs)
            y_true_labels.append(true_label)
            y_pred_labels.append(pred_label)

        y_true_labels = np.array(y_true_labels)
        y_pred_labels = np.array(y_pred_labels)
        accuracy = np.mean(y_true_labels == y_pred_labels)

        # Przygotowanie klasyfikatorów
        num_classes = self.mlp.layer_sizes[-1] if self.mlp.layer_sizes and len(self.mlp.layer_sizes) > 0 else 0
        if not class_names:
            class_names = [f"Klasa {i}" for i in range(num_classes)]

        # Obliczanie macierzy pomyłek
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true_labels, y_pred_labels):
            cm[true, pred] += 1

        print(f"\n--- Metryki Klasyfikacji ---")
        print(f"Dokładność (Accuracy): {accuracy:.4f} ({np.sum(y_true_labels == y_pred_labels)}/{len(y_true_labels)})")

        # Wyświetlenie macierzy pomyłek
        print("\nMacierz Pomyłek (Confusion Matrix):")
        header = "Pred: " + " ".join([f"{name[:3]}" for name in class_names])
        print(header)
        for i, name in enumerate(class_names):
            row_str = f"True {name[:3]}: " + " ".join([f"{cm[i, j]:3d}" for j in range(num_classes)])
            print(row_str)

        # Wyliczanie metryk per klasa
        print("\nMetryki per klasa:")
        precision_list, recall_list, f1_list = [], [], []

        for i in range(num_classes):
            TP = cm[i, i]
            FP = np.sum(cm[:, i]) - TP
            FN = np.sum(cm[i, :]) - TP

            precision = TP / (TP + FP) if (TP + FP) > 0 else 0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)

            print(f"  {class_names[i]}:")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall:    {recall:.4f}")
            print(f"    F1-score:  {f1:.4f}")

        # Wyliczanie średnich metryk
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        print("\nUśrednione metryki (Macro Average):")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  F1-score:  {avg_f1:.4f}")
        print("----------------------------")

        return {
            "accuracy": accuracy,
            "confusion_matrix": cm,
            "precision": precision_list,
            "recall": recall_list,
            "f1": f1_list
        }
