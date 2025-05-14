# Trainer.py
import random
import numpy as np
import json  # Do ładniejszego zapisu list w pliku CSV


class Trainer:
    def __init__(self, mlp, training_data, learning_rate=0.1, momentum=0.0, shuffle=True):
        self.mlp = mlp
        self.training_data = [(np.array(x, dtype=float), np.array(y, dtype=float)) for x, y in training_data]
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.shuffle = shuffle

    def train(self, max_epochs=1000, error_threshold=None, log_interval=10, log_file="training_log.txt"):
        print(
            f"\nRozpoczęcie treningu: max_epochs={max_epochs}, error_threshold={error_threshold}, lr={self.learning_rate}, momentum={self.momentum}")
        with open(log_file, 'w') as log:
            log.write("Epoch,AvgError\n")

            for epoch in range(max_epochs):
                epoch_total_error = 0.0

                current_training_data = self.training_data[:]
                if self.shuffle:
                    random.shuffle(current_training_data)

                for x_pattern, y_expected in current_training_data:
                    # 1. Propagacja w przód
                    y_actual = self.mlp.forward(x_pattern)

                    # 2. Obliczenie błędu dla wzorca (do sumowania błędu epoki)
                    pattern_error = self.mlp.calculate_pattern_error(y_actual, y_expected)
                    epoch_total_error += pattern_error

                    # 3. Propagacja wsteczna i aktualizacja wag
                    # (expected_output jest używane do obliczenia gradientu w MLP.backward)
                    self.mlp.backward(y_expected, self.learning_rate, self.momentum)

                avg_epoch_error = epoch_total_error / len(self.training_data)

                if epoch % log_interval == 0 or epoch == max_epochs - 1:
                    log.write(f"{epoch},{avg_epoch_error:.8f}\n")
                    log.flush()  # Wymuś zapis na dysk
                    print(f"Epoka: {epoch}, Średni błąd (MSE): {avg_epoch_error:.8f}")

                if error_threshold is not None and avg_epoch_error <= error_threshold:
                    print(
                        f"Trening zakończony: Osiągnięto próg błędu {avg_epoch_error:.8f} <= {error_threshold} w epoce {epoch}.")
                    log.write(f"Trening zakończony w epoce {epoch}, błąd: {avg_epoch_error:.8f}\n")
                    return  # Zakończ trening

            print(
                f"Trening zakończony: Osiągnięto maksymalną liczbę epok ({max_epochs}). Finalny błąd: {avg_epoch_error:.8f}")
            log.write(f"Trening zakończony po {max_epochs} epokach, błąd: {avg_epoch_error:.8f}\n")

    def test(self, test_data, log_file="testing_log.txt", log_elements=None):
        """
        Testuje sieć i zapisuje wyniki do pliku.
        :param test_data: Dane testowe (lista krotek (x, y)).
        :param log_file: Nazwa pliku do zapisu logów.
        :param log_elements: Lista stringów określająca, co logować, np.
                             ['input', 'expected', 'output', 'pattern_error', 'output_errors',
                              'hidden_outputs', 'hidden_weights', 'hidden_biases',
                              'output_weights', 'output_biases'].
                              Jeśli None, loguje wszystko.
        """
        if log_elements is None:
            log_elements = ['input', 'expected', 'output', 'pattern_error', 'output_errors',
                            'hidden_outputs', 'hidden_weights', 'hidden_biases',
                            'output_weights', 'output_biases']

        print(f"\nRozpoczęcie testowania. Wyniki zostaną zapisane do: {log_file}")

        # Przygotowanie danych testowych (konwersja na numpy arrays)
        processed_test_data = [(np.array(x, dtype=float), np.array(y, dtype=float)) for x, y in test_data]

        with open(log_file, 'w') as log:
            header_parts = ["PatternNum"]
            if 'input' in log_elements: header_parts.append("Input")
            if 'expected' in log_elements: header_parts.append("ExpectedOutput")
            if 'output' in log_elements: header_parts.append("ActualOutput")
            if 'pattern_error' in log_elements: header_parts.append("PatternMSEError")
            if 'output_errors' in log_elements: header_parts.append("OutputNodeErrors")
            if 'hidden_outputs' in log_elements: header_parts.append("HiddenLayerOutputs")
            if 'hidden_weights' in log_elements: header_parts.append("HiddenLayerWeights")
            if 'hidden_biases' in log_elements: header_parts.append("HiddenLayerBiases")
            if 'output_weights' in log_elements: header_parts.append("OutputLayerWeights")
            if 'output_biases' in log_elements: header_parts.append("OutputLayerBiases")
            log.write(",".join(header_parts) + "\n")

            total_mse = 0
            for i, (x_pattern, y_expected) in enumerate(processed_test_data):
                # Propagacja w przód (bez uczenia)
                y_actual = self.mlp.forward(x_pattern)

                pattern_mse_error = self.mlp.calculate_pattern_error(y_actual, y_expected)
                total_mse += pattern_mse_error
                output_node_errors = (y_expected - y_actual) ** 2  # Kwadraty błędów dla każdego neuronu wyjściowego

                params = self.mlp.get_all_parameters()  # Pobierz parametry PO forward pass

                log_line_parts = [str(i + 1)]
                if 'input' in log_elements: log_line_parts.append(f"\"{json.dumps(x_pattern.tolist())}\"")
                if 'expected' in log_elements: log_line_parts.append(f"\"{json.dumps(y_expected.tolist())}\"")
                if 'output' in log_elements: log_line_parts.append(f"\"{json.dumps(y_actual.tolist())}\"")
                if 'pattern_error' in log_elements: log_line_parts.append(f"{pattern_mse_error:.8f}")
                if 'output_errors' in log_elements: log_line_parts.append(
                    f"\"{json.dumps(output_node_errors.tolist())}\"")

                if 'hidden_outputs' in log_elements: log_line_parts.append(
                    f"\"{json.dumps(params['hidden_outputs'])}\"")
                if 'hidden_weights' in log_elements: log_line_parts.append(
                    f"\"{json.dumps(params['hidden_weights'])}\"")
                if 'hidden_biases' in log_elements: log_line_parts.append(f"\"{json.dumps(params['hidden_biases'])}\"")

                if 'output_weights' in log_elements: log_line_parts.append(
                    f"\"{json.dumps(params['output_weights'])}\"")
                if 'output_biases' in log_elements: log_line_parts.append(f"\"{json.dumps(params['output_biases'])}\"")

                log.write(",".join(log_line_parts) + "\n")

                if (i + 1) % 10 == 0 or (i + 1) == len(processed_test_data):
                    print(f"Testowanie wzorca {i + 1}/{len(processed_test_data)}")

            avg_mse_test = total_mse / len(processed_test_data)
            print(f"Testowanie zakończone. Średni błąd MSE na zbiorze testowym: {avg_mse_test:.8f}")
            log.write(f"\nOverall Average MSE on Test Set: {avg_mse_test:.8f}\n")

    def calculate_classification_metrics(self, data_set, class_names=None):
        """Oblicza metryki klasyfikacji: dokładność, macierz pomyłek, precision, recall, F1."""
        y_true_labels = []
        y_pred_labels = []

        # Konwersja na numpy array
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

        num_classes = self.mlp.layer_sizes[-1] if self.mlp.layer_sizes and len(self.mlp.layer_sizes) > 0 else 0
        if not class_names:
            class_names = [f"Klasa {i}" for i in range(num_classes)]

        # Macierz pomyłek (ręcznie lub z sklearn jeśli dozwolone)
        # Dla uproszczenia, zaimplementuję podstawową macierz pomyłek
        cm = np.zeros((num_classes, num_classes), dtype=int)
        for true, pred in zip(y_true_labels, y_pred_labels):
            cm[true, pred] += 1

        print(f"\n--- Metryki Klasyfikacji ---")
        print(f"Dokładność (Accuracy): {accuracy:.4f} ({np.sum(y_true_labels == y_pred_labels)}/{len(y_true_labels)})")

        print("\nMacierz Pomyłek (Confusion Matrix):")
        # Wyświetlanie macierzy pomyłek
        header = "Pred: " + " ".join([f"{name[:3]}" for name in class_names])
        print(header)
        for i, name in enumerate(class_names):
            row_str = f"True {name[:3]}: " + " ".join([f"{cm[i, j]:3d}" for j in range(num_classes)])
            print(row_str)

        # Precision, Recall, F1-score dla każdej klasy
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

        # Uśrednione metryki (makro)
        avg_precision = np.mean(precision_list)
        avg_recall = np.mean(recall_list)
        avg_f1 = np.mean(f1_list)
        print("\nUśrednione metryki (Macro Average):")
        print(f"  Precision: {avg_precision:.4f}")
        print(f"  Recall:    {avg_recall:.4f}")
        print(f"  F1-score:  {avg_f1:.4f}")
        print("----------------------------")
        return {"accuracy": accuracy, "confusion_matrix": cm,
                "precision": precision_list, "recall": recall_list, "f1": f1_list}